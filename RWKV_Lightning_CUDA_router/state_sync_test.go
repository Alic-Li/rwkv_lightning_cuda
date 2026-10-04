package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func testStateProxy(servers ...*httptest.Server) *proxy {
	s := &scheduler{sessions: map[string]*backend{}, cooldown: time.Millisecond}
	for i, server := range servers {
		u, _ := url.Parse(server.URL)
		s.backends = append(s.backends, &backend{name: fmt.Sprintf("node-%d", i), baseURL: u, weight: 1})
	}
	return &proxy{scheduler: s, client: servers[0].Client(), uploadPolicy: uploadRetryPolicy{attempts: 3, attemptTimeout: time.Second, baseDelay: 20 * time.Millisecond, maxDelay: 40 * time.Millisecond}}
}

func inferWithState(p *proxy, id, session string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/batch/completions", strings.NewReader(fmt.Sprintf(`{"contents":["hello"],"state_id":%q}`, id)))
	if session != "" {
		req.Header.Set("X-RWKV-Session-Id", session)
	}
	resp := httptest.NewRecorder()
	p.ServeHTTP(resp, req)
	return resp
}

func awaitReady(t *testing.T, p *proxy, id string, b *backend) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		p.scheduler.mu.Lock()
		ready := p.scheduler.stateReadyLocked(id, b)
		p.scheduler.mu.Unlock()
		if ready {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("State replica did not become ready")
}

func TestUploadRetryKeepsPendingReplicaOutOfInference(t *testing.T) {
	var healthyUploads, unstableUploads, inferA, inferB atomic.Int32
	stateID := make(chan string, 1)
	retryEntered := make(chan struct{})
	allowRetry := make(chan struct{})
	var allowOnce sync.Once

	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/state/upload" {
			healthyUploads.Add(1)
			fmt.Fprintf(w, `{"state_id":"state-%s"}`, r.Header.Get("X-RWKV-State-Upload-UUID"))
			return
		}
		inferA.Add(1)
		io.WriteString(w, `{}`)
	}))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/state/upload" {
			n := unstableUploads.Add(1)
			id := "state-" + r.Header.Get("X-RWKV-State-Upload-UUID")
			if n == 1 {
				stateID <- id
				w.WriteHeader(503)
				return
			}
			close(retryEntered)
			select {
			case <-allowRetry:
			case <-r.Context().Done():
				return
			}
			fmt.Fprintf(w, `{"state_id":%q}`, id)
			return
		}
		inferB.Add(1)
		io.WriteString(w, `{}`)
	}))
	defer b.Close()
	p := testStateProxy(a, b)
	defer allowOnce.Do(func() { close(allowRetry) })
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		resp := httptest.NewRecorder()
		p.ServeHTTP(resp, stateUploadRequest(t, "state.pth", "payload"))
		done <- resp
	}()
	var id string
	select {
	case id = <-stateID:
	case <-time.After(time.Second):
		t.Fatal("upload not started")
	}
	select {
	case <-retryEntered:
	case <-time.After(time.Second):
		t.Fatal("retry not started")
	}
	awaitReady(t, p, id, p.scheduler.backends[0])
	p.scheduler.mu.Lock()
	p.scheduler.sessions["pinned-to-b"] = p.scheduler.backends[1]
	p.scheduler.backends[1].unhealthy = time.Time{}
	p.scheduler.mu.Unlock()
	for i := 0; i < 4; i++ {
		if resp := inferWithState(p, id, "pinned-to-b"); resp.Code != 200 {
			t.Fatalf("inference: %d %s", resp.Code, resp.Body.String())
		}
	}
	if inferA.Load() != 4 || inferB.Load() != 0 {
		t.Fatalf("pending replica received State inference: a=%d b=%d", inferA.Load(), inferB.Load())
	}
	allowOnce.Do(func() { close(allowRetry) })
	select {
	case resp := <-done:
		if resp.Code != 200 {
			t.Fatalf("upload: %d %s", resp.Code, resp.Body.String())
		}
	case <-time.After(time.Second):
		t.Fatal("upload stuck")
	}
	if healthyUploads.Load() != 1 || unstableUploads.Load() != 2 {
		t.Fatal("retried a successful replica or missed failed replica")
	}
	p.scheduler.mu.Lock()
	p.scheduler.next = 1
	p.scheduler.mu.Unlock()
	if resp := inferWithState(p, id, ""); resp.Code != 200 || inferB.Load() != 1 {
		t.Fatal("confirmed replica did not rejoin State routing")
	}
}

func TestUploadLostResponseRetriesSameUUIDAndBytes(t *testing.T) {
	var attempts atomic.Int32
	var firstUUID, firstBody string
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		uuid := r.Header.Get("X-RWKV-State-Upload-UUID")
		mu.Lock()
		first := attempts.Add(1) == 1
		if first {
			firstUUID = uuid
			firstBody = string(body)
			mu.Unlock()
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			conn.Close()
			return
		}
		if uuid != firstUUID || string(body) != firstBody {
			t.Error("retry changed upload identity/content")
		}
		mu.Unlock()
		fmt.Fprintf(w, `{"state_id":"state-%s","created_ms":123}`, uuid)
	}))
	defer server.Close()
	p := testStateProxy(server)
	resp := httptest.NewRecorder()
	p.ServeHTTP(resp, stateUploadRequest(t, "state.pth", "same-content"))
	if resp.Code != 200 || attempts.Load() != 2 || !strings.Contains(resp.Body.String(), `"created_ms":123`) {
		t.Fatalf("lost response recovery: %d attempts=%d body=%s", resp.Code, attempts.Load(), resp.Body.String())
	}
}

func TestExhaustedReplicaStaysExcludedAfterCooldown(t *testing.T) {
	var attemptsB, inferB atomic.Int32
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/state/upload" {
			fmt.Fprintf(w, `{"state_id":"state-%s"}`, r.Header.Get("X-RWKV-State-Upload-UUID"))
			return
		}
		io.WriteString(w, `{}`)
	}))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/state/upload" {
			attemptsB.Add(1)
			w.WriteHeader(503)
			return
		}
		inferB.Add(1)
		io.WriteString(w, `{}`)
	}))
	defer b.Close()
	p := testStateProxy(a, b)
	p.uploadPolicy.attempts = 2
	resp := httptest.NewRecorder()
	p.ServeHTTP(resp, stateUploadRequest(t, "state.pth", "payload"))
	var result struct {
		ID     string   `json:"state_id"`
		Ready  []string `json:"ready_backends"`
		Failed []any    `json:"failed_backends"`
	}
	json.Unmarshal(resp.Body.Bytes(), &result)
	if resp.Code != 502 || result.ID == "" || len(result.Ready) != 1 || len(result.Failed) != 1 || attemptsB.Load() != 2 {
		t.Fatalf("partial response: %d %s", resp.Code, resp.Body.String())
	}
	p.scheduler.mu.Lock()
	p.scheduler.backends[1].unhealthy = time.Time{}
	p.scheduler.sessions["bad"] = p.scheduler.backends[1]
	p.scheduler.mu.Unlock()
	for i := 0; i < 6; i++ {
		if r := inferWithState(p, result.ID, "bad"); r.Code != 200 {
			t.Fatalf("healthy replica not usable: %d", r.Code)
		}
	}
	if inferB.Load() != 0 {
		t.Fatal("cooldown expiry re-enabled unconfirmed State")
	}
	other := p.scheduler.beginStateMutation("other", false)
	for _, node := range p.scheduler.backends {
		p.scheduler.confirmState(node, listedState{ID: "other"})
	}
	p.scheduler.finishStateMutation(other, false)
	p.scheduler.mu.Lock()
	p.scheduler.next = 1
	p.scheduler.mu.Unlock()
	if r := inferWithState(p, "other", ""); r.Code != 200 || inferB.Load() != 1 {
		t.Fatal("State-specific isolation blocked an unrelated ready State")
	}
}

func TestUnknownStateProbesPresenceAndFailsClosed(t *testing.T) {
	var inferA, inferB atomic.Int32
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/state/list" {
			io.WriteString(w, `{"object":"list","data":[]}`)
			return
		}
		inferA.Add(1)
		io.WriteString(w, `{}`)
	}))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/state/list" {
			if r.Header.Get("Authorization") != "Bearer secret" {
				t.Error("probe lost runtime authorization")
			}
			io.WriteString(w, `{"object":"list","data":[{"state_id":"existing","size_bytes":10}]}`)
			return
		}
		inferB.Add(1)
		io.WriteString(w, `{}`)
	}))
	defer b.Close()
	p := testStateProxy(a, b)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"state_id":"existing"}`))
	req.Header.Set("Authorization", "Bearer secret")
	resp := httptest.NewRecorder()
	p.ServeHTTP(resp, req)
	if resp.Code != 200 || inferA.Load() != 0 || inferB.Load() != 1 {
		t.Fatal("unknown State not routed to confirmed replica")
	}
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"state_id":"missing"}`))
	req.Header.Set("Authorization", "Bearer secret")
	resp = httptest.NewRecorder()
	p.ServeHTTP(resp, req)
	if resp.Code != 503 || inferA.Load() != 0 || inferB.Load() != 1 {
		t.Fatal("missing State reached inference")
	}
}

func TestStateUploadPermanentFailureDoesNotRetry(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(400)
		io.WriteString(w, `{"error":"invalid PTH"}`)
	}))
	defer server.Close()
	p := testStateProxy(server)
	resp := httptest.NewRecorder()
	p.ServeHTTP(resp, stateUploadRequest(t, "state.pth", "invalid"))
	if resp.Code != 400 || attempts.Load() != 1 || resp.Body.String() != `{"error":"invalid PTH"}` {
		t.Fatalf("validation error retried: %d %s", resp.Code, resp.Body.String())
	}
}

func TestStateUploadAttemptTimeoutCanRecover(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		if attempts.Add(1) == 1 {
			<-r.Context().Done()
			return
		}
		fmt.Fprintf(w, `{"state_id":"state-%s"}`, r.Header.Get("X-RWKV-State-Upload-UUID"))
	}))
	defer server.Close()
	p := testStateProxy(server)
	p.uploadPolicy.attemptTimeout = 40 * time.Millisecond
	resp := httptest.NewRecorder()
	p.ServeHTTP(resp, stateUploadRequest(t, "state.pth", "payload"))
	if resp.Code != 200 || attempts.Load() != 2 {
		t.Fatalf("attempt timeout recovery: %d attempts=%d", resp.Code, attempts.Load())
	}
}

func TestStateUploadCancellationStopsBackoff(t *testing.T) {
	var attempts atomic.Int32
	first := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { attempts.Add(1); close(first); w.WriteHeader(503) }))
	defer server.Close()
	p := testStateProxy(server)
	p.uploadPolicy.baseDelay = time.Second
	p.uploadPolicy.maxDelay = time.Second
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() {
		p.ServeHTTP(httptest.NewRecorder(), stateUploadRequest(t, "state.pth", "payload").WithContext(ctx))
		close(done)
	}()
	<-first
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cancelled upload stuck in backoff")
	}
	if attempts.Load() != 1 {
		t.Fatal("cancelled upload retried")
	}
}

func TestStateReadinessIgnoresStaleListsAndDeletion(t *testing.T) {
	b := &backend{name: "b"}
	s := &scheduler{backends: []*backend{b}}
	old := s.stateSnapshot()
	mutation := s.beginStateMutation("state", false)
	s.confirmState(b, listedState{ID: "state"})
	s.finishStateMutation(mutation, false)
	s.observeStateList(b, stateResponse{status: 200, body: []byte(`{"object":"list","data":[]}`)}, old)
	s.mu.Lock()
	ready := s.stateReadyLocked("state", b)
	s.mu.Unlock()
	if !ready {
		t.Fatal("stale list erased confirmed upload")
	}
	mutation = s.beginStateMutation("state", true)
	s.finishStateMutation(mutation, false)
	s.observeStateList(b, stateResponse{status: 200, body: []byte(`{"object":"list","data":[{"state_id":"state"}]}`)}, s.stateSnapshot())
	s.mu.Lock()
	ready = s.stateReadyLocked("state", b)
	s.mu.Unlock()
	if ready {
		t.Fatal("deleted State re-enabled by an orphan copy")
	}
}

func TestUploadRetryPolicyAndRetryAfter(t *testing.T) {
	p, err := configuredUploadPolicy(config{})
	if err != nil || p.attempts != 3 || p.attemptTimeout != 120*time.Second {
		t.Fatal("incorrect defaults")
	}
	for _, c := range []config{{StateUploadMaxAttempts: -1}, {StateUploadMaxAttempts: 11}, {StateUploadRetryBaseMS: 1000, StateUploadRetryMaxMS: 500}} {
		if _, err := configuredUploadPolicy(c); err == nil {
			t.Fatal("invalid policy accepted")
		}
	}
	for i := 0; i < 10; i++ {
		delay := p.delay(i)
		if delay <= 0 || delay > p.maxDelay {
			t.Fatal("backoff out of bounds")
		}
	}
	if retryAfter(http.Header{"Retry-After": []string{"2"}}) != 2*time.Second {
		t.Fatal("Retry-After ignored")
	}
	if retryAfter(http.Header{"Retry-After": []string{"invalid"}}) != 0 {
		t.Fatal("invalid Retry-After accepted")
	}
}

func TestRejectedDeletePreservesStateReadiness(t *testing.T) {
	var deletes, inferences atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/state/delete" {
			deletes.Add(1)
			w.WriteHeader(401)
			return
		}
		inferences.Add(1)
		io.WriteString(w, `{}`)
	}))
	defer server.Close()
	p := testStateProxy(server)
	mutation := p.scheduler.beginStateMutation("existing", false)
	p.scheduler.confirmState(p.scheduler.backends[0], listedState{ID: "existing"})
	p.scheduler.finishStateMutation(mutation, false)
	req := httptest.NewRequest(http.MethodDelete, "/v1/state/delete?state_id=existing", nil)
	resp := httptest.NewRecorder()
	p.ServeHTTP(resp, req)
	if resp.Code != 401 {
		t.Fatal("runtime rejection not preserved")
	}
	if resp := inferWithState(p, "existing", ""); resp.Code != 200 {
		t.Fatal("rejected delete erased readiness")
	}
	req = httptest.NewRequest(http.MethodDelete, "/v1/state/delete?state_id=existing", strings.NewReader(`{"state_id":"different"}`))
	resp = httptest.NewRecorder()
	p.ServeHTTP(resp, req)
	if resp.Code != 400 || deletes.Load() != 1 || inferences.Load() != 1 {
		t.Fatal("conflicting deletion mutated State routing")
	}
}

func TestListStartedDuringUploadCannotEraseCompletion(t *testing.T) {
	b := &backend{name: "b"}
	s := &scheduler{backends: []*backend{b}}
	mutation := s.beginStateMutation("state", false)
	during := s.stateSnapshot()
	s.confirmState(b, listedState{ID: "state"})
	s.finishStateMutation(mutation, false)
	s.observeStateList(b, stateResponse{status: 200, body: []byte(`{"object":"list","data":[]}`)}, during)
	s.mu.Lock()
	ready := s.stateReadyLocked("state", b)
	s.mu.Unlock()
	if !ready {
		t.Fatal("list started during upload erased later confirmation")
	}
}

func TestRuntimeLostStateInvalidatesReplicaWithoutInferenceRetry(t *testing.T) {
	var callsA, callsB atomic.Int32
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callsA.Add(1)
		w.WriteHeader(400)
		io.WriteString(w, `{"error":"uploaded state not found: existing"}`)
	}))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { callsB.Add(1); io.WriteString(w, `{}`) }))
	defer b.Close()
	p := testStateProxy(a, b)
	mutation := p.scheduler.beginStateMutation("existing", false)
	for _, node := range p.scheduler.backends {
		p.scheduler.confirmState(node, listedState{ID: "existing"})
	}
	p.scheduler.finishStateMutation(mutation, false)
	if resp := inferWithState(p, "existing", ""); resp.Code != 400 || callsA.Load() != 1 || callsB.Load() != 0 {
		t.Fatal("failed inference was silently replayed")
	}
	for i := 0; i < 4; i++ {
		if resp := inferWithState(p, "existing", ""); resp.Code != 200 {
			t.Fatal("remaining replica was not used")
		}
	}
	if callsA.Load() != 1 || callsB.Load() != 4 {
		t.Fatal("lost State replica remained eligible")
	}
}
