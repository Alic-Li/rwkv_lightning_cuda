package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const adapterV1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const adapterV2 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

func adapterProxy(servers ...*httptest.Server) *proxy {
	p := testStateProxy(servers...)
	p.adapterUploadPolicy = uploadRetryPolicy{Attempts: 3, AttemptTimeout: time.Second, BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond}
	return p
}

func adapterUploadRequest(t *testing.T, id string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	if err := form.WriteField("adapter_id", id); err != nil {
		t.Fatal(err)
	}
	for name, payload := range map[string]string{"training.pth": "PTH binary\x00bytes", "checkpoint.json": `{"method":"miss"}`} {
		part, err := form.CreateFormFile("file", name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(part, payload); err != nil {
			t.Fatal(err)
		}
	}
	if err := form.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/adapters", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	req.Header.Set("Authorization", "Bearer test-secret")
	return req
}

func adapterCall(p *proxy, method, path, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	p.ServeHTTP(w, r)
	return w
}

func inferAdapter(p *proxy, id, version, stateID string) *httptest.ResponseRecorder {
	body, _ := json.Marshal(map[string]any{"contents": []string{"hello"}, "adapter_id": id,
		"adapter_version": version, "state_id": stateID})
	return adapterCall(p, "POST", "/v1/batch/completions", string(body))
}

func TestAdapterMultipartBroadcastAndNineConcurrentRequests(t *testing.T) {
	var mu sync.Mutex
	uploads, inferences := make([]int, 3), make([]int, 3)
	var original []byte
	servers := make([]*httptest.Server, 3)
	for i := range servers {
		servers[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			mu.Lock()
			defer mu.Unlock()
			if r.URL.Path == "/v1/adapters" {
				uploads[i]++
				if r.Method != "POST" || r.Header.Get("Authorization") != "Bearer test-secret" {
					t.Error("request headers/method lost")
				}
				if original == nil {
					original = body
				} else if !bytes.Equal(original, body) {
					t.Error("multipart bytes changed")
				}
				fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
				return
			}
			var payload map[string]any
			json.Unmarshal(body, &payload)
			if payload["adapter_version"] != adapterV1 {
				t.Errorf("version not pinned: %s", body)
			}
			inferences[i]++
			io.WriteString(w, `{"choices":[]}`)
		}))
		defer servers[i].Close()
	}
	p := adapterProxy(servers...)
	w := httptest.NewRecorder()
	p.ServeHTTP(w, adapterUploadRequest(t, "miss"))
	if w.Code != 200 {
		t.Fatalf("upload: %d %s", w.Code, w.Body.String())
	}
	var wg sync.WaitGroup
	for i := 0; i < 9; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := inferAdapter(p, "miss", "", "")
			if w.Code != 200 {
				t.Errorf("inference: %d %s", w.Code, w.Body.String())
			}
		}()
	}
	wg.Wait()
	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(uploads, []int{1, 1, 1}) {
		t.Fatal(uploads)
	}
	total := 0
	for _, n := range inferences {
		total += n
		if n == 0 {
			t.Fatal("backend not used", inferences)
		}
	}
	if total != 9 {
		t.Fatal(inferences)
	}
	for _, b := range p.scheduler.backends {
		if b.inflight != 0 {
			t.Fatal("reservation leaked")
		}
	}
}

func TestAdapterRetriesOnlyFailedBackendAndSameBytes(t *testing.T) {
	for _, lost := range []bool{false, true} {
		t.Run(fmt.Sprint("lost-response=", lost), func(t *testing.T) {
			var aCalls, bCalls atomic.Int32
			var original []byte
			a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				aCalls.Add(1)
				fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
			}))
			defer a.Close()
			b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				data, _ := io.ReadAll(r.Body)
				n := bCalls.Add(1)
				if n == 1 {
					original = data
					if lost {
						conn, _, _ := w.(http.Hijacker).Hijack()
						conn.Close()
					} else {
						w.WriteHeader(502)
					}
					return
				}
				if !bytes.Equal(data, original) {
					t.Error("retry changed body")
				}
				fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
			}))
			defer b.Close()
			p := adapterProxy(a, b)
			w := httptest.NewRecorder()
			p.ServeHTTP(w, adapterUploadRequest(t, "miss"))
			if w.Code != 200 || aCalls.Load() != 1 || bCalls.Load() != 2 {
				t.Fatalf("%d %s calls=%d/%d", w.Code, w.Body.String(), aCalls.Load(), bCalls.Load())
			}
		})
	}
}

func TestAdapterPartialUploadNeverRoutesToMissingReplica(t *testing.T) {
	var callsA, callsB atomic.Int32
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/adapters" {
			fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
			return
		}
		callsA.Add(1)
		io.WriteString(w, `{}`)
	}))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/adapters" {
			w.WriteHeader(503)
			return
		}
		callsB.Add(1)
		http.Error(w, "unknown adapter id", 400)
	}))
	defer b.Close()
	p := adapterProxy(a, b)
	w := httptest.NewRecorder()
	p.ServeHTTP(w, adapterUploadRequest(t, "miss"))
	if w.Code != 502 || !strings.Contains(w.Body.String(), `"ready_backends":["node-0"]`) || !strings.Contains(w.Body.String(), `"name":"node-1"`) {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	p.scheduler.mu.Lock()
	p.scheduler.backends[1].unhealthy = time.Time{}
	p.scheduler.sessions["pinned"] = p.scheduler.backends[1]
	p.scheduler.mu.Unlock()
	for i := 0; i < 9; i++ {
		if w := adapterCall(p, "POST", "/state/chat/completions", `{"session_id":"pinned","contents":["hello"],"adapter_id":"miss"}`); w.Code != 200 {
			t.Fatalf("%d %s", w.Code, w.Body.String())
		}
	}
	if callsA.Load() != 9 || callsB.Load() != 0 {
		t.Fatalf("inference %d/%d", callsA.Load(), callsB.Load())
	}
}

func TestAdapterInconsistentVersionsReturnFailureAndPinCanonical(t *testing.T) {
	var callsB atomic.Int32
	servers := make([]*httptest.Server, 2)
	for i, v := range []string{adapterV1, adapterV2} {
		servers[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/v1/adapters" {
				fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, v)
				return
			}
			if i == 1 {
				callsB.Add(1)
			}
			io.WriteString(w, `{}`)
		}))
		defer servers[i].Close()
	}
	p := adapterProxy(servers...)
	w := adapterCall(p, "POST", "/v1/adapters", `{"adapter_id":"miss","path":"/shared/training.pth"}`)
	if w.Code != 502 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	for i := 0; i < 5; i++ {
		if w := inferAdapter(p, "miss", "", ""); w.Code != 200 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	if callsB.Load() != 0 {
		t.Fatal("inconsistent content used")
	}
}

func TestAdapterListComparesIdentityNotMetricsOrOrder(t *testing.T) {
	var inconsistent atomic.Bool
	makeServer := func(index int) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			data := []map[string]any{{"id": "a", "version": adapterV1, "manifest": map[string]any{"method": "miss"}}, {"id": "b", "version": adapterV2, "manifest": map[string]any{"method": "miss"}}}
			if index == 1 {
				data[0], data[1] = data[1], data[0]
				if inconsistent.Load() {
					data = data[:1]
				}
			}
			json.NewEncoder(w).Encode(map[string]any{"data": data, "gpu_hits": index, "ram_bytes": index * 1024})
		}))
	}
	a, b := makeServer(0), makeServer(1)
	defer a.Close()
	defer b.Close()
	p := adapterProxy(a, b)
	if w := adapterCall(p, "GET", "/v1/adapters", ""); w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	inconsistent.Store(true)
	if w := adapterCall(p, "GET", "/v1/adapters", ""); w.Code != 502 {
		t.Fatal("different lists accepted", w.Code, w.Body.String())
	}
}

func TestAdapterRestartDiscoversOnlyReadyVersions(t *testing.T) {
	var inferA, inferB atomic.Int32
	makeServer := func(index int) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/v1/adapters" {
				if index == 0 {
					fmt.Fprintf(w, `{"data":[{"id":"miss","version":%q}]}`, adapterV1)
				} else {
					io.WriteString(w, `{"data":[]}`)
				}
				return
			}
			if index == 0 {
				inferA.Add(1)
			} else {
				inferB.Add(1)
			}
			io.WriteString(w, `{}`)
		}))
	}
	a, b := makeServer(0), makeServer(1)
	defer a.Close()
	defer b.Close()
	p := adapterProxy(a, b)
	for i := 0; i < 9; i++ {
		if w := inferAdapter(p, "miss", "", ""); w.Code != 200 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	if inferA.Load() != 9 || inferB.Load() != 0 {
		t.Fatalf("%d/%d", inferA.Load(), inferB.Load())
	}
	if w := inferAdapter(p, "miss", adapterV2, ""); w.Code != 503 {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestAdapterMultipleDiscoveredVersionsRequireExplicitVersion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/adapters" {
			fmt.Fprintf(w, `{"data":[{"id":"miss","version":%q},{"id":"miss","version":%q}]}`, adapterV1, adapterV2)
			return
		}
		io.WriteString(w, `{}`)
	}))
	defer server.Close()
	p := adapterProxy(server)
	if w := inferAdapter(p, "miss", "", ""); w.Code != 503 {
		t.Fatal(w.Code, w.Body.String())
	}
	if w := inferAdapter(p, "miss", adapterV2, ""); w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestAdapterAndStateMustBeReadyOnSameBackend(t *testing.T) {
	var inference atomic.Int32
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { inference.Add(1); io.WriteString(w, `{}`) }))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { inference.Add(1); io.WriteString(w, `{}`) }))
	defer b.Close()
	p := adapterProxy(a, b)
	p.scheduler.states = map[string]*statePlacement{"state": {ready: map[*backend]bool{p.scheduler.backends[0]: true}}}
	p.scheduler.adapters = map[string]*adapterPlacement{"miss": {current: adapterV1, ready: map[string]map[*backend]bool{adapterV1: {p.scheduler.backends[1]: true}}}}
	if w := inferAdapter(p, "miss", "", "state"); w.Code != 503 {
		t.Fatal(w.Code, w.Body.String())
	}
	if inference.Load() != 0 {
		t.Fatal("routed without both resources")
	}
}

func TestAdapterValidationAndOptions(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		if r.Method == "OPTIONS" {
			w.WriteHeader(204)
			return
		}
		http.Error(w, "invalid PTH", 400)
	}))
	defer server.Close()
	p := adapterProxy(server)
	if w := adapterCall(p, "POST", "/v1/adapters", `{"adapter_id":"miss","path":"/bad"}`); w.Code != 400 {
		t.Fatal(w.Code, w.Body.String())
	}
	if attempts.Load() != 1 {
		t.Fatal("validation error retried")
	}
	for _, body := range []string{`{}`, `{"adapter_id":1}`, `{"adapter_id":null}`} {
		if w := adapterCall(p, "POST", "/v1/adapters", body); w.Code != 400 {
			t.Fatal(body, w.Code, w.Body.String())
		}
	}
	if w := adapterCall(p, "OPTIONS", "/v1/adapters", ""); w.Code != 204 {
		t.Fatal(w.Code)
	}
}

func TestAdapterDeleteBroadcastAndNeverResurrectPartialDeletion(t *testing.T) {
	for _, version := range []string{"", adapterV1} {
		t.Run("version="+version, func(t *testing.T) {
			var deletesA, deletesB atomic.Int32
			makeServer := func(index int) *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method == "POST" {
						fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
						return
					}
					if r.Method == "DELETE" {
						if index == 0 {
							deletesA.Add(1)
							io.WriteString(w, `{"deleted":true}`)
						} else {
							deletesB.Add(1)
							w.WriteHeader(503)
						}
						return
					}
					fmt.Fprintf(w, `{"data":[{"id":"miss","version":%q}]}`, adapterV1)
				}))
			}
			a, b := makeServer(0), makeServer(1)
			defer a.Close()
			defer b.Close()
			p := adapterProxy(a, b)
			if w := adapterCall(p, "POST", "/v1/adapters", `{"adapter_id":"miss","path":"/shared"}`); w.Code != 200 {
				t.Fatal(w.Code)
			}
			body, _ := json.Marshal(map[string]string{"adapter_id": "miss", "adapter_version": version})
			if w := adapterCall(p, "DELETE", "/v1/adapters", string(body)); w.Code != 502 {
				t.Fatal(w.Code, w.Body.String())
			}
			if deletesA.Load() != 1 || deletesB.Load() != 1 {
				t.Fatal("delete not broadcast")
			}
			adapterCall(p, "GET", "/v1/adapters", "")
			if w := inferAdapter(p, "miss", adapterV1, ""); w.Code != 503 {
				t.Fatal("deleted adapter resurrected", w.Code, w.Body.String())
			}
		})
	}
}

func TestAdapterRejectedMutationRestoresOldReadiness(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/adapters" {
			http.Error(w, "Unauthorized", 401)
			return
		}
		io.WriteString(w, `{}`)
	}))
	defer server.Close()
	for _, method := range []string{"POST", "DELETE"} {
		t.Run(method, func(t *testing.T) {
			p := adapterProxy(server)
			p.scheduler.adapters = map[string]*adapterPlacement{"miss": {current: adapterV1, ready: map[string]map[*backend]bool{adapterV1: {p.scheduler.backends[0]: true}}}}
			w := adapterCall(p, method, "/v1/adapters", `{"adapter_id":"miss","path":"/a"}`)
			if w.Code != 401 {
				t.Fatal(w.Code, w.Body.String())
			}
			if w := inferAdapter(p, "miss", "", ""); w.Code != 200 {
				t.Fatal("readiness not restored", w.Code, w.Body.String())
			}
		})
	}
}

func TestAdapterUploadConcurrentMutationAndStaleList(t *testing.T) {
	entered, proceed := make(chan struct{}), make(chan struct{})
	var once sync.Once
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		once.Do(func() { close(entered) })
		select {
		case <-proceed:
		case <-r.Context().Done():
			return
		}
		fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
	}))
	defer server.Close()
	var release sync.Once
	defer release.Do(func() { close(proceed) })
	p := adapterProxy(server)
	epoch := p.scheduler.adapterSnapshot()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- adapterCall(p, "POST", "/v1/adapters", `{"adapter_id":"miss","path":"/shared"}`) }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("upload not started")
	}
	if w := adapterCall(p, "POST", "/v1/adapters", `{"adapter_id":"miss","path":"/other"}`); w.Code != 409 {
		t.Fatal(w.Code, w.Body.String())
	}
	if w := inferAdapter(p, "miss", adapterV1, ""); w.Code != 503 {
		t.Fatal("mutation admitted inference", w.Code)
	}
	release.Do(func() { close(proceed) })
	if w := <-done; w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	p.scheduler.observeAdapterList(p.scheduler.backends[0], resourceResponse{status: 200, body: []byte(`{"data":[]}`)}, epoch)
	p.scheduler.mu.Lock()
	ready := p.scheduler.adapterReadyLocked("miss", adapterV1, p.scheduler.backends[0])
	p.scheduler.mu.Unlock()
	if !ready {
		t.Fatal("stale list erased uploaded readiness")
	}
}

func TestAdapterDiscoveryRetainsJSONPassword(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["password"] != "test-password" {
			http.Error(w, "Unauthorized", 401)
			return
		}
		if r.URL.Path == "/v1/adapters" {
			fmt.Fprintf(w, `{"data":[{"id":"miss","version":%q}]}`, adapterV1)
			return
		}
		if body["adapter_version"] != adapterV1 {
			t.Error("version not pinned")
		}
		io.WriteString(w, `{}`)
	}))
	defer server.Close()
	p := adapterProxy(server)
	w := adapterCall(p, "POST", "/v1/batch/completions", `{"adapter_id":"miss","password":"test-password","contents":["hi"]}`)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestAdapterSSEStopFieldsAndScalePassThrough(t *testing.T) {
	const response = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\",\"matched_stop\":\"</tool_call>\"}]}\n\ndata: [DONE]\n\n"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["adapter_scale"] != 0.5 || body["adapter_version"] != adapterV1 || !reflect.DeepEqual(body["stop"], []any{"</tool_call>"}) {
			t.Errorf("payload altered: %v", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, response)
	}))
	defer server.Close()
	p := adapterProxy(server)
	p.scheduler.adapters = map[string]*adapterPlacement{"miss": {current: adapterV1, ready: map[string]map[*backend]bool{adapterV1: {p.scheduler.backends[0]: true}}}}
	w := adapterCall(p, "POST", "/v1/chat/completions", `{"adapter_id":"miss","adapter_scale":0.5,"stop":["</tool_call>"],"stream":true}`)
	if w.Code != 200 || w.Body.String() != response || !w.Flushed {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestAdapterMissingRuntimeReplicaIsRemovedWithoutReplayingInference(t *testing.T) {
	var aCalls, bCalls atomic.Int32
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		aCalls.Add(1)
		http.Error(w, "unknown adapter version", 400)
	}))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { bCalls.Add(1); io.WriteString(w, `{}`) }))
	defer b.Close()
	p := adapterProxy(a, b)
	p.scheduler.adapters = map[string]*adapterPlacement{"miss": {current: adapterV1, ready: map[string]map[*backend]bool{adapterV1: {p.scheduler.backends[0]: true, p.scheduler.backends[1]: true}}}}
	if w := inferAdapter(p, "miss", "", ""); w.Code != 400 {
		t.Fatal(w.Code)
	}
	if aCalls.Load() != 1 || bCalls.Load() != 0 {
		t.Fatal("inference was replayed")
	}
	for i := 0; i < 5; i++ {
		if w := inferAdapter(p, "miss", "", ""); w.Code != 200 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	if aCalls.Load() != 1 || bCalls.Load() != 5 {
		t.Fatal("missing version was still routed")
	}
}

func TestAdapterUploadCancellationReleasesEveryReservation(t *testing.T) {
	var attempts atomic.Int32
	entered := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		io.Copy(io.Discard, r.Body)
		close(entered)
		<-r.Context().Done()
	}))
	defer server.Close()
	p := adapterProxy(server)
	req := adapterUploadRequest(t, "miss")
	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	req = req.WithContext(ctx)
	done := make(chan struct{})
	go func() { p.ServeHTTP(httptest.NewRecorder(), req); close(done) }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("upload not started")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cancelled operation stuck")
	}
	if attempts.Load() != 1 {
		t.Fatal("cancelled upload retried")
	}
	p.scheduler.mu.Lock()
	defer p.scheduler.mu.Unlock()
	if p.scheduler.backends[0].inflight != 0 || p.scheduler.adapters["miss"].busy {
		t.Fatal("reservation/mutation leaked")
	}
}

func TestAdapterUploadConfigValidation(t *testing.T) {
	for _, c := range []config{{AdapterUploadMaxAttempts: -1}, {AdapterUploadMaxAttempts: 11}, {AdapterUploadAttemptTimeoutSeconds: -1}, {AdapterUploadRetryBaseMS: 1000, AdapterUploadRetryMaxMS: 1}} {
		if _, err := configuredAdapterUploadPolicy(c); err == nil {
			t.Fatal("invalid policy accepted", c)
		}
	}
	if policy, err := configuredAdapterUploadPolicy(config{}); err != nil || policy.Attempts != 3 {
		t.Fatal(policy, err)
	}
}

func TestAdapterDeletionRequiresAcknowledgment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, `{"deleted":false}`) }))
	defer server.Close()
	p := adapterProxy(server)
	if w := adapterCall(p, "DELETE", "/v1/adapters", `{"adapter_id":"miss"}`); w.Code != 502 {
		t.Fatal(w.Code, w.Body.String())
	}
}
