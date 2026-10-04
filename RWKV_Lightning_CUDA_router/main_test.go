package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestBatchSize(t *testing.T) {
	tests := []struct {
		path, body string
		want       int64
	}{
		{"/v1/batch/completions", `{"contents":["a","b","c"]}`, 3},
		{"/translate/v1/batch-translate", `{"text_list":["a","b"]}`, 2},
		{"/big_batch/completions", `{"bsz":12}`, 12},
		{"/v1/chat/completions", `{"messages":[]}`, 1},
		{"/v1/models", `{"contents":["a","b"]}`, 1},
	}
	for _, tt := range tests {
		got, _ := batchSize(httptest.NewRequest(http.MethodPost, tt.path, nil), []byte(tt.body))
		if got != tt.want {
			t.Errorf("%s: got %d, want %d", tt.path, got, tt.want)
		}
	}
}

func TestBatchSizeSessionHeaderFallback(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/state/chat/completions", nil)
	req.Header.Set("X-RWKV-Session-Id", "session-header")
	if _, got := batchSize(req, []byte(`{"contents":["a"]}`)); got != "session-header" {
		t.Fatalf("header fallback got %q", got)
	}
	if _, got := batchSize(req, []byte(`{"contents":["a"],"session_id":"session-body"}`)); got != "session-body" {
		t.Fatalf("body session alone got %q", got)
	}
	reqWithout := httptest.NewRequest(http.MethodPost, "/state/chat/completions", nil)
	if _, got := batchSize(reqWithout, []byte(`{"contents":["a"]}`)); got != "" {
		t.Fatalf("no channel got %q", got)
	}
	other := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	other.Header.Set("X-RWKV-Session-Id", "session-header")
	if _, got := batchSize(other, nil); got != "" {
		t.Fatalf("non-affinity path must not read session headers, got %q", got)
	}
}

func TestWeightedLeastInflight(t *testing.T) {
	a, _ := url.Parse("http://a:8000")
	b, _ := url.Parse("http://b:8000")
	s := &scheduler{backends: []*backend{{name: "a", baseURL: a, weight: 1}, {name: "b", baseURL: b, weight: 2}}, sessions: map[string]*backend{}}
	first, releaseFirst, err := s.acquire(2, "", "")
	if err != nil || first.name != "a" {
		t.Fatalf("first=%v err=%v", first, err)
	}
	second, releaseSecond, err := s.acquire(2, "", "")
	if err != nil || second.name != "b" {
		t.Fatalf("second=%v err=%v", second, err)
	}
	releaseFirst()
	releaseSecond()
	if s.backends[0].inflight != 0 || s.backends[1].inflight != 0 {
		t.Fatal("load was not released")
	}
}

func TestSessionAffinityAndCooldown(t *testing.T) {
	a, _ := url.Parse("http://a:8000")
	b, _ := url.Parse("http://b:8000")
	s := &scheduler{backends: []*backend{{name: "a", baseURL: a, weight: 1}, {name: "b", baseURL: b, weight: 1}}, sessions: map[string]*backend{}, cooldown: time.Second}
	first, releaseFirst, _ := s.acquire(1, "session-1", "")
	releaseFirst()
	second, releaseSecond, err := s.acquire(1, "session-1", "")
	if err != nil || second != first {
		t.Fatal("session was not kept on the same backend")
	}
	releaseSecond()
	s.failed(first)
	third, releaseThird, err := s.acquire(1, "session-1", "")
	if err != nil || third == first {
		t.Fatal("unhealthy session backend was selected")
	}
	releaseThird()
}

func TestStateIDFromBody(t *testing.T) {
	if got := stateIDFromBody("/v1/batch/completions", []byte(`{"state_id":"state-test"}`)); got != "state-test" {
		t.Fatalf("got %q", got)
	}
	if got := stateIDFromBody("/v1/state/upload", []byte(`{"state_id":"ignored"}`)); got != "" {
		t.Fatalf("multipart upload body should not be parsed, got %q", got)
	}
}

func TestRequestStateID(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/batch/completions?state_id=state-query", nil)
	if got := requestStateID(req, []byte(`{}`)); got != "state-query" {
		t.Fatalf("query fallback got %q", got)
	}
	req.URL.RawQuery = ""
	req.Header.Set("X-RWKV-State-Id", "state-header")
	if got := requestStateID(req, []byte(`{}`)); got != "state-header" {
		t.Fatalf("header fallback got %q", got)
	}
	if got := requestStateID(req, []byte(`{"state_id":"state-body"}`)); got != "state-body" {
		t.Fatalf("body alone got %q", got)
	}
	upload := httptest.NewRequest(http.MethodPost, "/v1/state/upload", nil)
	upload.Header.Set("X-RWKV-State-Id", "state-header")
	if got := requestStateID(upload, nil); got != "" {
		t.Fatalf("upload must not read state ids, got %q", got)
	}
}

func TestProxySessionHeaderAffinity(t *testing.T) {
	var mu sync.Mutex
	requests := map[string]int{}
	newBackend := func(name string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			requests[name]++
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{}`)
		}))
	}
	serverA, serverB := newBackend("a"), newBackend("b")
	defer serverA.Close()
	defer serverB.Close()
	urlA, _ := url.Parse(serverA.URL)
	urlB, _ := url.Parse(serverB.URL)
	p := &proxy{
		scheduler: &scheduler{
			backends: []*backend{{name: "a", baseURL: urlA, weight: 1}, {name: "b", baseURL: urlB, weight: 1}},
			sessions: map[string]*backend{},
		},
		client: serverA.Client(),
	}
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodPost, "/state/chat/completions", strings.NewReader(`{"contents":["hi"]}`))
		req.Header.Set("X-RWKV-Session-Id", "session-header")
		resp := httptest.NewRecorder()
		p.ServeHTTP(resp, req)
		if resp.Code != http.StatusOK {
			t.Fatalf("request %d returned %d", i, resp.Code)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if requests["a"] != 2 || requests["b"] != 0 {
		t.Fatalf("header session was not pinned to one backend: %+v", requests)
	}
}

func TestProxySynchronizesUploadedState(t *testing.T) {
	requestsA := 0
	requestsB := 0
	var requestsMu sync.Mutex
	serverA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestsMu.Lock()
		requestsA++
		requestsMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/state/upload" {
			fmt.Fprintf(w, `{"state_id":"state-%s"}`, r.Header.Get("X-RWKV-State-Upload-UUID"))
			return
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	defer serverA.Close()
	serverB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestsMu.Lock()
		requestsB++
		requestsMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/state/upload" {
			fmt.Fprintf(w, `{"state_id":"state-%s"}`, r.Header.Get("X-RWKV-State-Upload-UUID"))
			return
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	defer serverB.Close()

	urlA, _ := url.Parse(serverA.URL)
	urlB, _ := url.Parse(serverB.URL)
	backendA := &backend{name: "a", baseURL: urlA, weight: 1}
	backendB := &backend{name: "b", baseURL: urlB, weight: 1}
	s := &scheduler{
		backends: []*backend{backendA, backendB},
		sessions: map[string]*backend{},
	}
	p := &proxy{scheduler: s, client: serverA.Client()}

	uploadReq := stateUploadRequest(t, "state.pth", "upload")
	uploadResp := httptest.NewRecorder()
	p.ServeHTTP(uploadResp, uploadReq)
	if uploadResp.Code != http.StatusOK || requestsA != 1 || requestsB != 1 {
		t.Fatalf("upload code=%d requestsA=%d requestsB=%d", uploadResp.Code, requestsA, requestsB)
	}

	var uploaded listedState
	json.Unmarshal(uploadResp.Body.Bytes(), &uploaded)
	inferReq := httptest.NewRequest(http.MethodPost, "/v1/batch/completions", strings.NewReader(fmt.Sprintf(`{"contents":["test"],"state_id":%q}`, uploaded.ID)))
	inferResp := httptest.NewRecorder()
	p.ServeHTTP(inferResp, inferReq)
	if inferResp.Code != http.StatusOK || requestsA != 2 || requestsB != 1 {
		t.Fatalf("inference code=%d requestsA=%d requestsB=%d", inferResp.Code, requestsA, requestsB)
	}
}

func TestProxySynchronizesStateListAndDelete(t *testing.T) {
	var mu sync.Mutex
	requests := map[string]int{}
	newBackend := func(name string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			requests[name+":"+r.URL.Path]++
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"object":"list","data":[]}`)
		}))
	}
	serverA, serverB := newBackend("a"), newBackend("b")
	defer serverA.Close()
	defer serverB.Close()
	urlA, _ := url.Parse(serverA.URL)
	urlB, _ := url.Parse(serverB.URL)
	p := &proxy{
		scheduler: &scheduler{backends: []*backend{{name: "a", baseURL: urlA, weight: 1}, {name: "b", baseURL: urlB, weight: 1}}},
		client:    serverA.Client(),
	}
	for _, request := range []*http.Request{
		httptest.NewRequest(http.MethodGet, "/v1/state/list", nil),
		httptest.NewRequest(http.MethodDelete, "/v1/state/delete?state_id=state.pth", nil),
	} {
		response := httptest.NewRecorder()
		p.ServeHTTP(response, request)
		if response.Code != http.StatusOK {
			t.Fatalf("%s returned %d", request.URL.Path, response.Code)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	for _, path := range []string{"/v1/state/list", "/v1/state/delete"} {
		if requests["a:"+path] != 1 || requests["b:"+path] != 1 {
			t.Fatalf("%s was not sent to every backend: %+v", path, requests)
		}
	}
}

func TestStateUploadUUIDSharedAcrossWorkers(t *testing.T) {
	original := stateUploadRequest(t, "state01.pth", "original-file-bytes")
	originalBody, _ := io.ReadAll(original.Body)
	contentType := original.Header.Get("Content-Type")
	var mu sync.Mutex
	var ids []string
	handler := func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-RWKV-State-Upload-UUID")
		mu.Lock()
		ids = append(ids, id)
		mu.Unlock()
		data, _ := io.ReadAll(r.Body)
		if !bytes.Equal(data, originalBody) {
			t.Error("upload body changed")
		}
		if r.Header.Get("Content-Type") != contentType {
			t.Error("content type changed")
		}
		fmt.Fprintf(w, `{"state_id":"state01-%s","size_bytes":123,"created_ms":1}`, id)
	}
	a := httptest.NewServer(http.HandlerFunc(handler))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(handler))
	defer b.Close()
	ua, _ := url.Parse(a.URL)
	ub, _ := url.Parse(b.URL)
	p := &proxy{scheduler: &scheduler{backends: []*backend{{name: "a", baseURL: ua, weight: 1}, {name: "b", baseURL: ub, weight: 1}}, sessions: map[string]*backend{}}, client: a.Client()}
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/state/upload", bytes.NewReader(originalBody))
		req.Header.Set("Content-Type", contentType)
		req.Header.Set("X-RWKV-State-Upload-UUID", "caller-value-must-be-replaced")
		resp := httptest.NewRecorder()
		p.ServeHTTP(resp, req)
		if resp.Code != 200 {
			t.Fatalf("upload returned %d: %s", resp.Code, resp.Body.String())
		}
		if !strings.Contains(resp.Body.String(), `"size_bytes":123`) {
			t.Error("metadata lost")
		}
	}
	if len(ids) != 4 || ids[0] != ids[1] || ids[2] != ids[3] || ids[0] == ids[2] {
		t.Fatalf("UUIDs not shared per upload: %v", ids)
	}
	pattern := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)
	if !pattern.MatchString(ids[0]) {
		t.Fatalf("invalid UUID-v7: %s", ids[0])
	}
	stamp, err := strconv.ParseInt(strings.ReplaceAll(ids[0][:13], "-", ""), 16, 64)
	if err != nil || time.Now().UnixMilli()-stamp > 5000 {
		t.Fatalf("invalid UUID timestamp: %d %v", stamp, err)
	}
}

func TestStateListConsistency(t *testing.T) {
	tests := []struct {
		name, a, b string
		want       int
	}{
		{"order and timestamps", `{"object":"list","data":[{"state_id":"a","size_bytes":1,"created_ms":1},{"state_id":"b","size_bytes":2}]}`, `{"object":"list","data":[{"state_id":"b","size_bytes":2},{"state_id":"a","size_bytes":1,"created_ms":2}]}`, 200},
		{"missing state", `{"object":"list","data":[{"state_id":"a"}]}`, `{"object":"list","data":[]}`, 502},
		{"size mismatch", `{"object":"list","data":[{"state_id":"a","size_bytes":1}]}`, `{"object":"list","data":[{"state_id":"a","size_bytes":2}]}`, 502},
		{"shape mismatch", `{"object":"list","data":[{"state_id":"a","heads":1}]}`, `{"object":"list","data":[{"state_id":"a","heads":2}]}`, 502},
		{"invalid list", `{"object":"list","data":[]}`, `{}`, 502},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, tc.a) }))
			defer a.Close()
			b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, tc.b) }))
			defer b.Close()
			ua, _ := url.Parse(a.URL)
			ub, _ := url.Parse(b.URL)
			p := &proxy{scheduler: &scheduler{backends: []*backend{{name: "a", baseURL: ua, weight: 1}, {name: "b", baseURL: ub, weight: 1}}}, client: a.Client()}
			resp := httptest.NewRecorder()
			p.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/v1/state/list", nil))
			if resp.Code != tc.want {
				t.Fatalf("status %d, want %d: %s", resp.Code, tc.want, resp.Body.String())
			}
		})
	}
}

func stateUploadRequest(t *testing.T, filename, content string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	part, err := form.CreateFormFile("file", filename)
	if err != nil {
		t.Fatal(err)
	}
	part.Write([]byte(content))
	form.Close()
	req := httptest.NewRequest(http.MethodPost, "/v1/state/upload", bytes.NewReader(body.Bytes()))
	req.Header.Set("Content-Type", form.FormDataContentType())
	return req
}

func TestProxyPreservesFinishReasons(t *testing.T) {
	for _, tc := range []struct{ contentType, body string }{
		{"application/json", `{"choices":[{"index":0,"finish_reason":"stop"},{"index":1,"finish_reason":"length"}]}`},
		{"text/event-stream", "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n"},
	} {
		t.Run(tc.contentType, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tc.contentType)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer server.Close()
			baseURL, _ := url.Parse(server.URL)
			p := &proxy{scheduler: &scheduler{backends: []*backend{{name: "test", baseURL: baseURL, weight: 1}}, sessions: map[string]*backend{}}, client: server.Client()}
			req := httptest.NewRequest(http.MethodPost, "/v1/batch/completions", strings.NewReader(`{"contents":["hi"]}`))
			resp := httptest.NewRecorder()
			p.ServeHTTP(resp, req)
			if resp.Code != http.StatusOK || resp.Body.String() != tc.body {
				t.Fatalf("finish reasons changed: status=%d body=%q", resp.Code, resp.Body.String())
			}
		})
	}
}
