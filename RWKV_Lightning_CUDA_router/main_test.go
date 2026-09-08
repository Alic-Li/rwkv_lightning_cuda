package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
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
		got, _ := batchSize(tt.path, []byte(tt.body))
		if got != tt.want {
			t.Errorf("%s: got %d, want %d", tt.path, got, tt.want)
		}
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
			_, _ = io.WriteString(w, `{"state_id":"state-uploaded"}`)
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
			_, _ = io.WriteString(w, `{"state_id":"state-uploaded"}`)
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

	uploadReq := httptest.NewRequest(http.MethodPost, "/v1/state/upload", strings.NewReader("upload"))
	uploadResp := httptest.NewRecorder()
	p.ServeHTTP(uploadResp, uploadReq)
	if uploadResp.Code != http.StatusOK || requestsA != 1 || requestsB != 1 {
		t.Fatalf("upload code=%d requestsA=%d requestsB=%d", uploadResp.Code, requestsA, requestsB)
	}

	inferReq := httptest.NewRequest(
		http.MethodPost,
		"/v1/batch/completions",
		strings.NewReader(`{"contents":["test"],"state_id":"state-uploaded"}`))
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
