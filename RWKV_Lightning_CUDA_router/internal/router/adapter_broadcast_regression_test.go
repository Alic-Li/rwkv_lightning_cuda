package router

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// Reproduce: upload reports 200, a later malformed/failed list reports 502,
// then the old router forgets all successful replicas and returns 503 inference.
func TestAdapterFailedListsPreserveBroadcastAcknowledgments(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{"gateway", 502, `<html>Bad Gateway</html>`},
		{"malformed", 200, `{"data":`},
		{"wrong-schema", 200, `{"unexpected":[]}`},
		{"unauthorized", 401, `{"error":"Unauthorized"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var uploads, inferences, lists atomic.Int32
			servers := make([]*httptest.Server, 3)
			for i := range servers {
				servers[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/v1/adapters" {
						if r.Method == "POST" {
							uploads.Add(1)
							fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
							return
						}
						lists.Add(1)
						w.WriteHeader(tc.status)
						io.WriteString(w, tc.body)
						return
					}
					inferences.Add(1)
					io.WriteString(w, `{"choices":[]}`)
				}))
				defer servers[i].Close()
			}
			p := adapterProxy(servers...)
			w := httptest.NewRecorder()
			p.ServeHTTP(w, adapterUploadRequest(t, "miss"))
			if w.Code != 200 || uploads.Load() != 3 {
				t.Fatalf("upload %d %s calls=%d", w.Code, w.Body.String(), uploads.Load())
			}
			listed := adapterCall(p, "GET", "/v1/adapters", "")
			expected := 502
			if tc.status == 401 {
				expected = 401
			}
			if listed.Code != expected {
				t.Fatal(listed.Code, listed.Body.String())
			}
			for i := 0; i < 9; i++ {
				if w := inferAdapter(p, "miss", "", ""); w.Code != 200 {
					t.Fatalf("lost successful upload: %d %s", w.Code, w.Body.String())
				}
			}
			if inferences.Load() != 9 || lists.Load() != 3 {
				t.Fatalf("unnecessary rediscovery: inferences=%d lists=%d", inferences.Load(), lists.Load())
			}
		})
	}
}

func TestAdapterAuthoritativeEmptyListStillRevokesReadiness(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
			return
		}
		io.WriteString(w, `{"data":[]}`)
	}))
	defer server.Close()
	p := adapterProxy(server)
	if w := adapterCall(p, "POST", "/v1/adapters", `{"adapter_id":"miss","path":"/shared"}`); w.Code != 200 {
		t.Fatal(w.Code)
	}
	if w := adapterCall(p, "GET", "/v1/adapters", ""); w.Code != 200 {
		t.Fatal(w.Code)
	}
	if w := inferAdapter(p, "miss", "", ""); w.Code != 503 {
		t.Fatal("actual absence was ignored", w.Code, w.Body.String())
	}
}

func TestAdapterListMismatchIdentifiesBackendAndVersion(t *testing.T) {
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"data":[{"id":"miss","version":%q}]}`, adapterV1)
	}))
	defer a.Close()
	b := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, `{"data":[]}`) }))
	defer b.Close()
	p := adapterProxy(a, b)
	w := adapterCall(p, "GET", "/v1/adapters", "")
	if w.Code != 502 {
		t.Fatal(w.Code)
	}
	var result struct {
		Failed []struct{ Name, Error string } `json:"failed_backends"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Failed) != 1 || result.Failed[0].Name != "node-1" || !strings.Contains(result.Failed[0].Error, "miss@"+adapterV1) || !strings.Contains(result.Failed[0].Error, "missing=") {
		t.Fatal(w.Body.String())
	}
}

// Real HTTP sockets on both sides of the router; each worker independently
// registers the uploaded ID and verifies it when accepting inference.
func TestAdapterBroadcastListAndConcurrentSessionAffinityOverHTTP(t *testing.T) {
	var mu sync.Mutex
	uploaded := make([]bool, 3)
	requestCount := make([]int, 3)
	servers := make([]*httptest.Server, 3)
	for i := range servers {
		servers[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			if r.URL.Path == "/v1/adapters" {
				if r.Method == "POST" {
					if err := r.ParseMultipartForm(1 << 20); err != nil {
						http.Error(w, err.Error(), 400)
						return
					}
					defer r.MultipartForm.RemoveAll()
					if r.FormValue("adapter_id") != "miss" {
						http.Error(w, "missing id", 400)
						return
					}
					files := r.MultipartForm.File["file"]
					if len(files) != 2 {
						http.Error(w, "missing files", 400)
						return
					}
					mu.Lock()
					uploaded[i] = true
					mu.Unlock()
					fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
					return
				}
				mu.Lock()
				ready := uploaded[i]
				mu.Unlock()
				if ready {
					fmt.Fprintf(w, `{"data":[{"id":"miss","version":%q}],"gpu_hits":%d}`, adapterV1, i)
				} else {
					io.WriteString(w, `{"data":[]}`)
				}
				return
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), 400)
				return
			}
			mu.Lock()
			defer mu.Unlock()
			if !uploaded[i] || body["adapter_id"] != "miss" || body["adapter_version"] != adapterV1 {
				http.Error(w, "unknown adapter", 400)
				return
			}
			requestCount[i]++
			// An upstream must not be able to spoof the router-selected worker header.
			w.Header().Set("X-RWKV-Backend", "spoofed")
			fmt.Fprintf(w, `{"worker":%d}`, i)
		}))
		defer servers[i].Close()
	}
	router := httptest.NewServer(adapterProxy(servers...))
	defer router.Close()
	upload := adapterUploadRequest(t, "miss")
	req, _ := http.NewRequest("POST", router.URL+"/v1/adapters", upload.Body)
	req.Header = upload.Header
	response, err := router.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, response.Body)
	response.Body.Close()
	if response.StatusCode != 200 {
		t.Fatal(response.StatusCode)
	}
	for i := 0; i < 6; i++ {
		response, err = router.Client().Get(router.URL + "/v1/adapters")
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(response.Body)
		response.Body.Close()
		if response.StatusCode != 200 {
			t.Fatalf("list %d %s", response.StatusCode, data)
		}
	}
	var wg sync.WaitGroup
	owners := make(chan string, 9)
	for i := 0; i < 9; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			request, _ := http.NewRequest("POST", router.URL+"/v1/batch/completions", strings.NewReader(`{"contents":["hello"],"adapter_id":"miss"}`))
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("X-RWKV-Session-Id", "same-session")
			response, err := router.Client().Do(request)
			if err != nil {
				t.Error(err)
				return
			}
			io.Copy(io.Discard, response.Body)
			response.Body.Close()
			if response.StatusCode != 200 {
				t.Error(response.StatusCode)
				return
			}
			owners <- response.Header.Get("X-RWKV-Backend")
		}()
	}
	wg.Wait()
	close(owners)
	owner := ""
	count := 0
	for value := range owners {
		count++
		if value == "" || value == "spoofed" {
			t.Fatal("missing selection header", value)
		}
		if owner == "" {
			owner = value
		} else if owner != value {
			t.Fatalf("session moved between workers: %s / %s", owner, value)
		}
	}
	if count != 9 {
		t.Fatal("missing successful requests", count)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, ready := range uploaded {
		if !ready {
			t.Fatal("broadcast omitted worker", uploaded)
		}
	}
	used := 0
	for _, count := range requestCount {
		if count > 0 {
			used++
		}
	}
	if used != 1 {
		t.Fatal("session was not pinned", requestCount)
	}
}

func TestAdapterTransportFailureDoesNotForgetConfirmedRegistration(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/adapters" {
			t.Error("unnecessary list rediscovery")
			http.Error(w, "list unavailable", 502)
			return
		}
		io.Copy(io.Discard, r.Body)
		if calls.Add(1) == 1 {
			conn, _, _ := w.(http.Hijacker).Hijack()
			conn.Close()
			return
		}
		io.WriteString(w, `{}`)
	}))
	defer server.Close()
	p := adapterProxy(server)
	p.scheduler.cooldown = 0
	p.scheduler.adapters = map[string]*adapterPlacement{"miss": {current: adapterV1, ready: map[string]map[*backend]bool{adapterV1: {p.scheduler.backends[0]: true}}}}
	if w := inferAdapter(p, "miss", "", ""); w.Code != 502 {
		t.Fatal(w.Code, w.Body.String())
	}
	if w := inferAdapter(p, "miss", "", ""); w.Code != 200 {
		t.Fatal("forgot registration on transport failure", w.Code, w.Body.String())
	}
	if calls.Load() != 2 {
		t.Fatal("unexpected retry", calls.Load())
	}
}
