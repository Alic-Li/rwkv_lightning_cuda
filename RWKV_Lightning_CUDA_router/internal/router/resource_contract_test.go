package router

import (
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

// Run the same HTTP contract against both resource types. Each worker has its
// own store; an upload accepted by one worker cannot mask a missing replica.
type resourceContract struct{ kind, listPath, deletePath string }

var resourceContracts = []resourceContract{
	{"State", "/v1/state/list", "/v1/state/delete"},
	{"Adapter", "/v1/adapters", "/v1/adapters"},
}

func (c resourceContract) upload(t *testing.T) *http.Request {
	if c.kind == "State" {
		return stateUploadRequest(t, "state.pth", "checkpoint")
	}
	return adapterUploadRequest(t, "miss")
}
func (c resourceContract) ack(w http.ResponseWriter, r *http.Request) string {
	id := "miss"
	if c.kind == "State" {
		id = "state-" + r.Header.Get("X-RWKV-State-Upload-UUID")
		fmt.Fprintf(w, `{"state_id":%q}`, id)
	} else {
		fmt.Fprintf(w, `{"adapter_id":"miss","version":%q}`, adapterV1)
	}
	return id
}
func (c resourceContract) list(w http.ResponseWriter, id string) {
	if c.kind == "State" {
		fmt.Fprintf(w, `{"object":"list","data":[{"state_id":%q}]}`, id)
	} else {
		fmt.Fprintf(w, `{"data":[{"id":"miss","version":%q}]}`, adapterV1)
	}
}
func (c resourceContract) body(id string) string {
	if c.kind == "State" {
		return fmt.Sprintf(`{"contents":["hello"],"state_id":%q}`, id)
	}
	return `{"contents":["hello"],"adapter_id":"miss"}`
}
func contractUpload(t *testing.T, p *proxy, c resourceContract) string {
	t.Helper()
	w := httptest.NewRecorder()
	p.ServeHTTP(w, c.upload(t))
	if w.Code != 200 {
		t.Fatalf("upload: %d %s", w.Code, w.Body.String())
	}
	var payload map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if c.kind == "State" {
		return payload["state_id"]
	}
	return payload["adapter_id"]
}

func TestResourceFailedObservationsPreserveConfirmedReplicas(t *testing.T) {
	for _, c := range resourceContracts {
		t.Run(c.kind, func(t *testing.T) {
			for _, bad := range []struct {
				name   string
				status int
				body   string
			}{
				{"gateway", 502, "upstream unavailable"}, {"auth", 401, `{"error":"Unauthorized"}`},
				{"malformed", 200, `{"data":`}, {"schema", 200, `{"other":[]}`},
			} {
				t.Run(bad.name, func(t *testing.T) {
					var uploads, inferences, lists atomic.Int32
					servers := make([]*httptest.Server, 3)
					for i := range servers {
						servers[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							if r.Method == "POST" && (r.URL.Path == "/v1/state/upload" || r.URL.Path == "/v1/adapters") {
								uploads.Add(1)
								c.ack(w, r)
								return
							}
							if r.URL.Path == c.listPath {
								lists.Add(1)
								w.WriteHeader(bad.status)
								io.WriteString(w, bad.body)
								return
							}
							inferences.Add(1)
							io.WriteString(w, `{}`)
						}))
						defer servers[i].Close()
					}
					p := adapterProxy(servers...)
					id := contractUpload(t, p, c)
					list := adapterCall(p, "GET", c.listPath, "")
					expected := 502
					if bad.status == 401 {
						expected = 401
					}
					if list.Code != expected {
						t.Fatal(list.Code, list.Body.String())
					}
					for range 9 {
						if w := adapterCall(p, "POST", "/v1/batch/completions", c.body(id)); w.Code != 200 {
							t.Fatal(w.Code, w.Body.String())
						}
					}
					if uploads.Load() != 3 || lists.Load() != 3 || inferences.Load() != 9 {
						t.Fatalf("upload/list/inference counts: %d/%d/%d", uploads.Load(), lists.Load(), inferences.Load())
					}
				})
			}
		})
	}
}

func TestResourceDiscoveryPreservesJSONAuthentication(t *testing.T) {
	for _, c := range resourceContracts {
		t.Run(c.kind, func(t *testing.T) {
			var lists atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == c.listPath {
					var body map[string]string
					if json.NewDecoder(r.Body).Decode(&body) != nil || body["password"] != "secret" || len(body) != 1 {
						t.Error("discovery lost authentication or forwarded inference body")
						w.WriteHeader(401)
						return
					}
					if r.Header.Get("Authorization") != "Bearer header" {
						t.Error("authorization header lost")
					}
					lists.Add(1)
					c.list(w, "existing")
					return
				}
				io.WriteString(w, `{}`)
			}))
			defer server.Close()
			p := adapterProxy(server)
			payload := strings.TrimSuffix(c.body("existing"), "}") + `,"password":"secret"}`
			req := httptest.NewRequest("POST", "/v1/batch/completions", strings.NewReader(payload))
			req.Header.Set("Authorization", "Bearer header")
			w := httptest.NewRecorder()
			p.ServeHTTP(w, req)
			if w.Code != 200 || lists.Load() != 1 {
				t.Fatal(w.Code, w.Body.String(), lists.Load())
			}
		})
	}
}

func TestResourceDeletionRequiresAcknowledgmentAndStaysDeleted(t *testing.T) {
	for _, c := range resourceContracts {
		t.Run(c.kind, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "DELETE" {
					io.WriteString(w, `{"deleted":false}`)
					return
				}
				c.ack(w, r)
			}))
			defer server.Close()
			p := adapterProxy(server)
			id := contractUpload(t, p, c)
			w := adapterCall(p, "DELETE", c.deletePath, c.body(id))
			if w.Code != 502 || !strings.Contains(w.Body.String(), "confirm resource deletion") {
				t.Fatal(w.Code, w.Body.String())
			}
			if w = adapterCall(p, "POST", "/v1/batch/completions", c.body(id)); w.Code != 503 {
				t.Fatal("uncertain deletion resurrected", w.Code)
			}
		})
	}
}

func TestResourceManagementTimeoutAndReservationRelease(t *testing.T) {
	for _, c := range resourceContracts {
		t.Run(c.kind, func(t *testing.T) {
			for _, method := range []string{"GET", "DELETE"} {
				t.Run(method, func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.Copy(io.Discard, r.Body); <-r.Context().Done() }))
					defer server.Close()
					p := adapterProxy(server)
					p.uploadPolicy.AttemptTimeout = 20 * time.Millisecond
					p.adapterUploadPolicy.AttemptTimeout = 20 * time.Millisecond
					path := c.listPath
					if method == "DELETE" {
						path = c.deletePath
					}
					started := time.Now()
					w := adapterCall(p, method, path, c.body("existing"))
					if w.Code != 502 || time.Since(started) > time.Second {
						t.Fatal("management request not bounded", w.Code)
					}
					for _, b := range p.scheduler.backends {
						if b.inflight != 0 {
							t.Fatal("reservation leaked")
						}
					}
				})
			}
		})
	}
}

func TestResourceConcurrentDeletionConflicts(t *testing.T) {
	for _, c := range resourceContracts {
		t.Run(c.kind, func(t *testing.T) {
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "DELETE" {
					once.Do(func() { close(entered) })
					<-release
					io.WriteString(w, `{"deleted":true}`)
					return
				}
				c.ack(w, r)
			}))
			defer server.Close()
			var releaseOnce sync.Once
			defer releaseOnce.Do(func() { close(release) })
			p := adapterProxy(server)
			id := contractUpload(t, p, c)
			done := make(chan *httptest.ResponseRecorder, 1)
			go func() { done <- adapterCall(p, "DELETE", c.deletePath, c.body(id)) }()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("delete not started")
			}
			if w := adapterCall(p, "DELETE", c.deletePath, c.body(id)); w.Code != 409 {
				t.Fatal("concurrent deletion accepted", w.Code)
			}
			if w := adapterCall(p, "POST", "/v1/batch/completions", c.body(id)); w.Code != 503 {
				t.Fatal("inference during deletion", w.Code)
			}
			releaseOnce.Do(func() { close(release) })
			if w := <-done; w.Code != 200 {
				t.Fatal(w.Code, w.Body.String())
			}
		})
	}
}

func TestStatePOSTManagementAndOptionsCompatibility(t *testing.T) {
	var deletes atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(204)
			return
		}
		switch r.URL.Path {
		case "/v1/state/upload":
			resourceContracts[0].ack(w, r)
		case "/v1/state/list":
			io.WriteString(w, `{"object":"list","data":[]}`)
		case "/v1/state/delete":
			deletes.Add(1)
			io.WriteString(w, `{"deleted":true}`)
		}
	}))
	defer server.Close()
	p := testStateProxy(server)
	id := contractUpload(t, p, resourceContracts[0])
	for _, path := range []string{"/v1/state/upload", "/v1/state/list", "/v1/state/delete"} {
		if w := adapterCall(p, "OPTIONS", path, ""); w.Code != 204 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	p.scheduler.mu.Lock()
	ready := p.scheduler.stateReadyLocked(id, p.scheduler.backends[0])
	p.scheduler.mu.Unlock()
	if !ready {
		t.Fatal("preflight changed readiness")
	}
	for _, path := range []string{"/v1/state/list", "/v1/state/delete"} {
		if w := adapterCall(p, "POST", path, resourceContracts[0].body(id)); w.Code != 200 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	if deletes.Load() != 1 {
		t.Fatal("delete not forwarded")
	}
}

func TestResourceTransportFailurePreservesRegistrationWithoutReplaying(t *testing.T) {
	for _, c := range resourceContracts {
		t.Run(c.kind, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/state/upload" || (r.URL.Path == "/v1/adapters" && r.Method == "POST") {
					c.ack(w, r)
					return
				}
				if r.URL.Path == c.listPath {
					t.Error("unnecessary rediscovery")
					w.WriteHeader(502)
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
			id := contractUpload(t, p, c)
			if w := adapterCall(p, "POST", "/v1/batch/completions", c.body(id)); w.Code != 502 {
				t.Fatal(w.Code)
			}
			if w := adapterCall(p, "POST", "/v1/batch/completions", c.body(id)); w.Code != 200 {
				t.Fatal(w.Code, w.Body.String())
			}
			if calls.Load() != 2 {
				t.Fatal("inference replayed", calls.Load())
			}
		})
	}
}

func TestStateAndAdapterBroadcastAndSessionAffinityOverHTTP(t *testing.T) {
	var mu sync.Mutex
	stateIDs := make([]string, 3)
	adapters := make([]bool, 3)
	inferences := make([]int, 3)
	servers := make([]*httptest.Server, 3)
	for i := range servers {
		servers[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			defer mu.Unlock()
			switch r.URL.Path {
			case "/v1/state/upload":
				if err := r.ParseMultipartForm(1 << 20); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				defer r.MultipartForm.RemoveAll()
				file, _, err := r.FormFile("file")
				if err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				defer file.Close()
				bytes, _ := io.ReadAll(file)
				if string(bytes) != "checkpoint" {
					t.Error("state bytes changed")
				}
				stateIDs[i] = resourceContracts[0].ack(w, r)
			case "/v1/state/list":
				resourceContracts[0].list(w, stateIDs[i])
			case "/v1/adapters":
				if r.Method == "POST" {
					if err := r.ParseMultipartForm(1 << 20); err != nil {
						t.Error(err)
						w.WriteHeader(400)
						return
					}
					defer r.MultipartForm.RemoveAll()
					if r.FormValue("adapter_id") != "miss" || len(r.MultipartForm.File["file"]) != 2 {
						t.Error("adapter payload incomplete")
					}
					resourceContracts[1].ack(w, r)
					adapters[i] = true
				} else {
					resourceContracts[1].list(w, "miss")
				}
			default:
				var payload struct {
					StateID   string `json:"state_id"`
					AdapterID string `json:"adapter_id"`
					Version   string `json:"adapter_version"`
				}
				if json.NewDecoder(r.Body).Decode(&payload) != nil || payload.StateID != stateIDs[i] || stateIDs[i] == "" || !adapters[i] || payload.AdapterID != "miss" || payload.Version != adapterV1 {
					http.Error(w, "resource missing", 400)
					return
				}
				inferences[i]++
				io.WriteString(w, `{}`)
			}
		}))
		defer servers[i].Close()
	}
	p := adapterProxy(servers...)
	router := httptest.NewServer(p)
	defer router.Close()
	client := router.Client()
	send := func(req *http.Request) (int, http.Header, []byte) {
		req.RequestURI = ""
		target, _ := url.Parse(router.URL + req.URL.Path)
		req.URL = target
		response, err := client.Do(req)
		if err != nil {
			t.Error(err)
			return 0, nil, nil
		}
		defer response.Body.Close()
		data, err := io.ReadAll(response.Body)
		if err != nil {
			t.Error(err)
		}
		return response.StatusCode, response.Header, data
	}
	var id string
	for _, c := range resourceContracts {
		status, _, data := send(c.upload(t))
		if status != 200 {
			t.Fatalf("upload: %d %s", status, data)
		}
		if c.kind == "State" {
			var payload map[string]string
			json.Unmarshal(data, &payload)
			id = payload["state_id"]
		}
	}
	for range 6 {
		for _, c := range resourceContracts {
			status, _, data := send(httptest.NewRequest("GET", c.listPath, nil))
			if status != 200 {
				t.Fatalf("list: %d %s", status, data)
			}
		}
	}
	var wg sync.WaitGroup
	selected := make(chan string, 9)
	for range 9 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest("POST", "/v1/batch/completions", strings.NewReader(fmt.Sprintf(`{"contents":["hello"],"state_id":%q,"adapter_id":"miss"}`, id)))
			req.Header.Set("X-RWKV-Session-Id", "same-session")
			status, headers, data := send(req)
			if status != 200 {
				t.Errorf("inference: %d %s", status, data)
			}
			selected <- headers.Get("X-RWKV-Backend")
		}()
	}
	wg.Wait()
	close(selected)
	first := ""
	for backend := range selected {
		if first == "" {
			first = backend
		}
		if backend == "" || backend != first {
			t.Fatal("session affinity lost", first, backend)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	total := 0
	for i, n := range inferences {
		if stateIDs[i] != id || !adapters[i] {
			t.Fatal("worker missing resource", i)
		}
		total += n
		if n != 0 && n != 9 {
			t.Fatal("session split across workers", inferences)
		}
	}
	if total != 9 {
		t.Fatal(inferences)
	}
}

type acknowledgmentObserver struct {
	*httptest.ResponseRecorder
	observe func(int)
}

func (w acknowledgmentObserver) WriteHeader(status int) {
	w.observe(status)
	w.ResponseRecorder.WriteHeader(status)
}

func TestResourceReadinessPublishedBeforeSuccessfulAcknowledgment(t *testing.T) {
	for _, c := range resourceContracts {
		t.Run(c.kind, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { c.ack(w, r) }))
			defer server.Close()
			p := adapterProxy(server)
			observed := false
			w := acknowledgmentObserver{ResponseRecorder: httptest.NewRecorder(), observe: func(status int) {
				observed = true
				if status != 200 {
					t.Errorf("upload status %d", status)
					return
				}
				p.scheduler.mu.Lock()
				defer p.scheduler.mu.Unlock()
				b := p.scheduler.backends[0]
				if c.kind == "State" {
					for id := range p.scheduler.states {
						if !p.scheduler.stateReadyLocked(id, b) {
							t.Error("upload acknowledged before State became routable")
						}
					}
				} else if !p.scheduler.adapterReadyLocked("miss", adapterV1, b) {
					t.Error("upload acknowledged before adapter became routable")
				}
			}}
			p.ServeHTTP(w, c.upload(t))
			if !observed {
				t.Fatal("no acknowledgment")
			}
		})
	}
}
