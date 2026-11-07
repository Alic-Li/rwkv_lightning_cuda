package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestRetiredAdaptersAndStateForwarding(t *testing.T) {
	var calls atomic.Int32
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("X-Received-Path", r.URL.Path)
		io.Copy(w, r.Body)
	}))
	defer worker.Close()
	u, _ := url.Parse(worker.URL)
	l := fsAgentLauncher(t)
	l.config.Port = u.Port()
	l.backends = openRegistry(filepath.Join(t.TempDir(), "launcher.json"))
	remote, err := l.backends.add("remote", worker.URL, "")
	if err != nil {
		t.Fatal(err)
	}
	h := l.handler()
	front := httptest.NewServer(h)
	defer front.Close()
	frontURL, _ := url.Parse(front.URL)
	l.listen = frontURL.Host
	for _, prefix := range []string{"", "/api/v1/backends/local", "/api/v1/backends/" + remote.ID} {
		for _, method := range []string{"GET", "POST", "DELETE", "OPTIONS"} {
			for _, suffix := range []string{"", "/", "/legacy"} {
				w := httptest.NewRecorder()
				h.ServeHTTP(w, httptest.NewRequest(method, "http://127.0.0.1:10721"+prefix+"/v1/adapters"+suffix, strings.NewReader(`{}`)))
				if w.Code != 404 {
					t.Fatalf("%s %s: %d %s", method, prefix+suffix, w.Code, w.Body.String())
				}
			}
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("retired route reached worker: %d", calls.Load())
	}
	// State uploads, lists, deletion and per-request selection keep byte-exact forwarding.
	for _, prefix := range []string{"", "/api/v1/backends/local", "/api/v1/backends/" + remote.ID} {
		for _, path := range []string{"/v1/state/upload", "/v1/state/list", "/v1/state/delete", "/v1/batch/completions"} {
			body := `{"state_id":"preserved", "session_id":"session"}`
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest("POST", "http://127.0.0.1:10721"+prefix+path, strings.NewReader(body)))
			if w.Code != 200 || w.Body.String() != body || w.Header().Get("X-Received-Path") != path {
				t.Fatalf("State forwarding changed: %s %d %s", prefix+path, w.Code, w.Body.String())
			}
		}
	}
	if calls.Load() != 12 {
		t.Fatalf("State calls: %d", calls.Load())
	}
}
