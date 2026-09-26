package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestValidateBaseURL(t *testing.T) {
	for _, ok := range []string{"http://100.64.0.10:18766", "https://node.example.com", "http://127.0.0.1:10721/"} {
		if _, err := validateBaseURL(ok); err != nil {
			t.Errorf("rejected %q: %v", ok, err)
		}
	}
	for _, bad := range []string{"", "100.64.0.10:18766", "ftp://x", "http://100.64.0.10:18766/v1", "http://100.64.0.10:18766/api"} {
		if _, err := validateBaseURL(bad); err == nil {
			t.Errorf("accepted %q", bad)
		}
	}
}

func TestRegistryPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "launcher.json")
	rg := openRegistry(path)
	e, err := rg.add("4090 节点", "http://100.64.0.10:18766", "abc123")
	if err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" {
		// Windows fakes permission bits; the 0600 guarantee is POSIX-only.
		if perm := st.Mode().Perm(); perm != 0o600 {
			t.Fatalf("registry file mode: %o", perm)
		}
	}
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "abc123") {
		t.Fatal("token must be persisted (client-side store)")
	}

	rg2 := openRegistry(path)
	if got, _ := rg2.lookup(e.ID); got.Token != "abc123" || got.BaseURL != "http://100.64.0.10:18766" {
		t.Fatalf("reload: %+v", got)
	}
	if err := rg2.remove(e.ID); err != nil {
		t.Fatal(err)
	}
	if err := rg2.remove("local"); err == nil || err != errLocalBackend {
		t.Fatalf("local backend must be pinned: %v", err)
	}
	if err := rg2.remove("missing"); err == nil || err != errUnknownBackend {
		t.Fatalf("unknown backend: %v", err)
	}
	// After removal the file no longer lists the entry.
	data, _ = os.ReadFile(path)
	if strings.Contains(string(data), "abc123") {
		t.Fatal("removed backend still persisted")
	}
}

// TestBackendsViewNeverLeaksToken is the M2 negative test #1.
func TestRegistryUpdate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "launcher.json")
	rg := openRegistry(path)
	e, err := rg.add("4090", "http://100.64.0.10:18766", "abc123")
	if err != nil {
		t.Fatal(err)
	}

	name := "pro6000 · 8"
	updated, err := rg.update(e.ID, &name, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The id is what the console holds on to; an edit must not move it.
	if updated.ID != e.ID {
		t.Errorf("id changed: %q -> %q", e.ID, updated.ID)
	}
	if updated.Name != name {
		t.Errorf("name = %q, want %q", updated.Name, name)
	}
	if updated.BaseURL != e.BaseURL || updated.Token != e.Token {
		t.Errorf("untouched fields changed: %+v", updated)
	}

	url := "https://node.example.com/"
	next, err := rg.update(e.ID, nil, &url, nil)
	if err != nil {
		t.Fatal(err)
	}
	if next.BaseURL != "https://node.example.com" {
		t.Errorf("base_url = %q, want the trailing slash trimmed", next.BaseURL)
	}
	if next.Name != name {
		t.Errorf("name = %q, want it left alone", next.Name)
	}

	// An empty token means "keep the stored one": the console never sees it.
	if _, err := rg.update(e.ID, nil, nil, ptr("")); err != nil {
		t.Fatal(err)
	}
	after, _ := rg.lookup(e.ID)
	if after.Token != "abc123" {
		t.Errorf("token = %q, want it preserved", after.Token)
	}
	// A new one replaces it.
	if _, err := rg.update(e.ID, nil, nil, ptr("new-token")); err != nil {
		t.Fatal(err)
	}
	if after, _ = rg.lookup(e.ID); after.Token != "new-token" {
		t.Errorf("token = %q, want it replaced", after.Token)
	}

	// The input is validated the same way adding is.
	bad := "http://100.64.0.10:18766/v1"
	if _, err := rg.update(e.ID, nil, &bad, nil); err == nil {
		t.Error("accepted a base_url with a /v1 suffix")
	}
	blank := "   "
	if _, err := rg.update(e.ID, &blank, nil, nil); err == nil {
		t.Error("accepted a blank name")
	}
	if _, err := rg.update("nope", &name, nil, nil); !errors.Is(err, errUnknownBackend) {
		t.Errorf("unknown id: err = %v", err)
	}
	if _, err := rg.update("local", &name, nil, nil); !errors.Is(err, errLocalBackend) {
		t.Errorf("local id: err = %v", err)
	}

	// Only the changed field is written, and it survives a reload.
	reloaded := openRegistry(path)
	got, ok := reloaded.lookup(e.ID)
	if !ok {
		t.Fatal("entry lost across reload")
	}
	if got.Name != name || got.BaseURL != next.BaseURL || got.Token != "new-token" {
		t.Errorf("reloaded %+v", got)
	}
}

func ptr[T any](v T) *T { return &v }

func TestBackendsViewNeverLeaksToken(t *testing.T) {
	l := newLauncher()
	l.clientOnly = true
	l.backends = openRegistry(filepath.Join(t.TempDir(), "launcher.json"))
	h := l.handler()
	_, body, raw := callJSON(t, h, "POST", "http://127.0.0.1:10721/api/v1/backends",
		strings.NewReader(`{"name":"4090 节点","base_url":"http://127.0.0.1:1","token":"abc123secret"}`), nil)
	if _, ok := body["id"]; !ok {
		t.Fatalf("add failed: %s", raw)
	}
	_, list, raw := callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/backends", nil, nil)
	if strings.Contains(raw, "abc123secret") {
		t.Fatal("token leaked in list response")
	}
	entries, _ := list["backends"].([]any)
	if len(entries) != 1 {
		t.Fatalf("list: %s", raw)
	}
	entry := entries[0].(map[string]any)
	if entry["has_token"] != true || entry["kind"] != "" {
		t.Fatalf("view fields: %v", entry)
	}
	// Empty-string id must not be assignable and unknown ids 404 on delete.
	if code, _, _ := callJSON(t, h, "DELETE", "http://127.0.0.1:10721/api/v1/backends/missing", nil, nil); code != 404 {
		t.Fatal("delete unknown backend must 404")
	}
}

func probeServer(t *testing.T, handler http.Handler) string {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv.URL
}

// TestProbeLadder covers M3: agent (with an unknown capability bit), bare
// inference node, a host that speaks only the removed pre-v1 surface, a
// Client, 401 and unreachable.
func TestProbeLadder(t *testing.T) {
	newAgent := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/node" {
			fmt.Fprint(w, `{"role":"agent","capabilities":["runtime","tuning_state","future_thing"]}`)
			return
		}
		w.WriteHeader(404)
	}))
	p := probeBackend(newAgent, "")
	if !p.Reachable || p.Kind != "agent" || len(p.Capabilities) != 3 || p.Capabilities[2] != "future_thing" {
		t.Fatalf("new agent probe: %+v", p)
	}

	// A host that answers only the removed pre-v1 surface is not an agent any
	// more. The ladder no longer has a rung for it, so it reads as
	// unreachable with the reason attached — never as a half-capable agent.
	preV1 := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/status" {
			fmt.Fprint(w, `{"status":"offline","running":false}`)
			return
		}
		w.WriteHeader(404)
	}))
	p = probeBackend(preV1, "")
	if p.Reachable || p.Kind != "" || p.ProbeError == "" {
		t.Fatalf("pre-v1 host must not classify as an agent: %+v", p)
	}

	bare := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/server/status" {
			fmt.Fprint(w, `{"status":"running"}`)
			return
		}
		w.WriteHeader(404)
	}))
	p = probeBackend(bare, "")
	if !p.Reachable || p.Kind != "inference_only" || fmt.Sprint(p.Capabilities) != fmt.Sprint([]string{"inference"}) {
		t.Fatalf("bare probe: %+v", p)
	}

	// A Client answers /api/v1/node with role=client and must never be taken
	// for an Agent.
	client := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/node" {
			fmt.Fprint(w, `{"status":"offline","role":"client"}`)
			return
		}
		w.WriteHeader(404)
	}))
	if p = probeBackend(client, ""); p.Reachable || !strings.Contains(p.ProbeError, "client") {
		t.Fatalf("client probe: %+v", p)
	}

	auth := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(401)
	}))
	if p = probeBackend(auth, "wrong"); p.Reachable || !strings.Contains(p.ProbeError, "401") {
		t.Fatalf("401 probe: %+v", p)
	}

	if p = probeBackend("http://127.0.0.1:1", ""); p.Reachable || p.ProbeError == "" {
		t.Fatalf("unreachable probe: %+v", p)
	}
}

// TestForwarding covers M2: prefix stripping, credential swap, byte-exact
// body passthrough (no re-marshal), status/header transparency, and SSE
// first-byte timing (FlushInterval).
func TestForwarding(t *testing.T) {
	var seenPath, seenAuth, seenBody, seenCustom string
	backend := probeServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenPath = r.URL.Path
		seenAuth = r.Header.Get("Authorization")
		seenCustom = r.Header.Get("X-Custom")
		body, _ := io.ReadAll(r.Body)
		seenBody = string(body)
		if r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(200)
			flusher := w.(http.Flusher)
			fmt.Fprint(w, "data: {\"delta\":\"first\"}\n\n")
			flusher.Flush() // backend flushes; the proxy must too
			time.Sleep(250 * time.Millisecond)
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("X-Backend", "yes")
		w.WriteHeader(201)
		fmt.Fprint(w, seenBody)
	}))

	l := newLauncher()
	l.clientOnly = true
	l.backends = openRegistry(filepath.Join(t.TempDir(), "launcher.json"))
	e, err := l.backends.add("fwd", backend, "tok123")
	if err != nil {
		t.Fatal(err)
	}
	h := l.handler()

	// Byte-exact POST passthrough with a body that DisallowUnknownFields
	// servers would reject if it were ever re-marshalled.
	rawBody := `{"zz_last":1,"aa_first":2,"unknown_field":{"nested":[1,2,3]}}`
	code, _, out := callJSONRaw(t, h, "POST", "http://127.0.0.1:10721/api/v1/backends/"+e.ID+"/v1/echo", rawBody, map[string]string{"X-Custom": "cval"})
	if code != 201 {
		t.Fatalf("forward status: %d %s", code, out)
	}
	if seenPath != "/v1/echo" {
		t.Fatalf("path not stripped: %s", seenPath)
	}
	if seenAuth != "Bearer tok123" {
		t.Fatalf("authorization not swapped: %q", seenAuth)
	}
	if seenCustom != "cval" {
		t.Fatal("custom header not passed through")
	}
	if seenBody != rawBody {
		t.Fatalf("body re-marshalled:\n got %s\nwant %s", seenBody, rawBody)
	}

	// The /api/v1 forward branch.
	code, _, _ = callJSONRaw(t, h, "GET", "http://127.0.0.1:10721/api/v1/backends/"+e.ID+"/api/v1/node", "", nil)
	if code != 201 || seenPath != "/api/v1/node" {
		t.Fatalf("api/v1 forward: %d %s", code, seenPath)
	}

	// SSE: first chunk must arrive while the stream is still open. This
	// needs a real socket (a recorder has no flush semantics for the
	// reverse proxy's streaming path).
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	start := time.Now()
	resp, err := http.Post(srv.URL+"/api/v1/backends/"+e.ID+"/v1/chat/completions", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := make([]byte, 64)
	n, _ := resp.Body.Read(buf)
	first := time.Since(start)
	if n == 0 || !strings.Contains(string(buf[:n]), "first") {
		t.Fatalf("no SSE data: %q", string(buf[:n]))
	}
	if first > 200*time.Millisecond {
		t.Fatalf("SSE first byte buffered (%v): FlushInterval is not in effect", first)
	}

	// Unknown backend and unsupported forward paths.
	if code, _, _ := callJSONRaw(t, h, "GET", "http://127.0.0.1:10721/api/v1/backends/nope/v1/models", "", nil); code != 404 {
		t.Fatal("unknown backend must 404")
	}
	if code, _, _ := callJSONRaw(t, h, "GET", "http://127.0.0.1:10721/api/v1/backends/"+e.ID+"/state/status", "", nil); code != 404 {
		t.Fatal("unsupported forward path must 404")
	}
}

func callJSONRaw(t *testing.T, h http.Handler, method, url, body string, header map[string]string) (int, map[string]any, string) {
	t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	r := httptest.NewRequest(method, url, rd)
	for k, v := range header {
		r.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	raw := w.Body.String()
	var out map[string]any
	_ = json.Unmarshal([]byte(raw), &out)
	return w.Code, out, raw
}

// The local backend carries no name: the WebUI supplies a localized label.
func TestLocalBackendHasNoHardcodedName(t *testing.T) {
	l := newLauncher()
	e, ok := l.localBackend()
	if !ok {
		t.Skip("no local runtime binary in this environment")
	}
	if e.Name != "" {
		t.Fatalf("local backend name must come from the WebUI catalogue, got %q", e.Name)
	}
}
