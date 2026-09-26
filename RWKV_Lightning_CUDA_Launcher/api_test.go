package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func callJSON(t *testing.T, h http.Handler, method, url string, body io.Reader, header map[string]string) (int, map[string]any, string) {
	t.Helper()
	r := httptest.NewRequest(method, url, body)
	for k, v := range header {
		r.Header.Set(k, v)
	}
	if body != nil {
		r.Header.Set("Content-Type", "application/json")
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	raw := w.Body.String()
	var out map[string]any
	_ = json.Unmarshal([]byte(raw), &out)
	return w.Code, out, raw
}

func TestTokenAuth(t *testing.T) {
	l := newLauncher()
	l.clientOnly = true // no binaries: exercise the client endpoints
	l.token = "abc123"
	h := l.handler()

	code, body, raw := callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/backends", nil, nil)
	if code != 401 || body["error"] != "unauthorized" {
		t.Fatalf("no token: %d %s", code, raw)
	}
	if len(body) != 1 {
		t.Fatalf("401 body must contain nothing but the error: %s", raw)
	}
	code, _, _ = callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/backends", nil, map[string]string{"Authorization": "Bearer wrong"})
	if code != 401 {
		t.Fatalf("wrong token: %d", code)
	}
	// Tokens must never be accepted via query string.
	code, _, _ = callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/backends?token=abc123", nil, nil)
	if code != 401 {
		t.Fatalf("query-string token accepted: %d", code)
	}
	code, _, _ = callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/backends", nil, map[string]string{"Authorization": "Bearer abc123"})
	if code != 200 {
		t.Fatalf("correct token rejected: %d", code)
	}
	// The local /v1 proxy is also token-gated when a token is configured.
	r := httptest.NewRequest("GET", "http://127.0.0.1:10721/v1/models", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 401 {
		t.Fatalf("local /v1 proxy not token-gated: %d", w.Code)
	}
}

func TestClientOnlyRole(t *testing.T) {
	l := newLauncher()
	l.clientOnly = true
	h := l.handler()

	for _, path := range []string{"/api/v1/node", "/api/v1/runtime", "/api/v1/jobs", "/api/v1/jobs/tuning", "/api/v1/node/metrics"} {
		code, body, raw := callJSON(t, h, "GET", "http://127.0.0.1:10721"+path, nil, nil)
		if code != 404 || body["error"] != "client_only" {
			t.Fatalf("%s: %d %s", path, code, raw)
		}
	}
	// Control actions answer the same way, so a Client is never probe-
	// classifiable as an Agent on any path.
	code, body, raw := callJSON(t, h, "POST", "http://127.0.0.1:10721/api/v1/runtime/start", strings.NewReader(`{}`), nil)
	if code != 404 || body["error"] != "client_only" {
		t.Fatalf("start in client form: %d %s", code, raw)
	}
	if l.role() != "client" || len(l.capabilities()) != 0 {
		t.Fatalf("client role: %s %v", l.role(), l.capabilities())
	}
}

func TestValidateStartup(t *testing.T) {
	if err := validateStartup("0.0.0.0:18766", "", false, true); err == nil || !strings.Contains(err.Error(), "--token") {
		t.Fatalf("open listen without token must refuse: %v", err)
	}
	if err := validateStartup("0.0.0.0:18766", "t", false, true); err != nil {
		t.Fatalf("open listen with token: %v", err)
	}
	if err := validateStartup(":18766", "", false, true); err == nil {
		t.Fatal("host-less listen binds all interfaces and must require a token")
	}
	if err := validateStartup("127.0.0.1:10721", "", true, false); err != nil {
		t.Fatalf("loopback client form: %v", err)
	}
	if err := validateStartup("0.0.0.0:18766", "t", true, false); err == nil || !strings.Contains(err.Error(), "loopback") {
		t.Fatalf("client form on non-loopback must refuse: %v", err)
	}
	if err := validateStartup("not-an-addr", "", false, true); err == nil {
		t.Fatal("invalid listen must refuse")
	}
}

func TestCapabilitiesAndNodePayload(t *testing.T) {
	for _, path := range []string{backendExecutable(), toolBinary("rwkv_state_tune"), toolBinary("rwkv_miss_tune"), toolBinary("rwkv_quantize")} {
		if err := os.WriteFile(path, []byte("#!/bin/sh\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		defer os.Remove(path)
	}
	l := newLauncher()
	if l.role() != "full" {
		t.Fatalf("role: %s", l.role())
	}
	caps := l.capabilities()
	want := []string{"runtime", "tuning_state", "tuning_miss", "quantization", "metrics", "fs", "state_import", "host_dialog"}
	if fmt.Sprint(caps) != fmt.Sprint(want) {
		t.Fatalf("capabilities: %v", caps)
	}
	_, body, raw := callJSON(t, l.handler(), "GET", "http://127.0.0.1:10721/api/v1/node", nil, nil)
	if body["role"] != "agent" || body["version"] == nil || fmt.Sprint(body["capabilities"]) == "" {
		t.Fatalf("node payload: %s", raw)
	}
	if _, ok := body["visible_devices"]; !ok {
		t.Fatalf("node payload missing visible_devices: %s", raw)
	}
	// /api/v1/node is a superset of the RuntimeState /api/v1/runtime reports:
	// one shape, never a second one that can drift.
	for _, k := range []string{"status", "running", "config", "base_url"} {
		if _, ok := body[k]; !ok {
			t.Fatalf("node payload missing %s: %s", k, raw)
		}
	}
}

func TestJobsCollectionAndItems(t *testing.T) {
	// The /api/v1 paths answer 404 in client-only form, so the launcher must
	// run in agent form.
	bin := backendExecutable()
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Remove(bin) })

	l := newLauncher()
	h := l.handler()
	_, job, raw := callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/jobs/tuning", nil, nil)
	if job["miss_available"] == nil {
		t.Fatalf("miss_available lost: %s", raw)
	}
	_, both, _ := callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/jobs", nil, nil)
	jobs, ok := both["jobs"].(map[string]any)
	if !ok || jobs["tuning"] == nil || jobs["quantization"] == nil {
		t.Fatalf("jobs map: %v", both)
	}
	if code, body, _ := callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/jobs/nope", nil, nil); code != 404 || body["error"] != "unknown job id" {
		t.Fatalf("unknown job: %v", body)
	}
}

func TestMetricsUnavailableIsExplicit(t *testing.T) {
	// On a machine without a GPU management interface (this test host
	// class), metrics must be available=false with a reason and an empty
	// GPU list — never zero-filled samples.
	m := sampleMetrics(0)
	if m.Available {
		// A real GPU host: then the sample must be complete instead.
		if len(m.GPUs) == 0 {
			t.Fatal("available without gpus")
		}
		return
	}
	if m.Reason == "" || m.Vendor == "" || len(m.GPUs) != 0 {
		t.Fatalf("unavailable metrics must carry a reason and no fake data: %+v", m)
	}
}

func TestWrongMethodAndUnknownAPIPaths(t *testing.T) {
	l := newLauncher()
	l.clientOnly = true
	h := l.handler()
	// Wrong-method requests on /api must answer JSON — never the static
	// file server's text 404.
	code, body, raw := callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/runtime/start", nil, nil)
	if code != 404 || body["error"] != "not found" {
		t.Fatalf("wrong method: %d %s", code, raw)
	}
	code, body, raw = callJSON(t, h, "POST", "http://127.0.0.1:10721/api/v1/node", strings.NewReader(`{}`), nil)
	if code != 404 || body["error"] != "not found" {
		t.Fatalf("wrong method GET-only: %d %s", code, raw)
	}
	code, body, _ = callJSON(t, h, "GET", "http://127.0.0.1:10721/api/v1/nonexistent", nil, nil)
	if code != 404 || body["error"] != "not found" {
		t.Fatalf("unknown api path: %d", code)
	}
	// Static hosting survives the catch-all.
	code, _, raw = callJSON(t, h, "GET", "http://127.0.0.1:10721/", nil, nil)
	if code != 200 || !strings.Contains(raw, "RWKV") {
		t.Fatalf("static broken: %d", code)
	}
}

func TestDialogUnsupportedForRemoteCallers(t *testing.T) {
	l := newLauncher()
	l.clientOnly = true
	code, body, _ := callJSON(t, l.handler(), "POST", "http://127.0.0.1:10721/api/v1/node/dialog/file", strings.NewReader(`{}`), nil)
	if code != 400 || body["error"] != "unsupported" || body["reason"] != "host-local only" {
		t.Fatalf("dialog in client form: %d %v", code, body)
	}
}

// No pre-/api/v1 path survives. The old WebUI is gone, and so is
// mixed-version support: this launcher speaks one protocol, and an unknown
// /api path answers JSON 404 rather than falling through to the static
// handler's text 404.
func TestPreV1SurfaceIsGone(t *testing.T) {
	for _, path := range []string{backendExecutable(), toolBinary("rwkv_state_tune")} {
		if err := os.WriteFile(path, []byte("#!/bin/sh\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		defer os.Remove(path)
	}
	l := newLauncher()
	h := l.handler()

	for _, c := range []struct{ method, path string }{
		{"GET", "/api/status"},
		{"POST", "/api/start"},
		{"POST", "/api/stop"},
		{"POST", "/api/restart"},
		{"POST", "/api/pick-file"},
		{"POST", "/api/pick-directory"},
		{"GET", "/api/tuning/status"},
		{"POST", "/api/tuning/validate"},
		{"POST", "/api/tuning/start"},
		{"POST", "/api/tuning/stop"},
		{"POST", "/api/tuning/open-folder"},
		{"GET", "/api/quantization/status"},
		{"POST", "/api/quantization/start"},
		{"POST", "/api/quantization/stop"},
	} {
		code, body, raw := callJSON(t, h, c.method, "http://127.0.0.1:10721"+c.path, nil, nil)
		if code != 404 || body["error"] != "not found" {
			t.Fatalf("%s %s should be gone: %d %s", c.method, c.path, code, raw)
		}
	}

	// /logs is replaced by /api/v1/runtime/logs. Nothing serves it now, so it
	// falls through to the static handler rather than to an SSE stream.
	r := httptest.NewRequest("GET", "http://127.0.0.1:10721/logs", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if ct := w.Header().Get("Content-Type"); strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("/logs still streams: %s", ct)
	}

	// The v1 replacements all answer.
	for _, c := range []struct{ method, path string }{
		{"GET", "/api/v1/node"},
		{"GET", "/api/v1/runtime"},
		{"GET", "/api/v1/jobs/tuning"},
		{"POST", "/api/v1/node/dialog/file"},
	} {
		if code, _, raw := callJSON(t, h, c.method, "http://127.0.0.1:10721"+c.path, nil, nil); code == 404 {
			t.Fatalf("%s %s must still exist: %s", c.method, c.path, raw)
		}
	}
}
