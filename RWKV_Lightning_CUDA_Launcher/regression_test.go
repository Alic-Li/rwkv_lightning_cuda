package main

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestTwoHopBrowserInference(t *testing.T) {
	for _, password := range []string{"runtime-secret", ""} {
		t.Run("password="+password, func(t *testing.T) {
			body := `{ "contents": ["hello"], "future_field": 123 }`
			received := make(chan string, 1)
			release := make(chan struct{})
			runtimeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				wantAuth := ""
				if password != "" {
					wantAuth = "Bearer " + password
				}
				if r.Header.Get("Authorization") != wantAuth || string(raw) != body || r.URL.Path != "/v1/batch/completions" || r.URL.RawQuery != "x=1" || r.Header.Get("X-RWKV-State-Id") != "state-a" {
					received <- fmt.Sprintf("auth=%q path=%s query=%s body=%s", r.Header.Get("Authorization"), r.URL.Path, r.URL.RawQuery, raw)
					w.WriteHeader(401)
					return
				}
				received <- "ok"
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("X-Runtime", "yes")
				fmt.Fprint(w, "data: first\n\n")
				w.(http.Flusher).Flush()
				select {
				case <-release:
				case <-r.Context().Done():
				}
			}))
			defer runtimeServer.Close()
			defer close(release)
			agent := newLauncher()
			agent.listen = "0.0.0.0:18766"
			agent.token = "agent-secret"
			agent.config.Password = password
			_, agent.config.Port, _ = net.SplitHostPort(strings.TrimPrefix(runtimeServer.URL, "http://"))
			agentServer := httptest.NewServer(agent.handler())
			defer agentServer.Close()
			client := newLauncher()
			client.clientOnly = true
			client.token = "client-secret"
			client.backends = openRegistry(filepath.Join(t.TempDir(), "launcher.json"))
			be, err := client.backends.add("agent", agentServer.URL, agent.token)
			if err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(client.handler())
			defer server.Close()
			requestURL := server.URL + "/api/v1/backends/" + be.ID + "/v1/batch/completions?x=1"
			req, _ := http.NewRequest("POST", requestURL, strings.NewReader(body))
			req.Header.Set("Origin", server.URL)
			req.Header.Set("Sec-Fetch-Site", "same-origin")
			req.Header.Set("Authorization", "Bearer client-secret")
			req.Header.Set("X-RWKV-State-Id", "state-a")
			httpClient := &http.Client{Timeout: 3 * time.Second}
			resp, err := httpClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != 200 {
				raw, _ := io.ReadAll(resp.Body)
				t.Fatalf("two-hop response: %d %s", resp.StatusCode, raw)
			}
			if got := <-received; got != "ok" {
				t.Fatal(got)
			}
			if resp.Header.Get("X-Runtime") != "yes" {
				t.Fatal("response header lost")
			}
			// The runtime cannot finish until release closes, so receiving this
			// line proves streaming works across both proxies without timing races.
			line, err := bufio.NewReader(resp.Body).ReadString('\n')
			if err != nil || line != "data: first\n" {
				t.Fatalf("SSE first event: %q %v", line, err)
			}
			resp.Body.Close()
			// Origin must still be checked at the browser-facing boundary.
			for _, origin := range []string{"https://evil.example", server.URL} {
				req, _ := http.NewRequest("POST", requestURL, strings.NewReader(body))
				req.Header.Set("Origin", origin)
				req.Header.Set("Authorization", "Bearer client-secret")
				if origin == server.URL {
					req.Header.Set("Sec-Fetch-Site", "cross-site")
				}
				resp, err := httpClient.Do(req)
				if err != nil {
					t.Fatal(err)
				}
				resp.Body.Close()
				if resp.StatusCode != 403 {
					t.Fatalf("cross-site accepted: %d", resp.StatusCode)
				}
			}
			req, _ = http.NewRequest("POST", agentServer.URL+"/v1/batch/completions", strings.NewReader(body))
			req.Header.Set("Origin", "https://evil.example")
			req.Header.Set("Authorization", "Bearer agent-secret")
			resp, err = httpClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			if resp.StatusCode != 403 {
				t.Fatalf("direct Agent cross-origin accepted: %d", resp.StatusCode)
			}
		})
	}
}

func TestRuntimeProxyCredentialHandoff(t *testing.T) {
	for _, caller := range []string{"", "Bearer caller-runtime-secret"} {
		t.Run(caller, func(t *testing.T) {
			seen := make(chan string, 1)
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				seen <- r.Header.Get("Authorization")
				w.WriteHeader(204)
			}))
			defer backend.Close()
			l := newLauncher()
			l.config.Password = "configured-secret"
			_, l.config.Port, _ = net.SplitHostPort(strings.TrimPrefix(backend.URL, "http://"))
			code, _, raw := callJSON(t, l.handler(), "GET", "http://127.0.0.1:10721/v1/models", nil, map[string]string{"Authorization": caller})
			if code != 204 {
				t.Fatalf("%d %s", code, raw)
			}
			want := caller
			if want == "" {
				want = "Bearer configured-secret"
			}
			if got := <-seen; got != want {
				t.Fatalf("auth=%q want=%q", got, want)
			}
		})
	}
}

func TestRegistryFailureDoesNotCommit(t *testing.T) {
	for _, failure := range []string{"parent", "rename"} {
		t.Run(failure, func(t *testing.T) {
			dir := t.TempDir()
			goodPath := filepath.Join(dir, "launcher.json")
			rg := openRegistry(goodPath)
			be, err := rg.add("original", "http://127.0.0.1:1", "keep-secret")
			if err != nil {
				t.Fatal(err)
			}
			rg.setProbe(be.ID, probeResult{Reachable: true, Kind: "agent"})
			bad := filepath.Join(dir, "blocked")
			if failure == "parent" {
				if err := os.WriteFile(bad, []byte("file"), 0600); err != nil {
					t.Fatal(err)
				}
				rg.path = filepath.Join(bad, "launcher.json")
			} else {
				if err := os.Mkdir(bad, 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(bad, "child"), nil, 0600); err != nil {
					t.Fatal(err)
				}
				rg.path = bad
			}
			l := newLauncher()
			l.clientOnly = true
			l.backends = rg
			code, _, raw := callJSON(t, l.handler(), "POST", "http://127.0.0.1:10721/api/v1/backends", strings.NewReader(`{"base_url":"http://127.0.0.1:2","token":"new-secret"}`), nil)
			if code != 400 || strings.Contains(raw, "new-secret") {
				t.Fatalf("failed add: %d %s", code, raw)
			}
			code, _, raw = callJSON(t, l.handler(), "DELETE", "http://127.0.0.1:10721/api/v1/backends/"+be.ID, nil, nil)
			if code != 400 {
				t.Fatalf("failed delete: %d %s", code, raw)
			}
			if len(rg.entries) != 1 || rg.entries[0] != be || !rg.probeOf(be.ID).Reachable {
				t.Fatal("failed persistence changed memory/probe")
			}
			if got, ok := openRegistry(goodPath).lookup(be.ID); !ok || got != be {
				t.Fatal("failed persistence changed disk")
			}
			temps, err := filepath.Glob(filepath.Join(dir, ".launcher-*.tmp"))
			if err != nil || len(temps) != 0 {
				t.Fatalf("temporary credentials left behind: %v %v", temps, err)
			}
			rg.path = goodPath
			if err := rg.remove(be.ID); err != nil {
				t.Fatal(err)
			}
			if _, ok := openRegistry(goodPath).lookup(be.ID); ok {
				t.Fatal("successful deletion not persisted")
			}
		})
	}
}

func TestIPv6Startup(t *testing.T) {
	for _, tc := range []struct {
		listen, token           string
		client, capable, wantOK bool
	}{
		{"[::1]:10721", "", true, false, true},
		{"[::1]:18766", "", false, true, true},
		{"[::]:18766", "secret", false, true, true},
		{"[::]:18766", "", false, true, false},
		{"[::]:18766", "secret", true, false, false},
		{"[::1]:0", "", true, false, false},
		{"[::1]:65536", "", true, false, false},
		{"[::1]:bad", "", true, false, false},
	} {
		t.Run(tc.listen+tc.token, func(t *testing.T) {
			err := validateStartup(tc.listen, tc.token, tc.client, tc.capable)
			if (err == nil) != tc.wantOK {
				t.Fatalf("startup error=%v wantOK=%v", err, tc.wantOK)
			}
		})
	}
}

func TestTwoHopBrowserControl(t *testing.T) {
	agent := newLauncher()
	agent.listen = "0.0.0.0:18766"
	agent.token = "agent-secret"
	backend := httptest.NewServer(agent.handler())
	defer backend.Close()
	client := newLauncher()
	client.clientOnly = true
	client.backends = openRegistry(filepath.Join(t.TempDir(), "registry.json"))
	be, err := client.backends.add("agent", backend.URL, agent.token)
	if err != nil {
		t.Fatal(err)
	}
	client.backends.setProbe(be.ID, probeResult{Kind: "agent", Reachable: true})
	code, body, raw := callJSONRaw(t, client.handler(), "POST", "http://127.0.0.1:10721/api/v1/backends/"+be.ID+"/api/v1/runtime/stop", `{}`, map[string]string{"Origin": "http://127.0.0.1:10721", "Sec-Fetch-Site": "same-origin"})
	if code != 200 || body["ok"] != true {
		t.Fatalf("two-hop control: %d %s", code, raw)
	}
	// The relaxed server-to-server Origin handling must not bypass Agent auth.
	bad := be
	bad.Token = "wrong-agent-token"
	r := httptest.NewRequest("POST", "http://127.0.0.1:10721/api/v1/runtime/stop", strings.NewReader(`{}`))
	w := httptest.NewRecorder()
	forwardToBackend(w, r, bad, "api/v1/runtime/stop")
	if w.Code != 401 {
		t.Fatalf("bad Agent token accepted: %d %s", w.Code, w.Body.String())
	}
}

// A runtime started outside this launcher (by hand, or by another tool) used
// to read as "offline" forever: the /v1/server/status probe only ran for
// processes the Agent spawned itself. The Agent must report what it can see,
// flagged managed=false, and keep "offline" when the port is dark.
func TestExternalRuntimeIsReportedReady(t *testing.T) {
	l := newLauncher()
	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"status":"running","model":{"name":"test-model"}}`)
	}))
	defer native.Close()
	u, _ := url.Parse(native.URL)
	l.config.Port = u.Port()

	out := l.status()
	if out["status"] != "ready" {
		t.Fatalf("external server reported %v, want ready", out["status"])
	}
	if managed, _ := out["managed"].(bool); managed {
		t.Fatal("external server reported as managed")
	}
	if out["running"] != true {
		t.Fatalf("running=%v, want true for a serving runtime", out["running"])
	}

	// --client-only never probes and is never "ready": it has no runtime.
	client := newLauncher()
	client.clientOnly = true
	client.config.Port = u.Port()
	if out := client.status(); out["status"] == "ready" {
		t.Fatal("client-only launcher reported an external runtime as ready")
	}

	// Dark port: back to plain offline, no phantom readiness.
	dark := newLauncher()
	unused, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	dark.config.Port = fmt.Sprint(unused.Addr().(*net.TCPAddr).Port)
	unused.Close()
	if out := dark.status(); out["status"] != "offline" {
		t.Fatalf("dark port reported %v, want offline", out["status"])
	}
}

// Card-switch loads need a saved runtime config to restart from: without one
// the Agent would silently start an empty server, and a --client-only host
// has no runtime to manage at all.
func TestRuntimeLoadValidation(t *testing.T) {
	client := newLauncher()
	client.clientOnly = true
	if _, err := client.runtimeLoad(runtimeLoadRequest{Model: "m", VisibleDevices: "1"}); err == nil {
		t.Fatal("client-only launcher accepted a runtime load")
	}
	l := newLauncher()
	if _, err := l.runtimeLoad(runtimeLoadRequest{VisibleDevices: "1"}); err == nil || !strings.Contains(err.Error(), "no saved runtime config") {
		t.Fatalf("load without a saved config: %v", err)
	}
}
