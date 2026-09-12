package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	switch os.Getenv("RWKV_TEST_CHILD") {
	case "logs":
		fmt.Print("[=                       ] 1/2 epoch=0/1 loss=1.2345 batch=1 tokens=10 lr=0.100000 tok/s=12.0 ETA=1s\r")
		fmt.Fprintln(os.Stderr, "stderr test secret-value")
		fmt.Println("saved: state_output/state-final.pth")
		return
	case "wait":
		fmt.Println("loading model")
		time.Sleep(30 * time.Second)
		return
	}
	os.Exit(m.Run())
}
func testFile(t *testing.T, name, contents string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if e := os.WriteFile(p, []byte(contents), 0600); e != nil {
		t.Fatal(e)
	}
	return p
}
func TestRuntimeArgs(t *testing.T) {
	req := startRequest{ModelPath: testFile(t, "with spaces.pth", "model"), VocabPath: testFile(t, "vocab.txt", "vocab"), Port: "8000", Password: "secret value", UseWKV32: true, ChunkLoad: true, ChunkSize: 64, StateDBPath: "cache path.db", TuneCache: "cache.tune"}
	args, e := runtimeArgs(req)
	if e != nil {
		t.Fatal(e)
	}
	want := []string{"--model-path", req.ModelPath, "--vocab-path", req.VocabPath, "--host", "127.0.0.1", "--port", "8000", "--chunk-size", "64", "--password", "secret value", "--state-db-path", "cache path.db", "--tune-cache", "cache.tune", "--wkv32", "--chunk-load"}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("%v", args)
	}
	for _, port := range []string{"0", "65536", "8088", "08088", "oops", "-1"} {
		req.Port = port
		if _, e = runtimeArgs(req); e == nil {
			t.Fatalf("accepted port %s", port)
		}
	}
	req.Port = "8000"
	req.EnableDynamicLoading = true
	if _, e = runtimeArgs(req); e == nil {
		t.Fatal("dynamic loading accepted a file")
	}
	req.ModelPath = t.TempDir()
	if _, e = runtimeArgs(req); e != nil {
		t.Fatal(e)
	}
}
func TestDatasetValidation(t *testing.T) {
	good := testFile(t, "data.jsonl", "{\"text\":\"你好\\nworld\"}\n\n{\"text\":\"\"}\n")
	if n, e := validateDataset(good); e != nil || n != 2 {
		t.Fatalf("%d %v", n, e)
	}
	for _, s := range []string{"", "{\"other\":\"a\"}", "{\"text\":3}", "{\"text\":null}", "{\"text\":\"a\",\"extra\":1}", "{\"text\":\"a\",\"text\":\"b\"}", "{\"text\":\"a\"}{}", "  ", "[]"} {
		if _, e := validateDataset(testFile(t, "bad.jsonl", s)); e == nil {
			t.Fatalf("accepted %q", s)
		}
	}
}
func TestTuningArgs(t *testing.T) {
	req := tuneRequest{Model: testFile(t, "model.pth", ""), Data: testFile(t, "data.jsonl", "{\"text\":\"hello\"}\n"), Output: "output with spaces", Ctx: 128, Chunk: 64, Epochs: 1, BatchSize: 1, LR: 1, LRFinal: .01, WarmupSteps: 10, Seed: 1234}
	args, e := tuningArgs(req)
	if e != nil {
		t.Fatal(e)
	}
	if !strings.Contains(strings.Join(args, " "), "--ctx 128 --chunk 64 --epochs 1 --batch-size 1 --max-steps 0") {
		t.Fatal(args)
	}
	req.LR = 0
	if _, e = tuningArgs(req); e == nil {
		t.Fatal("accepted zero LR")
	}
}
func TestProcessLogsAndProgress(t *testing.T) {
	t.Setenv("RWKV_TEST_CHILD", "logs")
	p := newProcess()
	exe, _ := os.Executable()
	if e := p.launch(exe, nil, "secret-value"); e != nil {
		t.Fatal(e)
	}
	p.mu.Lock()
	done := p.done
	p.mu.Unlock()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out")
	}
	s := p.snapshot()
	if s["status"] != "completed" {
		t.Fatal(s)
	}
	if s["progress"].(map[string]any)["loss"] != 1.2345 {
		t.Fatal(s)
	}
	if !strings.HasSuffix(s["checkpoint"].(string), "state-final.pth") {
		t.Fatal(s)
	}
	if strings.Contains(strings.Join(s["logs"].([]string), ""), "secret-value") {
		t.Fatal("leaked secret")
	}
}
func TestProcessStopAndMutualExclusion(t *testing.T) {
	t.Setenv("RWKV_TEST_CHILD", "wait")
	l := newLauncher()
	exe, _ := os.Executable()
	if e := l.tuning.launch(exe, nil, ""); e != nil {
		t.Fatal(e)
	}
	defer l.tuning.stop()
	if e := l.start(startRequest{}); e == nil || !strings.Contains(e.Error(), "tuning") {
		t.Fatal(e)
	}
	if e := l.tuning.stop(); e != nil {
		t.Fatal(e)
	}
	if l.tuning.active() || l.tuning.snapshot()["status"] != "offline" {
		t.Fatal(l.tuning.snapshot())
	}
}
func TestReadinessIsReal(t *testing.T) {
	t.Setenv("RWKV_TEST_CHILD", "wait")
	l := newLauncher()
	exe, _ := os.Executable()
	if e := l.runtime.launch(exe, nil, ""); e != nil {
		t.Fatal(e)
	}
	defer l.runtime.stop()
	if l.status()["status"] != "starting" {
		t.Fatal("reported ready without a backend")
	}
	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/server/status" {
			t.Error(r.URL.Path)
		}
		fmt.Fprint(w, `{"status":"running","model":{"name":"test-model"}}`)
	}))
	defer native.Close()
	u, _ := url.Parse(native.URL)
	l.config.Port = u.Port()
	if l.status()["status"] != "ready" {
		t.Fatal("backend status was not used")
	}
}
func TestProxyRawContinuationAndErrors(t *testing.T) {
	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if r.Header.Get("Authorization") != "Bearer managed-secret" {
			t.Error("missing managed auth")
		}
		if bytes.Contains(body, []byte(`"contents"`)) {
			if r.URL.Path != "/v1/batch/completions" {
				t.Error(r.URL.Path)
			}
		} else if r.URL.Path != "/v1/chat/completions" {
			t.Error(r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer native.Close()
	u, _ := url.Parse(native.URL)
	l := newLauncher()
	l.config.Port = u.Port()
	l.config.Password = "managed-secret"
	for _, body := range []string{`{"contents":["English: Hello\n\nChinese:"],"stream":true}`, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`} {
		r := httptest.NewRequest("POST", "http://127.0.0.1:8088/v1/chat/completions", strings.NewReader(body))
		w := httptest.NewRecorder()
		l.handler().ServeHTTP(w, r)
		if w.Code != 200 || !strings.Contains(w.Body.String(), "[DONE]") {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	native.Close()
	r := httptest.NewRequest("POST", "http://127.0.0.1:8088/v1/chat/completions", strings.NewReader(`{}`))
	w := httptest.NewRecorder()
	l.handler().ServeHTTP(w, r)
	if w.Code != 502 {
		t.Fatal(w.Code)
	}
}
func TestStaticHostAndSecurity(t *testing.T) {
	l := newLauncher()
	for _, c := range []struct {
		method, path, origin string
		want                 int
	}{{"GET", "/", "", 200}, {"GET", "/api/status", "", 200}, {"GET", "/api/start", "", 405}, {"POST", "/api/stop", "https://evil.example", 403}} {
		r := httptest.NewRequest(c.method, "http://127.0.0.1:8088"+c.path, nil)
		r.Header.Set("Origin", c.origin)
		w := httptest.NewRecorder()
		l.handler().ServeHTTP(w, r)
		if w.Code != c.want {
			t.Fatalf("%s: %d", c.path, w.Code)
		}
		if c.path == "/" && !strings.Contains(w.Body.String(), "RWKV Lightning") {
			t.Fatal("static app missing")
		}
	}
	r := httptest.NewRequest("GET", "http://evil.example/api/status", nil)
	w := httptest.NewRecorder()
	l.handler().ServeHTTP(w, r)
	if w.Code != 403 {
		t.Fatal("DNS rebinding host accepted")
	}
	data, _ := json.Marshal(l.status())
	if bytes.Contains(data, []byte("managed-secret")) {
		t.Fatal("status leaked secret")
	}
}
