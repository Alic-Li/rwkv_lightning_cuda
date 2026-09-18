package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
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
	req := tuneRequest{Model: testFile(t, "model.pth", ""), Data: testFile(t, "data.jsonl", "{\"text\":\"hello\"}\n"), Output: "output with spaces", Ctx: 128, Chunk: 64, Epochs: 1, BatchSize: 1, LR: 1, LRFinal: .01, WarmupSteps: 10, Seed: 1234, Optimizer: "muon", WKVTape: true}
	args, e := tuningArgs(req)
	if e != nil {
		t.Fatal(e)
	}
	if !strings.Contains(strings.Join(args, " "), "--ctx 128 --chunk 64 --epochs 1 --batch-size 1 --max-steps 0") {
		t.Fatal(args)
	}
	if !strings.Contains(strings.Join(args, " "), "--optimizer muon --wkv_tape") {
		t.Fatal(args)
	}
	req.LR = 0
	if _, e = tuningArgs(req); e == nil {
		t.Fatal("accepted zero LR")
	}
	req.LR = 1
	req.BatchSize = 129
	if _, e = tuningArgs(req); e == nil {
		t.Fatal("accepted batch size above 128")
	}
	req.BatchSize = 1
	req.Chunk = req.Ctx + 1
	if _, e = tuningArgs(req); e == nil {
		t.Fatal("accepted recompute chunk above context length")
	}
	req.Chunk = 64
	req.Optimizer = "sgd"
	if _, e = tuningArgs(req); e == nil {
		t.Fatal("accepted unsupported optimizer")
	}
}
func TestQuantizationArgs(t *testing.T) {
	input := testFile(t, "model.pth", "model")
	output := filepath.Join(t.TempDir(), "model.w4a16.rwkvq")
	args, err := quantizationArgs(quantizeRequest{InputPath: input, OutputPath: output, Format: "w4a16", GroupSize: 32})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--format", "w4a16", "--group-size", "32", input, output}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("%v", args)
	}
	args, err = quantizationArgs(quantizeRequest{InputPath: input, OutputPath: filepath.Join(t.TempDir(), "model.w8a16.rwkvq"), Format: "w8a16", GroupSize: 128})
	if err != nil || strings.Contains(strings.Join(args, " "), "group-size") {
		t.Fatalf("%v %v", args, err)
	}
	for _, req := range []quantizeRequest{
		{InputPath: input, OutputPath: output, Format: "w4a16", GroupSize: 64},
		{InputPath: input, OutputPath: output, Format: "fp8", GroupSize: 128},
		{InputPath: input, OutputPath: filepath.Join(t.TempDir(), "model.bin"), Format: "w8a16"},
		{InputPath: testFile(t, "model.rwkvq", "model"), OutputPath: output, Format: "w8a16"},
	} {
		if _, err = quantizationArgs(req); err == nil {
			t.Fatalf("accepted invalid request: %+v", req)
		}
	}
	if err = os.WriteFile(output, []byte("existing"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = quantizationArgs(quantizeRequest{InputPath: input, OutputPath: output, Format: "w4a16", GroupSize: 128}); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatal("accepted existing output", err)
	}
}

func TestMissTuningArgs(t *testing.T) {
	req := tuneRequest{Method: "miss", Model: testFile(t, "base.pth", ""), Data: testFile(t, "data.jsonl", "{\"text\":\"hello\"}\n"), Output: "miss output", Ctx: 4096, Chunk: 1024, Epochs: 1, BatchSize: 8, LR: .0001, LRFinal: .00001, Rank: 16, Alpha: 16, Targets: "all", WKVTape: true, Resume: t.TempDir(), State: testFile(t, "state.pth", "")}
	args, err := tuningArgs(req)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, part := range []string{"--rank 16 --alpha 16 --targets all", "--resume " + req.Resume, "--state " + req.State, "--optimizer adam", "--wkv_tape"} {
		if !strings.Contains(joined, part) {
			t.Fatal(args)
		}
	}
	for _, bad := range []string{"bad.weight", "ffn.key.weight,ffn.key.weight", ""} {
		req.Targets = bad
		if _, err = tuningArgs(req); err == nil {
			t.Fatal("accepted invalid targets", bad)
		}
	}
	req.Targets = "ffn.value.weight"
	req.Optimizer = "muon"
	if _, err = tuningArgs(req); err == nil {
		t.Fatal("accepted MiSS Muon")
	}
	req.Optimizer = "adam"
	req.Rank = 0
	if _, err = tuningArgs(req); err == nil {
		t.Fatal("accepted rank zero")
	}
	req.Rank = 16
	req.Method = "unknown"
	if _, err = tuningArgs(req); err == nil {
		t.Fatal("accepted unknown method")
	}
}

// Opt-in real CUDA integration using fixtures produced by tools/check_miss.py.
func TestNativeMissLifecycle(t *testing.T) {
	build, fixture := os.Getenv("RWKV_NATIVE_BUILD"), os.Getenv("RWKV_NATIVE_FIXTURE")
	if build == "" || fixture == "" {
		t.Skip("set RWKV_NATIVE_BUILD and RWKV_NATIVE_FIXTURE for native GPU test")
	}
	for _, name := range []string{"rwkv_miss_tune", "rwkv_lighting_cuda"} {
		path := filepath.Join(appDir(), name)
		if err := os.Symlink(filepath.Join(build, name), path); err != nil {
			t.Fatal(err)
		}
		defer os.Remove(path)
	}
	l := newLauncher()
	defer l.runtime.stop()
	defer l.tuning.stop()
	call := func(method, path string, body any) map[string]any {
		t.Helper()
		var data io.Reader
		if body != nil {
			raw, _ := json.Marshal(body)
			data = bytes.NewReader(raw)
		}
		r := httptest.NewRequest(method, "http://127.0.0.1:10721"+path, data)
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		l.handler().ServeHTTP(w, r)
		if w.Code != 200 {
			t.Fatalf("%s: %d %s", path, w.Code, w.Body.String())
		}
		var out map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	wait := func() {
		t.Helper()
		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			s := call("GET", "/api/tuning/status", nil)
			if s["running"] == false {
				if s["status"] != "completed" {
					t.Fatal(s)
				}
				return
			}
			time.Sleep(50 * time.Millisecond)
		}
		t.Fatal("native training timed out")
	}
	cfg := tuneRequest{Method: "miss", Model: filepath.Join(fixture, "tiny.pth"), Data: filepath.Join(fixture, "train.jsonl"), Vocab: filepath.Join(fixture, "vocab.txt"), Output: filepath.Join(t.TempDir(), "first"), Ctx: 30, Chunk: 7, Epochs: 4, BatchSize: 2, MaxSteps: 1, LR: .003, LRFinal: .003, SaveEvery: 1, Seed: 1234, Rank: 8, Alpha: 8, Targets: "all", Optimizer: "adam", WKVTape: true}
	if call("GET", "/api/tuning/status", nil)["miss_available"] != true {
		t.Fatal("missing trainer detection")
	}
	call("POST", "/api/tuning/start", cfg)
	wait()
	cfg.Resume = filepath.Join(cfg.Output, "checkpoint-1")
	cfg.Output = filepath.Join(t.TempDir(), "resumed")
	cfg.MaxSteps = 2
	call("POST", "/api/tuning/start", cfg)
	wait()
	final := filepath.Join(cfg.Output, "adapter-final.pth")
	if call("GET", "/api/tuning/status", nil)["checkpoint"] != final {
		t.Fatal("missing final export path")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := strconv.Itoa(listener.Addr().(*net.TCPAddr).Port)
	listener.Close()
	call("POST", "/api/start", startRequest{ModelPath: cfg.Model, VocabPath: cfg.Vocab, Port: port, Password: "miss-test-password", ChunkSize: 32, UseWKV32: true, StateDBPath: filepath.Join(t.TempDir(), "state.db")})
	deadline := time.Now().Add(30 * time.Second)
	for l.status()["status"] != "ready" {
		if time.Now().After(deadline) {
			t.Fatal(l.status())
		}
		time.Sleep(50 * time.Millisecond)
	}
	var payload bytes.Buffer
	writer := multipart.NewWriter(&payload)
	writer.WriteField("adapter_id", "test")
	part, _ := writer.CreateFormFile("file", "training.pth")
	file, err := os.Open(filepath.Join(cfg.Output, "checkpoint-2", "training.pth"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.Copy(part, file)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	writer.Close()
	r := httptest.NewRequest("POST", "http://127.0.0.1:10721/v1/adapters", &payload)
	r.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()
	l.handler().ServeHTTP(w, r)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	stats := call("GET", "/v1/adapters", nil)
	if stats["uploads"] != float64(0) {
		t.Fatal(stats)
	}
	var registration map[string]string
	json.Unmarshal(w.Body.Bytes(), &registration)
	body := map[string]any{"contents": []string{"abcabc"}, "adapter_id": "test", "adapter_version": registration["version"], "adapter_scale": 1, "max_tokens": 1, "temperature": 1, "top_k": 1, "stop_tokens": []int{}, "stream": false}
	call("POST", "/v1/chat/completions", body)
	call("POST", "/v1/chat/completions", body)
	if call("GET", "/v1/adapters", nil)["uploads"] != float64(1) {
		t.Fatal("repeated adapter H2D")
	}
	call("DELETE", "/v1/adapters", map[string]string{"adapter_id": "test", "adapter_version": registration["version"]})
	call("POST", "/api/stop", map[string]any{})
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
	unused, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	l.config.Port = fmt.Sprint(unused.Addr().(*net.TCPAddr).Port)
	if err := unused.Close(); err != nil {
		t.Fatal(err)
	}
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
