package main

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const defaultPort = "8000"
const defaultVocabPath = "./rwkv_vocab_v20230424.txt"
const listenAddr = "127.0.0.1:8088"

//go:embed dist/*
var webFiles embed.FS

type startRequest struct {
	ModelPath            string `json:"model_path"`
	VocabPath            string `json:"vocab_path"`
	Port                 string `json:"port"`
	Password             string `json:"password"`
	UseWKV32             bool   `json:"use_wkv32"`
	ChunkLoad            bool   `json:"chunk_load"`
	EnableDynamicLoading bool   `json:"enable_dynamic_loading"`
	ChunkSize            int    `json:"chunk_size"`
	StateDBPath          string `json:"state_db_path"`
	TuneCache            string `json:"tune_cache"`
}
type tuneRequest struct {
	Model       string  `json:"model"`
	Data        string  `json:"data"`
	Output      string  `json:"output"`
	Vocab       string  `json:"vocab"`
	Ctx         int     `json:"ctx"`
	Chunk       int     `json:"chunk"`
	Epochs      int     `json:"epochs"`
	BatchSize   int     `json:"batch_size"`
	MaxSteps    int     `json:"max_steps"`
	LR          float64 `json:"lr"`
	LRFinal     float64 `json:"lr_final"`
	WarmupSteps int     `json:"warmup_steps"`
	SaveEvery   int     `json:"save_every"`
	Seed        int     `json:"seed"`
}
type process struct {
	mu         sync.Mutex
	cmd        *exec.Cmd
	cancel     context.CancelFunc
	done       chan struct{}
	state      string
	errorText  string
	started    time.Time
	finished   time.Time
	logs       []string
	secret     string
	checkpoint string
	progress   map[string]any
	losses     []map[string]any
}

func newProcess() *process {
	return &process{state: "offline", logs: []string{}, losses: []map[string]any{}}
}

var progressRE = regexp.MustCompile(`\]\s+(\d+)/(\d+) epoch=(\d+)/(\d+) loss=([\d.eE+\-]+).* lr=([\d.eE+\-]+) tok/s=([\d.eE+\-]+) ETA=([\d.eE+\-]+)s`)

func (p *process) appendLog(line string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.secret != "" {
		line = strings.ReplaceAll(line, p.secret, "[redacted]")
	}
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	p.logs = append(p.logs, "["+time.Now().Format("15:04:05")+"] "+line)
	if len(p.logs) > 2000 {
		p.logs = append([]string{}, p.logs[len(p.logs)-2000:]...)
	}
	if m := progressRE.FindStringSubmatch(line); m != nil {
		keys := []string{"step", "total", "epoch", "epochs", "loss", "lr", "tokens_per_second", "eta"}
		v := map[string]any{}
		for i, k := range keys {
			n, _ := strconv.ParseFloat(m[i+1], 64)
			v[k] = n
		}
		p.progress = v
		p.losses = append(p.losses, map[string]any{"step": v["step"], "loss": v["loss"]})
		if len(p.losses) > 2000 {
			p.losses = p.losses[len(p.losses)-2000:]
		}
	}
	if i := strings.Index(line, "saved: "); i >= 0 {
		p.checkpoint = strings.TrimSpace(line[i+7:])
		if !filepath.IsAbs(p.checkpoint) {
			p.checkpoint = filepath.Join(appDir(), p.checkpoint)
		}
	}
}

// State tuning renders progress with carriage returns, not newline-delimited logs.
func splitProgress(data []byte, atEOF bool) (int, []byte, error) {
	if i := bytes.IndexAny(data, "\r\n"); i >= 0 {
		return i + 1, data[:i], nil
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}
func (p *process) launch(exe string, args []string, secret string) error {
	p.mu.Lock()
	if p.cmd != nil {
		p.mu.Unlock()
		return fmt.Errorf("process is already running")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Dir = appDir()
	cmd.Env = backendProcessEnv(cmd.Dir)
	// Drain both pipes before Wait to avoid losing the final checkpoint/log lines.
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		p.mu.Unlock()
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		p.mu.Unlock()
		return err
	}
	if err = cmd.Start(); err != nil {
		cancel()
		p.state = "error"
		p.errorText = err.Error()
		p.mu.Unlock()
		return err
	}
	p.cmd = cmd
	p.cancel = cancel
	p.done = make(chan struct{})
	p.state = "starting"
	p.errorText = ""
	p.secret = secret
	p.started = time.Now()
	p.finished = time.Time{}
	p.checkpoint = ""
	p.progress = nil
	p.losses = []map[string]any{}
	p.logs = []string{}
	done := p.done
	p.mu.Unlock()
	p.appendLog("started: " + exe) // Never log credential-bearing argv.
	var wg sync.WaitGroup
	for name, pipe := range map[string]io.ReadCloser{"stdout": stdout, "stderr": stderr} {
		wg.Add(1)
		go func(name string, pipe io.ReadCloser) {
			defer wg.Done()
			s := bufio.NewScanner(pipe)
			s.Buffer(make([]byte, 65536), 8*1024*1024)
			s.Split(splitProgress)
			for s.Scan() {
				p.appendLog(name + ": " + s.Text())
			}
			if e := s.Err(); e != nil {
				p.appendLog(name + " scanner error: " + e.Error())
			}
		}(name, pipe)
	}
	go func() {
		wg.Wait()
		err := cmd.Wait()
		cancel()
		p.mu.Lock()
		stopped := p.state == "stopping"
		p.cmd = nil
		p.cancel = nil
		p.finished = time.Now()
		if err != nil && !stopped {
			p.state = "error"
			p.errorText = err.Error()
		} else if stopped {
			p.state = "offline"
		} else {
			p.state = "completed"
		}
		p.mu.Unlock()
		if err != nil && !stopped {
			p.appendLog("process exited: " + err.Error())
		} else {
			p.appendLog("process exited")
		}
		close(done)
	}()
	return nil
}
func (p *process) active() bool { p.mu.Lock(); defer p.mu.Unlock(); return p.cmd != nil }
func (p *process) stop() error {
	p.mu.Lock()
	if p.cmd == nil {
		p.mu.Unlock()
		return nil
	}
	p.state = "stopping"
	p.cancel()
	done := p.done
	p.mu.Unlock()
	select {
	case <-done:
		return nil
	case <-time.After(10 * time.Second):
		return fmt.Errorf("process did not exit within 10 seconds")
	}
}
func (p *process) snapshot() map[string]any {
	p.mu.Lock()
	defer p.mu.Unlock()
	elapsed := 0.0
	if !p.started.IsZero() {
		end := p.finished
		if end.IsZero() {
			end = time.Now()
		}
		elapsed = end.Sub(p.started).Seconds()
	}
	return map[string]any{"status": p.state, "running": p.cmd != nil, "error": p.errorText, "logs": append([]string{}, p.logs...), "progress": p.progress, "losses": append([]map[string]any{}, p.losses...), "checkpoint": p.checkpoint, "elapsed": elapsed}
}

// Keep the legacy runtime log stream available to existing local clients.
func (p *process) sse(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", 500)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	ticker := time.NewTicker(300 * time.Millisecond)
	defer ticker.Stop()
	last := ""
	for {
		p.mu.Lock()
		lines := append([]string{}, p.logs...)
		p.mu.Unlock()
		start := 0
		if last != "" {
			for i := len(lines) - 1; i >= 0; i-- {
				if lines[i] == last {
					start = i + 1
					break
				}
			}
		}
		for _, line := range lines[start:] {
			if _, err := fmt.Fprintf(w, "data: %s\n\n", line); err != nil {
				return
			}
			last = line
		}
		flusher.Flush()
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
		}
	}
}

type launcher struct {
	mu      sync.Mutex
	runtime *process
	tuning  *process
	config  startRequest
}

func newLauncher() *launcher {
	return &launcher{runtime: newProcess(), tuning: newProcess(), config: startRequest{Port: defaultPort, VocabPath: defaultVocabPath, ChunkSize: 128, StateDBPath: "rwkv_sessions.db"}}
}
func existingPath(path string, dir bool) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("path is required")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(appDir(), path)
	}
	st, e := os.Stat(path)
	if e != nil {
		return e
	}
	if st.IsDir() != dir {
		return fmt.Errorf("wrong path type: %s", path)
	}
	return nil
}
func runtimeArgs(req startRequest) ([]string, error) {
	if err := existingPath(req.ModelPath, req.EnableDynamicLoading); err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	if req.VocabPath == "" {
		req.VocabPath = defaultVocabPath
	}
	if err := existingPath(req.VocabPath, false); err != nil {
		return nil, fmt.Errorf("vocab: %w", err)
	}
	if req.Port == "" {
		req.Port = defaultPort
	}
	port, e := strconv.Atoi(req.Port)
	if e != nil || port < 1 || port > 65535 || port == 8088 {
		return nil, fmt.Errorf("port must be 1–65535 and different from launcher port 8088")
	}
	if req.ChunkSize == 0 {
		req.ChunkSize = 128
	}
	if req.ChunkSize < 1 {
		return nil, fmt.Errorf("prefill chunk size must be positive")
	}
	args := []string{"--model-path", req.ModelPath, "--vocab-path", req.VocabPath, "--host", "127.0.0.1", "--port", req.Port, "--chunk-size", strconv.Itoa(req.ChunkSize)}
	for _, pair := range [][2]string{{"--password", req.Password}, {"--state-db-path", req.StateDBPath}, {"--tune-cache", req.TuneCache}} {
		if pair[1] != "" {
			args = append(args, pair[0], pair[1])
		}
	}
	for _, flag := range []struct {
		on   bool
		name string
	}{{req.UseWKV32, "--wkv32"}, {req.ChunkLoad, "--chunk-load"}, {req.EnableDynamicLoading, "--enable-dynamic-loading"}} {
		if flag.on {
			args = append(args, flag.name)
		}
	}
	return args, nil
}
func (l *launcher) start(req startRequest) error {
	if l.tuning.active() {
		return fmt.Errorf("state tuning is using the GPU; stop tuning first")
	}
	if l.runtime.active() {
		return fmt.Errorf("backend is already running")
	}
	args, err := runtimeArgs(req)
	if err != nil {
		return err
	}
	if req.Port == "" {
		req.Port = defaultPort
	}
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", req.Port), 300*time.Millisecond)
	if err == nil {
		conn.Close()
		return fmt.Errorf("port %s is already in use", req.Port)
	}
	if err = l.runtime.launch(backendExecutable(), args, req.Password); err != nil {
		return err
	}
	l.config = req
	return nil
}
func (l *launcher) status() map[string]any {
	l.mu.Lock()
	config := l.config
	l.mu.Unlock()
	out := l.runtime.snapshot()
	safe := config
	safe.Password = ""
	out["config"] = safe
	out["base_url"] = "http://127.0.0.1:" + config.Port
	out["translation_adapter"] = true
	if out["running"] == true && out["status"] != "stopping" {
		client := http.Client{Timeout: 1500 * time.Millisecond}
		resp, err := client.Get("http://127.0.0.1:" + config.Port + "/v1/server/status")
		if err == nil {
			defer resp.Body.Close()
			var data map[string]any
			if resp.StatusCode == 200 && json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&data) == nil && data["status"] == "running" {
				out["status"] = "ready"
				out["backend"] = data
			}
		}
	}
	if out["status"] == "completed" {
		out["status"] = "offline"
	}
	return out
}
func validateDataset(path string) (int, error) {
	if err := existingPath(path, false); err != nil {
		return 0, err
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(appDir(), path)
	}
	f, e := os.Open(path)
	if e != nil {
		return 0, e
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 16<<20)
	count, line := 0, 0
	for s.Scan() {
		line++
		if len(s.Bytes()) == 0 {
			continue
		}
		dec := json.NewDecoder(strings.NewReader(s.Text()))
		tok, err := dec.Token()
		if err != nil || tok != json.Delim('{') {
			return 0, fmt.Errorf("line %d: expected JSON object", line)
		}
		key, err := dec.Token()
		if err != nil || key != "text" {
			return 0, fmt.Errorf("line %d: only text field is supported", line)
		}
		value, valueErr := dec.Token()
		_, isString := value.(string)
		if valueErr != nil || !isString {
			return 0, fmt.Errorf("line %d: text must be a string", line)
		}
		if dec.More() {
			return 0, fmt.Errorf("line %d: exactly one text field is required", line)
		}
		if tok, err = dec.Token(); err != nil || tok != json.Delim('}') {
			return 0, fmt.Errorf("line %d: invalid object", line)
		}
		if _, err = dec.Token(); err != io.EOF {
			return 0, fmt.Errorf("line %d: trailing data", line)
		}
		count++
	}
	if e = s.Err(); e != nil {
		return 0, e
	}
	if count == 0 {
		return 0, fmt.Errorf("dataset is empty")
	}
	return count, nil
}
func tuningArgs(req tuneRequest) ([]string, error) {
	if err := existingPath(req.Model, false); err != nil {
		return nil, err
	}
	if !strings.EqualFold(filepath.Ext(req.Model), ".pth") {
		return nil, fmt.Errorf("state tuning requires a BF16 .pth base model")
	}
	if _, err := validateDataset(req.Data); err != nil {
		return nil, err
	}
	if strings.TrimSpace(req.Output) == "" {
		return nil, fmt.Errorf("output directory is required")
	}
	if req.Ctx < 1 || req.Chunk < 1 || req.Epochs < 1 || req.BatchSize < 1 || req.LR <= 0 || req.LRFinal <= 0 || req.MaxSteps < 0 || req.WarmupSteps < 0 || req.SaveEvery < 0 || req.Seed < 0 {
		return nil, fmt.Errorf("invalid training parameter; sizes and learning rates must be positive, counts nonnegative")
	}
	args := []string{"--model", req.Model, "--data", req.Data, "--output", req.Output}
	if req.Vocab != "" {
		if err := existingPath(req.Vocab, false); err != nil {
			return nil, err
		}
		args = append(args, "--vocab", req.Vocab)
	}
	for _, p := range []struct {
		name string
		n    int
	}{{"ctx", req.Ctx}, {"chunk", req.Chunk}, {"epochs", req.Epochs}, {"batch-size", req.BatchSize}, {"max-steps", req.MaxSteps}, {"warmup-steps", req.WarmupSteps}, {"save-every", req.SaveEvery}, {"seed", req.Seed}} {
		args = append(args, "--"+p.name, strconv.Itoa(p.n))
	}
	args = append(args, "--lr", strconv.FormatFloat(req.LR, 'g', -1, 64), "--lr-final", strconv.FormatFloat(req.LRFinal, 'g', -1, 64))
	return args, nil
}
func (l *launcher) proxy(w http.ResponseWriter, r *http.Request) {
	l.mu.Lock()
	config := l.config
	l.mu.Unlock()
	target, _ := url.Parse("http://127.0.0.1:" + config.Port)
	if r.URL.Path == "/v1/chat/completions" && r.Method == "POST" {
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 16<<20))
		if err != nil {
			writeJSON(w, 400, map[string]any{"error": err.Error()})
			return
		}
		var payload map[string]json.RawMessage
		if err = json.Unmarshal(body, &payload); err != nil {
			writeJSON(w, 400, map[string]any{"error": err.Error()})
			return
		}
		// CUDA chat adds a User/Assistant envelope. Raw contents must use the existing generic continuation handler.
		if _, raw := payload["contents"]; raw {
			if _, chat := payload["messages"]; !chat {
				r.URL.Path = "/v1/batch/completions"
			}
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = -1
	original := proxy.Director
	proxy.Director = func(req *http.Request) {
		original(req)
		if req.Header.Get("Authorization") == "" && config.Password != "" {
			req.Header.Set("Authorization", "Bearer "+config.Password)
		}
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, e error) {
		writeJSON(w, 502, map[string]any{"error": "runtime connection: " + e.Error()})
	}
	proxy.ServeHTTP(w, r)
}
func decode(w http.ResponseWriter, r *http.Request, v any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if e := d.Decode(v); e != nil {
		return e
	}
	if e := d.Decode(new(any)); e != io.EOF {
		return fmt.Errorf("unexpected trailing request data")
	}
	return nil
}
func (l *launcher) handler() http.Handler {
	mux := http.NewServeMux()
	web, _ := fs.Sub(webFiles, "dist")
	mux.Handle("/", http.FileServer(http.FS(web)))
	api := func(path, method string, f func(http.ResponseWriter, *http.Request) error) {
		mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != method {
				writeJSON(w, 405, map[string]any{"error": "method not allowed"})
				return
			}
			if err := f(w, r); err != nil {
				writeJSON(w, 400, map[string]any{"error": err.Error()})
			}
		})
	}
	api("/api/status", "GET", func(w http.ResponseWriter, r *http.Request) error { writeJSON(w, 200, l.status()); return nil })
	for _, action := range []string{"start", "stop", "restart"} {
		action := action
		api("/api/"+action, "POST", func(w http.ResponseWriter, r *http.Request) error {
			l.mu.Lock()
			defer l.mu.Unlock()
			req := l.config
			if action == "start" {
				if e := decode(w, r, &req); e != nil {
					return e
				}
			}
			if action != "start" {
				if e := l.runtime.stop(); e != nil {
					return e
				}
			}
			if action != "stop" {
				if e := l.start(req); e != nil {
					l.runtime.appendLog("start failed: " + e.Error())
					l.runtime.mu.Lock()
					if l.runtime.cmd == nil {
						l.runtime.state = "error"
						l.runtime.errorText = e.Error()
					}
					l.runtime.mu.Unlock()
					return e
				}
			}
			writeJSON(w, 200, map[string]any{"ok": true})
			return nil
		})
	}
	api("/api/pick-file", "POST", func(w http.ResponseWriter, r *http.Request) error {
		path, e := pickFile()
		if e != nil {
			return e
		}
		writeJSON(w, 200, map[string]any{"path": path})
		return nil
	})
	api("/api/tuning/validate", "POST", func(w http.ResponseWriter, r *http.Request) error {
		var req struct {
			Path string `json:"path"`
		}
		if e := decode(w, r, &req); e != nil {
			return e
		}
		n, e := validateDataset(req.Path)
		if e != nil {
			return e
		}
		writeJSON(w, 200, map[string]any{"samples": n})
		return nil
	})
	api("/api/tuning/status", "GET", func(w http.ResponseWriter, r *http.Request) error {
		out := l.tuning.snapshot()
		name := "rwkv_state_tune"
		if runtime.GOOS == "windows" {
			name += ".exe"
		}
		_, err := os.Stat(filepath.Join(appDir(), name))
		out["available"] = err == nil
		writeJSON(w, 200, out)
		return nil
	})
	api("/api/tuning/start", "POST", func(w http.ResponseWriter, r *http.Request) error {
		var req tuneRequest
		if e := decode(w, r, &req); e != nil {
			return e
		}
		args, e := tuningArgs(req)
		if e != nil {
			return e
		}
		l.mu.Lock()
		defer l.mu.Unlock()
		if l.runtime.active() {
			return fmt.Errorf("inference is using the GPU; stop inference before starting tuning")
		}
		name := "rwkv_state_tune"
		if runtime.GOOS == "windows" {
			name += ".exe"
		}
		if e = l.tuning.launch(filepath.Join(appDir(), name), args, ""); e != nil {
			return e
		}
		l.tuning.mu.Lock()
		if l.tuning.cmd != nil {
			l.tuning.state = "running"
		}
		l.tuning.mu.Unlock()
		writeJSON(w, 200, map[string]any{"ok": true})
		return nil
	})
	api("/api/tuning/stop", "POST", func(w http.ResponseWriter, r *http.Request) error {
		l.mu.Lock()
		defer l.mu.Unlock()
		if e := l.tuning.stop(); e != nil {
			return e
		}
		writeJSON(w, 200, map[string]any{"ok": true})
		return nil
	})
	api("/api/tuning/open-folder", "POST", func(w http.ResponseWriter, r *http.Request) error {
		l.tuning.mu.Lock()
		path := l.tuning.checkpoint
		l.tuning.mu.Unlock()
		if path == "" {
			return fmt.Errorf("no saved checkpoint")
		}
		folder := filepath.Dir(path)
		var cmd *exec.Cmd
		switch runtime.GOOS {
		case "windows":
			cmd = exec.Command("explorer", folder)
		case "darwin":
			cmd = exec.Command("open", folder)
		default:
			cmd = exec.Command("xdg-open", folder)
		}
		if e := cmd.Start(); e != nil {
			return e
		}
		go cmd.Wait()
		writeJSON(w, 200, map[string]any{"ok": true})
		return nil
	})
	mux.HandleFunc("/logs", l.runtime.sse)
	mux.HandleFunc("/v1/", l.proxy)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Loopback binding plus Host/Origin checks prevent cross-site launcher control and DNS rebinding.
		host, _, err := net.SplitHostPort(r.Host)
		if err != nil {
			host = r.Host
		}
		if host != "localhost" && host != "127.0.0.1" && host != "[::1]" && host != "::1" {
			writeJSON(w, 403, map[string]any{"error": "local host required"})
			return
		}
		if origin := r.Header.Get("Origin"); origin != "" {
			u, e := url.Parse(origin)
			if e != nil || u.Host != r.Host {
				writeJSON(w, 403, map[string]any{"error": "same-origin request required"})
				return
			}
		}
		if r.Header.Get("Sec-Fetch-Site") == "cross-site" {
			writeJSON(w, 403, map[string]any{"error": "cross-site request rejected"})
			return
		}
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		mux.ServeHTTP(w, r)
	})
}
func main() {
	l := newLauncher()
	url := "http://" + listenAddr
	if os.Getenv("RWKV_LAUNCHER_NO_BROWSER") != "1" {
		go func() { time.Sleep(350 * time.Millisecond); openBrowser(url) }()
	}
	log.Printf("RWKV Lightning Launcher: %s", url)
	server := http.Server{Addr: listenAddr, Handler: l.handler(), ReadHeaderTimeout: 5 * time.Second}
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-shutdown
		l.mu.Lock()
		_ = l.runtime.stop()
		_ = l.tuning.stop()
		l.mu.Unlock()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
func backendProcessEnv(baseDir string) []string {
	env := os.Environ()
	if runtime.GOOS != "windows" {
		return env
	}

	libDir := filepath.Join(baseDir, "lib")
	if st, err := os.Stat(libDir); err != nil || !st.IsDir() {
		return env
	}

	oldPath, key := getEnvCaseInsensitive("PATH")
	newPath := libDir
	if oldPath != "" {
		newPath = libDir + ";" + oldPath
	}

	prefix := key + "="
	replaced := false
	for i := range env {
		if strings.HasPrefix(strings.ToUpper(env[i]), "PATH=") {
			env[i] = prefix + newPath
			replaced = true
			break
		}
	}
	if !replaced {
		env = append(env, prefix+newPath)
	}
	return env
}

func getEnvCaseInsensitive(name string) (value string, key string) {
	for _, e := range os.Environ() {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) != 2 {
			continue
		}
		if strings.EqualFold(parts[0], name) {
			return parts[1], parts[0]
		}
	}
	return "", name
}

func backendExecutable() string {
	name := "rwkv_lighting_cuda"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(appDir(), name)
}

func appDir() string {
	exe, err := os.Executable()
	if err != nil {
		return "."
	}
	return filepath.Dir(exe)
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}

func pickFile() (string, error) {
	switch runtime.GOOS {
	case "windows":
		ps := `Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.OpenFileDialog; $d.Filter = 'All files (*.*)|*.*'; if ($d.ShowDialog() -eq 'OK') { Write-Output $d.FileName }`
		out, err := exec.Command("powershell", "-NoProfile", "-STA", "-Command", ps).Output()
		return strings.TrimSpace(string(out)), err
	case "darwin":
		out, err := exec.Command("osascript", "-e", `POSIX path of (choose file)`).Output()
		return strings.TrimSpace(string(out)), err
	default:
		for _, tool := range [][]string{
			{"zenity", "--file-selection"},
			{"kdialog", "--getopenfilename", "."},
			{"yad", "--file-selection"},
		} {
			if _, err := exec.LookPath(tool[0]); err == nil {
				out, err := exec.Command(tool[0], tool[1:]...).Output()
				return strings.TrimSpace(string(out)), err
			}
		}
		return "", fmt.Errorf("no native file picker found; install zenity/kdialog/yad or type path manually")
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
