package main

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
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
const defaultListen = "127.0.0.1:10721"

// launcherVersion is stamped by release builds via
// -ldflags "-X main.launcherVersion=v<VERSION>".
var launcherVersion = "dev"

//go:embed dist/*
var webFiles embed.FS

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
func (p *process) launch(exe string, args []string, secret string, deviceSpec string) error {
	p.mu.Lock()
	if p.cmd != nil {
		p.mu.Unlock()
		return fmt.Errorf("process is already running")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Dir = appDir()
	cmd.Env = backendProcessEnv(cmd.Dir)
	// §5.8: GPU selection is environment injection, never a child flag.
	if deviceSpec != "" {
		vendor, _ := gpuVendor()
		cmd.Env = applyVisibleDevices(cmd.Env, deviceSpec, vendor)
	}
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

// pid is the running child's PID, or 0 when nothing is running. The metrics
// sampler uses it to tell the runtime's memory from everyone else's.
func (p *process) pid() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cmd == nil || p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}
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

// SSE log stream shared by the runtime and both job processes.
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
	mu                 sync.Mutex
	runtime            *process
	tuning             *process
	quantization       *process
	config             startRequest
	quantizationOutput string

	// Last accepted job configs (fs whitelist + dashboard display only).
	tuningConfig       tuneRequest
	quantizationConfig quantizeRequest

	// §5.8 device state of the managed GPU processes (guarded by mu).
	// Quantization is absent on purpose — it never touches a GPU, see
	// handleQuantizationStart.
	runtimeDevices resolvedDevices
	tuningDevices  resolvedDevices

	// Startup form (§1): one binary, roles chosen at startup time.
	listen     string
	token      string
	clientOnly bool
	card       string
	configPath string

	backends *registry
	fs       *fsWhitelist
}

func newLauncher() *launcher {
	l := &launcher{
		runtime:      newProcess(),
		tuning:       newProcess(),
		quantization: newProcess(),
		config:       startRequest{Port: defaultPort, VocabPath: defaultVocabPath, ChunkSize: 128, StateDBPath: "rwkv_sessions.db"},
		listen:       defaultListen,
	}
	l.backends = openRegistry(defaultConfigPath())
	l.fs = openFSWhitelist(filepath.Join(appDir(), fsRootsFile))
	return l
}

// loopbackListen reports whether the HTTP listener is loopback-only (§2:
// the Client must only ever bind loopback; the Agent may bind wider with a
// mandatory token).
func (l *launcher) loopbackListen() bool {
	host, _, err := net.SplitHostPort(l.listen)
	if err != nil {
		return false
	}
	switch host {
	case "127.0.0.1", "::1", "localhost":
		return true
	}
	return false
}

// role reports the startup form: "full" (Client+Agent on this machine),
// "agent" (server node), or "client" (no local runtime — a legal state, not
// an error). Detection is lazy: dropping a runtime binary next to the
// launcher upgrades the form without a restart.
func (l *launcher) role() string {
	if l.clientOnly || !l.agentCapable() {
		return "client"
	}
	if l.loopbackListen() {
		return "full"
	}
	return "agent"
}

func (l *launcher) agentCapable() bool {
	if l.clientOnly {
		return false
	}
	return binaryPresent(backendExecutable())
}

func binaryPresent(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func toolBinary(name string) string {
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(appDir(), name)
}

// capabilities is an open set (§5.4): clients must ignore unknown bits.
// T2 decision: quantization stays a single bit — one tool binary serves
// both w8a16 and w4a16; the format is a per-job config, not a deployment
// property.
func (l *launcher) capabilities() []string {
	if l.role() == "client" {
		return []string{}
	}
	caps := []string{}
	if binaryPresent(backendExecutable()) {
		caps = append(caps, "runtime")
	}
	if binaryPresent(toolBinary("rwkv_state_tune")) {
		caps = append(caps, "tuning_state")
	}
	if binaryPresent(toolBinary("rwkv_miss_tune")) {
		caps = append(caps, "tuning_miss")
	}
	if binaryPresent(toolBinary("rwkv_quantize")) {
		caps = append(caps, "quantization")
	}
	// state_import: this Agent can hand a node-local .pth to the runtime's
	// multipart upload, which the native API offers no path form of. The
	// console hides the "import from this node" control on agents without it.
	caps = append(caps, "metrics", "fs", "state_import")
	if l.role() == "full" {
		caps = append(caps, "host_dialog")
	}
	return caps
}

// status is the RuntimeState payload shared by /api/v1/runtime and
// /api/v1/node (§6.1): process state plus available (can this host start a
// runtime at all), visible_devices (the raw spec the current runtime was
// pinned to, §5.8(b)), card and managed.
func (l *launcher) status() map[string]any {
	l.mu.Lock()
	config := l.config
	devices := l.runtimeDevices
	l.mu.Unlock()
	out := l.runtime.snapshot()
	safe := config
	safe.Password = ""
	out["config"] = safe
	out["base_url"] = "http://127.0.0.1:" + config.Port
	out["translation_adapter"] = true
	out["available"] = l.agentCapable()
	out["visible_devices"] = devices.spec
	// The --card pin, so clients can lock their device pickers to it; empty
	// when the launcher was started without --card (§5.8).
	out["card"] = l.card
	// A runtime the Agent did not spawn itself (started by hand or by another
	// launcher) used to read as "offline" forever: the /v1/server/status probe
	// only ran for managed processes. Probe the configured port regardless, so
	// an external server shows up as ready — with managed=false, since
	// stop/restart cannot act on a process we do not own.
	managed, _ := out["running"].(bool)
	out["managed"] = managed
	// A Client cannot start a runtime, so it has no port worth probing: keep
	// its status payload a pure reflection of process state (§6.1 contract).
	// Gate on the explicit --client-only flag, not role(): a host missing the
	// native binary is still exactly where someone starts the server by hand.
	if !l.clientOnly && out["status"] != "stopping" {
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
	out["running"] = out["status"] == "ready" || out["status"] == "running"
	if out["status"] == "completed" {
		out["status"] = "offline"
	}
	return out
}

// nodePayload is GET /api/v1/node (§6.1): everything status() reports plus
// the node identity fields the Client's capability probe reads. This is the
// only shape a node is ever identified by.
func (l *launcher) nodePayload() map[string]any {
	out := l.status()
	out["role"] = "agent"
	out["version"] = launcherVersion
	out["capabilities"] = l.capabilities()
	return out
}

func (l *launcher) tuningStatus() map[string]any {
	out := l.tuning.snapshot()
	out["available"] = binaryPresent(toolBinary("rwkv_state_tune"))
	out["miss_available"] = binaryPresent(toolBinary("rwkv_miss_tune"))
	return out
}

func (l *launcher) quantizationStatus() map[string]any {
	out := l.quantization.snapshot()
	out["available"] = binaryPresent(toolBinary("rwkv_quantize"))
	l.mu.Lock()
	out["output_path"] = l.quantizationOutput
	l.mu.Unlock()
	return out
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
func (l *launcher) runtimeArgs(req startRequest, devices resolvedDevices) ([]string, error) {
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
	if e != nil || port < 1 || port > 65535 {
		return nil, fmt.Errorf("port must be 1–65535")
	}
	// The launcher's own HTTP port is the only reserved one; the historical
	// hardcoded 8088 check is gone (the launcher no longer listens there).
	if _, own, err := net.SplitHostPort(l.listen); err == nil {
		if ownPort, err2 := strconv.Atoi(own); err2 == nil && ownPort == port {
			return nil, fmt.Errorf("port %d is used by the launcher HTTP server", port)
		}
	}
	if req.ChunkSize == 0 {
		req.ChunkSize = 128
	}
	if req.ChunkSize < 1 {
		return nil, fmt.Errorf("prefill chunk size must be positive")
	}
	args := []string{"--model-path", req.ModelPath, "--vocab-path", req.VocabPath, "--host", "127.0.0.1", "--port", req.Port, "--chunk-size", strconv.Itoa(req.ChunkSize)}
	tuneCache := req.TuneCache
	if tuneCache == "" {
		// §5.8(c): with a pinned card the default tune cache is card-bound
		// so a card switch cannot silently reuse another card's tuning.
		// `devices` was resolved once by the caller: re-resolving here would
		// let auto placement sample free VRAM twice and pin the process and
		// its tune cache to different cards.
		tuneCache = deviceTuneCache(req, devices)
	}
	for _, pair := range [][2]string{{"--password", req.Password}, {"--state-db-path", req.StateDBPath}, {"--tune-cache", tuneCache}} {
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
	if err := l.validateDeviceRequest(req.VisibleDevices); err != nil {
		return err
	}
	devices := l.resolveVisibleDevices(req.VisibleDevices)
	// §5.8(a): the global runtime/tuning exclusion is per-card now — only
	// processes whose resolved devices overlap are blocked.
	if l.tuning.active() && devicesOverlap(devices, l.tuningDevices) {
		return fmt.Errorf("state tuning is using the GPU; stop tuning first")
	}
	if l.runtime.active() {
		return fmt.Errorf("backend is already running")
	}
	args, err := l.runtimeArgs(req, devices)
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
	if err = l.runtime.launch(backendExecutable(), args, req.Password, devices.spec); err != nil {
		return err
	}
	if devices.auto {
		l.runtime.appendLog("auto device placement: GPU " + devices.spec + " (most free VRAM)")
	}
	if note := tuneCacheNote(req, devices); note != "" {
		l.runtime.appendLog(note)
	}
	l.config = req
	l.runtimeDevices = devices
	l.recordFSConfigs(req, l.tuningConfig, l.quantizationConfig)
	return nil
}

// runtimeLoadRequest is POST /api/v1/runtime/load: put the runtime (and, in
// dynamic-loading mode, a model) onto a chosen card. Restart-based by design
// — CUDA binds devices to the process at init — so a card switch inherently
// interrupts active inference; the request contract says so.
type runtimeLoadRequest struct {
	Model          string `json:"model"`
	VisibleDevices string `json:"visible_devices"`
}

const runtimeLoadReadyTimeout = 10 * time.Minute

// runtimeLoad switches the managed runtime onto a card using the saved
// config: stop → start with the card injected as CUDA_VISIBLE_DEVICES →
// wait for readiness → (dynamic mode) POST /v1/model/load. It runs with the
// saved config because that carries the real runtime password, which the
// WebUI never sees (status() blanks it on the wire).
func (l *launcher) runtimeLoad(req runtimeLoadRequest) (map[string]any, error) {
	l.mu.Lock()
	cfg := l.config
	l.mu.Unlock()
	if cfg.ModelPath == "" {
		return nil, errors.New("no saved runtime config: start the runtime once from the Runtime page first")
	}
	if spec := strings.TrimSpace(req.VisibleDevices); spec != "" {
		cfg.VisibleDevices = &spec
	} else {
		cfg.VisibleDevices = nil // no card chosen: §5.8 chain, auto placement included
	}
	// Refuse a pin violation before stopping the current runtime: a failed
	// start after the stop would leave the node with nothing serving.
	if err := l.validateDeviceRequest(cfg.VisibleDevices); err != nil {
		return nil, err
	}

	l.mu.Lock()
	if l.runtime.active() {
		if e := l.runtime.stop(); e != nil {
			l.mu.Unlock()
			return nil, fmt.Errorf("stopping the current runtime failed: %w", e)
		}
	}
	e := l.start(cfg)
	l.mu.Unlock()
	if e != nil {
		l.runtime.appendLog("card switch failed: " + e.Error())
		return nil, e
	}
	l.mu.Lock()
	spec := l.runtimeDevices.spec
	l.mu.Unlock()

	// Readiness is what status() reports: the native server answering
	// /v1/server/status. Non-dynamic cold loads of multi-GB models live
	// inside this window, hence the generous cap.
	deadline := time.Now().Add(runtimeLoadReadyTimeout)
	for {
		state, _ := l.status()["status"].(string)
		if state == "ready" {
			break
		}
		if state == "error" {
			return nil, errors.New("runtime failed to start on the requested card; see the runtime log")
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("runtime did not become ready within %s", runtimeLoadReadyTimeout)
		}
		time.Sleep(800 * time.Millisecond)
	}

	loaded := ""
	target := strings.TrimSpace(req.Model)
	if cfg.EnableDynamicLoading && target != "" {
		client := http.Client{Timeout: 15 * time.Minute}
		body, err := json.Marshal(map[string]string{"model": target})
		if err != nil {
			return nil, err
		}
		httpReq, err := http.NewRequest("POST",
			"http://127.0.0.1:"+cfg.Port+"/v1/model/load", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if cfg.Password != "" {
			httpReq.Header.Set("Authorization", "Bearer "+cfg.Password)
		}
		resp, err := client.Do(httpReq)
		if err != nil {
			return nil, fmt.Errorf("model load request failed: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			return nil, fmt.Errorf("model load failed: HTTP %d: %s",
				resp.StatusCode, strings.TrimSpace(string(snippet)))
		}
		loaded = target
		l.runtime.appendLog("model " + target + " loaded on GPU " + spec)
	} else if spec != "" {
		l.runtime.appendLog("runtime restarted on GPU " + spec)
	}
	return map[string]any{"ok": true, "visible_devices": spec, "model": loaded}, nil
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
	if req.Method != "" && req.Method != "state" && req.Method != "miss" {
		return nil, fmt.Errorf("training method must be state or miss")
	}
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
	if req.BatchSize < 1 || req.BatchSize > 128 {
		return nil, fmt.Errorf("batch size must be between 1 and 128")
	}
	if req.Optimizer == "" {
		req.Optimizer = "adam"
	}
	if req.Optimizer != "adam" && req.Optimizer != "muon" {
		return nil, fmt.Errorf("optimizer must be adam or muon")
	}
	if req.Ctx < 1 || req.Chunk < 1 || req.Chunk > req.Ctx || req.Epochs < 1 || req.LR <= 0 || req.LRFinal <= 0 || req.MaxSteps < 0 || req.WarmupSteps < 0 || req.SaveEvery < 0 || req.Seed < 0 {
		return nil, fmt.Errorf("invalid training parameter; sizes and learning rates must be positive, counts nonnegative")
	}
	args := []string{"--model", req.Model, "--data", req.Data, "--output", req.Output, "--optimizer", req.Optimizer}
	if req.Method == "miss" {
		if req.Optimizer != "adam" || req.Rank < 1 || req.Rank > 1024 {
			return nil, fmt.Errorf("MiSS requires Adam and rank between 1 and 1024")
		}
		allowed := map[string]bool{"att.receptance.weight": true, "att.key.weight": true, "att.value.weight": true, "att.output.weight": true, "ffn.key.weight": true, "ffn.value.weight": true}
		seen := map[string]bool{}
		if req.Targets != "all" {
			for _, target := range strings.Split(req.Targets, ",") {
				if !allowed[target] || seen[target] {
					return nil, fmt.Errorf("invalid or duplicate MiSS target: %s", target)
				}
				seen[target] = true
			}
		}
		args = append(args, "--rank", strconv.Itoa(req.Rank), "--alpha", strconv.FormatFloat(req.Alpha, 'g', -1, 64), "--targets", req.Targets)
		if req.State != "" {
			if err := existingPath(req.State, false); err != nil {
				return nil, err
			}
			args = append(args, "--state", req.State)
		}
		if req.Resume != "" {
			if err := existingPath(req.Resume, true); err != nil {
				return nil, err
			}
			args = append(args, "--resume", req.Resume)
		}
	}
	if req.WKVTape {
		args = append(args, "--wkv_tape")
	}
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
func quantizationArgs(req quantizeRequest) ([]string, error) {
	if err := existingPath(req.InputPath, false); err != nil {
		return nil, fmt.Errorf("input model: %w", err)
	}
	if !strings.EqualFold(filepath.Ext(req.InputPath), ".pth") {
		return nil, fmt.Errorf("quantization requires a BF16 .pth input model")
	}
	if strings.TrimSpace(req.OutputPath) == "" {
		return nil, fmt.Errorf("output path is required")
	}
	if !strings.EqualFold(filepath.Ext(req.OutputPath), ".rwkvq") {
		return nil, fmt.Errorf("quantized output must use the .rwkvq extension")
	}
	input := req.InputPath
	if !filepath.IsAbs(input) {
		input = filepath.Join(appDir(), input)
	}
	output := req.OutputPath
	if !filepath.IsAbs(output) {
		output = filepath.Join(appDir(), output)
	}
	input = filepath.Clean(input)
	output = filepath.Clean(output)
	if strings.EqualFold(input, output) {
		return nil, fmt.Errorf("input and output paths must differ")
	}
	parent := filepath.Dir(output)
	if err := existingPath(parent, true); err != nil {
		return nil, fmt.Errorf("output directory: %w", err)
	}
	if _, err := os.Stat(output); err == nil {
		return nil, fmt.Errorf("output already exists: %s", output)
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("output: %w", err)
	}
	if req.Format == "" {
		req.Format = "w4a16"
	}
	if req.Format != "w8a16" && req.Format != "w4a16" {
		return nil, fmt.Errorf("format must be w8a16 or w4a16")
	}
	args := []string{"--format", req.Format}
	if req.Format == "w4a16" {
		if req.GroupSize == 0 {
			req.GroupSize = 128
		}
		if req.GroupSize != 32 && req.GroupSize != 128 {
			return nil, fmt.Errorf("W4A16 group size must be 32 or 128")
		}
		args = append(args, "--group-size", strconv.Itoa(req.GroupSize))
	}
	return append(args, req.InputPath, req.OutputPath), nil
}
func (l *launcher) proxy(w http.ResponseWriter, r *http.Request) {
	l.mu.Lock()
	config := l.config
	l.mu.Unlock()
	target, _ := url.Parse("http://127.0.0.1:" + config.Port)
	// /v1 is a transparent pass-through. It used to sniff POSTs to
	// /v1/chat/completions and rewrite a `contents` body to
	// /v1/batch/completions, because the old WebUI sent raw continuations to
	// the chat path. That WebUI is gone — the current console posts raw
	// continuations to /v1/batch/completions itself (see
	// src/lib/api/inference.ts) — so the launcher no longer buffers and
	// re-parses up to 16 MiB of every chat request to guess what the caller
	// meant.
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = -1
	original := proxy.Director
	proxy.Director = func(req *http.Request) {
		original(req)
		// The Agent token authenticates this hop only and means nothing to
		// the runtime, so drop it. Without an Agent token the caller may be
		// supplying the runtime's own password — keep that one.
		if l.token != "" {
			req.Header.Del("Authorization")
		}
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

// defaultConfigPath is the T4 decision: the Client registry lives under
// the user's home so it survives launcher binary upgrades (and per-card
// deployment folders being swapped wholesale). macOS/Linux:
// ~/.rwkv_launcher/launcher.json, Windows: %USERPROFILE%\.rwkv_launcher\launcher.json.
func defaultConfigPath() string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".rwkv_launcher", "launcher.json")
	}
	return filepath.Join(appDir(), "launcher.json")
}

// validateStartup enforces the §5.1 hard constraints before anything binds:
// a non-loopback listen without a token refuses to start (the /api surface
// can spawn arbitrary processes — loopback makes that a feature, otherwise
// it is unauthenticated RCE), and a client-only form must stay loopback.
func validateStartup(listen, token string, clientOnly, agentCapable bool) error {
	host, portText, err := net.SplitHostPort(listen)
	if err != nil {
		return fmt.Errorf("--listen must be host:port, got %q", listen)
	}
	if port, err := strconv.Atoi(portText); err != nil || port < 1 || port > 65535 {
		return fmt.Errorf("--listen port must be 1–65535")
	}
	loopback := false
	switch host {
	case "127.0.0.1", "::1", "localhost":
		loopback = true
	case "":
		// ":port" binds every interface — treat as non-loopback.
	}
	if !loopback && token == "" {
		return fmt.Errorf("refusing to listen on %s without --token: the control API can start local processes; a non-loopback listener without a token is unauthenticated remote code execution", listen)
	}
	if (clientOnly || !agentCapable) && !loopback {
		return fmt.Errorf("a client-only launcher must bind loopback (Client serves the WebUI on 127.0.0.1; it never manages processes)")
	}
	return nil
}

func main() {
	listen := flag.String("listen", defaultListen, "HTTP listen address; non-loopback requires --token")
	token := flag.String("token", "", "bearer token protecting every control and inference path")
	client := flag.Bool("client", false, "run as a pure Client: WebUI + backend registry, no local runtime")
	configPath := flag.String("config", defaultConfigPath(), "backend registry file (JSON, 0600)")
	card := flag.String("card", "", "default GPU selection for managed processes (CUDA_VISIBLE_DEVICES style, e.g. 0 or 0,1)")
	flag.Parse()

	l := newLauncher()
	l.listen, l.token, l.clientOnly, l.card = *listen, *token, *client, *card
	if *configPath != "" {
		l.configPath = *configPath
		l.backends = openRegistry(l.configPath)
	}
	if err := validateStartup(l.listen, l.token, l.clientOnly, l.agentCapable()); err != nil {
		log.Fatalf("rwkv_launcher: %v", err)
	}
	url := "http://" + l.listen
	if l.loopbackListen() && os.Getenv("RWKV_LAUNCHER_NO_BROWSER") != "1" {
		go func() { time.Sleep(350 * time.Millisecond); openBrowser(url) }()
	}
	log.Printf("RWKV Lightning Launcher %s: %s (role: %s)", launcherVersion, url, l.role())
	server := http.Server{Addr: l.listen, Handler: l.handler(), ReadHeaderTimeout: 5 * time.Second}
	// Bind before probing. The "local" backend probe dials this very port, so
	// a probe that wins the race against the listener caches Reachable=false
	// for a full probeInterval — which greys out the runtime controls and the
	// composer on a freshly started launcher.
	ln, err := net.Listen("tcp", l.listen)
	if err != nil {
		log.Fatal(err)
	}
	// Display-only probes at startup and every probeInterval; results never
	// gate any logic (I3).
	go l.backends.probeAll(l.localBackend)
	go l.backends.probeLoop(l.localBackend)
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-shutdown
		l.mu.Lock()
		_ = l.runtime.stop()
		_ = l.tuning.stop()
		_ = l.quantization.stop()
		l.mu.Unlock()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()
	if err := server.Serve(ln); err != nil && err != http.ErrServerClosed {
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

func pickDirectory() (string, error) {
	switch runtime.GOOS {
	case "windows":
		ps := `Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.FolderBrowserDialog; if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }`
		out, err := exec.Command("powershell", "-NoProfile", "-STA", "-Command", ps).Output()
		return strings.TrimSpace(string(out)), err
	case "darwin":
		out, err := exec.Command("osascript", "-e", `POSIX path of (choose folder)`).Output()
		return strings.TrimSpace(string(out)), err
	default:
		for _, tool := range [][]string{
			{"zenity", "--file-selection", "--directory"},
			{"kdialog", "--getexistingdirectory", "."},
			{"yad", "--file-selection", "--directory"},
		} {
			if _, err := exec.LookPath(tool[0]); err == nil {
				out, err := exec.Command(tool[0], tool[1:]...).Output()
				return strings.TrimSpace(string(out)), err
			}
		}
		return "", fmt.Errorf("no native folder picker found; install zenity/kdialog/yad or type path manually")
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
