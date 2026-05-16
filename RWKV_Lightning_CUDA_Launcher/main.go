package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const defaultPort = "8000"
const defaultVocabPath = "./rwkv_vocab_v20230424.txt"
const listenAddr = "127.0.0.1:8088"

type startRequest struct {
	ModelPath string `json:"model_path"`
	VocabPath string `json:"vocab_path"`
	Port      string `json:"port"`
	Password  string `json:"password"`
	UseWKV32  bool   `json:"use_wkv32"`
}

type launcher struct {
	mu      sync.Mutex
	cmd     *exec.Cmd
	cancel  context.CancelFunc
	running bool

	logMu   sync.Mutex
	logs    []string
	clients map[chan string]struct{}
}

func newLauncher() *launcher {
	return &launcher{clients: make(map[chan string]struct{})}
}

func (l *launcher) appendLog(line string) {
	line = strings.TrimRight(line, "\r\n")
	if line == "" {
		return
	}

	l.logMu.Lock()
	stamp := time.Now().Format("15:04:05")
	msg := "[" + stamp + "] " + line
	l.logs = append(l.logs, msg)
	if len(l.logs) > 2000 {
		l.logs = l.logs[len(l.logs)-2000:]
	}
	for ch := range l.clients {
		select {
		case ch <- msg:
		default:
		}
	}
	l.logMu.Unlock()
}

func (l *launcher) start(req startRequest) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.running {
		return fmt.Errorf("backend is already running")
	}

	modelPath := strings.TrimSpace(req.ModelPath)
	if modelPath == "" {
		return fmt.Errorf("model path is required")
	}

	vocabPath := strings.TrimSpace(req.VocabPath)
	if vocabPath == "" {
		vocabPath = defaultVocabPath
	}

	port := strings.TrimSpace(req.Port)
	if port == "" {
		port = defaultPort
	}
	if _, err := strconv.Atoi(port); err != nil {
		return fmt.Errorf("invalid port: %s", port)
	}

	exe := backendExecutable()
	if _, err := os.Stat(exe); err != nil {
		return fmt.Errorf("backend executable not found: %s", exe)
	}

	// Adjust these flags if your backend CLI uses different names.
	args := []string{
		"--model-path", modelPath,
		"--vocab-path", vocabPath,
		"--port", port,
	}
	if strings.TrimSpace(req.Password) != "" {
		args = append(args, "--password", strings.TrimSpace(req.Password))
	}
	if req.UseWKV32 {
		args = append(args, "--wkv32")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Dir = appDir()
	cmd.Env = backendProcessEnv(cmd.Dir)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return err
	}

	if err := cmd.Start(); err != nil {
		cancel()
		return err
	}

	l.cmd = cmd
	l.cancel = cancel
	l.running = true
	l.appendLog("started: " + exe + " " + strings.Join(args, " "))

	go pipeScanner(l, "stdout", stdout)
	go pipeScanner(l, "stderr", stderr)

	go func() {
		err := cmd.Wait()
		l.mu.Lock()
		l.running = false
		l.cmd = nil
		l.cancel = nil
		l.mu.Unlock()
		if err != nil {
			l.appendLog("backend exited: " + err.Error())
		} else {
			l.appendLog("backend exited normally")
		}
	}()

	return nil
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

func pipeScanner(l *launcher, name string, r interface{ Read([]byte) (int, error) }) {
	s := bufio.NewScanner(r)
	buf := make([]byte, 0, 1024*1024)
	s.Buffer(buf, 8*1024*1024)
	for s.Scan() {
		l.appendLog(name + ": " + s.Text())
	}
	if err := s.Err(); err != nil {
		l.appendLog(name + " scanner error: " + err.Error())
	}
}

func (l *launcher) stop() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if !l.running || l.cmd == nil {
		return fmt.Errorf("backend is not running")
	}

	l.appendLog("stopping backend...")
	if l.cancel != nil {
		l.cancel()
	}
	if l.cmd.Process != nil {
		_ = l.cmd.Process.Kill()
	}
	return nil
}

func (l *launcher) status() map[string]any {
	l.mu.Lock()
	running := l.running
	l.mu.Unlock()
	return map[string]any{"running": running}
}

func (l *launcher) sse(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	ch := make(chan string, 256)

	l.logMu.Lock()
	for _, line := range l.logs {
		fmt.Fprintf(w, "data: %s\n\n", sseEscape(line))
	}
	l.clients[ch] = struct{}{}
	l.logMu.Unlock()
	flusher.Flush()

	defer func() {
		l.logMu.Lock()
		delete(l.clients, ch)
		l.logMu.Unlock()
		close(ch)
	}()

	for {
		select {
		case <-r.Context().Done():
			return
		case line := <-ch:
			fmt.Fprintf(w, "data: %s\n\n", sseEscape(line))
			flusher.Flush()
		}
	}
}

func sseEscape(s string) string {
	return strings.ReplaceAll(s, "\n", " ")
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

func main() {
	l := newLauncher()

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		_ = page.Execute(w, map[string]string{
			"DefaultPort":  defaultPort,
			"DefaultVocab": defaultVocabPath,
		})
	})

	http.HandleFunc("/api/start", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req startRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, 400, map[string]any{"ok": false, "error": err.Error()})
			return
		}
		if err := l.start(req); err != nil {
			l.appendLog("start failed: " + err.Error())
			writeJSON(w, 400, map[string]any{"ok": false, "error": err.Error()})
			return
		}
		writeJSON(w, 200, map[string]any{"ok": true})
	})

	http.HandleFunc("/api/stop", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if err := l.stop(); err != nil {
			writeJSON(w, 400, map[string]any{"ok": false, "error": err.Error()})
			return
		}
		writeJSON(w, 200, map[string]any{"ok": true})
	})

	http.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, l.status())
	})

	http.HandleFunc("/api/pick-file", func(w http.ResponseWriter, r *http.Request) {
		path, err := pickFile()
		if err != nil {
			writeJSON(w, 400, map[string]any{"ok": false, "error": err.Error()})
			return
		}
		writeJSON(w, 200, map[string]any{"ok": true, "path": path})
	})

	http.HandleFunc("/logs", l.sse)

	url := "http://" + listenAddr
	go func() {
		time.Sleep(350 * time.Millisecond)
		openBrowser(url)
	}()

	log.Printf("RWKV Lightning CUDA Launcher: %s", url)
	log.Printf("Backend executable: %s", backendExecutable())
	if err := http.ListenAndServe(listenAddr, nil); err != nil {
		log.Fatal(err)
	}
}

var page = template.Must(template.New("page").Parse(`<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RWKV Lightning CUDA Control Center</title>
  <style>
    :root {
      --bg: #f7fafc;
      --card: rgba(255,255,255,.86);
      --text: #0f172a;
      --muted: #64748b;
      --border: #dbe4ef;
      --primary: #2563eb;
      --primary2: #1d4ed8;
      --danger: #dc2626;
      --danger2: #b91c1c;
      --console: #0b1020;
      --consoleText: #d5e7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 10%, #dbeafe 0, transparent 30%),
        radial-gradient(circle at 90% 20%, #e0e7ff 0, transparent 28%),
        linear-gradient(135deg, #f8fafc, #eef6ff);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 28px;
    }
    .shell {
      width: min(980px, 100%);
      background: var(--card);
      border: 1px solid rgba(219, 228, 239, .9);
      border-radius: 24px;
      box-shadow: 0 24px 80px rgba(15, 23, 42, .12);
      backdrop-filter: blur(16px);
      overflow: hidden;
    }
    .header {
      padding: 24px 28px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .title { font-size: 22px; font-weight: 750; letter-spacing: -.02em; }
    .subtitle { margin-top: 4px; color: var(--muted); font-size: 13px; }
    .badge {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      color: var(--muted);
      background: rgba(255,255,255,.65);
      white-space: nowrap;
    }
    .content { padding: 14px 28px 28px; }
    .grid { display: grid; gap: 12px; }
    .row { display: grid; grid-template-columns: 130px minmax(0, 1fr) 96px; gap: 10px; align-items: center; }
    .row.small { grid-template-columns: 130px minmax(0, 1fr) 130px minmax(0, 1fr); }
    label { font-size: 13px; font-weight: 650; color: #334155; }
    input {
      width: 100%;
      height: 40px;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0 12px;
      color: var(--text);
      background: rgba(255,255,255,.85);
      outline: none;
      transition: border .15s, box-shadow .15s;
    }
    input:focus { border-color: #93c5fd; box-shadow: 0 0 0 4px rgba(147,197,253,.26); }
    button {
      height: 40px;
      border: 0;
      border-radius: 12px;
      padding: 0 16px;
      font-weight: 700;
      cursor: pointer;
      transition: transform .08s, background .15s, opacity .15s;
    }
    button:active { transform: translateY(1px); }
    .browse { background: #eaf2ff; color: #1e40af; }
    .browse:hover { background: #dbeafe; }
    .actions { display: flex; gap: 12px; margin-top: 16px; }
    .start { background: var(--primary); color: white; min-width: 120px; }
    .start:hover { background: var(--primary2); }
    .stop { background: #fee2e2; color: var(--danger); min-width: 120px; }
    .stop:hover { background: #fecaca; color: var(--danger2); }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .console-wrap { margin-top: 18px; border: 1px solid #101827; border-radius: 18px; overflow: hidden; background: var(--console); }
    .console-top { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #111827; color: #9ca3af; font-size: 12px; }
    #console {
      width: 100%;
      height: 390px;
      padding: 14px;
      background: var(--console);
      color: var(--consoleText);
      border: 0;
      resize: vertical;
      outline: none;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12.5px;
      line-height: 1.55;
    }
    .hint { margin-top: 10px; color: var(--muted); font-size: 12px; }
    @media (max-width: 720px) {
      body { padding: 12px; }
      .header { display: block; }
      .badge { margin-top: 12px; display: inline-block; }
      .row, .row.small { grid-template-columns: 1fr; }
      .actions { flex-direction: column; }
      button { width: 100%; }
    }
	input[type="checkbox"] {
	  margin-right: 6px;
	  width: 16px;
	  height: 16px;
	  vertical-align: middle;
	}
  </style>
</head>
<body>
  <main class="shell">
    <div class="header">
      <div>
        <div class="title">RWKV Lightning CUDA Control Center</div>
        <div class="subtitle">Local launcher for rwkv_lighting_cuda</div>
      </div>
      <div class="badge" id="status">Checking...</div>
    </div>

    <div class="content">
      <div class="grid">
        <div class="row">
          <label>Model Path</label>
          <input id="model" placeholder="Select model file..." />
          <button class="browse" onclick="pickFile('model')">Browse</button>
        </div>
        <div class="row">
          <label>Vocab Path</label>
          <input id="vocab" value="{{.DefaultVocab}}" />
          <button class="browse" onclick="pickFile('vocab')">Browse</button>
        </div>
		<div class="row small">
		  <label>Port</label>
		  <input id="port" value="{{.DefaultPort}}" inputmode="numeric" />
		  <label>Password</label>
		  <input id="password" type="password" placeholder="empty = disabled" />
		</div>
		<div class="row" style="align-items: center;">
		  <label></label>
		  <label style="display: flex; align-items: center; font-weight: 500; color: #334155; gap: 8px;">
		    <input type="checkbox" id="useWkv32" />
		    Use FP32 WKV (More accurate, Bsz=1 Have almost same speed, High concurrency use more VRAM)
		  </label>
		  <label></label>
		</div>
      </div>

      <div class="console-wrap">
        <div class="console-top">
          <span>terminal output</span>
          <span id="logState">connected</span>
        </div>
        <textarea id="console" readonly></textarea>
      </div>

      <div class="actions">
        <button class="start" id="startBtn" onclick="startBackend()">Start</button>
        <button class="stop" id="stopBtn" onclick="stopBackend()">Stop</button>
      </div>
      <div class="hint">Password is only passed to backend when non-empty. Backend executable must be next to this launcher.</div>
    </div>
  </main>

<script>
const consoleBox = document.getElementById('console');
const statusBox = document.getElementById('status');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

function log(line) {
  consoleBox.value += line + "\n";
  consoleBox.scrollTop = consoleBox.scrollHeight;
}

async function pickFile(target) {
  try {
    const r = await fetch('/api/pick-file');
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || 'file picker failed');
    if (j.path) document.getElementById(target).value = j.path;
  } catch (e) {
    log('[launcher] ' + e.message);
  }
}

async function startBackend() {
  const payload = {
    model_path: document.getElementById('model').value,
    vocab_path: document.getElementById('vocab').value,
    port: document.getElementById('port').value,
    password: document.getElementById('password').value,
  	use_wkv32: document.getElementById('useWkv32').checked,
  };
  try {
    const r = await fetch('/api/start', { method: 'POST', body: JSON.stringify(payload) });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || 'start failed');
    await refreshStatus();
  } catch (e) {
    log('[launcher] ' + e.message);
  }
}

async function stopBackend() {
  try {
    const r = await fetch('/api/stop', { method: 'POST' });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || 'stop failed');
    await refreshStatus();
  } catch (e) {
    log('[launcher] ' + e.message);
  }
}

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const j = await r.json();
    statusBox.textContent = j.running ? 'Backend running' : 'Backend stopped';
    statusBox.style.color = j.running ? '#166534' : '#64748b';
    startBtn.disabled = !!j.running;
    stopBtn.disabled = !j.running;
  } catch (_) {
    statusBox.textContent = 'Launcher offline';
  }
}

const es = new EventSource('/logs');
es.onmessage = (e) => log(e.data);
es.onerror = () => { document.getElementById('logState').textContent = 'reconnecting...'; };
es.onopen = () => { document.getElementById('logState').textContent = 'connected'; };

refreshStatus();
setInterval(refreshStatus, 1200);
</script>
</body>
</html>`))
