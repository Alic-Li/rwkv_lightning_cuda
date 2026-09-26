package main

// §5.3–§5.4 Client-side backend registry and forwarding. The registry is
// the only persistent Client state (0600 JSON file). Every registered node
// speaks /api/v1 — there is no protocol adapter any more. Forwarding strips
// the /api/v1/backends/{id} prefix, swaps the Authorization header for the
// backend's token, removes browser provenance after entrance validation,
// and passes remaining end-to-end headers, status, method and body through
// byte-for-byte — never re-marshalling JSON (the Agent's decode() uses
// DisallowUnknownFields, so any re-marshal is a cross-version 400 bomb).

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type backendEntry struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	BaseURL string `json:"base_url"`
	Token   string `json:"token,omitempty"`
}

type probeResult struct {
	Kind         string   `json:"kind"`
	Capabilities []string `json:"capabilities"`
	Reachable    bool     `json:"reachable"`
	LastProbe    int64    `json:"last_probe"`
	ProbeError   string   `json:"probe_error"`
}

type backendView struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	BaseURL      string   `json:"base_url"`
	HasToken     bool     `json:"has_token"`
	Kind         string   `json:"kind"`
	Capabilities []string `json:"capabilities"`
	Reachable    bool     `json:"reachable"`
	LastProbe    int64    `json:"last_probe"`
	ProbeError   string   `json:"probe_error"`
}

type registry struct {
	mu      sync.Mutex
	path    string
	loaded  bool
	entries []backendEntry
	probes  map[string]*probeResult
}

func openRegistry(path string) *registry {
	return &registry{path: path, probes: map[string]*probeResult{}}
}

func (rg *registry) ensureLoaded() {
	if rg.loaded {
		return
	}
	rg.loaded = true
	data, err := os.ReadFile(rg.path)
	if err != nil || len(data) > 1<<20 {
		return
	}
	var file struct {
		Backends []backendEntry `json:"backends"`
	}
	if json.Unmarshal(data, &file) != nil {
		return
	}
	for _, e := range file.Backends {
		if e.ID == "" || e.BaseURL == "" || e.ID == "local" {
			continue
		}
		rg.entries = append(rg.entries, e)
	}
}

// save commits a candidate snapshot before the caller changes in-memory state.
func (rg *registry) save(entries []backendEntry) error {
	if err := os.MkdirAll(filepath.Dir(rg.path), 0o700); err != nil {
		return fmt.Errorf("save backend registry: %w", err)
	}
	data, err := json.MarshalIndent(map[string]any{"backends": entries}, "", "  ")
	if err != nil {
		return fmt.Errorf("save backend registry: %w", err)
	}
	f, err := os.CreateTemp(filepath.Dir(rg.path), ".launcher-*.tmp")
	if err != nil {
		return fmt.Errorf("save backend registry: %w", err)
	}
	defer os.Remove(f.Name())
	// CreateTemp creates a new 0600 file; never follow an existing .tmp symlink.
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Rename(f.Name(), rg.path)
	}
	if err != nil {
		return fmt.Errorf("save backend registry: %w", err)
	}
	return nil
}

// validateBaseURL enforces §5.3: scheme + host, no /v1 suffix. base_url must
// reach both /api/* and /v1/*.
func validateBaseURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", fmt.Errorf("base_url must be an http(s) URL like http://100.64.0.10:18766")
	}
	if strings.HasSuffix(strings.TrimSuffix(u.Path, "/"), "/v1") || strings.HasSuffix(strings.TrimSuffix(u.Path, "/"), "/api") {
		return "", fmt.Errorf("base_url must not include a /v1 or /api suffix")
	}
	return strings.TrimSuffix(raw, "/"), nil
}

func (rg *registry) add(name, baseURL, token string) (backendEntry, error) {
	baseURL, err := validateBaseURL(baseURL)
	if err != nil {
		return backendEntry{}, err
	}
	buf := make([]byte, 3)
	if _, err := rand.Read(buf); err != nil {
		return backendEntry{}, err
	}
	e := backendEntry{
		ID:      hex.EncodeToString(buf),
		Name:    strings.TrimSpace(name),
		BaseURL: baseURL,
		Token:   strings.TrimSpace(token),
	}
	if e.Name == "" {
		e.Name = "backend " + e.ID
	}
	rg.mu.Lock()
	defer rg.mu.Unlock()
	rg.ensureLoaded()
	for _, existing := range rg.entries {
		if existing.ID == e.ID {
			return backendEntry{}, fmt.Errorf("backend id collision; retry the request")
		}
	}
	entries := append(append([]backendEntry{}, rg.entries...), e)
	if err := rg.save(entries); err != nil {
		return backendEntry{}, err
	}
	rg.entries = entries
	return e, nil
}

// update changes the fields the caller sent and leaves the rest alone. The id
// is random and survives an edit: renaming or re-addressing a node must not
// move it out from under whatever the console has selected.
func (rg *registry) update(id string, name, baseURL, token *string) (backendEntry, error) {
	if id == "local" {
		return backendEntry{}, errLocalBackend
	}
	rg.mu.Lock()
	defer rg.mu.Unlock()
	rg.ensureLoaded()
	for i, e := range rg.entries {
		if e.ID != id {
			continue
		}
		next := e
		if name != nil {
			next.Name = strings.TrimSpace(*name)
			if next.Name == "" {
				return backendEntry{}, fmt.Errorf("name must not be empty")
			}
		}
		if baseURL != nil {
			parsed, err := validateBaseURL(*baseURL)
			if err != nil {
				return backendEntry{}, err
			}
			next.BaseURL = parsed
		}
		if token != nil {
			// Empty means "leave the stored token alone": the console never
			// receives it, so an untouched field cannot mean "clear it".
			if trimmed := strings.TrimSpace(*token); trimmed != "" {
				next.Token = trimmed
			}
		}
		entries := append([]backendEntry{}, rg.entries...)
		entries[i] = next
		if err := rg.save(entries); err != nil {
			return backendEntry{}, err
		}
		rg.entries = entries
		return next, nil
	}
	return backendEntry{}, errUnknownBackend
}

func (rg *registry) remove(id string) error {
	rg.mu.Lock()
	defer rg.mu.Unlock()
	rg.ensureLoaded()
	for i, e := range rg.entries {
		if e.ID == id {
			entries := append([]backendEntry{}, rg.entries[:i]...)
			entries = append(entries, rg.entries[i+1:]...)
			if err := rg.save(entries); err != nil {
				return err
			}
			rg.entries = entries
			delete(rg.probes, id)
			return nil
		}
	}
	if id == "local" {
		return errLocalBackend
	}
	return errUnknownBackend
}

func (rg *registry) lookup(id string) (backendEntry, bool) {
	rg.mu.Lock()
	defer rg.mu.Unlock()
	rg.ensureLoaded()
	for _, e := range rg.entries {
		if e.ID == id {
			return e, true
		}
	}
	return backendEntry{}, false
}

var (
	errUnknownBackend = fmt.Errorf("unknown backend id")
	errLocalBackend   = fmt.Errorf("the local backend cannot be removed")
)

func (rg *registry) setProbe(id string, p probeResult) {
	rg.mu.Lock()
	defer rg.mu.Unlock()
	p.LastProbe = time.Now().Unix()
	rg.probes[id] = &p
}

func (rg *registry) probeOf(id string) probeResult {
	rg.mu.Lock()
	defer rg.mu.Unlock()
	if p, ok := rg.probes[id]; ok {
		return *p
	}
	return probeResult{Capabilities: []string{}, LastProbe: 0}
}

// localProvider synthesizes the "local" backend entry in full form (§5.3).
type localProvider func() (backendEntry, bool)

func (l *launcher) localBackend() (backendEntry, bool) {
	if l.role() != "full" {
		return backendEntry{}, false
	}
	host, port, _ := net.SplitHostPort(l.listen)
	if host == "" {
		host = "127.0.0.1"
	}
	// Name is left empty on purpose. The local node's label is the only one
	// the Client owns rather than the user, so the WebUI renders it from its
	// own message catalogue (id === "local") instead of the Go side hard-
	// coding one language. Registered backends keep their user-given names.
	e := backendEntry{ID: "local", BaseURL: "http://" + net.JoinHostPort(host, port)}
	if l.token != "" {
		e.Token = l.token
	}
	return e, true
}

func (rg *registry) list(local localProvider) []backendView {
	rg.mu.Lock()
	rg.ensureLoaded()
	entries := append([]backendEntry{}, rg.entries...)
	rg.mu.Unlock()
	views := []backendView{}
	convert := func(e backendEntry) backendView {
		p := rg.probeOf(e.ID)
		caps := p.Capabilities
		if caps == nil {
			caps = []string{}
		}
		return backendView{
			ID: e.ID, Name: e.Name, BaseURL: e.BaseURL, HasToken: e.Token != "",
			Kind: p.Kind, Capabilities: caps, Reachable: p.Reachable,
			LastProbe: p.LastProbe, ProbeError: p.ProbeError,
		}
	}
	if e, ok := local(); ok {
		views = append(views, convert(e))
	}
	for _, e := range entries {
		views = append(views, convert(e))
	}
	return views
}

// probeBackend implements the §5.4 ladder: /api/v1/node → /v1/server/status.
// A host that answers neither is unreachable; one that answers only the
// latter is a bare inference server, not a node this launcher can drive.
// The capability set is open: unknown values pass through untouched and must
// never fail a client (M3).
func probeBackend(baseURL, token string) probeResult {
	result := probeResult{Capabilities: []string{}}
	fetch := func(path string, out *map[string]any) (int, error) {
		req, err := http.NewRequest("GET", baseURL+path, nil)
		if err != nil {
			return 0, err
		}
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := probeClient.Do(req)
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if err != nil {
			return resp.StatusCode, err
		}
		if resp.StatusCode == 200 {
			if json.Unmarshal(body, out) != nil {
				return resp.StatusCode, fmt.Errorf("invalid JSON from %s", path)
			}
		}
		return resp.StatusCode, nil
	}

	var node map[string]any
	status, err := fetch("/api/v1/node", &node)
	if err == nil && status == 200 {
		if role, _ := node["role"].(string); role == "agent" {
			result.Reachable = true
			result.Kind = "agent"
			result.Capabilities = stringSlice(node["capabilities"])
			return result
		}
		return result.withError(fmt.Sprintf("unexpected /api/v1/node role %q", node["role"]))
	}
	if status == http.StatusUnauthorized {
		return result.withError("authentication failed (HTTP 401): check the backend token")
	}

	var server map[string]any
	status2, err2 := fetch("/v1/server/status", &server)
	if err2 == nil && status2 == 200 {
		result.Reachable = true
		result.Kind = "inference_only"
		result.Capabilities = []string{"inference"}
		return result
	}
	if status2 == http.StatusUnauthorized {
		return result.withError("authentication failed (HTTP 401): check the backend password")
	}
	detail := "HTTP " + strconv.Itoa(status2)
	if err2 != nil {
		detail = err2.Error() // connection-level failure is the informative case
	} else if err != nil {
		detail = err.Error()
	}
	return result.withError("probe failed: " + detail)
}

func (p probeResult) withError(msg string) probeResult {
	p.ProbeError = msg
	return p
}

var probeClient = &http.Client{Timeout: 5 * time.Second}

func stringSlice(v any) []string {
	items, ok := v.([]any)
	if !ok {
		return []string{}
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// probeInterval keeps reachability and last_probe fresh without manual
// re-probes; the frontend registry poll only lists, it never probes.
const probeInterval = 30 * time.Second

// probeAll runs display-only probes (I3: results are shown, never acted on).
// Runs in parallel; a dead backend cannot delay the others.
func (rg *registry) probeAll(local localProvider) {
	rg.mu.Lock()
	rg.ensureLoaded()
	entries := append([]backendEntry{}, rg.entries...)
	rg.mu.Unlock()
	var wg sync.WaitGroup
	probe := func(e backendEntry) {
		defer wg.Done()
		rg.setProbe(e.ID, probeBackend(e.BaseURL, e.Token))
	}
	for _, e := range entries {
		wg.Add(1)
		go probe(e)
	}
	if e, ok := local(); ok {
		wg.Add(1)
		go probe(e)
	}
	wg.Wait()
}

// probeLoop repeats probeAll forever so the registry reflects a backend that
// came up or went down on its own, not only at startup.
func (rg *registry) probeLoop(local localProvider) {
	ticker := time.NewTicker(probeInterval)
	defer ticker.Stop()
	for range ticker.C {
		rg.probeAll(local)
	}
}

// forwardTransport deliberately has no overall or response-header timeout:
// /v1/model/load waits for active inference to drain before swapping VRAM
// (§3.3) and cold loads of multi-GB models take tens of seconds (§8.3).
var forwardTransport = http.DefaultTransport.(*http.Transport).Clone()

// forwardToBackend strips /api/v1/backends/{id} and proxies the rest to the
// backend. The Authorization header is replaced (never forwarded: it holds
// this Client's credentials, which mean nothing to the backend), and
// FlushInterval = -1 keeps SSE streams unbuffered (§5.4, §8.4).
func forwardToBackend(w http.ResponseWriter, r *http.Request, be backendEntry, rest string) {
	base, err := url.Parse(be.BaseURL)
	if err != nil {
		writeJSON(w, 502, map[string]any{"error": "invalid backend base_url"})
		return
	}
	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.Out.URL.Path = "/" + rest
			pr.Out.URL.RawPath = ""
			pr.SetURL(base)
			// Browser provenance was checked at the Client entrance. This is
			// now a server-to-server request with a different Host.
			pr.Out.Header.Del("Origin")
			pr.Out.Header.Del("Sec-Fetch-Site")
			pr.Out.Header.Del("Authorization")
			if be.Token != "" {
				pr.Out.Header.Set("Authorization", "Bearer "+be.Token)
			}
		},
		FlushInterval: -1,
		Transport:     forwardTransport,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			writeJSON(w, 502, map[string]any{"error": "backend unreachable: " + err.Error()})
		},
	}
	proxy.ServeHTTP(w, r)
}
