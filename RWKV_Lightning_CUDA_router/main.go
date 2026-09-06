package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/BurntSushi/toml"
)

type config struct {
	Listen                 string          `toml:"listen"`
	MaxRequestBodyBytes    int64           `toml:"max_request_body_bytes"`
	RequestTimeoutSeconds  int             `toml:"request_timeout_seconds"`
	FailureCooldownSeconds int             `toml:"failure_cooldown_seconds"`
	Backends               []backendConfig `toml:"backends"`
}

type backendConfig struct {
	Name   string  `toml:"name"`
	URL    string  `toml:"url"`
	Weight float64 `toml:"weight"`
}

type backend struct {
	name      string
	baseURL   *url.URL
	weight    float64
	inflight  int64
	unhealthy time.Time
}

type scheduler struct {
	mu       sync.Mutex
	backends []*backend
	next     uint64
	sessions map[string]*backend
	states   map[string]*backend
	cooldown time.Duration
}

func newScheduler(c config) (*scheduler, error) {
	if c.Listen == "" {
		return nil, errors.New("listen is required")
	}
	if _, _, err := net.SplitHostPort(c.Listen); err != nil {
		return nil, fmt.Errorf("listen must be host:port: %w", err)
	}
	if c.MaxRequestBodyBytes < 0 || c.RequestTimeoutSeconds < 0 || c.FailureCooldownSeconds < 0 {
		return nil, errors.New("timeout, cooldown, and body limit cannot be negative")
	}
	if len(c.Backends) == 0 {
		return nil, errors.New("at least one [[backends]] entry is required")
	}
	s := &scheduler{
		sessions: make(map[string]*backend),
		states:   make(map[string]*backend),
		cooldown: time.Duration(c.FailureCooldownSeconds) * time.Second,
	}
	for i, item := range c.Backends {
		u, err := url.Parse(item.URL)
		if err != nil || u.Scheme == "" || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
			return nil, fmt.Errorf("backends[%d].url must be an absolute http(s) URL", i)
		}
		if item.Weight == 0 {
			item.Weight = 1
		}
		if item.Weight < 0 || math.IsNaN(item.Weight) || math.IsInf(item.Weight, 0) {
			return nil, fmt.Errorf("backends[%d].weight must be positive", i)
		}
		name := item.Name
		if name == "" {
			name = u.Host
		}
		s.backends = append(s.backends, &backend{name: name, baseURL: u, weight: item.Weight})
	}
	return s, nil
}

func (s *scheduler) acquire(bsz int64, session, stateID string) (*backend, func(), error) {
	if bsz < 1 {
		bsz = 1
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	if stateID != "" {
		if b := s.states[stateID]; b != nil {
			if now.Before(b.unhealthy) {
				return nil, nil, errors.New("backend holding the uploaded state is temporarily unavailable")
			}
			b.inflight += bsz
			if session != "" {
				s.sessions[session] = b
			}
			return b, s.releaseFunc(b, bsz), nil
		}
	}
	if session != "" {
		if b := s.sessions[session]; b != nil && !now.Before(b.unhealthy) {
			b.inflight += bsz
			return b, s.releaseFunc(b, bsz), nil
		}
	}
	var chosen *backend
	best := math.Inf(1)
	for offset := 0; offset < len(s.backends); offset++ {
		b := s.backends[(int(s.next)+offset)%len(s.backends)]
		if now.Before(b.unhealthy) {
			continue
		}
		load := float64(b.inflight) / b.weight
		if load < best {
			best, chosen = load, b
		}
	}
	if chosen == nil {
		return nil, nil, errors.New("all upstream backends are temporarily unavailable")
	}
	s.next = (s.next + 1) % uint64(len(s.backends))
	chosen.inflight += bsz
	if session != "" {
		s.sessions[session] = chosen
	}
	return chosen, s.releaseFunc(chosen, bsz), nil
}

func (s *scheduler) bindState(stateID string, b *backend) {
	if stateID == "" || b == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.states == nil {
		s.states = make(map[string]*backend)
	}
	s.states[stateID] = b
}

func (s *scheduler) unbindState(stateID string) {
	if stateID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.states, stateID)
}

func (s *scheduler) releaseFunc(b *backend, bsz int64) func() {
	var once sync.Once
	return func() { once.Do(func() { s.mu.Lock(); b.inflight -= bsz; s.mu.Unlock() }) }
}

func (s *scheduler) failed(b *backend) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cooldown > 0 {
		b.unhealthy = time.Now().Add(s.cooldown)
	}
}

type proxy struct {
	scheduler *scheduler
	client    *http.Client
	maxBody   int64
	timeout   time.Duration
}

func (p *proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, err := p.readBody(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusRequestEntityTooLarge)
		return
	}
	bsz, session := batchSize(r.URL.Path, body)
	stateID := stateIDFromBody(r.URL.Path, body)
	if stateID == "" {
		stateID = r.URL.Query().Get("state_id")
	}
	b, release, err := p.scheduler.acquire(bsz, session, stateID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer release()
	ctx := r.Context()
	if p.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, p.timeout)
		defer cancel()
	}
	target := *b.baseURL
	target.Path = joinPath(b.baseURL.Path, r.URL.Path)
	target.RawQuery = r.URL.RawQuery
	upstream, err := http.NewRequestWithContext(ctx, r.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		http.Error(w, "invalid upstream request", http.StatusBadGateway)
		return
	}
	copyHeaders(upstream.Header, r.Header)
	upstream.Host = b.baseURL.Host
	resp, err := p.client.Do(upstream)
	if err != nil {
		p.scheduler.failed(b)
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	copyHeaders(w.Header(), resp.Header)
	if r.URL.Path == "/v1/state/upload" && resp.StatusCode >= 200 && resp.StatusCode < 300 {
		responseBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			p.scheduler.failed(b)
			http.Error(w, "upstream response read failed", http.StatusBadGateway)
			return
		}
		var uploaded struct {
			StateID string `json:"state_id"`
		}
		if json.Unmarshal(responseBody, &uploaded) == nil {
			p.scheduler.bindState(uploaded.StateID, b)
		}
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(responseBody)
		return
	}
	if r.URL.Path == "/v1/state/delete" && resp.StatusCode >= 200 && resp.StatusCode < 300 {
		p.scheduler.unbindState(stateID)
	}
	w.WriteHeader(resp.StatusCode)
	destination := io.Writer(w)
	if strings.HasPrefix(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		destination = flushingWriter{ResponseWriter: w}
	}
	if _, err := io.Copy(destination, resp.Body); err != nil && !errors.Is(err, context.Canceled) {
		log.Printf("proxy %s: response copy: %v", b.name, err)
	}
}

func stateIDFromBody(path string, body []byte) string {
	if path == "/v1/state/upload" || len(body) == 0 {
		return ""
	}
	var payload struct {
		StateID string `json:"state_id"`
	}
	if json.Unmarshal(body, &payload) != nil {
		return ""
	}
	return payload.StateID
}

type flushingWriter struct{ http.ResponseWriter }

func (w flushingWriter) Write(p []byte) (int, error) {
	n, err := w.ResponseWriter.Write(p)
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
	return n, err
}

func (p *proxy) readBody(r *http.Request) ([]byte, error) {
	if p.maxBody > 0 && r.ContentLength > p.maxBody {
		return nil, errors.New("request body exceeds configured limit")
	}
	reader := io.Reader(r.Body)
	if p.maxBody > 0 {
		reader = io.LimitReader(r.Body, p.maxBody+1)
	}
	body, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	if p.maxBody > 0 && int64(len(body)) > p.maxBody {
		return nil, errors.New("request body exceeds configured limit")
	}
	return body, nil
}

func batchSize(path string, body []byte) (int64, string) {
	if !strings.HasPrefix(path, "/state/") && !strings.Contains(path, "completion") && !strings.Contains(path, "translate") && !strings.Contains(path, "FIM") && !strings.Contains(path, "big_batch") {
		return 1, ""
	}
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) != nil {
		return 1, ""
	}
	session := ""
	if raw := payload["session_id"]; raw != nil {
		_ = json.Unmarshal(raw, &session)
	}
	for _, key := range []string{"contents", "text_list", "prompts", "inputs"} {
		var values []json.RawMessage
		if raw := payload[key]; raw != nil && json.Unmarshal(raw, &values) == nil && len(values) > 0 {
			return int64(len(values)), session
		}
	}
	for _, key := range []string{"bsz", "batch_size"} {
		var value int64
		if raw := payload[key]; raw != nil && json.Unmarshal(raw, &value) == nil && value > 0 {
			return value, session
		}
	}
	return 1, session
}

func copyHeaders(dst, src http.Header) {
	for k, values := range src {
		if hopHeader(k) {
			continue
		}
		dst.Del(k)
		for _, v := range values {
			dst.Add(k, v)
		}
	}
}
func hopHeader(k string) bool {
	switch strings.ToLower(k) {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade":
		return true
	}
	return false
}
func joinPath(base, request string) string {
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(request, "/")
}

func main() {
	path := flag.String("config", "config.toml", "TOML configuration file")
	flag.Parse()
	var c config
	meta, err := toml.DecodeFile(*path, &c)
	if err != nil {
		log.Fatalf("read config: %v", err)
	}
	if unknown := meta.Undecoded(); len(unknown) > 0 {
		log.Fatalf("read config: unknown TOML key %q", unknown[0])
	}
	s, err := newScheduler(c)
	if err != nil {
		log.Fatalf("invalid config: %v", err)
	}
	p := &proxy{scheduler: s, maxBody: c.MaxRequestBodyBytes, timeout: time.Duration(c.RequestTimeoutSeconds) * time.Second, client: &http.Client{Transport: &http.Transport{Proxy: http.ProxyFromEnvironment, MaxIdleConns: 1024, MaxIdleConnsPerHost: 256, IdleConnTimeout: 90 * time.Second, ResponseHeaderTimeout: 0, ExpectContinueTimeout: time.Second}}}
	server := &http.Server{Addr: c.Listen, Handler: p, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 120 * time.Second}
	log.Printf("RWKV router listening on %s with %d backends", c.Listen, len(s.backends))
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
