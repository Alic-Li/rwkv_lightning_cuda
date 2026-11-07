package router

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

type proxy struct {
	scheduler    *scheduler
	client       *http.Client
	maxBody      int64
	timeout      time.Duration
	uploadPolicy uploadRetryPolicy
}

func (p *proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Retired endpoints must not reach legacy upstream workers.
	if r.URL.Path == "/v1/adapters" || strings.HasPrefix(r.URL.Path, "/v1/adapters/") {
		http.NotFound(w, r)
		return
	}
	body, err := p.readBody(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusRequestEntityTooLarge)
		return
	}
	if isSynchronizedStateRoute(r.URL.Path) {
		p.serveSynchronizedState(w, r, body)
		return
	}
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) == nil {
		for _, key := range []string{"adapter_id", "adapter_version", "adapter_scale"} {
			if _, exists := payload[key]; exists {
				http.Error(w, "request adapters are no longer supported", http.StatusBadRequest)
				return
			}
		}
	}
	bsz, session := batchSize(r, body)
	stateID, err := resolvedRequestStateID(r, body)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if stateID != "" && p.scheduler.needsStateRefresh(stateID) {
		p.refreshState(r, body)
	}
	b, release, err := p.scheduler.acquire(bsz, session, stateID)
	if err != nil {
		log.Printf("router route_unavailable path=%q session_id=%q state_id=%q backends=%s",
			r.URL.Path, session, stateID, p.scheduler.routeDiagnostic(stateID))
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer release()
	if stateID != "" || session != "" {
		log.Printf("router route_selected backend=%q path=%q session_id=%q state_id=%q",
			b.name, r.URL.Path, session, stateID)
	}
	w.Header().Set("X-RWKV-Backend", b.name)
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
		// Transport failure says nothing about registered resources.
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	if stateID != "" && (resp.StatusCode == 400 || resp.StatusCode == 404) {
		// A restarted runtime can lose its temporary State store.
		errorBody, readErr := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		if readErr == nil && bytes.Contains(errorBody, []byte("uploaded state not found")) {
			p.scheduler.stateUnavailable(stateID, b)
		}
		copyHeaders(w.Header(), resp.Header)
		w.Header().Set("X-RWKV-Backend", b.name)
		w.WriteHeader(resp.StatusCode)
		w.Write(errorBody)
		return
	}
	copyHeaders(w.Header(), resp.Header)
	w.Header().Set("X-RWKV-Backend", b.name)
	w.WriteHeader(resp.StatusCode)
	destination := io.Writer(w)
	if strings.HasPrefix(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		destination = flushingWriter{ResponseWriter: w}
	}
	if _, err := io.Copy(destination, resp.Body); err != nil && !errors.Is(err, context.Canceled) {
		if r.Context().Err() == nil {
			p.scheduler.failed(b)
		}
		log.Printf("proxy %s: response copy: %v", b.name, err)
	}
}
