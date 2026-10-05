package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"
)

type statePlacement struct {
	ready     map[*backend]bool
	active    bool
	deleted   bool
	signature *listedState
}

func (s *scheduler) stateReadyLocked(id string, b *backend) bool {
	if id == "" {
		return true
	}
	entry := s.states[id]
	return entry != nil && !entry.deleted && entry.ready[b]
}

func (s *scheduler) needsStateRefresh(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry := s.states[id]
	if entry == nil {
		return true
	}
	if entry.active || entry.deleted {
		return false
	}
	for _, ready := range entry.ready {
		if ready {
			return false
		}
	}
	return true
}

func (s *scheduler) stateSnapshot() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stateEpoch
}

type stateMutation struct {
	id       string
	previous *statePlacement
	current  *statePlacement
}

func (s *scheduler) beginStateMutation(id string, deleted bool) stateMutation {
	if id == "" {
		return stateMutation{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.states == nil {
		s.states = make(map[string]*statePlacement)
	}
	s.stateEpoch++
	mutation := stateMutation{id: id, previous: s.states[id], current: &statePlacement{ready: make(map[*backend]bool), active: true, deleted: deleted}}
	s.states[id] = mutation.current
	return mutation
}

func (s *scheduler) finishStateMutation(mutation stateMutation, rollback bool) {
	if mutation.id == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.states[mutation.id] != mutation.current {
		return
	}
	// Lists begun during the mutation are stale too, not only those begun before it.
	s.stateEpoch++
	if rollback {
		if mutation.previous == nil {
			delete(s.states, mutation.id)
		} else {
			s.states[mutation.id] = mutation.previous
		}
	} else {
		mutation.current.active = false
	}
}

// Only deterministic pre-mutation rejections can restore previous readiness.
func rejectedStateDeletion(responses []stateResponse) bool {
	for _, r := range responses {
		if r.err != nil {
			return false
		}
		switch r.status {
		case 400, 401, 403, 405, 415, 422:
		default:
			return false
		}
	}
	return true
}

func resolvedRequestStateID(r *http.Request, body []byte) (string, error) {
	values := []string{r.Header.Get("X-RWKV-State-Id"), r.URL.Query().Get("state_id")}
	if len(body) > 0 {
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(body, &payload); err != nil {
			if r.URL.Path == "/v1/state/delete" {
				return "", errors.New("invalid JSON request")
			}
		} else if raw, ok := payload["state_id"]; ok && string(raw) != "null" {
			var id string
			if json.Unmarshal(raw, &id) != nil {
				return "", errors.New("state_id must be a string")
			}
			values = append(values, id)
		}
	}
	id := ""
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if id != "" && id != value {
			return "", errors.New("conflicting state_id values")
		}
		id = value
	}
	return id, nil
}

func (s *scheduler) confirmState(b *backend, info listedState) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry := s.states[info.ID]
	if entry == nil || entry.deleted {
		return false
	}
	if entry.signature != nil && *entry.signature != info {
		return false
	}
	copy := info
	entry.signature = &copy
	entry.ready[b] = true
	b.unhealthy = time.Time{}
	return true
}

func (s *scheduler) stateUnavailable(id string, b *backend) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if entry := s.states[id]; entry != nil {
		entry.ready[b] = false
	}
}

// A stale list started before an upload/delete must not undo the mutation.
func (s *scheduler) observeStateList(b *backend, response stateResponse, epoch uint64) {
	states, err := decodeStateList(response.body)
	if response.err != nil || response.status < 200 || response.status >= 300 {
		err = errors.New("state list unavailable")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stateEpoch != epoch {
		return
	}
	if s.states == nil {
		s.states = make(map[string]*statePlacement)
	}
	for id, entry := range s.states {
		if entry.active || entry.deleted {
			continue
		}
		info, present := states[id]
		entry.ready[b] = err == nil && present && (entry.signature == nil || *entry.signature == info)
	}
	if err != nil {
		return
	}
	for id, info := range states {
		entry := s.states[id]
		if entry == nil {
			copy := info
			entry = &statePlacement{ready: make(map[*backend]bool), signature: &copy}
			s.states[id] = entry
		}
		if !entry.active && !entry.deleted && (entry.signature == nil || *entry.signature == info) {
			entry.ready[b] = true
		}
	}
}

type uploadRetryPolicy struct {
	attempts       int
	attemptTimeout time.Duration
	baseDelay      time.Duration
	maxDelay       time.Duration
}

func configuredUploadPolicy(c config) (uploadRetryPolicy, error) {
	if c.StateUploadMaxAttempts < 0 || c.StateUploadMaxAttempts > 10 || c.StateUploadAttemptTimeoutSeconds < 0 || c.StateUploadAttemptTimeoutSeconds > 86400 || c.StateUploadRetryBaseMS < 0 || c.StateUploadRetryBaseMS > 60000 || c.StateUploadRetryMaxMS < 0 || c.StateUploadRetryMaxMS > 60000 {
		return uploadRetryPolicy{}, errors.New("invalid state upload retry configuration")
	}
	p := uploadRetryPolicy{attempts: c.StateUploadMaxAttempts, attemptTimeout: time.Duration(c.StateUploadAttemptTimeoutSeconds) * time.Second, baseDelay: time.Duration(c.StateUploadRetryBaseMS) * time.Millisecond, maxDelay: time.Duration(c.StateUploadRetryMaxMS) * time.Millisecond}
	p = p.defaults()
	if p.maxDelay < p.baseDelay {
		return p, errors.New("state upload retry max must be at least retry base")
	}
	return p, nil
}

func (p uploadRetryPolicy) defaults() uploadRetryPolicy {
	if p.attempts == 0 {
		p.attempts = 3
	}
	if p.attemptTimeout == 0 {
		p.attemptTimeout = 120 * time.Second
	}
	if p.baseDelay == 0 {
		p.baseDelay = 250 * time.Millisecond
	}
	if p.maxDelay == 0 {
		p.maxDelay = 2 * time.Second
	}
	return p
}

func (p uploadRetryPolicy) delay(attempt int) time.Duration {
	delay := p.baseDelay
	for i := 0; i < attempt && delay < p.maxDelay; i++ {
		delay *= 2
	}
	if delay > p.maxDelay {
		delay = p.maxDelay
	}
	var random [1]byte
	if _, err := rand.Read(random[:]); err == nil {
		delay = delay/2 + time.Duration(random[0])*delay/256
	}
	if delay > p.maxDelay {
		delay = p.maxDelay
	}
	return delay
}

func retryableStateResponse(r stateResponse) bool {
	return r.err != nil || r.status == 408 || r.status == 429 || r.status == 500 || r.status == 502 || r.status == 503 || r.status == 504
}

func retryAfter(header http.Header) time.Duration {
	value := header.Get("Retry-After")
	if seconds, err := strconv.ParseInt(value, 10, 32); err == nil && seconds > 0 {
		return time.Duration(seconds) * time.Second
	}
	if date, err := http.ParseTime(value); err == nil && time.Until(date) > 0 {
		return time.Until(date)
	}
	return 0
}

func waitStateRetry(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// Mirror the runtime's basename/stem rules before registering a pending upload.
func uploadedStateID(r *http.Request, body []byte, uuid string) (string, error) {
	media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || media != "multipart/form-data" || params["boundary"] == "" {
		return "", errors.New("invalid multipart/form-data request")
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	name := ""
	count := 0
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", errors.New("invalid multipart/form-data request")
		}
		if filename := part.FileName(); filename != "" {
			count++
			name = path.Base(strings.ReplaceAll(filename, "\\", "/"))
		}
		if err := part.Close(); err != nil {
			return "", err
		}
	}
	if count != 1 {
		return "", errors.New("request must contain exactly one state file")
	}
	if name == "" || name == "." || name == ".." {
		name = "state.pth"
	}
	stem := name
	if index := strings.LastIndexByte(name, '.'); index > 0 {
		stem = name[:index]
	}
	if stem == "" {
		stem = "state"
	}
	return stem + "-" + uuid, nil
}

func (p *proxy) stateRequest(ctx context.Context, r *http.Request, body []byte, b *backend, uuid string) stateResponse {
	result := stateResponse{backend: b}
	target := *b.baseURL
	target.Path = joinPath(b.baseURL.Path, r.URL.Path)
	target.RawQuery = r.URL.RawQuery
	upstream, err := http.NewRequestWithContext(ctx, r.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		result.err = err
		return result
	}
	copyHeaders(upstream.Header, r.Header)
	upstream.Host = b.baseURL.Host
	if uuid != "" {
		upstream.Header.Set("X-RWKV-State-Upload-UUID", uuid)
	}
	resp, err := p.client.Do(upstream)
	if err != nil {
		result.err = err
		if ctx.Err() == nil {
			p.scheduler.failed(b)
		}
		return result
	}
	defer resp.Body.Close()
	result.status = resp.StatusCode
	result.header = resp.Header.Clone()
	result.body, result.err = io.ReadAll(io.LimitReader(resp.Body, (8<<20)+1))
	if len(result.body) > 8<<20 {
		result.err = errors.New("state response exceeds 8 MiB")
	}
	if result.err != nil && ctx.Err() == nil {
		p.scheduler.failed(b)
	}
	return result
}

func (p *proxy) uploadToBackend(ctx context.Context, r *http.Request, body []byte, b *backend, uuid, id string, policy uploadRetryPolicy) stateResponse {
	var result stateResponse
	for attempt := 0; attempt < policy.attempts; attempt++ {
		if ctx.Err() != nil {
			return stateResponse{backend: b, err: ctx.Err()}
		}
		attemptCtx, cancel := context.WithTimeout(ctx, policy.attemptTimeout)
		result = p.stateRequest(attemptCtx, r, body, b, uuid)
		cancel()
		if result.err == nil && result.status >= 200 && result.status < 300 {
			var info listedState
			if json.Unmarshal(result.body, &info) != nil || info.ID != id || !p.scheduler.confirmState(b, info) {
				result.err = errors.New("backend returned inconsistent uploaded State identity or structure")
				p.scheduler.stateUnavailable(id, b)
				return result
			}
			return result
		}
		p.scheduler.stateUnavailable(id, b)
		if !retryableStateResponse(result) {
			return result
		}
		if ctx.Err() == nil {
			p.scheduler.failed(b)
		}
		if attempt+1 == policy.attempts {
			return result
		}
		delay := policy.delay(attempt)
		if hint := retryAfter(result.header); hint > delay {
			delay = hint
		}
		if !waitStateRetry(ctx, delay) {
			return stateResponse{backend: b, err: ctx.Err()}
		}
	}
	return result
}

func (p *proxy) serveStateUpload(w http.ResponseWriter, r *http.Request, body []byte) {
	uuid, err := newUploadUUID()
	if err != nil {
		http.Error(w, "cannot generate state upload UUID", 500)
		return
	}
	id, err := uploadedStateID(r, body, uuid)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	mutation := p.scheduler.beginStateMutation(id, false)
	defer p.scheduler.finishStateMutation(mutation, false)
	backends, release := p.scheduler.acquireAll()
	defer release()
	policy := p.uploadPolicy.defaults()
	// Bound the operation even when inference has no total request timeout.
	budget := time.Duration(policy.attempts)*policy.attemptTimeout + time.Duration(policy.attempts-1)*policy.maxDelay
	if p.timeout > 0 && p.timeout < budget {
		budget = p.timeout
	}
	ctx, cancel := context.WithTimeout(r.Context(), budget)
	defer cancel()
	responses := make([]stateResponse, len(backends))
	var wg sync.WaitGroup
	for i, b := range backends {
		wg.Add(1)
		go func(i int, b *backend) {
			defer wg.Done()
			responses[i] = p.uploadToBackend(ctx, r, body, b, uuid, id, policy)
		}(i, b)
	}
	wg.Wait()
	ready := []string{}
	failed := []map[string]any{}
	for _, resp := range responses {
		if resp.err == nil && resp.status >= 200 && resp.status < 300 {
			ready = append(ready, resp.backend.name)
		} else {
			failed = append(failed, map[string]any{"name": resp.backend.name, "status": resp.status})
		}
	}
	if len(failed) == 0 {
		copyHeaders(w.Header(), responses[0].header)
		w.WriteHeader(responses[0].status)
		w.Write(responses[0].body)
		return
	}
	// Preserve a deterministic runtime validation/auth error when no worker accepted.
	if len(ready) == 0 {
		first := responses[0]
		same := first.err == nil && !retryableStateResponse(first)
		for _, resp := range responses {
			if resp.err != nil || resp.status != first.status {
				same = false
			}
		}
		if same {
			copyHeaders(w.Header(), first.header)
			w.WriteHeader(first.status)
			w.Write(first.body)
			return
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadGateway)
	json.NewEncoder(w).Encode(map[string]any{"error": "State synchronization incomplete", "state_id": id, "ready_backends": ready, "failed_backends": failed})
}

// Unknown States (including after router restart) must be confirmed, never guessed.
func (p *proxy) refreshState(r *http.Request, id string) {
	epoch := p.scheduler.stateSnapshot()
	backends, release := p.scheduler.acquireAll()
	defer release()
	timeout := 5 * time.Second
	if p.timeout > 0 && p.timeout < timeout {
		timeout = p.timeout
	}
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()
	probe := r.Clone(ctx)
	probe.Method = http.MethodGet
	probe.URL.Path = "/v1/state/list"
	probe.URL.RawQuery = ""
	probe.Header = r.Header.Clone()
	probe.Header.Del("Content-Type")
	probe.Header.Del("X-RWKV-State-Id")
	var wg sync.WaitGroup
	for _, b := range backends {
		wg.Add(1)
		go func(b *backend) {
			defer wg.Done()
			resp := p.stateRequest(ctx, probe, nil, b, "")
			p.scheduler.observeStateList(b, resp, epoch)
		}(b)
	}
	wg.Wait()
}
