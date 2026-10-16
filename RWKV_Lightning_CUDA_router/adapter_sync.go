package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"reflect"
	"sync"
	"time"
)

type adapterPlacement struct {
	ready           map[string]map[*backend]bool
	current         string
	busy            bool
	deleted         bool
	deletedVersions map[string]bool
}

type adapterMutation struct {
	id                string
	previous, current *adapterPlacement
}

func (s *scheduler) beginAdapterMutation(id string) (adapterMutation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.adapters == nil {
		s.adapters = make(map[string]*adapterPlacement)
	}
	old := s.adapters[id]
	if old != nil && old.busy {
		return adapterMutation{}, errors.New("adapter synchronization already in progress")
	}
	entry := &adapterPlacement{ready: make(map[string]map[*backend]bool), deletedVersions: make(map[string]bool), busy: true}
	if old != nil {
		entry.current = old.current
		for v, deleted := range old.deletedVersions {
			entry.deletedVersions[v] = deleted
		}
		for version, nodes := range old.ready {
			entry.ready[version] = make(map[*backend]bool)
			for b, ready := range nodes {
				entry.ready[version][b] = ready
			}
		}
	}
	s.adapters[id] = entry
	s.adapterEpoch++
	return adapterMutation{id, old, entry}, nil
}

func (s *scheduler) finishAdapterMutation(m adapterMutation, rollback bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.adapters[m.id] != m.current {
		return
	}
	s.adapterEpoch++
	if rollback {
		if m.previous == nil {
			delete(s.adapters, m.id)
		} else {
			s.adapters[m.id] = m.previous
		}
	} else {
		m.current.busy = false
	}
}

func (s *scheduler) adapterReadyLocked(id, version string, b *backend) bool {
	if id == "" {
		return true
	}
	e := s.adapters[id]
	return e != nil && !e.busy && !e.deleted && e.ready[version][b]
}

func (s *scheduler) adapterUnavailable(id, version string, b *backend) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.adapters[id]; e != nil && e.ready[version] != nil {
		e.ready[version][b] = false
	}
}

// Always pin inference to an explicit content version. Runtime lists do not
// expose "latest", so after restart multiple versions require a caller choice.
func (s *scheduler) resolveAdapterVersion(id, version string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.adapters[id]
	if e == nil {
		return version, true
	}
	if e.busy || e.deleted || e.deletedVersions[version] {
		return version, false
	}
	if version == "" {
		version = e.current
	}
	if version == "" && len(e.ready) == 1 {
		for v := range e.ready {
			version = v
		}
	}
	for _, ready := range e.ready[version] {
		if ready {
			return version, false
		}
	}
	return version, true
}

func requestAdapter(body []byte) (string, string, error) {
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) != nil {
		return "", "", nil
	} // Runtime validates other malformed bodies.
	read := func(key string) (string, error) {
		raw, ok := payload[key]
		if !ok {
			return "", nil
		}
		var value string
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) || json.Unmarshal(raw, &value) != nil {
			return "", errors.New(key + " must be a string")
		}
		return value, nil
	}
	id, err := read("adapter_id")
	if err != nil {
		return "", "", err
	}
	version, err := read("adapter_version")
	if err != nil {
		return "", "", err
	}
	if id == "" && version != "" {
		return "", "", errors.New("adapter_id is required with adapter_version")
	}
	return id, version, nil
}

func uploadedAdapterID(r *http.Request, body []byte) (string, error) {
	media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err == nil && media == "multipart/form-data" {
		reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
		id := ""
		count := 0
		for {
			part, err := reader.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				return "", errors.New("invalid multipart adapter upload")
			}
			if part.FormName() == "adapter_id" && part.FileName() == "" {
				value, err := io.ReadAll(io.LimitReader(part, 4097))
				if err != nil || len(value) > 4096 {
					return "", errors.New("adapter_id too large")
				}
				count++
				id = string(value)
			}
			if err := part.Close(); err != nil {
				return "", err
			}
		}
		if count != 1 || id == "" {
			return "", errors.New("upload requires exactly one non-empty adapter_id")
		}
		return id, nil
	}
	id, _, err := requestAdapter(body)
	if err == nil && id == "" {
		err = errors.New("missing adapter_id")
	}
	return id, err
}

type listedAdapter struct {
	ID       string         `json:"id"`
	Version  string         `json:"version"`
	Manifest map[string]any `json:"manifest"`
}

type adapterKey struct{ ID, Version string }

func decodeAdapterList(body []byte) (map[adapterKey]listedAdapter, error) {
	var payload struct {
		Data *[]listedAdapter `json:"data"`
	}
	if json.Unmarshal(body, &payload) != nil || payload.Data == nil {
		return nil, errors.New("invalid adapter list")
	}
	result := make(map[adapterKey]listedAdapter)
	for _, item := range *payload.Data {
		key := adapterKey{item.ID, item.Version}
		if item.ID == "" || item.Version == "" {
			return nil, errors.New("invalid adapter identity")
		}
		if _, exists := result[key]; exists {
			return nil, errors.New("duplicate adapter version")
		}
		result[key] = item
	}
	return result, nil
}

func (s *scheduler) adapterSnapshot() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.adapterEpoch
}

func (s *scheduler) observeAdapterList(b *backend, response stateResponse, epoch uint64) {
	items, err := decodeAdapterList(response.body)
	if response.err != nil || response.status < 200 || response.status >= 300 {
		err = errors.New("adapter list unavailable")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.adapterEpoch != epoch {
		return
	}
	if s.adapters == nil {
		s.adapters = make(map[string]*adapterPlacement)
	}
	for id, entry := range s.adapters {
		if entry.busy || entry.deleted {
			continue
		}
		for version, nodes := range entry.ready {
			_, present := items[adapterKey{id, version}]
			nodes[b] = err == nil && present
		}
	}
	if err != nil {
		return
	}
	for key := range items {
		entry := s.adapters[key.ID]
		if entry == nil {
			entry = &adapterPlacement{ready: make(map[string]map[*backend]bool)}
			s.adapters[key.ID] = entry
		}
		if entry.busy || entry.deleted || entry.deletedVersions[key.Version] {
			continue
		}
		if entry.ready[key.Version] == nil {
			entry.ready[key.Version] = make(map[*backend]bool)
		}
		entry.ready[key.Version][b] = true
	}
}

func (p *proxy) refreshAdapters(r *http.Request, body []byte) {
	epoch := p.scheduler.adapterSnapshot()
	backends, release := p.scheduler.acquireAll()
	defer release()
	timeout := 5 * time.Second
	if p.timeout > 0 && p.timeout < timeout {
		timeout = p.timeout
	}
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()
	probe := r.Clone(ctx)
	url := *r.URL
	probe.URL = &url
	probe.Method = http.MethodGet
	probe.URL.Path = "/v1/adapters"
	probe.URL.RawQuery = ""
	probe.Header = r.Header.Clone()
	probe.Header.Del("Content-Type")
	probe.Header.Del("X-RWKV-State-Id")
	// Discovery must honor the runtime's JSON-password authentication too.
	var probeBody []byte
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) == nil && payload["password"] != nil {
		probeBody, _ = json.Marshal(map[string]json.RawMessage{"password": payload["password"]})
		probe.Header.Set("Content-Type", "application/json")
	}
	var wg sync.WaitGroup
	for _, b := range backends {
		wg.Add(1)
		go func(b *backend) {
			defer wg.Done()
			p.scheduler.observeAdapterList(b, p.stateRequest(ctx, probe, probeBody, b, ""), epoch)
		}(b)
	}
	wg.Wait()
}

func configuredAdapterUploadPolicy(c config) (uploadRetryPolicy, error) {
	p, err := configuredUploadPolicy(config{
		StateUploadMaxAttempts:           c.AdapterUploadMaxAttempts,
		StateUploadAttemptTimeoutSeconds: c.AdapterUploadAttemptTimeoutSeconds,
		StateUploadRetryBaseMS:           c.AdapterUploadRetryBaseMS,
		StateUploadRetryMaxMS:            c.AdapterUploadRetryMaxMS,
	})
	if err != nil {
		return p, errors.New("invalid adapter upload retry configuration")
	}
	return p, nil
}

func (p *proxy) adapterRequest(ctx context.Context, r *http.Request, body []byte, b *backend, retry bool) stateResponse {
	policy := p.adapterUploadPolicy.defaults()
	var result stateResponse
	for attempt := 0; attempt < policy.attempts; attempt++ {
		attemptCtx, cancel := context.WithTimeout(ctx, policy.attemptTimeout)
		result = p.stateRequest(attemptCtx, r, body, b, "")
		cancel()
		if !retry || !retryableStateResponse(result) || attempt+1 == policy.attempts {
			return result
		}
		p.scheduler.failed(b)
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

func (p *proxy) serveSynchronizedAdapters(w http.ResponseWriter, r *http.Request, body []byte) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost && r.Method != http.MethodDelete && r.Method != http.MethodOptions {
		w.Header().Set("Allow", "GET, POST, DELETE, OPTIONS")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	epoch := p.scheduler.adapterSnapshot()
	var mutation adapterMutation
	version := ""
	rollback := false
	if r.Method == http.MethodPost || r.Method == http.MethodDelete {
		var id string
		var err error
		if r.Method == http.MethodPost {
			id, err = uploadedAdapterID(r, body)
		} else {
			id, version, err = requestAdapter(body)
			if err == nil && id == "" {
				err = errors.New("missing adapter_id")
			}
		}
		if err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		mutation, err = p.scheduler.beginAdapterMutation(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		defer func() { p.scheduler.finishAdapterMutation(mutation, rollback) }()
		// An uncertain deletion must never route new work to a removed replica.
		if r.Method == http.MethodDelete {
			p.scheduler.mu.Lock()
			if version == "" {
				mutation.current.ready = make(map[string]map[*backend]bool)
				mutation.current.current = ""
				mutation.current.deleted = true
			} else {
				delete(mutation.current.ready, version)
				mutation.current.deletedVersions[version] = true
				if mutation.current.current == version {
					mutation.current.current = ""
				}
			}
			p.scheduler.mu.Unlock()
		}
	}
	backends, release := p.scheduler.acquireAll()
	defer release()
	policy := p.adapterUploadPolicy.defaults()
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
			responses[i] = p.adapterRequest(ctx, r, body, b, r.Method == http.MethodPost)
			if r.Method == http.MethodGet {
				p.scheduler.observeAdapterList(b, responses[i], epoch)
			}
		}(i, b)
	}
	wg.Wait()
	// Preserve old readiness only if every backend deterministically rejected the mutation.
	if mutation.id != "" {
		rollback = rejectedStateDeletion(responses)
	}
	ready := []string{}
	failed := []map[string]any{}
	canonicalVersion := ""
	var baseline map[adapterKey]listedAdapter
	for i := range responses {
		response := &responses[i]
		if response.err == nil && response.status >= 200 && response.status < 300 {
			switch r.Method {
			case http.MethodPost:
				var info struct {
					ID      string `json:"adapter_id"`
					Version string `json:"version"`
				}
				if json.Unmarshal(response.body, &info) != nil || info.ID != mutation.id || info.Version == "" {
					response.err = errors.New("invalid adapter upload identity")
				} else if canonicalVersion != "" && canonicalVersion != info.Version {
					response.err = errors.New("adapter versions differ across backends")
				} else {
					canonicalVersion = info.Version
					p.scheduler.mu.Lock()
					entry := mutation.current
					if entry.ready[info.Version] == nil {
						entry.ready[info.Version] = make(map[*backend]bool)
					}
					entry.ready[info.Version][response.backend] = true
					entry.current = info.Version
					delete(entry.deletedVersions, info.Version)
					response.backend.unhealthy = time.Time{}
					p.scheduler.mu.Unlock()
				}
			case http.MethodDelete:
				var result struct {
					Deleted bool `json:"deleted"`
				}
				if json.Unmarshal(response.body, &result) != nil || !result.Deleted {
					response.err = errors.New("backend did not confirm adapter deletion")
				}
			case http.MethodGet:
				items, err := decodeAdapterList(response.body)
				if err != nil {
					response.err = err
				} else if baseline == nil {
					baseline = items
				} else if !reflect.DeepEqual(baseline, items) {
					response.err = errors.New("adapter lists differ across backends")
				}
			}
		}
		if response.err == nil && response.status >= 200 && response.status < 300 {
			ready = append(ready, response.backend.name)
		} else {
			failure := map[string]any{"name": response.backend.name, "status": response.status}
			if response.err != nil {
				failure["error"] = response.err.Error()
			}
			failed = append(failed, failure)
		}
	}
	if len(failed) == 0 {
		for _, response := range responses[1:] {
			if response.status != responses[0].status {
				http.Error(w, "adapter synchronization returned inconsistent backend statuses", http.StatusBadGateway)
				return
			}
		}
		copyHeaders(w.Header(), responses[0].header)
		w.WriteHeader(responses[0].status)
		w.Write(responses[0].body)
		return
	}
	// Keep a common deterministic validation/auth response, but never a partial 2xx.
	if len(ready) == 0 {
		first := responses[0]
		same := first.err == nil && first.status >= 400 && !retryableStateResponse(first)
		for _, response := range responses {
			if response.err != nil || response.status != first.status {
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
	json.NewEncoder(w).Encode(map[string]any{"error": "Adapter synchronization incomplete", "adapter_id": mutation.id,
		"version": canonicalVersion, "ready_backends": ready, "failed_backends": failed})
}
