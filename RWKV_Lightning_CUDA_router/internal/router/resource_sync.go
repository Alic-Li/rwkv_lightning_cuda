package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"time"

	"rwkv-lightning-cuda-router/internal/broadcast"
)

type resourceResponse struct {
	backend *backend
	status  int
	header  http.Header
	body    []byte
	err     error
}

func (r resourceResponse) failure() error {
	return (broadcast.Response{Status: r.status, Err: r.err}).Failure()
}

// broadcastResource is the sole outbound path for resource upload, list,
// delete and discovery. All configured workers participate, including workers
// in cooldown; inference eligibility is maintained separately.
func (p *proxy) broadcastResource(r *http.Request, body []byte, policy uploadRetryPolicy, retry bool, uuid string) []resourceResponse {
	backends, release := p.scheduler.acquireAll()
	defer release()
	budget := policy.Budget(retry)
	if p.timeout > 0 && p.timeout < budget {
		budget = p.timeout
	}
	ctx, cancel := context.WithTimeout(r.Context(), budget)
	defer cancel()
	targets := make([]broadcast.Target, len(backends))
	for i, b := range backends {
		targets[i] = broadcast.Target{Name: b.name, URL: b.baseURL}
	}
	headers := make(http.Header)
	copyHeaders(headers, r.Header)
	if uuid != "" {
		headers.Set("X-RWKV-State-Upload-UUID", uuid)
	}
	results := broadcast.Run(ctx, p.client, targets, broadcast.Request{Method: r.Method, Path: r.URL.Path, RawQuery: r.URL.RawQuery, Header: headers, Body: body}, policy, retry, func(i int) { p.scheduler.failed(backends[i]) })
	responses := make([]resourceResponse, len(results))
	for i, result := range results {
		responses[i] = resourceResponse{backend: backends[i], status: result.Status, header: result.Header, body: result.Body, err: result.Err}
	}
	return responses
}

// Only definitive pre-mutation rejections may restore previous readiness.
func rejectedMutation(responses []resourceResponse) bool {
	if len(responses) == 0 {
		return false
	}
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

func confirmDeletion(body []byte) error {
	var result struct {
		Deleted bool `json:"deleted"`
	}
	if json.Unmarshal(body, &result) != nil || !result.Deleted {
		return errors.New("backend did not confirm resource deletion")
	}
	return nil
}

// Validate each successful acknowledgment before building a single response.
// Domain validators may register confirmed replicas, but cannot turn partial
// failure into success. Mutation guards keep inference out until completion.
func finishResourceSync(w http.ResponseWriter, r *http.Request, kind string, metadata map[string]any, responses []resourceResponse, validate func(*resourceResponse) error, complete func()) {
	ready := []string{}
	failed := []map[string]any{}
	for i := range responses {
		response := &responses[i]
		if response.failure() == nil && validate != nil {
			response.err = validate(response)
		}
		if err := response.failure(); err != nil {
			failed = append(failed, map[string]any{"name": response.backend.name, "status": response.status, "error": err.Error()})
			log.Printf("router resource_sync_failed resource=%s backend=%q method=%s status=%d reason=%q", kind, response.backend.name, r.Method, response.status, err.Error())
		} else {
			ready = append(ready, response.backend.name)
		}
	}
	// Publish final readiness before any acknowledgment can reach the client.
	if complete != nil {
		complete()
	}
	if len(responses) > 0 {
		first := responses[0]
		same := true
		for _, response := range responses {
			if response.err != nil || response.status != first.status {
				same = false
			}
		}
		if same && (len(failed) == 0 || (len(ready) == 0 && first.status >= 400 && !retryableResourceResponse(first))) {
			copyHeaders(w.Header(), first.header)
			w.WriteHeader(first.status)
			_, _ = w.Write(first.body)
			return
		}
		if len(failed) == 0 {
			// Different 2xx statuses are not a uniform acknowledgment.
			for _, response := range responses[1:] {
				if response.status != first.status {
					failed = append(failed, map[string]any{"name": response.backend.name, "status": response.status, "error": "inconsistent backend status"})
				}
			}
		}
	}
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadata["error"] = fmt.Sprintf("%s synchronization incomplete", kind)
	metadata["ready_backends"] = ready
	metadata["failed_backends"] = failed
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadGateway)
	_ = json.NewEncoder(w).Encode(metadata)
}

// Failed observations never mean deletion, irrespective of resource type.
func validObservation(kind string, b *backend, response resourceResponse, decodeErr error) bool {
	err := response.failure()
	if err == nil {
		err = decodeErr
	}
	if err != nil {
		log.Printf("router resource_list resource=%s backend=%q status=%d content_type=%q bytes=%d error=%q readiness=preserved", kind, b.name, response.status, response.header.Get("Content-Type"), len(response.body), err.Error())
		return false
	}
	return true
}

func (p *proxy) discoverResource(r *http.Request, body []byte, path string, policy uploadRetryPolicy, observe func(*backend, resourceResponse)) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	probe := r.Clone(ctx)
	probe.Method = http.MethodGet
	probe.URL.Path = path
	probe.URL.RawQuery = ""
	probe.Header.Del("Content-Type")
	probe.Header.Del("X-RWKV-State-Id")
	var probeBody []byte
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) == nil && payload["password"] != nil {
		probeBody, _ = json.Marshal(map[string]json.RawMessage{"password": payload["password"]})
		probe.Header.Set("Content-Type", "application/json")
	}
	for _, response := range p.broadcastResource(probe, probeBody, policy, false, "") {
		observe(response.backend, response)
	}
}
