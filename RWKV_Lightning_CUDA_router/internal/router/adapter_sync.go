package router

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"reflect"
	"time"
)

func (p *proxy) refreshAdapters(r *http.Request, body []byte) {
	epoch := p.scheduler.adapterSnapshot()
	p.discoverResource(r, body, "/v1/adapters", p.adapterUploadPolicy, func(b *backend, response resourceResponse) { p.scheduler.observeAdapterList(b, response, epoch) })
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
	var complete func()
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
		complete = func() { p.scheduler.finishAdapterMutation(mutation, rollback) }
		defer complete()
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
	responses := p.broadcastResource(r, body, p.adapterUploadPolicy, r.Method == http.MethodPost, "")
	if r.Method == http.MethodGet {
		for _, response := range responses {
			p.scheduler.observeAdapterList(response.backend, response, epoch)
		}
	}
	// Preserve old readiness only if every backend deterministically rejected the mutation.
	if mutation.id != "" {
		rollback = rejectedMutation(responses)
	}
	canonicalVersion := ""
	var baseline map[adapterKey]listedAdapter
	baselineBackend := ""
	metadata := map[string]any{"adapter_id": mutation.id, "version": ""}
	finishResourceSync(w, r, "Adapter", metadata, responses, func(response *resourceResponse) error {
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
				log.Printf("router adapter_uploaded backend=%q adapter_id=%q version=%q", response.backend.name, info.ID, info.Version)
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
			response.err = confirmDeletion(response.body)
		case http.MethodGet:
			items, err := decodeAdapterList(response.body)
			if err != nil {
				response.err = err
			} else if baseline == nil {
				baseline = items
				baselineBackend = response.backend.name
			} else if !reflect.DeepEqual(baseline, items) {
				response.err = fmt.Errorf("adapter lists differ from backend %q: %s", baselineBackend, adapterListDifference(baseline, items))
			}
		}
		metadata["version"] = canonicalVersion
		return response.err
	}, complete)
}
