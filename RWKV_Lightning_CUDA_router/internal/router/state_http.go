package router

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
)

func isSynchronizedStateRoute(path string) bool {
	return path == "/v1/state/upload" || path == "/v1/state/list" || path == "/v1/state/delete"
}

func (p *proxy) serveSynchronizedState(w http.ResponseWriter, r *http.Request, body []byte) {
	// OPTIONS is only a preflight and must not create a resource mutation.
	upload := r.URL.Path == "/v1/state/upload"
	deleting := r.URL.Path == "/v1/state/delete"
	allowed := "GET, POST, OPTIONS"
	valid := r.Method == http.MethodGet || r.Method == http.MethodPost
	if upload {
		allowed = "POST, OPTIONS"
		valid = r.Method == http.MethodPost
	}
	if deleting {
		allowed = "DELETE, POST, OPTIONS"
		valid = r.Method == http.MethodDelete || r.Method == http.MethodPost
	}
	if !valid && r.Method != http.MethodOptions {
		w.Header().Set("Allow", allowed)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	isDelete := deleting && r.Method != http.MethodOptions
	isList := !upload && !deleting && r.Method != http.MethodOptions
	if upload && r.Method == http.MethodPost {
		p.serveStateUpload(w, r, body)
		return
	}
	epoch := p.scheduler.stateSnapshot()
	rollback := false
	var complete func()
	id := ""
	if isDelete {
		var err error
		id, err = resolvedRequestStateID(r, body)
		if err != nil || id == "" {
			http.Error(w, "missing or invalid state_id", 400)
			return
		}
		mutation, err := p.scheduler.beginStateMutation(id, true)
		if err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		complete = func() { p.scheduler.finishStateMutation(mutation, rollback) }
		defer complete()
	}
	responses := p.broadcastResource(r, body, p.uploadPolicy, false, "")
	rollback = rejectedMutation(responses)
	if isList {
		for _, response := range responses {
			p.scheduler.observeStateList(response.backend, response, epoch)
		}
	}
	var baseline map[string]listedState
	baselineBackend := ""
	finishResourceSync(w, r, "State", map[string]any{"state_id": id}, responses, func(response *resourceResponse) error {
		if isDelete {
			return confirmDeletion(response.body)
		}
		if !isList {
			return nil
		}
		items, err := decodeStateList(response.body)
		if err != nil {
			return err
		}
		if baseline == nil {
			baseline = items
			baselineBackend = response.backend.name
			return nil
		}
		if !reflect.DeepEqual(baseline, items) {
			return fmt.Errorf("state lists differ from backend %q", baselineBackend)
		}
		return nil
	}, complete)
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

func requestStateID(r *http.Request, body []byte) string {
	if r.URL.Path == "/v1/state/upload" {
		return ""
	}
	stateID := stateIDFromBody(r.URL.Path, body)
	if stateID == "" {
		stateID = r.URL.Query().Get("state_id")
	}
	if stateID == "" {
		stateID = strings.TrimSpace(r.Header.Get("X-RWKV-State-Id"))
	}
	return stateID
}
