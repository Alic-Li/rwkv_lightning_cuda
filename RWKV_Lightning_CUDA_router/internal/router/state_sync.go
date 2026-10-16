package router

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
)

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
	mutation, err := p.scheduler.beginStateMutation(id, false)
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	complete := func() { p.scheduler.finishStateMutation(mutation, false) }
	defer complete()
	responses := p.broadcastResource(r, body, p.uploadPolicy, true, uuid)
	finishResourceSync(w, r, "State", map[string]any{"state_id": id}, responses, func(response *resourceResponse) error {
		var info listedState
		if json.Unmarshal(response.body, &info) != nil || info.ID != id || !p.scheduler.confirmState(response.backend, info) {
			return errors.New("backend returned inconsistent uploaded State identity or structure")
		}
		log.Printf("router resource_uploaded resource=State backend=%q state_id=%q", response.backend.name, id)
		return nil
	}, complete)
}
func (p *proxy) refreshState(r *http.Request, body []byte) {
	epoch := p.scheduler.stateSnapshot()
	p.discoverResource(r, body, "/v1/state/list", p.uploadPolicy, func(b *backend, response resourceResponse) { p.scheduler.observeStateList(b, response, epoch) })
}
