package router

import (
	"encoding/json"
	"time"
)

func (s *scheduler) routeDiagnostic(id, version, stateID string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry := s.adapters[id]
	state := s.states[stateID]
	nodes := []map[string]any{}
	for _, b := range s.backends {
		node := map[string]any{"backend": b.name, "healthy": !time.Now().Before(b.unhealthy),
			"state_ready": s.stateReadyLocked(stateID, b), "adapter_ready": s.adapterReadyLocked(id, version, b)}
		if entry != nil {
			node["busy"] = entry.busy
			node["deleted"] = entry.deleted || entry.deletedVersions[version]
		}
		if state != nil {
			node["state_busy"] = state.active
			node["state_deleted"] = state.deleted
		}
		nodes = append(nodes, node)
	}
	value, _ := json.Marshal(nodes)
	return string(value)
}
