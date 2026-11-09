package router

import (
	"errors"
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
	if s.adapters[m.id] != m.current || !m.current.busy {
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

func (s *scheduler) adapterSnapshot() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.adapterEpoch
}

func (s *scheduler) observeAdapterList(b *backend, response resourceResponse, epoch uint64) {
	items, err := decodeAdapterList(response.body)
	if !validObservation("Adapter", b, response, err) {
		return
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
			nodes[b] = present
		}
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
