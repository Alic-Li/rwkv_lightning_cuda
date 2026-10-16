package router

import (
	"errors"
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
	return entry != nil && !entry.active && !entry.deleted && entry.ready[b]
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

func (s *scheduler) beginStateMutation(id string, deleted bool) (stateMutation, error) {
	if id == "" {
		return stateMutation{}, errors.New("missing state_id")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.states == nil {
		s.states = make(map[string]*statePlacement)
	}
	if old := s.states[id]; old != nil && old.active {
		return stateMutation{}, errors.New("state synchronization already in progress")
	}
	s.stateEpoch++
	mutation := stateMutation{id: id, previous: s.states[id], current: &statePlacement{ready: make(map[*backend]bool), active: true, deleted: deleted}}
	s.states[id] = mutation.current
	return mutation, nil
}

func (s *scheduler) finishStateMutation(mutation stateMutation, rollback bool) {
	if mutation.id == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.states[mutation.id] != mutation.current || !mutation.current.active {
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
func (s *scheduler) observeStateList(b *backend, response resourceResponse, epoch uint64) {
	states, err := decodeStateList(response.body)
	if !validObservation("State", b, response, err) {
		return
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
		entry.ready[b] = present && (entry.signature == nil || *entry.signature == info)
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
