package router

import (
	"errors"
	"math"
	"net/url"
	"sync"
	"time"
)

type backend struct {
	name      string
	baseURL   *url.URL
	weight    float64
	inflight  int64
	unhealthy time.Time
}

type scheduler struct {
	mu           sync.Mutex
	backends     []*backend
	next         uint64
	sessions     map[string]*backend
	cooldown     time.Duration
	states       map[string]*statePlacement
	stateEpoch   uint64
	adapters     map[string]*adapterPlacement
	adapterEpoch uint64
}

func (s *scheduler) acquire(bsz int64, session, stateID string) (*backend, func(), error) {
	return s.acquireWithAdapter(bsz, session, stateID, "", "")
}

func (s *scheduler) acquireWithAdapter(bsz int64, session, stateID, adapterID, adapterVersion string) (*backend, func(), error) {
	if bsz < 1 {
		bsz = 1
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	if session != "" {
		if b := s.sessions[session]; b != nil && !now.Before(b.unhealthy) && s.stateReadyLocked(stateID, b) && s.adapterReadyLocked(adapterID, adapterVersion, b) {
			b.inflight += bsz
			return b, s.releaseFunc(b, bsz), nil
		}
	}
	var chosen *backend
	best := math.Inf(1)
	for offset := 0; offset < len(s.backends); offset++ {
		b := s.backends[(int(s.next)+offset)%len(s.backends)]
		if now.Before(b.unhealthy) || !s.stateReadyLocked(stateID, b) || !s.adapterReadyLocked(adapterID, adapterVersion, b) {
			continue
		}
		load := float64(b.inflight) / b.weight
		if load < best {
			best, chosen = load, b
		}
	}
	if chosen == nil {
		if adapterID != "" {
			return nil, nil, errors.New("no healthy upstream backend has the requested adapter/version and State ready; specify adapter_version when multiple versions are registered")
		}
		if stateID == "" {
			return nil, nil, errors.New("all upstream backends are temporarily unavailable")
		}
		return nil, nil, errors.New("no healthy upstream backend has the requested State ready")
	}
	s.next = (s.next + 1) % uint64(len(s.backends))
	chosen.inflight += bsz
	if session != "" {
		s.sessions[session] = chosen
	}
	return chosen, s.releaseFunc(chosen, bsz), nil
}

func (s *scheduler) releaseFunc(b *backend, bsz int64) func() {
	var once sync.Once
	return func() { once.Do(func() { s.mu.Lock(); b.inflight -= bsz; s.mu.Unlock() }) }
}

func (s *scheduler) failed(b *backend) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cooldown > 0 {
		b.unhealthy = time.Now().Add(s.cooldown)
	}
}

// acquireAll reserves every backend while a state-store operation is in
// progress. Uploaded states live in each backend's local filesystem, so these
// operations must not be load-balanced to only one of them.
func (s *scheduler) acquireAll() ([]*backend, func()) {
	s.mu.Lock()
	backends := append([]*backend(nil), s.backends...)
	for _, b := range backends {
		b.inflight++
	}
	s.mu.Unlock()
	var once sync.Once
	return backends, func() {
		once.Do(func() {
			s.mu.Lock()
			for _, b := range backends {
				b.inflight--
			}
			s.mu.Unlock()
		})
	}
}
