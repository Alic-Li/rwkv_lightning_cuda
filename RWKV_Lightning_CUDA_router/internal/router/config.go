package router

import (
	"errors"
	"fmt"
	"math"
	"net"
	"net/url"
	"time"
)

type config struct {
	Listen                             string          `toml:"listen"`
	MaxRequestBodyBytes                int64           `toml:"max_request_body_bytes"`
	RequestTimeoutSeconds              int             `toml:"request_timeout_seconds"`
	FailureCooldownSeconds             int             `toml:"failure_cooldown_seconds"`
	StateUploadMaxAttempts             int             `toml:"state_upload_max_attempts"`
	StateUploadAttemptTimeoutSeconds   int             `toml:"state_upload_attempt_timeout_seconds"`
	StateUploadRetryBaseMS             int             `toml:"state_upload_retry_base_ms"`
	StateUploadRetryMaxMS              int             `toml:"state_upload_retry_max_ms"`
	AdapterUploadMaxAttempts           int             `toml:"adapter_upload_max_attempts"`
	AdapterUploadAttemptTimeoutSeconds int             `toml:"adapter_upload_attempt_timeout_seconds"`
	AdapterUploadRetryBaseMS           int             `toml:"adapter_upload_retry_base_ms"`
	AdapterUploadRetryMaxMS            int             `toml:"adapter_upload_retry_max_ms"`
	Backends                           []backendConfig `toml:"backends"`
}

type backendConfig struct {
	Name   string  `toml:"name"`
	URL    string  `toml:"url"`
	Weight float64 `toml:"weight"`
}

func newScheduler(c config) (*scheduler, error) {
	if c.Listen == "" {
		return nil, errors.New("listen is required")
	}
	if _, _, err := net.SplitHostPort(c.Listen); err != nil {
		return nil, fmt.Errorf("listen must be host:port: %w", err)
	}
	if _, err := configuredUploadPolicy(c); err != nil {
		return nil, err
	}
	if _, err := configuredAdapterUploadPolicy(c); err != nil {
		return nil, err
	}
	if c.MaxRequestBodyBytes < 0 || c.RequestTimeoutSeconds < 0 || c.FailureCooldownSeconds < 0 {
		return nil, errors.New("timeout, cooldown, and body limit cannot be negative")
	}
	if len(c.Backends) == 0 {
		return nil, errors.New("at least one [[backends]] entry is required")
	}
	s := &scheduler{
		sessions: make(map[string]*backend),
		cooldown: time.Duration(c.FailureCooldownSeconds) * time.Second,
	}
	for i, item := range c.Backends {
		u, err := url.Parse(item.URL)
		if err != nil || u.Scheme == "" || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
			return nil, fmt.Errorf("backends[%d].url must be an absolute http(s) URL", i)
		}
		if item.Weight == 0 {
			item.Weight = 1
		}
		if item.Weight < 0 || math.IsNaN(item.Weight) || math.IsInf(item.Weight, 0) {
			return nil, fmt.Errorf("backends[%d].weight must be positive", i)
		}
		name := item.Name
		if name == "" {
			name = u.Host
		}
		s.backends = append(s.backends, &backend{name: name, baseURL: u, weight: item.Weight})
	}
	return s, nil
}
