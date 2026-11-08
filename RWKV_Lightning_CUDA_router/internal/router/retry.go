package router

import (
	"errors"
	"net/http"
	"time"

	"rwkv-lightning-cuda-router/internal/broadcast"
)

type uploadRetryPolicy = broadcast.Policy

func configuredUploadPolicy(c config) (uploadRetryPolicy, error) {
	if c.StateUploadMaxAttempts < 0 || c.StateUploadMaxAttempts > 10 || c.StateUploadAttemptTimeoutSeconds < 0 || c.StateUploadAttemptTimeoutSeconds > 86400 || c.StateUploadRetryBaseMS < 0 || c.StateUploadRetryBaseMS > 60000 || c.StateUploadRetryMaxMS < 0 || c.StateUploadRetryMaxMS > 60000 {
		return uploadRetryPolicy{}, errors.New("invalid state upload retry configuration")
	}
	p := uploadRetryPolicy{Attempts: c.StateUploadMaxAttempts, AttemptTimeout: time.Duration(c.StateUploadAttemptTimeoutSeconds) * time.Second, BaseDelay: time.Duration(c.StateUploadRetryBaseMS) * time.Millisecond, MaxDelay: time.Duration(c.StateUploadRetryMaxMS) * time.Millisecond}
	p = p.Defaults()
	if p.MaxDelay < p.BaseDelay {
		return p, errors.New("state upload retry max must be at least retry base")
	}
	return p, nil
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

func retryableResourceResponse(r resourceResponse) bool { return broadcast.Retryable(r.status, r.err) }
func retryAfter(h http.Header) time.Duration            { return broadcast.RetryAfter(h) }
