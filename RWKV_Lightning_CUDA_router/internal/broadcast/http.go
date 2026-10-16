// Package broadcast implements the bounded, replayable HTTP fan-out used by
// resource synchronization. It knows nothing about State or adapter identities.
package broadcast

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

type Target struct {
	Name string
	URL  *url.URL
}
type Request struct {
	Method, Path, RawQuery string
	Header                 http.Header
	Body                   []byte // immutable; each attempt gets an independent reader
}
type Response struct {
	Status int
	Header http.Header
	Body   []byte
	Err    error
}

// Run waits for every target, preserving target order. Successful targets are
// never replayed. onFailure must be concurrency-safe and only affects health,
// never resource registration. The caller owns the total operation deadline.
func Run(ctx context.Context, client *http.Client, targets []Target, req Request, policy Policy, retry bool, onFailure func(int)) []Response {
	policy = policy.Defaults()
	results := make([]Response, len(targets))
	var wg sync.WaitGroup
	for i, target := range targets {
		wg.Add(1)
		go func(i int, target Target) {
			defer wg.Done()
			attempts := 1
			if retry {
				attempts = policy.Attempts
			}
			for attempt := 0; attempt < attempts; attempt++ {
				if ctx.Err() != nil {
					results[i] = Response{Err: ctx.Err()}
					return
				}
				attemptCtx, cancel := context.WithTimeout(ctx, policy.AttemptTimeout)
				started := time.Now()
				result := request(attemptCtx, client, target, req)
				cancel()
				results[i] = result
				reason := ""
				if result.Err != nil {
					reason = result.Err.Error()
				}
				log.Printf("router broadcast backend=%q method=%s path=%q attempt=%d status=%d elapsed_ms=%d bytes=%d error=%q", target.Name, req.Method, req.Path, attempt+1, result.Status, time.Since(started).Milliseconds(), len(result.Body), reason)
				if !Retryable(result.Status, result.Err) {
					return
				}
				if ctx.Err() == nil && onFailure != nil && (result.Err != nil || retry) {
					onFailure(i)
				}
				if attempt+1 == attempts {
					return
				}
				delay := policy.Delay(attempt)
				if hint := RetryAfter(result.Header); hint > delay {
					delay = hint
				}
				if !waitRetry(ctx, delay) {
					results[i].Err = ctx.Err()
					return
				}
			}
		}(i, target)
	}
	wg.Wait()
	return results
}

func request(ctx context.Context, client *http.Client, target Target, req Request) Response {
	u := *target.URL
	u.Path = strings.TrimRight(u.Path, "/") + "/" + strings.TrimLeft(req.Path, "/")
	u.RawQuery = req.RawQuery
	upstream, err := http.NewRequestWithContext(ctx, req.Method, u.String(), bytes.NewReader(req.Body))
	if err != nil {
		return Response{Err: err}
	}
	upstream.Header = req.Header.Clone()
	upstream.Host = u.Host
	// A redirect is not an acknowledgment from the configured worker.
	transport := *client
	transport.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := transport.Do(upstream)
	if err != nil {
		return Response{Err: err}
	}
	defer resp.Body.Close()
	result := Response{Status: resp.StatusCode, Header: resp.Header.Clone()}
	result.Body, result.Err = io.ReadAll(io.LimitReader(resp.Body, (8<<20)+1))
	if len(result.Body) > 8<<20 {
		result.Err = errors.New("resource response exceeds 8 MiB")
	}
	return result
}

func (r Response) Failure() error {
	if r.Err != nil {
		return r.Err
	}
	if r.Status < 200 || r.Status >= 300 {
		return fmt.Errorf("upstream HTTP %d", r.Status)
	}
	return nil
}
