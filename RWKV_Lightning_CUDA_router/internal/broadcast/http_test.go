package broadcast

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"
)

func TestBroadcastRejectsRedirectInsteadOfRegisteringElsewhere(t *testing.T) {
	var redirected atomic.Int32
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { redirected.Add(1) }))
	defer destination.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()
	u, _ := url.Parse(source.URL)
	results := Run(context.Background(), source.Client(), []Target{{Name: "worker", URL: u}}, Request{Method: "POST", Path: "/upload", Body: []byte("checkpoint")}, Policy{}, true, nil)
	if len(results) != 1 || results[0].Status != 307 || results[0].Failure() == nil || redirected.Load() != 0 {
		t.Fatal(results, redirected.Load())
	}
}

func TestBroadcastRetryAfterIsBoundedByCancellation(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Retry-After", "3600")
		w.WriteHeader(503)
	}))
	defer server.Close()
	u, _ := url.Parse(server.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	start := time.Now()
	result := Run(ctx, server.Client(), []Target{{Name: "worker", URL: u}}, Request{Method: "POST", Path: "/upload"}, Policy{}, true, nil)[0]
	if result.Err != context.DeadlineExceeded || calls.Load() != 1 || time.Since(start) > time.Second {
		t.Fatal(result, calls.Load())
	}
}

func TestBroadcastRejectsOversizedAcknowledgment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chunk := make([]byte, 1<<20)
		for range 9 {
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
	}))
	defer server.Close()
	u, _ := url.Parse(server.URL)
	result := Run(context.Background(), server.Client(), []Target{{Name: "worker", URL: u}}, Request{Method: "GET", Path: "/list"}, Policy{}, false, nil)[0]
	if result.Err == nil || len(result.Body) != (8<<20)+1 {
		t.Fatal("unbounded or accepted response", len(result.Body), result.Err)
	}
}

func TestBroadcastTimeoutRetriesIdenticalBodyAndHeaders(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		if string(data) != "checkpoint" || r.Header.Get("X-Upload-Id") != "same-id" {
			t.Error("retry changed upload")
		}
		if calls.Add(1) == 1 {
			<-r.Context().Done()
			return
		}
		io.WriteString(w, `{}`)
	}))
	defer server.Close()
	u, _ := url.Parse(server.URL)
	policy := Policy{Attempts: 2, AttemptTimeout: 30 * time.Millisecond, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond}
	result := Run(context.Background(), server.Client(), []Target{{Name: "worker", URL: u}}, Request{Method: "POST", Path: "/upload", Header: http.Header{"X-Upload-Id": []string{"same-id"}}, Body: []byte("checkpoint")}, policy, true, nil)[0]
	if result.Failure() != nil || calls.Load() != 2 {
		t.Fatal(result, calls.Load())
	}
}
