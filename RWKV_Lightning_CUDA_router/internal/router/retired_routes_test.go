package router

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/BurntSushi/toml"
)

func TestRetiredAdaptersNeverReachWorkers(t *testing.T) {
	var calls atomic.Int32
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(200)
	}))
	defer worker.Close()
	p := testStateProxy(worker)
	for _, method := range []string{"GET", "POST", "DELETE", "OPTIONS"} {
		for _, path := range []string{"/v1/adapters", "/v1/adapters/", "/v1/adapters/legacy"} {
			if w := proxyCall(p, method, path, `{}`); w.Code != 404 {
				t.Fatalf("%s %s: %d", method, path, w.Code)
			}
		}
	}
	for _, key := range []string{"adapter_id", "adapter_version", "adapter_scale"} {
		for _, value := range []string{`"legacy"`, `""`, `null`, `0`} {
			for _, path := range []string{"/v1/chat/completions", "/v1/batch/completions", "/v1/completions"} {
				// Unknown State must not trigger discovery for an invalid generation request.
				body := fmt.Sprintf(`{"contents":["hello"],"state_id":"unknown",%q:%s}`, key, value)
				if w := proxyCall(p, "POST", path, body); w.Code != 400 {
					t.Fatalf("%s %s: %d", path, key, w.Code)
				}
			}
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("retired requests reached workers: %d", calls.Load())
	}
	if w := proxyCall(p, "POST", "/v1/batch/completions", `{"contents":["hello"]}`); w.Code != 200 {
		t.Fatal(w.Code)
	}
	if calls.Load() != 1 {
		t.Fatal("base inference did not reach worker")
	}
}

func TestRetiredAdapterConfigurationIsUnknown(t *testing.T) {
	for _, key := range []string{"adapter_upload_max_attempts", "adapter_upload_attempt_timeout_seconds", "adapter_upload_retry_base_ms", "adapter_upload_retry_max_ms"} {
		var c config
		meta, err := toml.Decode(key+" = 1", &c)
		if err != nil {
			t.Fatal(err)
		}
		if len(meta.Undecoded()) != 1 || meta.Undecoded()[0].String() != key {
			t.Fatalf("retired config accepted: %s", key)
		}
	}
}
