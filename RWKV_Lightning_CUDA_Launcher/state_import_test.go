package main

import (
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func stateImportCall(t *testing.T, l *launcher, body string) (int, map[string]any, string) {
	t.Helper()
	r := httptest.NewRequest("POST", "http://127.0.0.1:10721/api/v1/runtime/state/import", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	l.handler().ServeHTTP(w, r)
	raw := w.Body.String()
	var out map[string]any
	_ = json.Unmarshal([]byte(raw), &out)
	return w.Code, out, raw
}

// A state file that lives on the node reaches the runtime as a multipart
// upload, because the native API has no path form of /v1/state/upload.
func TestStateImportStreamsNodeFileToRuntime(t *testing.T) {
	l := fsAgentLauncher(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "agentic.pth")
	if err := os.WriteFile(path, []byte("PK-state-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Whitelist the directory the same way the browser does.
	l.config.ModelPath = filepath.Join(dir, "model.pth")

	var gotName, gotBody, gotAuth string
	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/state/upload" {
			t.Errorf("path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		_, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil {
			t.Fatal(err)
		}
		part, err := multipart.NewReader(r.Body, params["boundary"]).NextPart()
		if err != nil {
			t.Fatal(err)
		}
		gotName = part.FileName()
		data, _ := io.ReadAll(part)
		gotBody = string(data)
		w.WriteHeader(201)
		fmt.Fprint(w, `{"object":"rwkv.state","state_id":"agentic.pth"}`)
	}))
	defer native.Close()
	u, _ := url.Parse(native.URL)
	l.config.Port = u.Port()
	l.config.Password = "managed-secret"

	code, out, raw := stateImportCall(t, l, `{"path":`+fmt.Sprintf("%q", path)+`}`)
	// The runtime's own status and body pass through untouched.
	if code != 201 || out["state_id"] != "agentic.pth" {
		t.Fatalf("import: %d %s", code, raw)
	}
	if gotName != "agentic.pth" || gotBody != "PK-state-bytes" {
		t.Fatalf("upload carried %q / %q", gotName, gotBody)
	}
	// Multipart authenticates with a Bearer header; the JSON password field
	// is not read on this route.
	if gotAuth != "Bearer managed-secret" {
		t.Fatalf("auth header: %q", gotAuth)
	}
}

func TestStateImportRefusesWhatTheRuntimeWouldReject(t *testing.T) {
	l := fsAgentLauncher(t)
	dir := t.TempDir()
	l.config.ModelPath = filepath.Join(dir, "model.pth")

	wrongExt := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(wrongExt, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if code, _, raw := stateImportCall(t, l, `{"path":`+fmt.Sprintf("%q", wrongExt)+`}`); code != 400 || !strings.Contains(raw, ".pth") {
		t.Fatalf("non-pth: %d %s", code, raw)
	}

	empty := filepath.Join(dir, "empty.pth")
	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if code, _, raw := stateImportCall(t, l, `{"path":`+fmt.Sprintf("%q", empty)+`}`); code != 400 || !strings.Contains(raw, "empty") {
		t.Fatalf("empty: %d %s", code, raw)
	}

	// A directory is not a state, and neither is a path nobody may browse:
	// the refusal never echoes the path back (§5.5).
	if code, _, raw := stateImportCall(t, l, `{"path":`+fmt.Sprintf("%q", dir)+`}`); code != 400 {
		t.Fatalf("directory: %d %s", code, raw)
	}
	outside := filepath.Join(t.TempDir(), "elsewhere.pth")
	if err := os.WriteFile(outside, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	code, _, raw := stateImportCall(t, l, `{"path":`+fmt.Sprintf("%q", outside)+`}`)
	if code != 403 || strings.Contains(raw, outside) {
		t.Fatalf("outside whitelist: %d %s", code, raw)
	}
}

// A Client has no runtime to upload to, so the route must 404 like every
// other agent endpoint rather than answering an error payload.
func TestStateImportIsAgentOnly(t *testing.T) {
	l := newLauncher()
	l.clientOnly = true
	if code, _, raw := stateImportCall(t, l, `{"path":"/tmp/x.pth"}`); code != 404 {
		t.Fatalf("client form: %d %s", code, raw)
	}
}
