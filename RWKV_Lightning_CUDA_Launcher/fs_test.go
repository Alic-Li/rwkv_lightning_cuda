package main

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakeHome points the launcher's home directory at a throwaway dir. The
// whitelist includes ~ by design (the WebUI picker starts there), and on
// Windows runners the real temp dir lives *under* the real home — which
// would silently make every "outside" path in these tests browsable. A fake
// home keeps "outside the whitelist" deterministic on every OS.
func fakeHome(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", t.TempDir())
		return
	}
	t.Setenv("HOME", t.TempDir())
}

// fsAgentLauncher builds a launcher whose appDir contains a fake runtime
// binary so the agent endpoints are live for fs tests.
func fsAgentLauncher(t *testing.T) *launcher {
	t.Helper()
	fakeHome(t)
	bin := backendExecutable()
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Remove(bin) })
	l := newLauncher()
	l.fs = openFSWhitelist(filepath.Join(t.TempDir(), "roots.json"))
	return l
}

func fsCall(t *testing.T, l *launcher, body string) (int, map[string]any, string) {
	t.Helper()
	r := httptest.NewRequest("POST", "http://127.0.0.1:10721/api/v1/node/fs", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	l.handler().ServeHTTP(w, r)
	var out map[string]any
	raw := w.Body.String()
	_ = json.Unmarshal([]byte(raw), &out)
	return w.Code, out, raw
}

func TestFSWhitelistBasics(t *testing.T) {
	l := fsAgentLauncher(t)
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "model.pth"), []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(modelDir, "sub"), 0o700); err != nil {
		t.Fatal(err)
	}
	l.config.ModelPath = filepath.Join(modelDir, "model.pth")

	code, out, raw := fsCall(t, l, `{}`)
	if code != 200 {
		t.Fatalf("roots: %d %s", code, raw)
	}
	roots, _ := out["roots"].([]any)
	if len(roots) == 0 {
		t.Fatalf("roots empty: %s", raw)
	}
	// The WebUI picker opens here when the form field is still empty.
	if out["default"] != appDir() {
		t.Fatalf("default browse dir: %v", out["default"])
	}

	code, out, raw = fsCall(t, l, fmt.Sprintf(`{"path":%q}`, modelDir))
	if code != 200 {
		t.Fatalf("list model dir: %d %s", code, raw)
	}
	if out["parent"] != "" {
		t.Fatalf("root must have no parent: %s", raw)
	}
	entries, _ := out["entries"].([]any)
	if len(entries) != 2 {
		t.Fatalf("entries: %s", raw)
	}
	var modelEntry, subEntry map[string]any
	for _, e := range entries {
		m, _ := e.(map[string]any)
		switch m["name"] {
		case "model.pth":
			modelEntry = m
		case "sub":
			subEntry = m
		}
	}
	if modelEntry == nil || modelEntry["is_dir"] != false {
		t.Fatalf("file entry: %v", modelEntry)
	}
	if _, ok := modelEntry["size"]; !ok {
		t.Fatalf("file entries carry size: %v", modelEntry)
	}
	if subEntry == nil || subEntry["is_dir"] != true {
		t.Fatalf("dir entry: %v", subEntry)
	}
	if _, ok := subEntry["size"]; ok {
		t.Fatalf("dir entries omit size: %v", subEntry)
	}

	// Subdirectory browses with a parent link back into the whitelist.
	code, out, _ = fsCall(t, l, fmt.Sprintf(`{"path":%q}`, filepath.Join(modelDir, "sub")))
	if code != 200 || out["parent"] != modelDir {
		t.Fatalf("sub dir: %d %v", code, out)
	}
}

// TestFSOutsideWhitelist is the M4 negative suite: outside path, traversal,
// and symlink escape — all 403, none echoing the rejected path.
func TestFSOutsideWhitelist(t *testing.T) {
	l := fsAgentLauncher(t)
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "model.pth"), []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	l.config.ModelPath = filepath.Join(modelDir, "model.pth")

	secretDir := t.TempDir()
	secret := filepath.Join(secretDir, "secret.txt")
	if err := os.WriteFile(secret, []byte("s"), 0o600); err != nil {
		t.Fatal(err)
	}

	for _, path := range []string{
		secret,
		"/etc",
		modelDir + "/../../../etc",
		"/etc/passwd",
	} {
		code, _, raw := fsCall(t, l, fmt.Sprintf(`{"path":%q}`, path))
		if code != 403 {
			t.Fatalf("path %q: got %d %s", path, code, raw)
		}
		if strings.Contains(raw, path) {
			t.Fatalf("403 body echoed the rejected path: %s", raw)
		}
	}

	// A symlink inside the whitelist pointing outside must not be followed.
	escape := filepath.Join(modelDir, "escape")
	if err := os.Symlink(secretDir, escape); err != nil {
		t.Fatal(err)
	}
	code, _, raw := fsCall(t, l, fmt.Sprintf(`{"path":%q}`, escape))
	if code != 403 || strings.Contains(raw, escape) {
		t.Fatalf("symlink escape: %d %s", code, raw)
	}
	// ...but the entry itself is still listed, without target metadata.
	code, out, raw := fsCall(t, l, fmt.Sprintf(`{"path":%q}`, modelDir))
	if code != 200 {
		t.Fatalf("listing: %d %s", code, raw)
	}
	for _, e := range out["entries"].([]any) {
		m, _ := e.(map[string]any)
		if m["name"] == "escape" {
			if m["is_dir"] != false {
				t.Fatalf("outside-pointing symlink must not present as a dir: %v", m)
			}
			if _, ok := m["size"]; ok {
				t.Fatalf("outside-pointing symlink must not leak target size: %v", m)
			}
		}
	}

	// A file inside the whitelist lists its containing directory: the browser
	// opens straight into the folder when seeded with a model path. (A file
	// outside the whitelist is covered by /etc/passwd above: still 403.)
	code, out, raw = fsCall(t, l, fmt.Sprintf(`{"path":%q}`, filepath.Join(modelDir, "model.pth")))
	if code != 200 {
		t.Fatalf("file list: %d %s", code, raw)
	}
	if out["path"] != modelDir {
		t.Fatalf("file path must resolve to its parent listing: %v", out["path"])
	}
}

func TestFSRecordPersistsAcrossRestarts(t *testing.T) {
	dir := t.TempDir()
	l := newLauncher()
	l.fs = openFSWhitelist(filepath.Join(dir, "roots.json"))
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "model.pth"), []byte("m"), 0o600); err != nil {
		t.Fatal(err)
	}
	l.recordFSConfigs(startRequest{ModelPath: filepath.Join(modelDir, "model.pth")}, tuneRequest{}, quantizeRequest{})
	if !containsStr(l.allFSRoots(), modelDir) {
		t.Fatalf("recorded roots: %v", l.allFSRoots())
	}
	l2 := newLauncher()
	l2.fs = openFSWhitelist(filepath.Join(dir, "roots.json"))
	if !containsStr(l2.allFSRoots(), modelDir) {
		t.Fatalf("roots not persisted: %v", l2.allFSRoots())
	}
}

func containsStr(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
