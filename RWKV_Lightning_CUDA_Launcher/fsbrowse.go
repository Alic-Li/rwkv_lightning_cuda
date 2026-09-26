package main

// §5.5 remote file browsing. This endpoint is a token-gated remote read of
// the host filesystem, confined to a whitelist: the user's home directory,
// appDir, and every directory the launcher has actually used in a
// runtime/tuning/quantization config.
//
// Home is in that set deliberately (see allFSRoots): model trees routinely
// live under ~/models and the picker has to open somewhere useful. So the
// whitelist is a navigation boundary, not a secrecy one — anyone holding the
// launcher token can enumerate names and sizes anywhere under $HOME. The
// token, and the fact that this console is not meant to face the open
// internet, are what protect this surface.
//
// Out-of-whitelist requests fail with 403 and never echo the rejected path;
// symlinks are resolved before the check, so a link pointing outside the
// whitelist cannot be followed.

import (
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

const fsRootsFile = "launcher_fs_roots.json"

type fsWhitelist struct {
	mu     sync.Mutex
	file   string
	roots  []string // persisted roots seen across restarts
	loaded bool
}

func openFSWhitelist(path string) *fsWhitelist {
	return &fsWhitelist{file: path}
}

func (w *fsWhitelist) ensureLoaded() {
	if w.loaded {
		return
	}
	w.loaded = true
	data, err := os.ReadFile(w.file)
	if err != nil || len(data) > 1<<20 {
		return
	}
	var roots []string
	if json.Unmarshal(data, &roots) == nil {
		for _, r := range roots {
			if r = filepath.Clean(r); filepath.IsAbs(r) && !w.contains(w.roots, r) {
				w.roots = append(w.roots, r)
			}
		}
	}
}

func (w *fsWhitelist) contains(list []string, dir string) bool {
	for _, r := range list {
		if r == dir {
			return true
		}
	}
	return false
}

// record adds directories referenced by a validated config so the next
// browse can walk them. Only already-existing directories are recorded:
// validation has passed, so the path was real at start time.
func (w *fsWhitelist) record(dirs ...string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.ensureLoaded()
	changed := false
	for _, dir := range dirs {
		dir = strings.TrimSpace(dir)
		if dir == "" || !filepath.IsAbs(dir) {
			continue
		}
		dir = filepath.Clean(dir)
		if st, err := os.Stat(dir); err != nil || !st.IsDir() {
			continue
		}
		if !w.contains(w.roots, dir) && len(w.roots) < 64 {
			w.roots = append(w.roots, dir)
			changed = true
		}
	}
	if changed {
		w.save()
	}
}

func (w *fsWhitelist) save() {
	data, err := json.MarshalIndent(w.roots, "", "  ")
	if err != nil {
		return
	}
	tmp := w.file + ".tmp"
	if os.WriteFile(tmp, data, 0o600) != nil {
		return
	}
	_ = os.Chmod(tmp, 0o600)
	_ = os.Rename(tmp, w.file)
}

// allRoots returns the live whitelist: persisted roots plus appDir plus the
// directories referenced by the current stored configs.
func (l *launcher) allFSRoots() []string {
	l.mu.Lock()
	rc, tc, qc := l.config, l.tuningConfig, l.quantizationConfig
	l.mu.Unlock()
	l.fs.mu.Lock()
	defer l.fs.mu.Unlock()
	l.fs.ensureLoaded()
	roots := append([]string{}, l.fs.roots...)
	add := func(path string, dir bool) {
		path = strings.TrimSpace(path)
		if path == "" {
			return
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(appDir(), path)
		}
		if dir {
			roots = append(roots, path)
		} else {
			roots = append(roots, filepath.Dir(path))
		}
	}
	add(appDir(), true)
	// The user's home is browsable by design: model trees routinely live under
	// ~/models, and the WebUI picker should be able to start from ~ (still
	// token-gated; §5.5 applies to the whole whitelist).
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		add(home, true)
	}
	add(rc.ModelPath, rc.EnableDynamicLoading)
	add(rc.VocabPath, false)
	add(rc.StateDBPath, false)
	add(rc.TuneCache, false)
	add(tc.Model, false)
	add(tc.Data, false)
	add(tc.Output, true)
	add(tc.Vocab, false)
	add(tc.State, false)
	add(tc.Resume, true)
	add(qc.InputPath, false)
	add(qc.OutputPath, false)
	seen := map[string]bool{}
	out := []string{}
	for _, r := range roots {
		r = filepath.Clean(r)
		if !seen[r] {
			seen[r] = true
			out = append(out, r)
		}
	}
	sort.Strings(out)
	return out
}

// within reports whether path is root or lies under it.
func fsWithin(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// resolveFS validates a requested path against the whitelist. It returns
// the cleaned path and whether it is a whitelist root (parent == "").
// Errors never include the requested path (§5.5).
func (l *launcher) resolveFS(path string) (clean string, parent string, err error) {
	if strings.TrimSpace(path) == "" {
		return "", "", errors.New("path is required")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(appDir(), path)
	}
	clean = filepath.Clean(path)
	resolved, rerr := filepath.EvalSymlinks(clean)
	if rerr != nil {
		// Nonexistent (or unresolvable) paths are treated exactly like
		// out-of-whitelist ones: 403, no echo — otherwise the error
		// difference would act as a filesystem existence oracle for
		// paths outside the whitelist.
		return "", "", errFSOutside
	}
	roots := l.allFSRoots()
	for _, root := range roots {
		rootResolved, err := filepath.EvalSymlinks(root)
		if err != nil {
			continue
		}
		if fsWithin(rootResolved, resolved) {
			if resolved == rootResolved {
				return clean, "", nil // at a whitelist root: no parent to walk up to
			}
			return clean, filepath.Dir(clean), nil
		}
	}
	return "", "", errFSOutside
}

var errFSOutside = errors.New("path is outside the browsable roots")

// defaultBrowseDir is where the WebUI directory browser opens when the form
// field is still empty: the launcher binary's own folder (models usually sit
// next to it), falling back to the user's home.
func defaultBrowseDir() string {
	if st, err := os.Stat(appDir()); err == nil && st.IsDir() {
		return appDir()
	}
	if home, err := os.UserHomeDir(); err == nil {
		return home
	}
	return ""
}

// recordFSConfigs whitelists the directories referenced by validated
// configs so remote browsing can walk them on the next request. Files
// contribute their parent directory; directory fields contribute
// themselves. Only existing directories are recorded.
func (l *launcher) recordFSConfigs(rc startRequest, tc tuneRequest, qc quantizeRequest) {
	var dirs []string
	add := func(path string, isDir bool) {
		path = strings.TrimSpace(path)
		if path == "" {
			return
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(appDir(), path)
		}
		if isDir {
			dirs = append(dirs, path)
		} else {
			dirs = append(dirs, filepath.Dir(path))
		}
	}
	add(rc.ModelPath, rc.EnableDynamicLoading)
	add(rc.VocabPath, false)
	add(rc.StateDBPath, false)
	add(rc.TuneCache, false)
	add(tc.Model, false)
	add(tc.Data, false)
	add(tc.Output, true)
	add(tc.Vocab, false)
	add(tc.State, false)
	add(tc.Resume, true)
	add(qc.InputPath, false)
	add(qc.OutputPath, false)
	l.fs.record(dirs...)
}

type fsEntry struct {
	Name  string `json:"name"`
	IsDir bool   `json:"is_dir"`
	Size  *int64 `json:"size,omitempty"`
}

func (l *launcher) handleNodeFS(w http.ResponseWriter, r *http.Request) error {
	if !l.agentGate(w) {
		return nil
	}
	var req struct {
		Path *string `json:"path"`
	}
	if e := decode(w, r, &req); e != nil {
		return e
	}
	if req.Path == nil || *req.Path == "" {
		roots := []string{}
		for _, root := range l.allFSRoots() {
			if _, err := os.Stat(root); err == nil {
				roots = append(roots, root)
			}
		}
		writeJSON(w, 200, map[string]any{"roots": roots, "default": defaultBrowseDir()})
		return nil
	}
	clean, parent, err := l.resolveFS(*req.Path)
	if err != nil {
		if errors.Is(err, errFSOutside) {
			writeJSON(w, 403, map[string]any{"error": "forbidden"})
			return nil
		}
		return err
	}
	if st, serr := os.Stat(clean); serr != nil {
		return errors.New("path is not a directory")
	} else if !st.IsDir() {
		// A file path (a model .pth seeded from the form) lists its containing
		// directory: the browser only ever shows directories, and the file's
		// directory is whitelisted whenever the file itself resolved.
		clean, parent, err = l.resolveFS(filepath.Dir(clean))
		if err != nil {
			writeJSON(w, 403, map[string]any{"error": "forbidden"})
			return nil
		}
	}
	entries, err := os.ReadDir(clean)
	if err != nil {
		return errors.New("directory is not readable")
	}
	out := []fsEntry{}
	for _, e := range entries {
		entry := fsEntry{Name: e.Name(), IsDir: e.IsDir()}
		if !e.IsDir() {
			if size, serr := e.Info(); serr == nil {
				s := size.Size()
				entry.Size = &s
			}
		}
		// Symlinks: keep the entry name (it lives in a whitelisted
		// directory) but reveal nothing about the target unless the
		// target itself is whitelisted.
		if e.Type()&fs.ModeSymlink != 0 {
			_, _, rerr := l.resolveFS(filepath.Join(clean, e.Name()))
			if rerr != nil {
				entry.IsDir = false
				entry.Size = nil
			} else {
				if st, serr := os.Stat(filepath.Join(clean, e.Name())); serr == nil {
					entry.IsDir = st.IsDir()
					if !st.IsDir() {
						s := st.Size()
						entry.Size = &s
					} else {
						entry.Size = nil
					}
				}
			}
		}
		out = append(out, entry)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].IsDir != out[j].IsDir {
			return out[i].IsDir
		}
		return out[i].Name < out[j].Name
	})
	writeJSON(w, 200, map[string]any{"path": clean, "parent": parent, "entries": out})
	return nil
}
