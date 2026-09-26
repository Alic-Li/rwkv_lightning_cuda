package main

// Importing a node-local state file (§ state).
//
// The native runtime registers an adapter from a server-side path
// (`POST /v1/adapters {adapter_id, path}`) but a *state* only through a
// multipart upload — there is no path form of /v1/state/upload. Without this
// endpoint the console could only send state files that happen to sit on the
// machine running the browser, which on a remote GPU box is the wrong
// machine: the .pth produced by a tuning run lives next to the trainer.
//
// So the Agent does the hop the protocol is missing: open the file inside the
// browsable whitelist, stream it to the local runtime as multipart, and hand
// the runtime's own answer back unchanged.

import (
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// The native upload cap (docs/http-api.zh-CN.md: 上传上限 512 MiB). Checked
// here so a 30 GB model path picked by mistake fails immediately instead of
// after streaming for minutes.
const stateImportMaxBytes = 512 << 20

type stateImportRequest struct {
	Path string `json:"path"`
}

func (l *launcher) handleStateImport(w http.ResponseWriter, r *http.Request) error {
	if !l.agentGate(w) {
		return nil
	}
	var req stateImportRequest
	if err := decode(w, r, &req); err != nil {
		return err
	}
	clean, _, err := l.resolveFS(req.Path)
	if err != nil {
		if errors.Is(err, errFSOutside) {
			writeJSON(w, 403, map[string]any{"error": "forbidden"})
			return nil
		}
		return err
	}
	info, err := os.Stat(clean)
	if err != nil || info.IsDir() {
		return errors.New("not a file on this node")
	}
	if info.Size() == 0 {
		return errors.New("file is empty")
	}
	if info.Size() > stateImportMaxBytes {
		return fmt.Errorf("file is larger than the %d MiB upload limit", stateImportMaxBytes>>20)
	}
	if !strings.EqualFold(filepath.Ext(clean), ".pth") {
		return errors.New("state files must be .pth")
	}

	status, body, err := l.uploadStateFile(clean)
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
	return nil
}

// uploadStateFile streams one file to the runtime's multipart upload and
// returns its verbatim status and body, so a rejection ("already exists",
// "not a state archive") reaches the console in the runtime's own words.
func (l *launcher) uploadStateFile(path string) (int, []byte, error) {
	l.mu.Lock()
	config := l.config
	l.mu.Unlock()

	file, err := os.Open(path)
	if err != nil {
		return 0, nil, errors.New("cannot read that file on this node")
	}
	defer file.Close()

	// Pipe the multipart body: a 512 MiB state must never be buffered twice.
	pr, pw := io.Pipe()
	form := multipart.NewWriter(pw)
	go func() {
		part, perr := form.CreateFormFile("file", filepath.Base(path))
		if perr == nil {
			_, perr = io.Copy(part, file)
		}
		if perr != nil {
			_ = pw.CloseWithError(perr)
			return
		}
		_ = pw.CloseWithError(form.Close())
	}()

	url := "http://127.0.0.1:" + config.Port + "/v1/state/upload"
	request, err := http.NewRequest(http.MethodPost, url, pr)
	if err != nil {
		return 0, nil, err
	}
	request.Header.Set("Content-Type", form.FormDataContentType())
	// Multipart uploads authenticate with a Bearer header; the JSON
	// `password` field is not read on this route.
	if config.Password != "" {
		request.Header.Set("Authorization", "Bearer "+config.Password)
	}

	client := http.Client{Timeout: 30 * time.Minute}
	response, err := client.Do(request)
	if err != nil {
		return 0, nil, fmt.Errorf("runtime connection: %w", err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return 0, nil, err
	}
	return response.StatusCode, body, nil
}
