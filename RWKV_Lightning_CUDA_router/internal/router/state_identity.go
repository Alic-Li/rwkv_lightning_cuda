package router

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"path"
	"strings"
	"time"
)

// UUID-v7 per RFC 9562, with cryptographic randomness for its 74 random bits.
func newUploadUUID() (string, error) {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return "", err
	}
	timestamp := uint64(time.Now().UnixMilli())
	for i := 5; i >= 0; i-- {
		id[i] = byte(timestamp)
		timestamp >>= 8
	}
	id[6] = (id[6] & 0x0f) | 0x70
	id[8] = (id[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", id[:4], id[4:6], id[6:8], id[8:10], id[10:]), nil
}

// Compare file identity/structure, allowing per-worker upload timestamps/order.
type listedState struct {
	ID       string `json:"state_id"`
	Size     uint64 `json:"size_bytes"`
	Tensors  int    `json:"tensor_count"`
	Layers   int    `json:"layers"`
	Heads    int    `json:"heads"`
	HeadSize int    `json:"head_size"`
}

func decodeStateList(body []byte) (map[string]listedState, error) {
	var payload struct {
		Object string         `json:"object"`
		Data   *[]listedState `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	if payload.Object != "list" || payload.Data == nil {
		return nil, errors.New("invalid state list")
	}
	states := make(map[string]listedState)
	for _, state := range *payload.Data {
		if state.ID == "" {
			return nil, errors.New("empty state ID")
		}
		if _, ok := states[state.ID]; ok {
			return nil, errors.New("duplicate state ID")
		}
		states[state.ID] = state
	}
	return states, nil
}
func resolvedRequestStateID(r *http.Request, body []byte) (string, error) {
	values := []string{r.Header.Get("X-RWKV-State-Id"), r.URL.Query().Get("state_id")}
	if len(body) > 0 {
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(body, &payload); err != nil {
			if r.URL.Path == "/v1/state/delete" {
				return "", errors.New("invalid JSON request")
			}
		} else if raw, ok := payload["state_id"]; ok && string(raw) != "null" {
			var id string
			if json.Unmarshal(raw, &id) != nil {
				return "", errors.New("state_id must be a string")
			}
			values = append(values, id)
		}
	}
	id := ""
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if id != "" && id != value {
			return "", errors.New("conflicting state_id values")
		}
		id = value
	}
	return id, nil
}

// Mirror the runtime's basename/stem rules before registering a pending upload.
func uploadedStateID(r *http.Request, body []byte, uuid string) (string, error) {
	media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || media != "multipart/form-data" || params["boundary"] == "" {
		return "", errors.New("invalid multipart/form-data request")
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	name := ""
	count := 0
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", errors.New("invalid multipart/form-data request")
		}
		if filename := part.FileName(); filename != "" {
			count++
			name = path.Base(strings.ReplaceAll(filename, "\\", "/"))
		}
		if err := part.Close(); err != nil {
			return "", err
		}
	}
	if count != 1 {
		return "", errors.New("request must contain exactly one state file")
	}
	if name == "" || name == "." || name == ".." {
		name = "state.pth"
	}
	stem := name
	if index := strings.LastIndexByte(name, '.'); index > 0 {
		stem = name[:index]
	}
	if stem == "" {
		stem = "state"
	}
	return stem + "-" + uuid, nil
}
