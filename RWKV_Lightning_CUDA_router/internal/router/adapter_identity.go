package router

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"reflect"
	"sort"
)

func requestAdapter(body []byte) (string, string, error) {
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) != nil {
		return "", "", nil
	} // Runtime validates other malformed bodies.
	read := func(key string) (string, error) {
		raw, ok := payload[key]
		if !ok {
			return "", nil
		}
		var value string
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) || json.Unmarshal(raw, &value) != nil {
			return "", errors.New(key + " must be a string")
		}
		return value, nil
	}
	id, err := read("adapter_id")
	if err != nil {
		return "", "", err
	}
	version, err := read("adapter_version")
	if err != nil {
		return "", "", err
	}
	if id == "" && version != "" {
		return "", "", errors.New("adapter_id is required with adapter_version")
	}
	return id, version, nil
}

func uploadedAdapterID(r *http.Request, body []byte) (string, error) {
	media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err == nil && media == "multipart/form-data" {
		reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
		id := ""
		count := 0
		for {
			part, err := reader.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				return "", errors.New("invalid multipart adapter upload")
			}
			if part.FormName() == "adapter_id" && part.FileName() == "" {
				value, err := io.ReadAll(io.LimitReader(part, 4097))
				if err != nil || len(value) > 4096 {
					return "", errors.New("adapter_id too large")
				}
				count++
				id = string(value)
			}
			if err := part.Close(); err != nil {
				return "", err
			}
		}
		if count != 1 || id == "" {
			return "", errors.New("upload requires exactly one non-empty adapter_id")
		}
		return id, nil
	}
	id, _, err := requestAdapter(body)
	if err == nil && id == "" {
		err = errors.New("missing adapter_id")
	}
	return id, err
}

type listedAdapter struct {
	ID       string         `json:"id"`
	Version  string         `json:"version"`
	Manifest map[string]any `json:"manifest"`
}

type adapterKey struct{ ID, Version string }

func decodeAdapterList(body []byte) (map[adapterKey]listedAdapter, error) {
	var payload struct {
		Data *[]listedAdapter `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("invalid adapter list JSON: %w", err)
	}
	if payload.Data == nil {
		return nil, errors.New("invalid adapter list: data must be an array")
	}
	result := make(map[adapterKey]listedAdapter)
	for _, item := range *payload.Data {
		key := adapterKey{item.ID, item.Version}
		if item.ID == "" || item.Version == "" {
			return nil, errors.New("invalid adapter identity")
		}
		if _, exists := result[key]; exists {
			return nil, errors.New("duplicate adapter version")
		}
		result[key] = item
	}
	return result, nil
}

// Report identities only, never full manifests, upload contents, or credentials.
func adapterListDifference(expected, actual map[adapterKey]listedAdapter) string {
	missing, extra, changed := []string{}, []string{}, []string{}
	for key, value := range expected {
		other, found := actual[key]
		label := key.ID + "@" + key.Version
		if !found {
			missing = append(missing, label)
		} else if !reflect.DeepEqual(value, other) {
			changed = append(changed, label)
		}
	}
	for key := range actual {
		if _, found := expected[key]; !found {
			extra = append(extra, key.ID+"@"+key.Version)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	sort.Strings(changed)
	return fmt.Sprintf("missing=%q extra=%q manifest_mismatch=%q", missing, extra, changed)
}
