package router

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
)

type flushingWriter struct{ http.ResponseWriter }

func (w flushingWriter) Write(p []byte) (int, error) {
	n, err := w.ResponseWriter.Write(p)
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
	return n, err
}

func (p *proxy) readBody(r *http.Request) ([]byte, error) {
	if p.maxBody > 0 && r.ContentLength > p.maxBody {
		return nil, errors.New("request body exceeds configured limit")
	}
	reader := io.Reader(r.Body)
	if p.maxBody > 0 {
		reader = io.LimitReader(r.Body, p.maxBody+1)
	}
	body, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	if p.maxBody > 0 && int64(len(body)) > p.maxBody {
		return nil, errors.New("request body exceeds configured limit")
	}
	return body, nil
}

func usesSessionAffinity(path string) bool {
	return strings.HasPrefix(path, "/state/") || strings.Contains(path, "completion") || strings.Contains(path, "translate") || strings.Contains(path, "FIM") || strings.Contains(path, "big_batch")
}

func batchSize(r *http.Request, body []byte) (int64, string) {
	path := r.URL.Path
	if !usesSessionAffinity(path) {
		return 1, ""
	}
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) != nil {
		return 1, ""
	}
	session := ""
	if raw := payload["session_id"]; raw != nil {
		_ = json.Unmarshal(raw, &session)
	}
	if session == "" {
		session = strings.TrimSpace(r.Header.Get("X-RWKV-Session-Id"))
	}
	for _, key := range []string{"contents", "text_list", "prompts", "inputs"} {
		var values []json.RawMessage
		if raw := payload[key]; raw != nil && json.Unmarshal(raw, &values) == nil && len(values) > 0 {
			return int64(len(values)), session
		}
	}
	for _, key := range []string{"bsz", "batch_size"} {
		var value int64
		if raw := payload[key]; raw != nil && json.Unmarshal(raw, &value) == nil && value > 0 {
			return value, session
		}
	}
	return 1, session
}

func copyHeaders(dst, src http.Header) {
	for k, values := range src {
		if hopHeader(k) {
			continue
		}
		dst.Del(k)
		for _, v := range values {
			dst.Add(k, v)
		}
	}
}
func hopHeader(k string) bool {
	switch strings.ToLower(k) {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade":
		return true
	}
	return false
}
func joinPath(base, request string) string {
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(request, "/")
}
