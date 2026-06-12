#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
PASSWORD="${PASSWORD:-}"
HTTP_TIMEOUT="${HTTP_TIMEOUT:-30}"
STREAM_TIMEOUT="${STREAM_TIMEOUT:-120}"
STOP_DELAY="${STOP_DELAY:-1}"
PAUSE_DELAY="${PAUSE_DELAY:-1}"
LONG_MAX_TOKENS="${LONG_MAX_TOKENS:-512}"
TMPDIR="${TMPDIR:-/tmp}/rwkv_api_test.$$"
KEEP_TMP="${KEEP_TMP:-0}"
PRINT_API_OUTPUT="${PRINT_API_OUTPUT:-1}"

mkdir -p "$TMPDIR"
if [[ "$KEEP_TMP" == "1" ]]; then
  trap 'printf "Keeping test artifacts in %s\n" "$TMPDIR"' EXIT
else
  trap 'rm -rf "$TMPDIR"' EXIT
fi

auth_args=()
if [[ -n "$PASSWORD" ]]; then
  auth_args=(-H "Authorization: Bearer $PASSWORD")
fi

log() {
  printf '\n== %s ==\n' "$*"
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

print_response() {
  local method="$1"
  local path="$2"
  local status="$3"
  local file="$4"

  [[ "$PRINT_API_OUTPUT" == "1" ]] || return 0

  printf -- '--- %s %s HTTP %s response begin ---\n' "$method" "$path" "$status"
  if [[ -s "$file" ]]; then
    cat "$file"
    printf '\n'
  else
    printf '<empty body>\n'
  fi
  printf -- '--- %s %s response end ---\n' "$method" "$path"
}

request() {
  local method="$1"
  local path="$2"
  local body="$3"
  local out="$4"
  local expected="${5:-200}"
  local status

  if [[ -n "$body" ]]; then
    status="$(
      curl -sS \
        --connect-timeout 3 \
        --max-time "$HTTP_TIMEOUT" \
        -X "$method" \
        -H "Content-Type: application/json" \
        "${auth_args[@]}" \
        -o "$out" \
        -w "%{http_code}" \
        --data "$body" \
        "$BASE_URL$path"
    )"
  else
    status="$(
      curl -sS \
        --connect-timeout 3 \
        --max-time "$HTTP_TIMEOUT" \
        -X "$method" \
        "${auth_args[@]}" \
        -o "$out" \
        -w "%{http_code}" \
        "$BASE_URL$path"
    )"
  fi

  [[ "$status" == "$expected" ]] || {
    printf 'Response body from %s %s:\n' "$method" "$path" >&2
    cat "$out" >&2 || true
    fail "expected HTTP $expected, got $status"
  }

  print_response "$method" "$path" "$status" "$out"
}

request_any_status() {
  local method="$1"
  local path="$2"
  local body="$3"
  local out="$4"
  shift 4
  local status
  local expected

  if [[ -n "$body" ]]; then
    status="$(
      curl -sS \
        --connect-timeout 3 \
        --max-time "$HTTP_TIMEOUT" \
        -X "$method" \
        -H "Content-Type: application/json" \
        "${auth_args[@]}" \
        -o "$out" \
        -w "%{http_code}" \
        --data "$body" \
        "$BASE_URL$path"
    )"
  else
    status="$(
      curl -sS \
        --connect-timeout 3 \
        --max-time "$HTTP_TIMEOUT" \
        -X "$method" \
        "${auth_args[@]}" \
        -o "$out" \
        -w "%{http_code}" \
        "$BASE_URL$path"
    )"
  fi

  for expected in "$@"; do
    if [[ "$status" == "$expected" ]]; then
      print_response "$method" "$path" "$status" "$out"
      return 0
    fi
  done

  printf 'Response body from %s %s:\n' "$method" "$path" >&2
  cat "$out" >&2 || true
  fail "expected HTTP one of: $*, got $status"
}

stream_request() {
  local path="$1"
  local body="$2"
  local out="$3"
  local expected="${4:-200}"
  local status

  status="$(
    curl -sS -N \
      --connect-timeout 3 \
      --max-time "$STREAM_TIMEOUT" \
      -X POST \
      -H "Content-Type: application/json" \
      "${auth_args[@]}" \
      -o "$out" \
      -w "%{http_code}" \
      --data "$body" \
      "$BASE_URL$path"
  )"

  echo "$status" >"$out.status"
  if [[ "$status" == "$expected" ]]; then
    print_response POST "$path" "$status" "$out"
    return 0
  fi
  return 1
}

assert_contains() {
  local file="$1"
  local pattern="$2"
  grep -q "$pattern" "$file" || {
    printf 'File %s did not contain pattern %s. Body:\n' "$file" "$pattern" >&2
    cat "$file" >&2 || true
    fail "missing expected response content"
  }
}

assert_sse_done() {
  local file="$1"
  assert_contains "$file" 'data: \[DONE\]'
}

extract_json_string() {
  local key="$1"
  local file="$2"
  sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$file" | head -n 1
}

require_cmd curl
require_cmd grep
require_cmd sed

session_id="api-test-$(date +%s)-$$"

paths=(
  "/v1/batch/completions"
  "/translate/v1/batch-translate"
  "/state/chat/completions"
  "/state/status"
  "/state/delete"
  "/v1/server/status"
  "/v1/server/stop"
  "/v1/server/pause"
  "/v1/server/resume"
  "/v1/tokens/count"
  "/v1/chat/completions"
  "/v1/models"
)

log "OPTIONS preflight"
for path in "${paths[@]}"; do
  request_any_status OPTIONS "$path" "" "$TMPDIR/options-${path//\//_}.json" 200 204
done

log "GET /v1/server/status"
request GET "/v1/server/status" "" "$TMPDIR/status.json"
assert_contains "$TMPDIR/status.json" '"status":"running"'

log "GET /v1/models"
request GET "/v1/models" "" "$TMPDIR/models.json"
assert_contains "$TMPDIR/models.json" '"object":"list"'

log "POST /v1/tokens/count text"
request POST "/v1/tokens/count" '{"text":"hello RWKV"}' "$TMPDIR/tokens-text.json"
assert_contains "$TMPDIR/tokens-text.json" '"tokens":'

log "POST /v1/tokens/count messages"
request POST "/v1/tokens/count" '{"messages":[{"role":"user","content":"hello"}]}' "$TMPDIR/tokens-messages.json"
assert_contains "$TMPDIR/tokens-messages.json" '"tokens":'

log "POST /v1/server/stop without active request"
request POST "/v1/server/stop" '{}' "$TMPDIR/stop-idle.json"
assert_contains "$TMPDIR/stop-idle.json" '"ok":true'

log "POST /v1/server/pause without active request"
request POST "/v1/server/pause" '{}' "$TMPDIR/pause-idle.json"
assert_contains "$TMPDIR/pause-idle.json" '"ok":true'

log "POST /v1/chat/completions non-stream"
request POST "/v1/chat/completions" '{
  "model":"api-test",
  "messages":[{"role":"user","content":"Say hello in one short sentence."}],
  "stream":false,
  "max_tokens":8,
  "temperature":1.0,
  "top_k":5,
  "top_p":0.3,
  "alpha_presence":0.2,
  "alpha_frequency":0.2,
  "alpha_decay":0.99,
  "stop_tokens":[0,261,24281],
  "chunk_size":1
}' "$TMPDIR/chat.json"
assert_contains "$TMPDIR/chat.json" '"choices":'

log "POST /v1/chat/completions stream"
stream_request "/v1/chat/completions" '{
  "model":"api-test",
  "messages":[{"role":"user","content":"Say hello in one short sentence."}],
  "stream":true,
  "max_tokens":8,
  "temperature":1.0,
  "top_k":5,
  "top_p":0.3,
  "alpha_presence":0.2,
  "alpha_frequency":0.2,
  "alpha_decay":0.99,
  "stop_tokens":[0,261,24281],
  "chunk_size":1
}' "$TMPDIR/chat-stream.sse" || fail "chat stream request failed"
assert_sse_done "$TMPDIR/chat-stream.sse"

log "POST /v1/batch/completions non-stream"
request POST "/v1/batch/completions" '{
  "contents":["English: Hello\n\nChinese:","English: Good morning\n\nChinese:"],
  "stream":false,
  "max_tokens":8,
  "temperature":1.0,
  "top_k":5,
  "top_p":0.3,
  "alpha_presence":0.2,
  "alpha_frequency":0.2,
  "alpha_decay":0.99,
  "stop_tokens":[0,261,24281],
  "chunk_size":1
}' "$TMPDIR/batch.json"
assert_contains "$TMPDIR/batch.json" '"choices":'

log "POST /v1/batch/completions stream"
stream_request "/v1/batch/completions" '{
  "contents":["English: Hello\n\nChinese:","English: Good morning\n\nChinese:"],
  "stream":true,
  "max_tokens":8,
  "temperature":1.0,
  "top_k":5,
  "top_p":0.3,
  "alpha_presence":0.2,
  "alpha_frequency":0.2,
  "alpha_decay":0.99,
  "stop_tokens":[0,261,24281],
  "chunk_size":1
}' "$TMPDIR/batch-stream.sse" || fail "batch stream request failed"
assert_sse_done "$TMPDIR/batch-stream.sse"

log "POST /translate/v1/batch-translate"
request POST "/translate/v1/batch-translate" '{
  "source_lang":"English",
  "target_lang":"Chinese",
  "text_list":["Hello","Good morning"]
}' "$TMPDIR/translate.json"
assert_contains "$TMPDIR/translate.json" '"translations":'

log "POST /state/chat/completions non-stream"
request POST "/state/chat/completions" "{
  \"session_id\":\"$session_id\",
  \"contents\":[\"User: remember the word albatross.\\nAssistant:\"],
  \"stream\":false,
  \"max_tokens\":8,
  \"temperature\":1.0,
  \"top_k\":5,
  \"top_p\":0.3,
  \"alpha_presence\":0.2,
  \"alpha_frequency\":0.2,
  \"alpha_decay\":0.99,
  \"stop_tokens\":[0,261,24281],
  \"chunk_size\":1
}" "$TMPDIR/state-chat.json"
assert_contains "$TMPDIR/state-chat.json" '"choices":'

log "POST /state/chat/completions stream"
stream_request "/state/chat/completions" "{
  \"session_id\":\"$session_id\",
  \"contents\":[\"User: continue.\\nAssistant:\"],
  \"stream\":true,
  \"max_tokens\":8,
  \"temperature\":1.0,
  \"top_k\":5,
  \"top_p\":0.3,
  \"alpha_presence\":0.2,
  \"alpha_frequency\":0.2,
  \"alpha_decay\":0.99,
  \"stop_tokens\":[0,261,24281],
  \"chunk_size\":1
}" "$TMPDIR/state-stream.sse" || fail "state stream request failed"
assert_sse_done "$TMPDIR/state-stream.sse"

log "POST /state/status"
request POST "/state/status" '{}' "$TMPDIR/state-status.json"
assert_contains "$TMPDIR/state-status.json" '"sessions":'

log "POST /state/delete"
request POST "/state/delete" "{\"session_id\":\"$session_id\"}" "$TMPDIR/state-delete.json"
assert_contains "$TMPDIR/state-delete.json" '"status":'

log "POST /v1/server/stop during active stream"
stream_request "/v1/chat/completions" "{
  \"model\":\"api-test\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Write a long numbered list.\"}],
  \"stream\":true,
  \"max_tokens\":$LONG_MAX_TOKENS,
  \"temperature\":1.0,
  \"top_k\":5,
  \"top_p\":0.3,
  \"alpha_presence\":0.2,
  \"alpha_frequency\":0.2,
  \"alpha_decay\":0.99,
  \"stop_tokens\":[0,261,24281],
  \"chunk_size\":1
}" "$TMPDIR/stop-stream.sse" &
stop_pid=$!
sleep "$STOP_DELAY"
request POST "/v1/server/stop" '{}' "$TMPDIR/stop-active.json"
assert_contains "$TMPDIR/stop-active.json" '"ok":true'
wait "$stop_pid" || fail "stream request did not finish cleanly after stop"
assert_sse_done "$TMPDIR/stop-stream.sse"

log "POST /v1/server/pause and /v1/server/resume"
stream_request "/v1/chat/completions" "{
  \"model\":\"api-test\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Write a long paragraph about RWKV.\"}],
  \"stream\":true,
  \"max_tokens\":$LONG_MAX_TOKENS,
  \"temperature\":1.0,
  \"top_k\":5,
  \"top_p\":0.3,
  \"alpha_presence\":0.2,
  \"alpha_frequency\":0.2,
  \"alpha_decay\":0.99,
  \"stop_tokens\":[0,261,24281],
  \"chunk_size\":1
}" "$TMPDIR/pause-stream.sse" &
pause_pid=$!
sleep "$PAUSE_DELAY"
request POST "/v1/server/pause" '{}' "$TMPDIR/pause-active.json"
assert_contains "$TMPDIR/pause-active.json" '"ok":true'
wait "$pause_pid" || fail "stream request did not finish cleanly after pause"
assert_sse_done "$TMPDIR/pause-stream.sse"

request_id="$(extract_json_string request_id "$TMPDIR/pause-active.json")"
if [[ -z "$request_id" ]]; then
  printf 'WARN: pause returned no request_id, likely generation finished before pause; skipping resume body check.\n' >&2
else
  stream_request "/v1/server/resume" "{
    \"request_id\":\"$request_id\",
    \"stream\":true
  }" "$TMPDIR/resume-stream.sse" || fail "resume stream request failed"
  assert_sse_done "$TMPDIR/resume-stream.sse"
fi

log "Final /v1/server/status"
request GET "/v1/server/status" "" "$TMPDIR/status-final.json"
assert_contains "$TMPDIR/status-final.json" '"status":"running"'

if [[ "$KEEP_TMP" == "1" ]]; then
  printf '\nAll API endpoint checks passed. Artifacts kept under %s.\n' "$TMPDIR"
else
  printf '\nAll API endpoint checks passed.\n'
fi
