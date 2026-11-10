#!/usr/bin/env python3
"""Live regression: BASE_URL=http://127.0.0.1:8000 python3 test/state_nonstream_stop_test.py."""

import concurrent.futures
import http.client
import json
import os
import time
import uuid
from urllib.parse import urlsplit


BASE = urlsplit(os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("PASSWORD"):
    HEADERS["Authorization"] = "Bearer " + os.environ["PASSWORD"]


def call(path, body=None):
    connection_type = http.client.HTTPSConnection if BASE.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(BASE.hostname, BASE.port, timeout=30)
    try:
        connection.request("POST" if body is not None else "GET", BASE.path.rstrip("/") + path,
                           json.dumps(body) if body is not None else None, HEADERS)
        response = connection.getresponse()
        data = response.read()
        if response.status != 200:
            raise AssertionError((path, response.status, data.decode()))
        return json.loads(data)
    finally:
        connection.close()


def main():
    session = "state-stop-regression-" + uuid.uuid4().hex
    body = {
        "session_id": session,
        "contents": ["User: Continue the exact sequence.\n\nAssistant: 1, 2, 3,"],
        "stream": False,
        "max_tokens": 128,
        "temperature": 1,
        "top_k": 1,
        "alpha_presence": 0,
        "alpha_frequency": 0,
        "stop_tokens": [],
    }
    try:
        # Compile/warm the model kernels before timing the mid-decode stop.
        call("/v1/chat/completions", body | {"max_tokens": 1})
        # Several connections keep status/stop usable if one lands on the IO
        # thread currently running the synchronous generation handler.
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            generation = pool.submit(call, "/state/chat/completions", body)
            probes = {pool.submit(call, "/v1/server/status") for _ in range(4)}
            request_id = None
            deadline = time.monotonic() + 10
            while request_id is None and not generation.done() and time.monotonic() < deadline:
                done, probes = concurrent.futures.wait(
                    probes, timeout=0.01, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    for active in future.result()["active_requests"]:
                        if active.get("state_key", "").startswith(session + ":"):
                            request_id = active["id"]
                            break
                if request_id is None:
                    while len(probes) < 4:
                        probes.add(pool.submit(call, "/v1/server/status"))
            assert request_id is not None, "generation finished before the stop test could run"
            time.sleep(0.05)
            stops = [pool.submit(call, "/v1/server/stop", {"request_id": request_id})
                     for _ in range(8)]
            response = generation.result()
            assert any(future.result()["stopped"] for future in stops), "stop was not accepted"
            choice = response["choices"][0]
            assert choice["finish_reason"] == "stop", response
            assert choice["message"]["content"], "expected a partial response before stopping"
            print("manual interruption:", json.dumps(choice, ensure_ascii=False))

        # The interrupted state must be cached and available to the next turn.
        assert any(entry["session_id"].startswith(session + ":")
                   for entry in call("/state/status", {})["sessions"])
        continuation = call("/state/chat/completions", body | {
            "contents": ["\n\nUser: Continue.\n\nAssistant: "], "max_tokens": 1})
        assert continuation["choices"][0]["finish_reason"] == "length", continuation
        print("session continuation: length")
    finally:
        # Cleanup only this test's session, including any model namespace suffix.
        for entry in call("/state/status", {})["sessions"]:
            if entry["session_id"].startswith(session + ":"):
                call("/state/delete", {"session_id": entry["session_id"]})
    print("state non-stream stop regression passed")


if __name__ == "__main__":
    main()
