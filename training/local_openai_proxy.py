"""Keychain-pipe OpenAI proxy for a private, same-container OpenCode process.

The key enters on stdin, stays in memory, and is never logged, written to disk,
or passed to a child environment. The CLI intentionally supports only a
read-only model probe; live chat serving requires a separate reviewed budget
and rollout lease before it is exposed.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import BinaryIO

MODEL = "gpt-5.6-sol"
UPSTREAM = "https://api.openai.com"
MAX_BODY = 16_777_216


def read_key(stream: BinaryIO) -> str:
    raw = stream.readline(4096)
    if not raw.endswith(b"\n") or len(raw) < 10 or len(raw) > 1024:
        raise ValueError("credential pipe did not deliver one bounded key")
    key = raw[:-1]
    if any(byte < 33 or byte > 126 for byte in key):
        raise ValueError("credential pipe contains an invalid key")
    return key.decode("ascii")


def probe_model(key: str, *, opener=urllib.request.urlopen, upstream: str = UPSTREAM) -> bool:
    request = urllib.request.Request(
        upstream + "/v1/models/" + MODEL,
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    with opener(request, timeout=20) as response:
        return json.load(response).get("id") == MODEL


def handler_for(key: str, *, upstream: str = UPSTREAM, opener=urllib.request.urlopen):
    """Testable transport, not a live launcher. Bind only to container loopback."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # Never emit request paths, bodies, or credentials.
            pass

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(204)
            self.end_headers()

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            length = self.headers.get("Content-Length")
            if (not length or not length.isdecimal() or not 0 < int(length) <= MAX_BODY
                    or self.headers.get("Transfer-Encoding")):
                self.send_error(413)
                return
            body = self.rfile.read(int(length))
            try:
                request_body = json.loads(body)
            except (UnicodeDecodeError, ValueError):
                self.send_error(400)
                return
            if not isinstance(request_body, dict) or request_body.get("model") != MODEL:
                self.send_error(400)
                return
            request = urllib.request.Request(
                upstream + self.path, data=body,
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json",
                         "Accept": "text/event-stream"},
                method="POST",
            )
            try:
                with opener(request, timeout=60) as response:
                    self.send_response(response.status)
                    self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                    self.end_headers()
                    while chunk := response.read(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except Exception:
                # Do not echo an upstream error body that may contain private text.
                self.close_connection = True

    return Handler


def main() -> int:
    if sys.argv[1:] != ["--probe-model"]:
        print("Only the read-only --probe-model action is enabled", file=sys.stderr)
        return 2
    try:
        available = probe_model(read_key(sys.stdin.buffer))
    except Exception:
        print("model probe failed", file=sys.stderr)
        return 1
    print("model_available=" + str(available).lower())
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
