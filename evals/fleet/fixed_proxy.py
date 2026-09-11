"""Secret-holding fixed-upstream proxy for the isolated Qwen Code container."""

from __future__ import annotations

import http.client
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def completion_overrides(value: dict) -> dict:
    """Validate operator-owned sampling; never let agent requests change the arm."""
    if set(value) != {"model", "temperature", "top_p", "seed", "max_tokens"}:
        raise ValueError("incomplete fixed completion policy")
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("fixed model required")
    for field, minimum, maximum in (("temperature", 0, 2), ("top_p", 0, 1)):
        v = value[field]
        if type(v) not in (int, float) or not math.isfinite(v) or not minimum <= v <= maximum:
            raise ValueError("invalid sampling value")
    if value["top_p"] == 0:
        raise ValueError("top_p must be positive")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**31:
        raise ValueError("seed must be a nonnegative int32")
    if type(value["max_tokens"]) is not int or value["max_tokens"] <= 0:
        raise ValueError("positive max_tokens required")
    return value


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    request_count = 0
    request_count_lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def do_DELETE(self) -> None:  # noqa: N802
        self._forward()

    def _forward(self) -> None:
        if self.path == "/healthz":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        bare_path = self.path.split("?", 1)[0]
        allowed_paths = {
            path for path in os.environ.get("FIXED_ALLOWED_PATHS", "").split(",") if path
        }
        allowed_prefix = os.environ.get("FIXED_ALLOWED_PREFIX", "/")
        if (allowed_paths and bare_path not in allowed_paths) or (
            not allowed_paths and not self.path.startswith(allowed_prefix)
        ):
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", "0"))
        max_request_bytes = int(os.environ.get("FIXED_MAX_REQUEST_BYTES", "16777216"))
        if length > max_request_bytes:
            self.send_error(413)
            return
        max_requests = int(os.environ.get("FIXED_MAX_REQUESTS", "0"))
        with self.request_count_lock:
            type(self).request_count += 1
            request_number = type(self).request_count
        if max_requests and request_number > max_requests:
            self.send_error(429)
            return
        upstream = urlsplit(os.environ["FIXED_UPSTREAM"])
        prefix = upstream.path.rstrip("/")
        path = f"{prefix}{self.path}"
        body = self.rfile.read(length) if length else None
        policy = os.environ.get("FIXED_COMPLETION_JSON")
        if policy and bare_path == "/v1/chat/completions":
            try:
                overrides = completion_overrides(json.loads(policy))
                payload = json.loads(body or b"")
                if not isinstance(payload, dict):
                    raise ValueError("completion request must be an object")
                payload.pop("max_completion_tokens", None)
                payload.update(overrides)
                body = json.dumps(payload, allow_nan=False).encode()
            except (ValueError, TypeError):
                self.send_error(400)
                return
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP | {"host", "authorization", "content-length"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))
        auth_header = os.environ.get("FIXED_AUTH_HEADER")
        auth_value = os.environ.get("FIXED_AUTH_VALUE")
        if auth_header and auth_value:
            headers[auth_header] = auth_value

        connection_type = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(upstream.hostname, upstream.port, timeout=3600)
        started_at = time.time()
        status = 502
        response_bytes = 0
        try:
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
            status = response.status
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in _HOP_BY_HOP | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read(65536):
                response_bytes += len(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "method": self.command,
                        "path": self.path.split("?", 1)[0],
                        "request_bytes": len(body or b""),
                        "request_number": request_number,
                        "response_bytes": response_bytes,
                        "status": status,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            self.close_connection = True
            connection.close()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


if __name__ == "__main__":
    port = int(os.environ.get("FIXED_PROXY_PORT", "8080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
