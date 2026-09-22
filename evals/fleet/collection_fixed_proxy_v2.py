"""Fixed non-thinking proxy with an exact chat-completion budget.

The historical collection proxy counts ``/v1/models`` discovery calls against
the same ceiling as generation calls.  This successor leaves those bytes
untouched and separates the two limits:

* ``FIXED_MAX_REQUESTS`` bounds every forwarded request; and
* ``FIXED_MAX_COMPLETIONS`` bounds only ``/v1/chat/completions``.

When the completion limit is reached, the proxy writes one content-free,
create-once receipt before returning 429.  A collection-only runtime can use
that receipt to classify the terminal condition as an output limit without
reading model output or treating a generic process error as scientific data.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
COMPLETION_PATH = "/v1/chat/completions"
BUDGET_RECEIPT_SCHEMA = "cyber_fleet_collection_completion_budget_exhausted_v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def completion_overrides(value: dict) -> dict:
    """Validate the operator-owned visible-action sampling policy."""
    if set(value) != {"model", "temperature", "top_p", "seed", "max_tokens"}:
        raise ValueError("incomplete fixed completion policy")
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("fixed model required")
    for field, minimum, maximum in (("temperature", 0, 2), ("top_p", 0, 1)):
        item = value[field]
        if (
            type(item) not in (int, float)
            or not math.isfinite(item)
            or not minimum <= item <= maximum
        ):
            raise ValueError("invalid sampling value")
    if value["top_p"] == 0:
        raise ValueError("top_p must be positive")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**31:
        raise ValueError("seed must be a nonnegative int32")
    if type(value["max_tokens"]) is not int or value["max_tokens"] <= 0:
        raise ValueError("positive max_tokens required")
    return value


def rewrite_completion(body: bytes, policy: dict) -> bytes:
    """Apply fixed sampling and force the exact non-thinking template input."""
    overrides = completion_overrides(policy)
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("completion request must be an object")
    payload.pop("max_completion_tokens", None)
    payload.pop("enable_thinking", None)
    payload.update(overrides)
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    return json.dumps(payload, allow_nan=False).encode()


def _write_budget_receipt(path: Path, body: dict[str, object]) -> None:
    unsigned = _canonical(body)
    receipt = {
        **body,
        "sha256": "sha256:" + hashlib.sha256(unsigned).hexdigest(),
    }
    payload = _canonical(receipt) + b"\n"
    try:
        parent = path.parent.lstat()
    except OSError as error:
        raise RuntimeError("completion budget evidence directory is unavailable") from error
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise RuntimeError("completion budget evidence directory must be a real directory")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError("completion budget receipt changed") from None
        return
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    request_count = 0
    completion_count = 0
    request_count_lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def do_DELETE(self) -> None:  # noqa: N802
        self._forward()

    def _error(self, status: int, code: str) -> None:
        body = _canonical({"error": {"code": code}})
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _reserve(self, bare_path: str) -> tuple[int, int | None, str | None]:
        max_requests = int(os.environ.get("FIXED_MAX_REQUESTS", "0"))
        max_completions = int(os.environ.get("FIXED_MAX_COMPLETIONS", "0"))
        if max_requests < 0 or max_completions < 0:
            raise ValueError("request limits must be nonnegative")
        is_completion = bare_path == COMPLETION_PATH
        with self.request_count_lock:
            request_number = type(self).request_count + 1
            completion_number = type(self).completion_count + 1 if is_completion else None
            if max_requests and request_number > max_requests:
                return request_number, completion_number, "total_request_budget_exhausted"
            if is_completion and max_completions and completion_number > max_completions:
                receipt_path = os.environ.get("FIXED_BUDGET_EVIDENCE_PATH")
                if not receipt_path:
                    raise RuntimeError("completion budget evidence path is required")
                _write_budget_receipt(
                    Path(receipt_path),
                    {
                        "schema": BUDGET_RECEIPT_SCHEMA,
                        "reason": "exact_completion_budget_exhausted",
                        "maximum_total_requests": max_requests,
                        "completed_total_requests": type(self).request_count,
                        "maximum_completions": max_completions,
                        "completed_completions": type(self).completion_count,
                        "rejected_completion_number": completion_number,
                        "observed_request_number": request_number,
                        "prompts_responses_scores_or_credentials_included": False,
                    },
                )
                return request_number, completion_number, "completion_budget_exhausted"
            type(self).request_count = request_number
            if is_completion:
                type(self).completion_count = completion_number
            return request_number, completion_number, None

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
        request_number, completion_number, rejection = self._reserve(bare_path)
        if rejection is not None:
            self._error(429, f"fleet_{rejection}")
            return
        upstream = urlsplit(os.environ["FIXED_UPSTREAM"])
        prefix = upstream.path.rstrip("/")
        path = f"{prefix}{self.path}"
        body = self.rfile.read(length) if length else None
        policy = os.environ.get("FIXED_COMPLETION_JSON")
        if policy and bare_path == COMPLETION_PATH:
            try:
                body = rewrite_completion(body or b"", json.loads(policy))
            except (json.JSONDecodeError, ValueError, TypeError):
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
                        "completion_number": completion_number,
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "method": self.command,
                        "path": bare_path,
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
