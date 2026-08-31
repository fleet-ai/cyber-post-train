"""Secret-holding fixed-upstream proxy for the isolated Qwen Code container."""

from __future__ import annotations

import http.client
import json
import os
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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        allowed_prefix = os.environ.get("FIXED_ALLOWED_PREFIX", "/")
        if not self.path.startswith(allowed_prefix):
            self.send_error(404)
            return
        upstream = urlsplit(os.environ["FIXED_UPSTREAM"])
        prefix = upstream.path.rstrip("/")
        path = f"{prefix}{self.path}"
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length) if length else None
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
