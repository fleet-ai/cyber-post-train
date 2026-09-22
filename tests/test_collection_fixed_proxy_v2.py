"""Regressions for the collection-only completion-budget proxy."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from evals.fleet import collection_fixed_proxy_v2 as proxy


def test_discovery_does_not_consume_completion_budget(
    monkeypatch,
    tmp_path,
) -> None:
    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = b'{"data":[]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length", "0"))
            self.rfile.read(length)
            body = b'{"choices":[]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    evidence = tmp_path / "completion-budget.json"
    monkeypatch.setenv("FIXED_UPSTREAM", f"http://127.0.0.1:{upstream.server_port}")
    monkeypatch.setenv("FIXED_ALLOWED_PATHS", "/v1/chat/completions,/v1/models")
    monkeypatch.setenv("FIXED_MAX_REQUESTS", "30")
    monkeypatch.setenv("FIXED_MAX_COMPLETIONS", "2")
    monkeypatch.setenv("FIXED_BUDGET_EVIDENCE_PATH", str(evidence))
    monkeypatch.setenv("FIXED_MAX_REQUEST_BYTES", "4096")
    monkeypatch.setenv(
        "FIXED_COMPLETION_JSON",
        json.dumps(
            {
                "model": "qwen",
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": 43,
                "max_tokens": 1024,
            }
        ),
    )
    proxy.Handler.request_count = 0
    proxy.Handler.completion_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    request = {"model": "agent-choice", "messages": []}
    try:
        assert httpx.get(base + "/v1/models").status_code == 200
        assert httpx.post(base + "/v1/chat/completions", json=request).status_code == 200
        assert httpx.post(base + "/v1/chat/completions", json=request).status_code == 200
        assert httpx.get(base + "/v1/models").status_code == 200
        rejected = httpx.post(base + "/v1/chat/completions", json=request)
        assert rejected.status_code == 429
        assert rejected.json() == {"error": {"code": "fleet_completion_budget_exhausted"}}
    finally:
        server.shutdown()
        upstream.shutdown()

    receipt = json.loads(evidence.read_text())
    assert receipt == {
        "schema": proxy.BUDGET_RECEIPT_SCHEMA,
        "reason": "exact_completion_budget_exhausted",
        "maximum_total_requests": 30,
        "completed_total_requests": 4,
        "maximum_completions": 2,
        "completed_completions": 2,
        "rejected_completion_number": 3,
        "observed_request_number": 5,
        "prompts_responses_scores_or_credentials_included": False,
        "sha256": receipt["sha256"],
    }
    assert proxy.Handler.request_count == 4
    assert proxy.Handler.completion_count == 2


def test_budget_receipt_is_create_once(monkeypatch, tmp_path) -> None:
    path = tmp_path / "budget.json"
    body = {
        "schema": proxy.BUDGET_RECEIPT_SCHEMA,
        "reason": "exact_completion_budget_exhausted",
        "maximum_total_requests": 630,
        "completed_total_requests": 601,
        "maximum_completions": 600,
        "completed_completions": 600,
        "rejected_completion_number": 601,
        "observed_request_number": 602,
        "prompts_responses_scores_or_credentials_included": False,
    }
    proxy._write_budget_receipt(path, body)  # noqa: SLF001
    first = path.read_bytes()
    proxy._write_budget_receipt(path, body)  # noqa: SLF001
    assert path.read_bytes() == first

    changed = dict(body, observed_request_number=603)
    try:
        proxy._write_budget_receipt(path, changed)  # noqa: SLF001
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("budget evidence must refuse a changed overwrite")
