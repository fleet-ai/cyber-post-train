"""Real loopback HTTP requests; all bodies and credentials are synthetic."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from evals.fleet.fixed_proxy import Handler, completion_overrides

POLICY = {"model": "frozen", "temperature": 0.6, "top_p": 0.95, "seed": 42, "max_tokens": 100}


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", float("nan")),
        ("temperature", True),
        ("temperature", 3),
        ("top_p", 0),
        ("top_p", -1),
        ("seed", -1),
        ("seed", True),
        ("seed", 2**31),
        ("max_tokens", 0),
        ("max_tokens", 1.5),
        ("model", ""),
        ("unknown", 1),
    ],
)
def test_bad_sampling_is_rejected(field, value):
    with pytest.raises(ValueError):
        completion_overrides({**POLICY, field: value})


@pytest.mark.parametrize(
    "policy,body,expected",
    [
        (POLICY, {"model": "agent-changed", "temperature": 2, "messages": [], "stream": True}, 200),
        (POLICY, {"model": "agent-changed", "max_completion_tokens": 99999}, 200),
        (None, {"model": "legacy"}, 200),
        (POLICY, [], 400),
        ({"model": "incomplete"}, {}, 400),
    ],
)
def test_wire_enforces_sampling_and_secret_boundary(monkeypatch, capsys, policy, body, expected):
    received = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                    self.headers["Authorization"],
                )
            )
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    monkeypatch.setenv("FIXED_UPSTREAM", f"http://127.0.0.1:{upstream.server_port}")
    monkeypatch.setenv("FIXED_ALLOWED_PATHS", "/v1/chat/completions")
    monkeypatch.setenv("FIXED_AUTH_HEADER", "Authorization")
    monkeypatch.setenv("FIXED_AUTH_VALUE", "synthetic-upstream-secret")
    monkeypatch.setenv("FIXED_COMPLETION_JSON", json.dumps(policy) if policy else "")
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, proxy)]
    for thread in threads:
        thread.start()
    try:
        response = httpx.post(
            f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
            json=body,
            headers={"Authorization": "agent-placeholder"},
            timeout=5,
        )
        assert response.status_code == expected
        if expected == 200:
            payload, authorization = received[0]
            assert authorization == "synthetic-upstream-secret"
            expected_body = {**body, **POLICY} if policy else body
            if policy:
                expected_body.pop("max_completion_tokens", None)
            assert payload == expected_body
        else:
            assert received == []
    finally:
        for server, thread in zip((upstream, proxy), threads, strict=True):
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
    assert "synthetic-upstream-secret" not in capsys.readouterr().out
