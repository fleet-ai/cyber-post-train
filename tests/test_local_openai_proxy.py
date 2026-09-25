import io
import json
import threading
import unittest
from contextlib import redirect_stderr
from email.message import Message
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from training import local_openai_proxy as proxy


class FakeResponse(io.BytesIO):
    status = 200
    headers = Message()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ProxyTest(unittest.TestCase):
    def test_key_requires_one_bounded_private_line(self):
        self.assertEqual(proxy.read_key(io.BytesIO(b"synthetic-key\n")), "synthetic-key")
        for value in (b"", b"short\n", b"secret", b"bad key\n", b"x" * 1024 + b"\n"):
            with self.subTest(value=value[:10]), self.assertRaises(ValueError):
                proxy.read_key(io.BytesIO(value))

    def test_probe_is_authenticated_read_only_and_checks_exact_model(self):
        seen = []

        def open_request(request, timeout):
            seen.append((request.full_url, request.get_method(),
                         request.get_header("Authorization"), timeout))
            return FakeResponse(json.dumps({"id": proxy.MODEL}).encode())

        self.assertTrue(proxy.probe_model("synthetic-key", opener=open_request))
        self.assertEqual(seen, [(proxy.UPSTREAM + "/v1/models/" + proxy.MODEL,
                                 "GET", "Bearer synthetic-key", 20)])

    def test_proxy_contract_has_no_live_cli(self):
        with patch("sys.argv", ["proxy", "--serve"]), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(proxy.main(), 2)
        self.assertIn("Only the read-only", errors.getvalue())
        handler = proxy.handler_for("synthetic-key", upstream="http://127.0.0.1:9")
        self.assertEqual(handler.log_message.__code__.co_consts, (None,))

    def test_synthetic_proxy_forwards_only_exact_model_with_in_memory_key(self):
        seen = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                seen.append((self.path, self.headers.get("Authorization"),
                             json.loads(self.rfile.read(int(self.headers["Content-Length"])))) )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        relay = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            proxy.handler_for("synthetic-key", upstream=f"http://127.0.0.1:{server.server_port}"),
        )
        threads = [threading.Thread(target=service.serve_forever, daemon=True)
                   for service in (server, relay)]
        for thread in threads:
            thread.start()
        try:
            connection = HTTPConnection("127.0.0.1", relay.server_port, timeout=2)
            connection.request("POST", "/v1/chat/completions",
                               json.dumps({"model": proxy.MODEL, "messages": []}),
                               {"Authorization": "Bearer wrong-client-key"})
            response = connection.getresponse()
            self.assertEqual((response.status, response.read()), (200, b'{"ok":true}'))
            connection.close()
            self.assertEqual(seen, [("/v1/chat/completions", "Bearer synthetic-key",
                                     {"model": proxy.MODEL, "messages": []})])
            connection = HTTPConnection("127.0.0.1", relay.server_port, timeout=2)
            connection.request("POST", "/v1/chat/completions", '{"model":"other"}')
            self.assertEqual(connection.getresponse().status, 400)
            connection.close()
            self.assertEqual(len(seen), 1)
        finally:
            for service in (relay, server):
                service.shutdown()
                service.server_close()
            for thread in threads:
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
