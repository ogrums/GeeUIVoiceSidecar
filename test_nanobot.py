"""Tests for optional nanobot chat forwarding in emo_server."""

import importlib.util
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location(
    "emo_server", Path(__file__).with_name("emo_server.py")
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class RecordingHandler(BaseHTTPRequestHandler):
    """Shared fake upstream; subclass sets `owner` to a FakeUpstream."""

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.owner.requests.append(
            {
                "path": self.path,
                "body": body,
                "auth": self.headers.get("Authorization"),
                "content_type": self.headers.get("Content-Type"),
            }
        )
        self.owner.handle_post(self, body)


class FakeUpstream:
    def __init__(self, role):
        self.role = role
        self.requests = []
        self.mode = "ok"  # ok | down | no_done | error
        self._httpd = None
        self.url = None

    def start(self):
        owner = self

        class Handler(RecordingHandler):
            pass

        Handler.owner = owner
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        host, port = self._httpd.server_address
        self.url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()
        return self

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()

    def handle_post(self, handler, body):
        if self.mode == "down":
            handler.send_response(503)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(b'{"error":"down"}')
            return
        if self.role == "nanobot":
            self._nanobot(handler, body)
        else:
            self._lemonade(handler, body)

    def _nanobot(self, handler, body):
        data = json.loads(body.decode())
        text = (data.get("messages") or [{}])[0].get("content", "")
        stream = bool(data.get("stream"))
        if text.startswith("/model"):
            payload = {
                "id": "cmd",
                "choices": [{"message": {"role": "assistant", "content": f"ok:{text}"}}],
            }
            raw = json.dumps(payload).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(raw)))
            handler.end_headers()
            handler.wfile.write(raw)
            return
        if stream:
            chunks = [
                b'data: {"id":"1","choices":[{"delta":{"content":"[happy] Salut"}}]}\n\n',
            ]
            if self.mode != "no_done":
                chunks.append(b"data: [DONE]\n\n")
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            for chunk in chunks:
                handler.wfile.write(chunk)
                handler.wfile.flush()
            return
        payload = {
            "id": "chat",
            "choices": [{"message": {"role": "assistant", "content": f"[happy] {text}"}}],
        }
        raw = json.dumps(payload).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    def _lemonade(self, handler, body):
        # Lemonade host already includes /api/v1; sidecar strips /v1 prefix.
        payload = {
            "id": "lem",
            "choices": [{"message": {"role": "assistant", "content": "[neutral] lemonade"}}],
        }
        raw = json.dumps(payload).encode()
        if b'"stream": true' in body or b'"stream":true' in body:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.wfile.write(
                b'data: {"choices":[{"delta":{"content":"[neutral] lemonade"}}]}\n\n'
            )
            handler.wfile.write(b"data: [DONE]\n\n")
            return
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)


class SidecarHarness:
    def __init__(self, env_extra):
        self.env_extra = env_extra
        self.nanobot = FakeUpstream("nanobot").start()
        self.lemonade = FakeUpstream("lemonade").start()
        self.emo = None
        self._httpd = None
        self._old_env = {}

    def __enter__(self):
        mapping = {
            "EMO_HOST": "http://127.0.0.1:0",
            "LEMONADE_HOST": self.lemonade.url,
            "NANOBOT_URL": "",
            "NANOBOT_API_KEY": "",
            "NANOBOT_SESSION": "rux",
            "NANOBOT_FAST": "rapide",
            "NANOBOT_HEAVY": "lourd",
            **self.env_extra,
        }
        if mapping.get("NANOBOT_URL") == "AUTO":
            mapping["NANOBOT_URL"] = self.nanobot.url + "/v1"
        for key, value in mapping.items():
            self._old_env[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Reset mood between tests
        server.mood.update({"emotion": "neutral", "intensity": 0.0})
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        host, port = self._httpd.server_address
        self.emo = f"http://127.0.0.1:{port}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        self.nanobot.stop()
        self.lemonade.stop()
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def post_chat(self, payload, timeout=5):
        raw = json.dumps(payload).encode()
        req = Request(
            self.emo + "/v1/chat/completions",
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)


class HelperTests(unittest.TestCase):
    def test_is_heavy_keywords_and_length(self):
        self.assertTrue(server.is_heavy("explique-moi la photosynthèse", words=["explique"], max_light=200))
        self.assertTrue(server.is_heavy("x" * 201, words=["explique"], max_light=200))
        self.assertFalse(server.is_heavy("salut ça va", words=["explique"], max_light=200))

    def test_last_user_text_takes_last_user(self):
        messages = [
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
        self.assertEqual("second", server.last_user_text(messages))

    def test_heard_emotion_prefix_only_when_intensity(self):
        server.mood.update({"emotion": "happy", "intensity": 0.5, "at": __import__("time").time()})
        self.assertEqual("(ton entendu : happy) ", server.heard_emotion_prefix())
        server.mood.update({"emotion": "happy", "intensity": 0.0, "at": __import__("time").time()})
        self.assertEqual("", server.heard_emotion_prefix())


class NanobotForwardTests(unittest.TestCase):
    def test_disabled_passthrough_to_lemonade(self):
        with SidecarHarness({"NANOBOT_URL": ""}) as h:
            status, body, _ = h.post_chat(
                {
                    "model": "gemma",
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "bonjour"},
                    ],
                }
            )
            self.assertEqual(200, status)
            self.assertIn(b"lemonade", body)
            self.assertEqual(0, len(h.nanobot.requests))
            self.assertEqual(1, len(h.lemonade.requests))
            lem = json.loads(h.lemonade.requests[0]["body"])
            self.assertEqual("gemma", lem.get("model"))
            self.assertEqual(2, len(lem["messages"]))

    def test_transform_single_user_session_bearer_and_emotion(self):
        with SidecarHarness(
            {
                "NANOBOT_URL": "AUTO",
                "NANOBOT_API_KEY": "secret-key",
                "NANOBOT_SESSION": "rux",
            }
        ) as h:
            server.observe("sad", 0.9, __import__("time").time())
            status, body, _ = h.post_chat(
                {
                    "model": "should-drop",
                    "messages": [
                        {"role": "system", "content": "prompt"},
                        {"role": "user", "content": "ancienne"},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "Salut"},
                    ],
                }
            )
            self.assertEqual(200, status)
            self.assertIn(b"Salut", body)
            self.assertEqual(0, len(h.lemonade.requests))
            self.assertEqual(1, len(h.nanobot.requests))
            req = h.nanobot.requests[0]
            self.assertEqual("Bearer secret-key", req["auth"])
            payload = json.loads(req["body"])
            self.assertEqual("rux", payload["session_id"])
            self.assertNotIn("model", payload)
            self.assertEqual(1, len(payload["messages"]))
            self.assertEqual("user", payload["messages"][0]["role"])
            self.assertEqual("(ton entendu : sad) Salut", payload["messages"][0]["content"])

    def test_heavy_routing_sequence(self):
        with SidecarHarness(
            {
                "NANOBOT_URL": "AUTO",
                "NANOBOT_API_KEY": "k",
                "NANOBOT_HEAVY_WORDS": "explique",
            }
        ) as h:
            status, body, _ = h.post_chat(
                {
                    "messages": [{"role": "user", "content": "explique la gravité"}],
                }
            )
            self.assertEqual(200, status)
            texts = [
                json.loads(r["body"])["messages"][0]["content"] for r in h.nanobot.requests
            ]
            self.assertEqual(
                ["/model lourd", "explique la gravité", "/model rapide"],
                texts,
            )
            self.assertTrue(all(not json.loads(r["body"]).get("stream") for r in h.nanobot.requests if json.loads(r["body"])["messages"][0]["content"].startswith("/model")))

    def test_fallback_when_nanobot_down(self):
        with SidecarHarness(
            {"NANOBOT_URL": "AUTO", "NANOBOT_API_KEY": "k"}
        ) as h:
            h.nanobot.mode = "down"
            status, body, _ = h.post_chat(
                {"messages": [{"role": "user", "content": "bonjour"}]}
            )
            self.assertEqual(200, status)
            self.assertIn(b"lemonade", body)
            self.assertGreaterEqual(len(h.nanobot.requests), 1)
            self.assertEqual(1, len(h.lemonade.requests))

    def test_stream_passthrough_with_done(self):
        with SidecarHarness(
            {"NANOBOT_URL": "AUTO", "NANOBOT_API_KEY": "k"}
        ) as h:
            status, body, headers = h.post_chat(
                {
                    "stream": True,
                    "messages": [{"role": "user", "content": "salut"}],
                }
            )
            self.assertEqual(200, status)
            self.assertIn(b"[happy] Salut", body)
            self.assertIn(b"data: [DONE]", body)
            self.assertEqual(0, len(h.lemonade.requests))

    def test_stream_appends_done_when_missing(self):
        with SidecarHarness(
            {"NANOBOT_URL": "AUTO", "NANOBOT_API_KEY": "k"}
        ) as h:
            h.nanobot.mode = "no_done"
            status, body, _ = h.post_chat(
                {
                    "stream": True,
                    "messages": [{"role": "user", "content": "salut"}],
                }
            )
            self.assertEqual(200, status)
            self.assertIn(b"[happy] Salut", body)
            self.assertIn(b"data: [DONE]", body)


if __name__ == "__main__":
    unittest.main()
