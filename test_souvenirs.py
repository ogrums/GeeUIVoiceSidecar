"""Souvenirs : [oubli:voix] écrit par nanobot (skill « oublie ») efface l'empreinte de la personne."""

import json
import tempfile
import time
import unittest

import emo_server as server
from test_nanobot import SidecarHarness
from test_speaker import MARIE, PAUL, service, wav


class ForgetVoiceTests(unittest.TestCase):
    def setUp(self):
        self.s = service(tempfile.mkdtemp(), {"paul": ("Paul", PAUL), "marie": ("Marie", MARIE)})
        server.set_speakers(self.s)

    def tearDown(self):
        server.set_speakers(None)

    def chat(self, h, reply, stream):
        def nanobot(handler, body):
            if stream:
                handler.send_response(200)
                handler.send_header("Content-Type", "text/event-stream")
                handler.end_headers()
                for piece in [reply[:20], reply[20:]]:
                    handler.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n")
                handler.wfile.write(b"data: [DONE]\n\n")
                return
            raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(raw)))
            handler.end_headers()
            handler.wfile.write(raw)
        h.nanobot.handle_post = nanobot
        h.post_chat({"stream": stream, "messages": [{"role": "user", "content": "oui"}]})
        for _ in range(50):
            if "paul" not in self.s.store.people():
                break
            time.sleep(0.05)

    def test_tag_removes_only_the_speaker(self):
        for stream in (False, True):
            self.s.store.put("paul", {"name": "Paul", "embedding": PAUL})
            self.s.observe(wav(2, 1))
            with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
                self.chat(h, "[neutral] C'est fait, j'ai tout oublié. [oubli:voix]", stream)
            self.assertEqual(["marie"], list(self.s.store.people()), stream)

    def test_no_tag_or_guest_keeps_prints(self):
        self.s.observe(wav(2, 1))
        with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
            self.chat(h, "[happy] D'accord.", False)
        self.s.observe(wav(2, 5))  # invité
        with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
            self.chat(h, "[neutral] Non. [oubli:voix]", False)
        self.assertEqual({"paul", "marie"}, set(self.s.store.people()))


if __name__ == "__main__":
    unittest.main()
