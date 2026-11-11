"""Mode enfant / invité : rôles dans speakers.json, balises permises, préfixe, routes."""

import json
import os
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import emo_server as server
import speaker_id
from test_nanobot import SidecarHarness
from test_speaker import MARIE, PAUL, Clock, _json, service, vec, wav

LEA = vec(0, 0, 0, 1)


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clock = Clock()
        self.s = service(self.tmp, {"paul": ("Paul", PAUL), "lea": ("Léa", LEA)}, self.clock)
        self.s.extractor.table = {**self.s.extractor.table, 7: LEA}

    def test_default_actions_per_role(self):
        self.assertEqual(["move", "volume", "light", "photo", "home"], speaker_id.allowed_actions("adulte"))
        self.assertEqual(["volume", "light", "photo"], speaker_id.allowed_actions("enfant"))
        self.assertEqual(["volume", "light"], speaker_id.allowed_actions("invite"))
        self.assertIsNone(speaker_id.allowed_actions(None))
        with mock.patch.dict(os.environ, {"ACTIONS_ENFANT": "light, bogus"}):
            self.assertEqual(["light"], speaker_id.allowed_actions("enfant"))

    def test_stt_fields_carry_role(self):
        out = self.s.observe(wav(2, 1))
        self.assertEqual(("adulte", 5), (out["speaker_role"], len(out["allowed_actions"])))
        self.s.store.set_role("lea", "enfant")
        out = self.s.observe(wav(2, 7))
        self.assertEqual(("lea", "enfant"), (out["speaker"], out["speaker_role"]))
        self.assertNotIn("move", out["allowed_actions"])
        self.clock.t += 120
        out = self.s.observe(wav(2, 5))
        self.assertEqual(("invite", ["volume", "light"]), (out["speaker_role"], out["allowed_actions"]))

    def test_only_a_known_adult_sets_roles(self):
        self.s.observe(wav(2, 1))  # Paul (adulte : pas de rôle = adulte)
        self.assertIn("Léa est enfant", self.s.handle_chat("Léa est une enfant"))
        self.assertEqual("enfant", self.s.role_of("lea"))
        self.assertIn("autre adulte", self.s.handle_chat("Paul est un enfant"))
        self.assertIsNone(self.s.handle_chat("Mon fils est un enfant"))  # pas une voix connue : nanobot
        self.s.observe(wav(2, 7))  # Léa
        self.assertIn("Seul un adulte", self.s.handle_chat("Léa est une adulte"))
        self.assertEqual("enfant", self.s.role_of("lea"))

    def test_child_prefix(self):
        self.s.store.set_role("lea", "enfant")
        server.set_speakers(self.s)
        try:
            self.s.observe(wav(2, 7))
            self.assertEqual(("rux-lea", "(Léa parle, c'est un enfant) ", False), server.speaker_route("rux", ""))
        finally:
            server.set_speakers(None)

    def test_first_voice_is_adult_next_ones_child(self):
        s = service(tempfile.mkdtemp(), clock=self.clock)
        for name, tag in (("Paul", 1), ("Marie", 2)):
            s.observe(wav(2, tag))
            s.handle_chat(f"Retiens ma voix, je suis {name}")
            for _ in range(3):
                s.observe(wav(2.5, tag))
                s.handle_chat("phrase")
            self.clock.t += 5
        self.assertEqual("adulte", s.role_of("paul"))
        self.assertEqual("enfant", s.role_of("marie"))


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.s = service(tempfile.mkdtemp(), {"paul": ("Paul", PAUL), "marie": ("Marie", MARIE)})
        server.set_speakers(self.s)

    def tearDown(self):
        server.set_speakers(None)

    def test_role_route_needs_key(self):
        with SidecarHarness({"SIDECAR_ADMIN_KEY": "k"}) as h:
            def post(body, key="k"):
                req = Request(h.emo + "/v1/speakers/marie/role", data=json.dumps(body).encode(), method="POST",
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                try:
                    with urlopen(req, timeout=5) as r:
                        return r.status, json.loads(r.read())
                except HTTPError as exc:
                    return exc.code, None
            self.assertEqual(401, post({"role": "enfant"}, key="x")[0])
            self.assertEqual(400, post({"role": "roi"})[0])
            self.assertEqual((200, {"id": "marie", "role": "enfant"}), post({"role": "enfant"}))
            self.assertEqual("enfant", self.s.role_of("marie"))

    def test_guest_goes_to_lemonade_by_default(self):
        with SidecarHarness({"NANOBOT_URL": "AUTO", "SPEAKER_GUEST_ROUTE": None}) as h:
            self.s.observe(wav(2, 5))
            _, body, _ = h.post_chat({"messages": [{"role": "user", "content": "Salut"}]})
            self.assertIn(b"lemonade", body)
            self.assertEqual(0, len(h.nanobot.requests))


if __name__ == "__main__":
    unittest.main()
