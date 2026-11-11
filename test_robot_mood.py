"""Humeur de RUX : dynamique lente, ennui, compliments, nuit, routes et préfixe nanobot."""

import json
import os
import tempfile
import unittest
from urllib.request import urlopen

import emo_server as server
import robot_mood
from test_nanobot import SidecarHarness

H = 3600


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def make(clock, enabled=True, night=lambda stamp: False):
    cfg = robot_mood.settings()
    cfg.update(enabled=enabled, file=os.path.join(tempfile.mkdtemp(), "m.json"))
    return robot_mood.RobotMood(cfg, clock, night)


class MoodTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.m = make(self.clock)

    def test_rest_is_calm_and_disabled_says_nothing(self):
        self.assertEqual("calme", self.m.snapshot()["label"])
        off = make(self.clock, enabled=False)
        self.assertFalse(off.user_turn("merci"))
        self.assertEqual("", off.prefix())

    def test_compliments_make_it_joyful_then_it_relaxes(self):
        for _ in range(3):
            self.m.user_turn("Merci, t'es trop mignon")
        self.assertEqual("joyeux", self.m.snapshot()["label"])
        self.assertIn("très bonne humeur", self.m.prefix())
        self.clock.t += 6 * H  # deux demi-vies
        self.assertNotEqual("joyeux", self.m.snapshot()["label"])

    def test_rudeness_makes_it_sad(self):
        self.m.user_turn("t'es nul")
        self.m.user_turn("idiot")
        self.assertEqual("triste", self.m.snapshot()["label"])

    def test_alone_for_hours_bored_then_sulky_then_missed(self):
        self.clock.t += 5 * H
        self.assertEqual("ennuye", self.m.snapshot()["label"])
        self.clock.t += 5 * H
        self.assertEqual("boudeur", self.m.snapshot()["label"])
        self.assertTrue(self.m.user_turn("coucou"))
        self.assertIn("longue absence", self.m.prefix(missed=True))
        self.assertFalse(self.m.user_turn("ça va ?"))

    def test_night_is_tired_and_not_bored(self):
        m = make(self.clock, night=lambda stamp: True)
        self.clock.t += 10 * H
        self.assertEqual("fatigue", m.snapshot()["label"])

    def test_state_survives_restart(self):
        for _ in range(3):
            self.m.user_turn("bravo")
        again = robot_mood.RobotMood(self.m.cfg, self.clock)
        self.assertEqual("joyeux", again.snapshot()["label"])


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.m = make(Clock())
        server.set_robot_mood(self.m)

    def tearDown(self):
        server.set_robot_mood(None)

    def test_mood_route_pending_and_prefix(self):
        with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
            for _ in range(3):
                h.post_chat({"messages": [{"role": "user", "content": "merci, tu es génial"}]})
            sent = json.loads(h.nanobot.requests[-1]["body"])["messages"][0]["content"]
            self.assertTrue(sent.startswith("(humeur de RUX : de très bonne humeur) "), sent)
            mood = json.loads(urlopen(h.emo + "/v1/robot/mood", timeout=5).read())
            self.assertEqual("joyeux", mood["label"])
            pending = json.loads(urlopen(h.emo + "/v1/robot/pending", timeout=5).read())
            self.assertEqual("joyeux", pending["mood"]["label"])


if __name__ == "__main__":
    unittest.main()
