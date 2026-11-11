"""Initiative : horloge simulée, règles, rappels, suivis, tension, tâche du soir, routes."""

import datetime as dt
import json
import os
import tempfile
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import emo_server as server
import initiative as ini
from test_nanobot import SidecarHarness

PARIS = ini._zone("Europe/Paris")


def at(text):
    return dt.datetime.fromisoformat(text).replace(tzinfo=PARIS).timestamp()


class Clock:
    def __init__(self, when):
        self.t = at(when)

    def __call__(self):
        return self.t

    def set(self, when):
        self.t = at(when)


MEMORY = """# Mémoire de RUX

## Faits de la maison
- Le chat s'appelle Luna.

## À suivre
- 2026-11-17 | Paul | entretien d'embauche chez Dupont | demander comment ça s'est passé
- 2026-11-17 | Hélène | examen de piano | demander si ça a marché
- 17/11/2026 | Paul | mal formée | rien
- 2026-11-40 | Paul | date impossible | rien
- 2026-11-17 | Paul | trop de champs
"""


def make(clock, **over):
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "ws", "memory"))
    with open(os.path.join(tmp, "ws", "memory", "MEMORY.md"), "w", encoding="utf-8") as f:
        f.write(MEMORY)
    cfg = ini.settings()
    cfg.update(enabled=True, state_dir=os.path.join(tmp, "state"), workspace=os.path.join(tmp, "ws"),
               tz="Europe/Paris")
    cfg.update(over)
    calls = []

    def generate(person, instruction):
        calls.append((person, instruction))
        return None  # phrase de secours

    commands = []
    s = ini.Initiative(cfg, clock, generate, lambda: {"paul": "Paul", "helene": "Hélène"},
                       lambda person, text: commands.append((person, text)) or True)
    s.calls, s.commands = calls, commands
    return s


class ParseTests(unittest.TestCase):
    def test_followups(self):
        items = ini.parse_followups(MEMORY)
        self.assertEqual(2, len(items))
        self.assertEqual(("paul", dt.date(2026, 11, 17)), (items[0]["person"], items[0]["date"]))
        self.assertEqual("helene", items[1]["person"])
        self.assertNotEqual(items[0]["id"], items[1]["id"])
        self.assertEqual([], ini.parse_followups("## Autre\n- 2026-11-17 | Paul | a | b\n"))

    def test_reminder_tags(self):
        now = dt.datetime(2026, 11, 10, 18, 0, tzinfo=PARIS)
        got = ini.parse_reminders("[happy] Ok ! [rappel:2026-11-10T19:30|sortir le linge]", now)
        self.assertEqual([(dt.datetime(2026, 11, 10, 19, 30, tzinfo=PARIS), "sortir le linge")], got)
        self.assertEqual(now + dt.timedelta(minutes=60), ini.parse_reminders("[rappel:+60|thé]", now)[0][0])
        self.assertEqual([], ini.parse_reminders("[rappel:2026-11-09T19:30|passé]", now))
        self.assertEqual([], ini.parse_reminders("[rappel:demain|flou]", now))

    def test_dnd_phrases(self):
        now = dt.datetime(2026, 11, 10, 14, 0, tzinfo=PARIS)
        self.assertEqual(("dnd", now + dt.timedelta(hours=2)), ini.parse_dnd("Silence pendant deux heures", now))
        self.assertEqual(("dnd", now.replace(hour=17)), ini.parse_dnd("ne me dérange pas jusqu'à 17 h", now))
        self.assertEqual(("dnd", dt.datetime(2026, 11, 11, 8, 0, tzinfo=PARIS)),
                         ini.parse_dnd("tais-toi jusqu'à demain", now))
        self.assertEqual(("wake", None), ini.parse_dnd("tu peux reparler", now))
        self.assertIsNone(ini.parse_dnd("quelle heure est-il", now))


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock("2026-11-17T09:00")
        self.s = make(self.clock)

    def test_disabled_changes_nothing(self):
        off = make(self.clock, enabled=False)
        self.assertEqual("", off.user_turn("paul"))
        self.assertIsNone(off.phrase("ne me dérange pas"))
        self.assertEqual(0, off.reminders_from_reply("[rappel:+5|x]", "paul"))
        self.assertEqual([], off.tick())
        self.assertEqual([], off.items())

    def test_greeting_once_a_day_after_six(self):
        self.clock.set("2026-11-17T05:30")
        self.assertEqual("", self.s.user_turn("paul"))
        self.clock.set("2026-11-17T07:00")
        self.assertIn("première fois aujourd'hui que Paul", self.s.user_turn("paul"))
        self.assertEqual("", self.s.user_turn("paul"))
        self.assertEqual("", self.s.user_turn("invite"))
        self.clock.set("2026-11-18T07:00")
        self.assertIn("Paul", self.s.user_turn("paul"))

    def test_followup_needs_presence_and_is_asked_once(self):
        self.assertEqual([], self.s.tick())  # Paul n'est pas encore venu
        self.s.user_turn("paul")
        made = self.s.tick()
        self.assertEqual(["followup"], [m["kind"] for m in made])
        self.assertIn("entretien d'embauche", made[0]["text"])
        self.assertIn("entretien", self.s.calls[0][1])
        self.assertEqual("paul", self.s.calls[0][0])
        item = self.s.items()[0]
        self.assertNotIn("expires", item)
        self.s.ack(item["id"], "spoken")
        self.clock.t += 3 * 3600
        self.assertEqual([], self.s.tick())

    def test_skipped_followup_comes_back(self):
        self.s.user_turn("paul")
        item = self.s.tick()[0]
        self.s.ack(item["id"], "skipped")
        self.clock.t += 60
        self.assertEqual(["followup"], [m["kind"] for m in self.s.tick()])

    def test_quiet_hours_gap_and_max(self):
        self.s.user_turn("paul")
        self.s.user_turn("helene")
        self.clock.set("2026-11-17T22:00")
        self.assertEqual([], self.s.tick())
        self.clock.set("2026-11-17T10:00")
        first = self.s.tick()
        self.s.ack(first[0]["id"], "spoken")
        self.clock.set("2026-11-17T10:30")
        self.assertEqual([], self.s.tick())  # écart minimal de 90 min
        self.clock.set("2026-11-17T12:00")
        self.assertEqual(1, len(self.s.tick()))
        capped = make(self.clock, max_per_day=1)
        capped.user_turn("paul")
        capped.user_turn("helene")
        capped.ack(capped.tick()[0]["id"], "spoken")
        self.clock.t += 4 * 3600
        state = capped._state()
        state["last_at"] = 0
        capped._save_state(state)
        self.assertEqual(["helene"], [m["person"] for m in capped.tick()])  # max par personne

    def test_dnd_blocks_initiative_but_not_reminders(self):
        self.s.user_turn("paul")
        self.assertIn("jusqu'à 11 h", self.s.phrase("silence pendant deux heures"))
        self.s.reminders_from_reply("[rappel:+30|sortir le linge]", "paul")
        self.assertEqual([], self.s.tick())
        self.clock.t += 31 * 60
        made = self.s.tick()
        self.assertEqual(["reminder"], [m["kind"] for m in made])
        self.assertIn("Paul, petit rappel : sortir le linge", made[0]["text"])
        self.assertIn("de nouveau", self.s.phrase("tu peux reparler"))
        self.s.ack(made[0]["id"], "spoken")
        self.assertEqual(["followup"], [m["kind"] for m in self.s.tick()])

    def test_reminder_in_quiet_hours_and_never_late(self):
        self.clock.set("2026-11-17T23:00")
        self.s.reminders_from_reply("[rappel:2026-11-17T23:10|prendre le médicament]", "paul")
        self.s.reminders_from_reply("[rappel:2026-11-17T23:15|vieux]", "paul")
        self.clock.set("2026-11-17T23:11")
        self.assertEqual(["reminder"], [m["kind"] for m in self.s.tick()])
        self.clock.set("2026-11-18T01:00")
        self.assertEqual([], self.s.tick())  # le second est trop vieux : jeté

    def test_pending_expires(self):
        self.s.user_turn("paul")
        self.s.tick()
        self.clock.t += 31 * 60
        self.assertEqual([], self.s.items())

    def test_tension_three_days(self):
        s = make(Clock("2026-11-20T10:00"))
        for day in ("2026-11-18", "2026-11-19", "2026-11-20"):
            for i in range(10):
                emotion = "sad" if i < 5 else "neutral"
                s.record_emotion("helene", emotion, 0.8, at(f"{day}T09:{i:02d}"))
        s.record_emotion("invite", "sad", 0.9, at("2026-11-20T09:30"))
        s.user_turn("helene")
        made = s.tick()
        self.assertEqual(["tension"], [m["kind"] for m in made])
        self.assertIn("sous pression", made[0]["text"])
        s.ack(made[0]["id"], "spoken")
        s.clock.t += 2 * 3600
        self.assertEqual([], s.tick())  # au plus une fois par semaine

    def test_tension_needs_enough_turns(self):
        s = make(Clock("2026-11-20T10:00"))
        for day in ("2026-11-18", "2026-11-19", "2026-11-20"):
            for i in range(4):
                s.record_emotion("helene", "sad", 0.9, at(f"{day}T09:{i:02d}"))
        s.user_turn("helene")
        self.assertEqual([], s.tick())

    def test_nightly_compact_then_dream_once(self):
        self.clock.set("2026-11-18T03:05")
        self.s.tick()
        self.s.tick()
        self.assertEqual([("helene", "/compact"), ("paul", "/compact"), (None, "/dream")], self.s.commands)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock("2026-11-17T09:00")
        self.s = make(self.clock)
        server.set_initiative(self.s)

    def tearDown(self):
        server.set_initiative(None)

    def get(self, h, path):
        with urlopen(h.emo + path, timeout=5) as r:
            return json.loads(r.read())

    def post(self, h, path, payload):
        req = Request(h.emo + path, data=json.dumps(payload).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def test_pending_and_ack(self):
        with SidecarHarness({}) as h:
            self.assertEqual({"items": []}, self.get(h, "/v1/robot/pending"))
            self.s.reminders_from_reply("[rappel:+1|boire de l'eau]", None)
            self.clock.t += 61
            self.s.tick()
            items = self.get(h, "/v1/robot/pending")["items"]
            self.assertEqual("reminder", items[0]["kind"])
            self.assertTrue(self.post(h, "/v1/robot/ack", {"id": items[0]["id"], "status": "spoken"})["ok"])
            self.assertEqual([], self.get(h, "/v1/robot/pending")["items"])
            with self.assertRaises(HTTPError):
                self.post(h, "/v1/robot/ack", {"id": items[0]["id"]})

    def test_reminder_tag_split_across_sse_chunks_is_saved(self):
        with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
            def nanobot(handler, body):
                handler.send_response(200)
                handler.send_header("Content-Type", "text/event-stream")
                handler.end_headers()
                for piece in ["[happy] D'accord, je te le rappellerai. [rap", "pel:+90|sortir le", " linge]"]:
                    handler.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n")
                    handler.wfile.flush()
                handler.wfile.write(b"data: [DONE]\n\n")
            h.nanobot.handle_post = nanobot
            h.post_chat({"stream": True, "messages": [{"role": "user", "content": "rappelle-moi dans 1 h 30 de sortir le linge"}]})
            path = os.path.join(self.s.cfg["state_dir"], "reminders.json")
            for _ in range(50):  # le sidecar lit les balises juste après la fin du flux
                if os.path.exists(path):
                    break
                time.sleep(0.05)
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual("sortir le linge", saved[0]["text"])

    def test_dnd_phrase_answered_locally(self):
        with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
            _, body, _ = h.post_chat({"messages": [{"role": "user", "content": "Ne me dérange pas"}]})
            self.assertIn("je ne parle plus de moi-même", json.loads(body)["choices"][0]["message"]["content"])
            self.assertEqual(0, len(h.nanobot.requests))


if __name__ == "__main__":
    unittest.main()
