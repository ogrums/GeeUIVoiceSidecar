"""Qui parle : identification, enrôlement à la voix, routes. Sans modèle (faux extracteur),
plus un test avec le vrai ResNet34-LM s'il est présent (SPEAKER_TEST_MODEL)."""

import glob
import io
import json
import os
import stat
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import emo_server as server
import speaker_id
from test_nanobot import SidecarHarness


def wav(seconds, tag=0):
    """WAV 16 kHz ; le premier échantillon porte `tag` pour que le faux extracteur s'y retrouve."""
    frames = int(16000 * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(struct.pack("<h", tag) + b"\x00\x00" * (frames - 1))
    return buffer.getvalue()


def vec(*head):
    return list(head) + [0.0] * (8 - len(head))


PAUL = vec(1, 0)
MARIE = vec(0, 1)


class FakeExtractor:
    name = "fake.onnx"
    error = None

    def __init__(self, table):
        self.table = table  # tag -> vecteur

    def available(self):
        return True

    def embed(self, data):
        with wave.open(io.BytesIO(data), "rb") as pcm:
            tag = struct.unpack("<h", pcm.readframes(1))[0]
        return self.table.get(tag)


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


TABLE = {
    1: PAUL, 2: MARIE,
    3: vec(0.9, 0.1, 0.3),      # Paul un peu enrhumé
    4: vec(0.6, 0.58),          # entre les deux
    5: vec(0, 0, 1),            # inconnu
    6: vec(0.4, 0, 0.9),        # douteux pour Paul (≈0,41)
}


def service(tmp, people=None, clock=None):
    cfg = speaker_id.settings()
    cfg.update(model="fake.onnx", file=os.path.join(tmp, "speakers.json"))
    store = speaker_id.Store(cfg["file"], "fake.onnx")
    for pid, (name, emb) in (people or {}).items():
        store.put(pid, {"name": name, "embedding": emb})
    return speaker_id.Speakers(FakeExtractor(TABLE), store, cfg, clock or Clock())


class IdentifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clock = Clock()
        self.s = service(self.tmp, {"paul": ("Paul", PAUL), "marie": ("Marie", MARIE)}, self.clock)

    def test_threshold_margin_and_guest(self):
        self.assertEqual("paul", self.s.observe(wav(2, 1))["speaker"])
        self.assertEqual("marie", self.s.observe(wav(2, 2))["speaker_name"].lower())
        self.assertEqual("paul", self.s.observe(wav(2, 3))["speaker"])
        self.clock.t += 120
        self.assertEqual("invite", self.s.observe(wav(2, 4))["speaker"])  # pas assez d'écart
        self.assertEqual("invite", self.s.observe(wav(2, 5))["speaker"])

    def test_short_phrase_keeps_last_person(self):
        self.s.observe(wav(2, 1))
        self.clock.t += 10
        self.assertEqual("paul", self.s.observe(wav(0.6, 5))["speaker"])
        self.clock.t += 120
        self.assertEqual("invite", self.s.observe(wav(0.6, 5))["speaker"])

    def test_doubtful_score_keeps_recent_person(self):
        self.s.observe(wav(2, 1))
        self.clock.t += 30
        self.assertEqual("paul", self.s.observe(wav(2, 6))["speaker"])
        self.clock.t += 120
        self.assertEqual("invite", self.s.observe(wav(2, 6))["speaker"])

    def test_route_session_and_prefix(self):
        server.set_speakers(self.s)
        try:
            self.s.observe(wav(2, 1))
            self.assertEqual(("rux-paul", "(Paul parle, ton entendu : sad) ", False),
                             server.speaker_route("rux", "(ton entendu : sad) "))
            self.clock.t += 120
            self.s.observe(wav(2, 5))
            self.assertEqual(("rux-invite", "(un invité parle) ", True), server.speaker_route("rux", ""))
        finally:
            server.set_speakers(None)


class EnrollTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clock = Clock()
        self.s = service(self.tmp, clock=self.clock)

    def say(self, text, audio=None):
        """Un tour : STT (empreinte) puis chat."""
        if audio is not None:
            self.s.observe(audio)
        return self.s.handle_chat(text)

    def test_enroll_three_phrases(self):
        self.assertIsNone(self.s.who())  # personne d'inscrit : session « rux » comme avant
        reply = self.say("Retiens ma voix, je suis Paul.", wav(2, 5))
        self.assertIn("D'accord Paul", reply)
        self.assertIn("empreinte", reply)
        self.assertIn("Encore", self.say("Il fait beau aujourd'hui.", wav(2.5, 1)))
        self.assertIn("trop courte", self.say("Oui.", wav(1, 1)))
        self.assertIn("dernière", self.say("J'aime le jazz et le café.", wav(2.5, 3)))
        self.assertIn("C'est noté, Paul", self.say("Et voilà la troisième phrase.", wav(2.5, 1)))
        people = self.s.store.people()
        self.assertEqual(["paul"], list(people))
        self.assertEqual("voice", people["paul"]["consent"])
        self.assertEqual(3, people["paul"]["samples"])
        self.assertEqual(("paul", "Paul"), self.s.who())
        self.assertIsNone(self.s.handle_chat("Quelle heure est-il ?"))  # retour à nanobot
        mode = stat.S_IMODE(os.stat(os.path.join(self.tmp, "speakers.json")).st_mode)
        self.assertEqual(0o600, mode)
        self.assertEqual([], glob.glob(os.path.join(self.tmp, ".speakers-*")))

    def test_two_voices_are_refused(self):
        self.say("je m'appelle Paul, retiens ma voix", wav(2, 5))
        self.say("une", wav(2.5, 1))
        self.say("deux", wav(2.5, 2))
        self.assertIn("ne se ressemblent pas", self.say("trois", wav(2.5, 1)))
        self.assertEqual({}, self.s.store.people())

    def test_cancel_and_timeout(self):
        self.say("retiens ma voix je suis Paul", wav(2, 5))
        self.assertIn("j'arrête", self.say("annule", wav(2.5, 1)))
        self.say("retiens ma voix je suis Paul", wav(2, 5))
        self.clock.t += 200
        self.s.observe(wav(2.5, 1))
        self.assertIsNone(self.s.enroll)

    def test_cannot_overwrite_someone_else(self):
        self.s.store.put("paul", {"name": "Paul", "embedding": PAUL})
        self.s.observe(wav(2, 2))  # inconnu (Marie n'est pas inscrite)
        self.assertIn("Je connais déjà la voix de Paul", self.s.handle_chat("retiens ma voix, je suis Paul"))
        self.s.observe(wav(2, 1))  # Paul lui-même peut réenregistrer
        self.assertIn("D'accord Paul", self.s.handle_chat("retiens ma voix, je suis Paul"))

    def test_forget_only_own_voice_and_whoami(self):
        self.s.store.put("paul", {"name": "Paul", "embedding": PAUL})
        self.s.store.put("marie", {"name": "Marie", "embedding": MARIE})
        self.s.observe(wav(2, 1))
        self.assertIn("Tu es Paul", self.s.handle_chat("Tu me reconnais ?"))
        self.assertIn("J'ai oublié ta voix, Paul", self.s.handle_chat("Oublie ma voix."))
        self.assertEqual(["marie"], list(self.s.store.people()))
        self.s.observe(wav(2, 5))
        self.assertIn("rien à oublier", self.s.handle_chat("oublie ma voix"))

    def test_accents_in_name(self):
        reply = self.say("retiens ma voix, je suis hélène", wav(2, 5))
        self.assertIn("Hélène", reply)
        self.assertEqual("helene", self.s.enroll["id"])

    def test_disabled_without_model(self):
        cfg = speaker_id.settings()
        cfg.update(model="", file=os.path.join(self.tmp, "x.json"))
        off = speaker_id.Speakers(cfg=cfg)
        self.assertFalse(off.enabled())
        self.assertIsNone(off.observe(wav(2, 1)))
        self.assertIsNone(off.handle_chat("retiens ma voix, je suis Paul"))
        self.assertIsNone(off.who())

    def test_model_change_ignores_old_prints(self):
        self.s.store.put("paul", {"name": "Paul", "embedding": PAUL})
        other = speaker_id.Store(self.s.store.path, "autre.onnx")
        self.assertEqual({}, other.people())


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.s = service(self.tmp, {"paul": ("Paul", PAUL)})
        server.set_speakers(self.s)

    def tearDown(self):
        server.set_speakers(None)

    def transcribe(self, h, audio):
        boundary = "XyZ"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                "Content-Type: audio/wav\r\n\r\n").encode() + audio + f"\r\n--{boundary}--\r\n".encode()
        req = Request(h.emo + "/v1/audio/transcriptions", data=body, method="POST",
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def test_stt_fields_and_session_per_person(self):
        with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
            h.lemonade.handle_post = lambda handler, body: _json(handler, {"text": "Bonjour"})
            out = self.transcribe(h, wav(2, 1))
            self.assertEqual("paul", out["speaker"])
            self.assertEqual("Paul", out["speaker_name"])
            self.assertGreater(out["speaker_score"], 0.9)
            h.post_chat({"messages": [{"role": "user", "content": "Salut"}]})
            sent = json.loads(h.nanobot.requests[-1]["body"])
            self.assertEqual("rux-paul", sent["session_id"])
            self.assertTrue(sent["messages"][0]["content"].startswith("(Paul parle"))

    def test_guest_to_lemonade_when_asked(self):
        with SidecarHarness({"NANOBOT_URL": "AUTO", "SPEAKER_GUEST_ROUTE": "lemonade"}) as h:
            self.s.observe(wav(2, 5))
            _, body, _ = h.post_chat({"messages": [{"role": "user", "content": "Salut"}]})
            self.assertIn(b"lemonade", body)
            self.assertEqual(0, len(h.nanobot.requests))

    def test_local_reply_is_openai_sse_and_skips_nanobot(self):
        with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
            self.s.observe(wav(2, 1))
            status, body, headers = h.post_chat(
                {"stream": True, "messages": [{"role": "user", "content": "Qui suis-je ?"}]})
            self.assertEqual(200, status)
            self.assertIn("text/event-stream", headers["Content-Type"])
            self.assertIn("Tu es Paul", body.decode())
            self.assertTrue(body.decode().rstrip().endswith("data: [DONE]"))
            self.assertEqual(0, len(h.nanobot.requests))
            _, body, _ = h.post_chat({"messages": [{"role": "user", "content": "qui suis-je"}]})
            self.assertIn("Tu es Paul", json.loads(body)["choices"][0]["message"]["content"])

    def test_admin_routes_need_key(self):
        with SidecarHarness({"SIDECAR_ADMIN_KEY": "k"}) as h:
            with self.assertRaises(HTTPError) as err:
                urlopen(h.emo + "/v1/speakers", timeout=5)
            self.assertEqual(401, err.exception.code)
            req = Request(h.emo + "/v1/speakers", headers={"Authorization": "Bearer k"})
            listing = json.loads(urlopen(req, timeout=5).read())
            self.assertEqual("paul", listing["people"][0]["id"])
            self.assertNotIn("embedding", listing["people"][0])
            req = Request(h.emo + "/v1/speakers/paul", method="DELETE", headers={"X-Admin-Key": "k"})
            self.assertTrue(json.loads(urlopen(req, timeout=5).read())["deleted"])
            self.assertEqual({}, self.s.store.people())


def _json(handler, payload):
    raw = json.dumps(payload).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


SAMPLES = os.environ.get("SPEAKER_TEST_SAMPLES", "tests/samples/ravdess")
MODEL = os.environ.get("SPEAKER_TEST_MODEL", "")


@unittest.skipUnless(MODEL and os.path.isfile(MODEL) and glob.glob(SAMPLES + "/*.wav"),
                     "SPEAKER_TEST_MODEL et les extraits RAVDESS sont nécessaires")
class RealModelTests(unittest.TestCase):
    def test_actor_01_vs_02(self):
        extractor = speaker_id.SherpaExtractor(MODEL)
        self.assertTrue(extractor.available(), extractor.error)
        files = sorted(glob.glob(SAMPLES + "/*.wav"))
        by_actor = {}
        for path in files:
            by_actor.setdefault(path[-6:-4], []).append(path)
        a, b = sorted(by_actor)[:2]
        tmp = tempfile.mkdtemp()
        cfg = speaker_id.settings()
        cfg.update(model=MODEL, file=os.path.join(tmp, "s.json"))
        s = speaker_id.Speakers(extractor, speaker_id.Store(cfg["file"], extractor.name), cfg)
        for actor in (a, b):
            prints = [extractor.embed(Path(p).read_bytes()) for p in by_actor[actor][:3]]
            s.store.put(actor, {"name": actor, "embedding": speaker_id.mean_vector(prints)})
        wrong = 0
        for actor in (a, b):
            for path in by_actor[actor][3:]:
                s.current = None
                got = s.observe(Path(path).read_bytes())["speaker"]
                wrong += got not in (actor, speaker_id.GUEST)
        self.assertEqual(0, wrong)


if __name__ == "__main__":
    unittest.main()
