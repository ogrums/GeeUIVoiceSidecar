"""Pocket TTS routing with Kokoro fallback, and Whisper text cleanup."""

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import emo_server as server


class Fake:
    """Tiny upstream: records POST bodies, answers with `reply(path, body)`."""

    def __init__(self, reply):
        self.reply = reply
        self.posts = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self._answer(self.path, b"")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                owner.posts.append((self.path, body))
                self._answer(self.path, body)

            def _answer(self, path, body):
                code, ctype, raw = owner.reply(path, body)
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def pocket_ok(path, body):
    if path == "/v1/voices":
        return 200, "application/json", b'{"data":[{"id":"alba"},{"id":"cosette"}]}'
    return 200, "audio/wav", b"RIFFpocket"


def pocket_down(path, body):
    return 500, "application/json", b'{"error":"boom"}'


def lemonade(path, body):
    if path.endswith("/audio/transcriptions"):
        return 200, "application/json", b'{"text":" Avance la voiture.\\n","segments":[{"text":" Avance\\n"}]}'
    return 200, "audio/mpeg", b"ID3kokoro"


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.lemonade = Fake(lemonade)
        self.pocket = None
        self.old = dict(os.environ)
        server._pocket_voices.update({"at": 0.0, "url": "", "names": set()})

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.lemonade.close()
        if self.pocket:
            self.pocket.close()
        os.environ.clear()
        os.environ.update(self.old)

    def start(self, pocket=None, **env):
        if pocket:
            self.pocket = Fake(pocket)
        os.environ.update({
            "LEMONADE_HOST": self.lemonade.url + "/api/v1",
            "POCKET_TTS_URL": self.pocket.url if self.pocket else "",
            "POCKET_TTS_VOICE": "alba",
            "POCKET_TTS_TIMEOUT": "2",
            "EMO_AUDIO_ONNX": "",
            "EMO_AUDIO_CMD": "",
            **env,
        })
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.emo = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def speech(self, payload):
        request = Request(self.emo + "/v1/audio/speech", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=5) as response:
            return response.read(), response.headers.get("X-TTS-Engine")

    def test_without_pocket_url_kokoro_gets_the_request_unchanged(self):
        self.start()
        audio, engine = self.speech({"model": "kokoro-v1", "input": "Salut", "voice": "ff_siwis"})
        self.assertEqual((b"ID3kokoro", "kokoro"), (audio, engine))
        sent = json.loads(self.lemonade.posts[0][1])
        self.assertEqual(("/api/v1/audio/speech", "kokoro-v1", "ff_siwis"),
                         (self.lemonade.posts[0][0], sent["model"], sent["voice"]))

    def test_pocket_first_with_default_voice_for_kokoro_names(self):
        self.start(pocket_ok)
        audio, engine = self.speech({"model": "kokoro-v1", "input": "Bonjour é", "voice": "ff_siwis",
                                     "response_format": "wav"})
        self.assertEqual((b"RIFFpocket", "pocket"), (audio, engine))
        sent = json.loads(self.pocket.posts[0][1])
        self.assertEqual(("alba", "Bonjour é", "wav"), (sent["voice"], sent["input"], sent["response_format"]))
        self.assertEqual([], self.lemonade.posts)

    def test_known_pocket_voice_passes_and_unknown_uses_default(self):
        self.start(pocket_ok)
        self.speech({"input": "a", "voice": "cosette"})
        self.speech({"input": "b", "voice": "Hélène"})
        voices = [json.loads(body)["voice"] for path, body in self.pocket.posts if path == "/v1/audio/speech"]
        self.assertEqual(["cosette", "alba"], voices)

    def test_english_kokoro_voice_skips_pocket(self):
        self.start(pocket_ok)
        audio, engine = self.speech({"model": "kokoro-v1", "input": "Hello", "voice": "af_heart"})
        self.assertEqual("kokoro", engine)
        self.assertEqual([], self.pocket.posts)
        self.assertEqual("af_heart", json.loads(self.lemonade.posts[0][1])["voice"])

    def test_pocket_error_falls_back_to_kokoro_with_kokoro_model_and_voice(self):
        self.start(pocket_down)
        audio, engine = self.speech({"model": "tts-1", "input": "Salut", "voice": "alba", "speed": 1.1})
        self.assertEqual((b"ID3kokoro", "kokoro"), (audio, engine))
        sent = json.loads(self.lemonade.posts[0][1])
        self.assertEqual(("kokoro-v1", "ff_siwis", 1.1), (sent["model"], sent["voice"], sent["speed"]))

    def test_pocket_unreachable_falls_back(self):
        self.start(None, POCKET_TTS_URL="http://127.0.0.1:9")
        audio, engine = self.speech({"input": "Salut", "voice": "ff_siwis"})
        self.assertEqual((b"ID3kokoro", "kokoro"), (audio, engine))

    def test_transcription_text_is_trimmed_and_utf8(self):
        self.start()
        boundary = "b0undary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"t.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + b"RIFF0000WAVE" + f"\r\n--{boundary}--\r\n".encode()
        request = Request(self.emo + "/v1/audio/transcriptions", data=body, method="POST",
                          headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urlopen(request, timeout=5) as response:
            raw = response.read()
        payload = json.loads(raw)
        self.assertEqual("Avance la voiture.", payload["text"])
        self.assertEqual("Avance", payload["segments"][0]["text"])
        self.assertNotIn(b"\\n", raw)


class HelperTests(unittest.TestCase):
    def test_clean_transcription(self):
        self.assertEqual({"text": "voiture"}, server.clean_transcription({"text": " voiture\n"}))
        self.assertEqual("x", server.clean_transcription("x"))

    def test_nonspeech_transcript_is_dropped(self):
        cases = [
            "[MUSIC]",
            "[Musique]",
            "(musique)",
            "[Applause]",
            "[BLANK_AUDIO]",
            "♪",
            "♪♪♪",
            "♪ musique ♪",
            "[MUSIC] [Applause]",
            " (musique) \n [MUSIC] ",
            "Sous-titres réalisés par la communauté d'Amara.org",
            "Sous-titres realises par l'Amara.org",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertTrue(server.is_nonspeech_transcript(raw), raw)
                self.assertEqual("", server.clean_transcription({"text": raw})["text"])

    def test_speech_with_annotation_is_kept(self):
        kept = [
            "J'aime la musique",
            "[MUSIC] et ensuite une phrase",
            "Il y a (musique) dans le jardin",
            "Bonjour",
            "",
            "  ",
        ]
        for raw in kept:
            with self.subTest(raw=raw):
                self.assertFalse(server.is_nonspeech_transcript(raw), raw)
        # Empty / whitespace stays empty after trim, and is not classified as
        # a non-speech annotation (the app already treats empty as no turn).
        self.assertEqual("J'aime la musique", server.clean_transcription({"text": " J'aime la musique\n"})["text"])
        self.assertEqual("[MUSIC] et ensuite une phrase", server.clean_transcription({"text": "[MUSIC] et ensuite une phrase"})["text"])
        self.assertEqual("", server.clean_transcription({"text": "  \n"})["text"])

    def test_json_reply_keeps_accents_as_utf8(self):
        self.assertIn("é", json.dumps({"t": "é"}, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
