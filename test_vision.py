"""Vision on demand: a photo in the chat turn becomes "(RUX voit : …)" for nanobot."""

import json
import os
import unittest
from unittest import mock

import vision
from test_nanobot import SidecarHarness

JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="


def turn(question="Qu'est-ce que tu vois ?", stream=False):
    return {
        "model": "x",
        "stream": stream,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": JPEG}},
        ]}],
    }


def answer(text):
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


class RewriteTests(unittest.TestCase):
    def test_no_image_is_untouched(self):
        body = json.dumps({"messages": [{"role": "user", "content": "salut"}]}).encode()
        self.assertIs(body, vision.rewrite(body, "http://lem", post=lambda *a: self.fail("called")))
        self.assertEqual(b"not json", vision.rewrite(b"not json"))

    def test_image_becomes_description(self):
        seen = {}

        def post(url, payload, cfg):
            seen.update(url=url, payload=payload)
            return answer("Une tasse rouge (en céramique) sur un bureau.\nTexte : « Lundi ».")

        with mock.patch.dict(os.environ, {"VISION_MODEL": "Gemma-4-E2B-it-GGUF", "VISION_URL": ""}):
            out = json.loads(vision.rewrite(json.dumps(turn()).encode(), "http://lem/api/v1", post=post, log=lambda m: None))
        self.assertEqual("http://lem/api/v1/chat/completions", seen["url"])
        self.assertEqual("Gemma-4-E2B-it-GGUF", seen["payload"]["model"])
        parts = seen["payload"]["messages"][0]["content"]
        self.assertEqual(JPEG, parts[1]["image_url"]["url"])
        self.assertIn("Qu'est-ce que tu vois ?", parts[0]["text"])
        self.assertEqual(
            "(RUX voit : Une tasse rouge en céramique sur un bureau. Texte : « Lundi ».) Qu'est-ce que tu vois ?",
            out["messages"][0]["content"],
        )
        self.assertNotIn("base64", json.dumps(out))

    def test_vision_off_or_failing_says_so(self):
        with mock.patch.dict(os.environ, {"VISION_MODEL": ""}):
            out = json.loads(vision.rewrite(json.dumps(turn("lis ça")).encode(), log=lambda m: None))
        self.assertTrue(out["messages"][0]["content"].startswith("(RUX a voulu regarder mais n'a pas pu : la vue n'est pas branchée) lis ça"))

        def boom(*a):
            raise TimeoutError("slow")

        with mock.patch.dict(os.environ, {"VISION_MODEL": "m"}):
            out = json.loads(vision.rewrite(json.dumps(turn()).encode(), "http://lem", post=boom, log=lambda m: None))
        self.assertIn("ne répond pas", out["messages"][0]["content"])

    def test_external_url_and_key(self):
        seen = {}

        def post(url, payload, cfg):
            seen.update(url=url, key=cfg["key"])
            return answer("Un chat.")

        env = {"VISION_MODEL": "qwen", "VISION_URL": "http://gpu:8000/v1/", "VISION_API_KEY": "k"}
        with mock.patch.dict(os.environ, env):
            vision.rewrite(json.dumps(turn()).encode(), "http://lem", post=post, log=lambda m: None)
        self.assertEqual({"url": "http://gpu:8000/v1/chat/completions", "key": "k"}, seen)


class IntentTests(unittest.TestCase):
    def test_seeing_phrases(self):
        for phrase in (
            "Qu'est ce que tu vois ?",
            "tu vois quoi",
            "regarde ça",
            "décris ce que tu vois",
            "what do you see",
        ):
            self.assertTrue(vision.is_vision_question(phrase), phrase)
        self.assertFalse(vision.is_vision_question("quelle heure est-il"))
        self.assertFalse(vision.is_vision_question("allume la lumière"))

    def test_turn_detects_image_or_prefix(self):
        self.assertTrue(vision.is_vision_turn(turn(), had_image=True))
        self.assertTrue(vision.is_vision_turn({
            "messages": [{"role": "user", "content": "(RUX voit : une tasse.) Qu'est-ce que tu vois ?"}]
        }))
        self.assertTrue(vision.is_vision_turn({
            "messages": [{"role": "user", "content": "(ton entendu : neutral) tu vois quoi"}]
        }))
        self.assertFalse(vision.is_vision_turn({
            "messages": [{"role": "user", "content": "quelle heure est-il"}]
        }))


class HttpTests(unittest.TestCase):
    def test_vision_skips_nanobot(self):
        with mock.patch.dict(os.environ, {"VISION_MODEL": "Gemma-4-E2B-it-GGUF"}):
            with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
                status, _, _ = h.post_chat(turn())
                self.assertEqual(200, status)
                self.assertEqual([], h.nanobot.requests)
                # First lemonade call describes the photo (it may contain the data URL).
                # The chat turn that follows is text-only: "(RUX voit : ...)", no pixels.
                bodies = [
                    r["body"] if isinstance(r["body"], bytes) else r["body"].encode()
                    for r in h.lemonade.requests
                ]
                joined = b"".join(bodies)
                self.assertIn(b"Gemma-4-E2B-it-GGUF", joined)
                chats = [b for b in bodies if b"(RUX voit :" in b]
                self.assertTrue(chats, "expected a rewritten chat turn on Lemonade")
                self.assertNotIn(b"base64", chats[0])

    def test_vision_off_still_skips_nanobot(self):
        with mock.patch.dict(os.environ, {"VISION_MODEL": ""}):
            with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
                status, _, _ = h.post_chat(turn())
                self.assertEqual(200, status)
                self.assertEqual([], h.nanobot.requests)
                joined = b"".join(r["body"] for r in h.lemonade.requests)
                self.assertIn("la vue n'est pas branchée".encode(), joined)
                self.assertNotIn(b"base64", joined)

    def test_seeing_question_without_image_skips_nanobot(self):
        with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
            status, _, _ = h.post_chat({
                "model": "x",
                "messages": [{"role": "user", "content": "tu vois quoi"}],
            })
            self.assertEqual(200, status)
            self.assertEqual([], h.nanobot.requests)


if __name__ == "__main__":
    unittest.main()
