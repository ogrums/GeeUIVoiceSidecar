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


class HttpTests(unittest.TestCase):
    def test_nanobot_gets_text_only(self):
        with mock.patch.dict(os.environ, {"VISION_MODEL": "Gemma-4-E2B-it-GGUF"}):
            with SidecarHarness({"NANOBOT_URL": "AUTO"}) as h:
                status, _, _ = h.post_chat(turn())
                self.assertEqual(200, status)
                sent = json.loads(h.nanobot.requests[-1]["body"])["messages"][0]["content"]
                self.assertEqual("(RUX voit : [neutral] lemonade) Qu'est-ce que tu vois ?", sent)
                lem = json.loads(h.lemonade.requests[0]["body"])
                self.assertEqual("Gemma-4-E2B-it-GGUF", lem["model"])
                self.assertNotIn(b"base64", h.nanobot.requests[-1]["body"])


if __name__ == "__main__":
    unittest.main()
