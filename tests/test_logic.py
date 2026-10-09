import unittest

from geeui_cosy.logic import MODEL_ID, MODEL_NAME, instruct_for, parse_label, route


class LogicTest(unittest.TestCase):
    def test_model_is_cosyvoice3_0_5b(self):
        self.assertEqual(MODEL_ID, "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
        self.assertEqual(MODEL_NAME, "Fun-CosyVoice3-0.5B-2512")

    def test_happy_instruct_has_the_end_token(self):
        text = instruct_for("cheerful")
        self.assertIn("<|endofprompt|>", text)
        self.assertIn("开心", text)

    def test_unknown_label_is_neutral(self):
        self.assertEqual(parse_label("nope"), "neutral")
        self.assertIn("平静", instruct_for("nope"))

    def test_short_sad_line_uses_cosyvoice3(self):
        chosen = route("j'y vais doucement", "sad", cosy_enabled=True)
        self.assertEqual(chosen["engine"], "cosyvoice")
        self.assertEqual(chosen["model"], MODEL_NAME)
        self.assertEqual(chosen["voice"], "sad")

    def test_neutral_or_long_or_disabled_stays_kokoro(self):
        self.assertEqual(route("salut", "neutral", True)["engine"], "kokoro")
        self.assertEqual(route("a" * 200, "happy", True)["engine"], "kokoro")
        self.assertEqual(route("salut", "happy", False)["engine"], "kokoro")


if __name__ == "__main__":
    unittest.main()
