import tempfile
import unittest
from pathlib import Path

from geeui_cosy.download import PROMPT_URL, ensure_prompt, ready


class DownloadTest(unittest.TestCase):
    def test_missing_prompt_is_taken_from_the_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "llm.pt").write_bytes(b"weights")
            self.assertFalse(ready(model))
            wav = b"RIFF" + b"\x00" * 80
            ensure_prompt(model, fetch=lambda url=PROMPT_URL: wav)
            self.assertTrue(ready(model))
            self.assertTrue((model / "asset" / "zero_shot_prompt.wav").is_file())

    def test_prompt_already_there_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "llm.pt").write_bytes(b"weights")
            dest = model / "asset" / "zero_shot_prompt.wav"
            dest.parent.mkdir()
            dest.write_bytes(b"RIFF" + b"\x01" * 80)

            def fail(url=PROMPT_URL):
                raise AssertionError(url)

            self.assertEqual(ensure_prompt(model, fetch=fail), dest)


if __name__ == "__main__":
    unittest.main()
