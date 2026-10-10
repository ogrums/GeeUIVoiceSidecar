import struct
import tempfile
import unittest
from pathlib import Path

from geeui_cosy.download import PROMPT_URL, ensure_prompt, ready, to_pcm16


def _pcm(rate=24000):
    return to_pcm16(_float_wav([0.0, 0.5, -0.5], rate))


def _float_wav(samples, rate):
    frames = b"".join(struct.pack("<f", s) for s in samples)
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(frames)) + frames
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


class DownloadTest(unittest.TestCase):
    def test_float_wav_becomes_pcm16(self):
        pcm = to_pcm16(_float_wav([0.0, 1.0], 24000))
        self.assertTrue(pcm.startswith(b"RIFF"))
        self.assertEqual(struct.unpack_from("<H", pcm, 20)[0], 1)

    def test_lfs_pointer_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "llm.pt").write_bytes(b"weights")
            dest = model / "asset"
            dest.mkdir()
            pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n"
            (dest / "zero_shot_prompt.wav").write_bytes(pointer)
            self.assertFalse(ready(model))
            ensure_prompt(model, fetch=lambda url=PROMPT_URL: _pcm())
            self.assertTrue(ready(model))
            self.assertTrue((dest / "zero_shot_prompt.wav").read_bytes().startswith(b"RIFF"))

    def test_prompt_already_there_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "llm.pt").write_bytes(b"weights")
            dest = model / "asset" / "zero_shot_prompt.wav"
            dest.parent.mkdir()
            dest.write_bytes(_pcm())

            def fail(url=PROMPT_URL):
                raise AssertionError(url)

            self.assertEqual(ensure_prompt(model, fetch=fail), dest)


if __name__ == "__main__":
    unittest.main()
