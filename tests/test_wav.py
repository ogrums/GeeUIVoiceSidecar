import unittest

from geeui_cosy.wav import wav_bytes


class WavTest(unittest.TestCase):
    def test_list_is_a_mono_wav(self):
        data = wav_bytes([0.0, 0.5, -1.0], 24000)
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertIn(b"WAVE", data)

    def test_cosyvoice_dict_uses_tts_speech(self):
        data = wav_bytes({"tts_speech": [0.25, -0.25]}, 22050)
        self.assertTrue(data.startswith(b"RIFF"))


if __name__ == "__main__":
    unittest.main()
