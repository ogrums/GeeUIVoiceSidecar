import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location("emo_server", Path(__file__).with_name("emo_server.py"))
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class LabelTest(unittest.TestCase):
    def test_emotion2vec_order_maps_to_six_labels(self):
        names = server.ONNX_LABELS
        mapped = [server.parse_label(name) for name in names]
        self.assertEqual([
            "angry", "neutral", "fear", "happy", "neutral", "neutral", "sad",
            "surprise", "neutral",
        ], mapped)
        self.assertEqual(names[2], "fearful")
        self.assertEqual(names[7], "surprised")


class AudioTest(unittest.TestCase):
    def wav(self, seconds=1, amplitude=1000):
        raw = io.BytesIO()
        with wave.open(raw, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(struct.pack("<h", amplitude) * int(seconds * 16000))
        return raw.getvalue()

    def test_short_clip_skips_onnx_runner(self):
        with patch.object(server, "_onnx_session") as model:
            self.assertEqual(("neutral", 0.0), server.onnx_audio_vote(self.wav(0.1)))
        model.assert_not_called()

    def test_silence_skips_onnx_and_returns_neutral(self):
        with patch.dict(os.environ, {"EMO_AUDIO_ONNX": "/fake/model.onnx"}, clear=False), \
             patch.object(server, "_onnx_session") as model:
            self.assertEqual(("neutral", 0.0), server.onnx_audio_vote(self.wav(1.2, amplitude=0)))
            self.assertEqual("silence", server._onnx_local.source)
            self.assertEqual(("neutral", 0.0), server.onnx_audio_vote(self.wav(1.2, amplitude=50)))
        model.assert_not_called()

    def test_onnx_receives_batched_float32_signal(self):
        session = FakeSession()
        with patch.dict(os.environ, {"EMO_AUDIO_ONNX": "/fake/model.onnx"}, clear=False), \
             patch.object(server, "_onnx_session", return_value=(session, ["audio"])), \
             patch.object(server, "classify_logits", return_value=[0, 0, 0, 10]), \
             patch.object(server, "_onnx_labels", return_value=["angry", "sad", "fear", "happy"]):
            label, confidence = server.onnx_audio_vote(self.wav())
        self.assertEqual("happy", label)
        self.assertGreater(confidence, 0.99)
        self.assertEqual((1, 16000), session.inputs["audio"].shape)
        self.assertEqual("float32", str(session.inputs["audio"].dtype))

    def test_external_head_pool_and_linear_layer(self):
        head = {"weight": [[1.0, 0.0], [0.0, 1.0]], "bias": [1.0, 2.0]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(head, tmp)
            path = tmp.name
        try:
            with patch.dict(os.environ, {"EMO_AUDIO_HEAD": path}):
                server._onnx_head = None
                result = server.classify_logits(FakeArray([
                    [[1.0, 2.0], [3.0, 4.0]]
                ]))
                self.assertEqual([3.0, 5.0], result.tolist())
                server._onnx_head = None
        finally:
            os.unlink(path)


class FakeSession:
    def run(self, _, inputs):
        self.inputs = inputs
        return [[[[1.0, 2.0], [3.0, 4.0]]]]


class FakeArray:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class ProviderTest(unittest.TestCase):
    def test_picks_first_available_accelerator(self):
        self.assertEqual(["ROCMExecutionProvider", "CPUExecutionProvider"], server.select_providers([
            "AzureExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider",
        ]))

    def test_cpu_fallback_and_override(self):
        self.assertEqual(["CPUExecutionProvider"], server.select_providers(["CPUExecutionProvider"]))
        self.assertEqual(["custom", "CPUExecutionProvider"], server.select_providers(
            ["CUDAExecutionProvider"], "custom,CPUExecutionProvider"
        ))


if __name__ == "__main__":
    unittest.main()


class StatusTest(unittest.TestCase):
    def test_head_labels_override_default_and_normalize(self):
        head = {"weight": [[1.0], [2.0]], "bias": [0.0, 0.0],
                "labels": ["生气/angry", "快乐/happy"]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(head, tmp)
            path = tmp.name
        try:
            with patch.dict(os.environ, {"EMO_AUDIO_HEAD": path}, clear=False):
                os.environ.pop("EMO_AUDIO_LABELS", None)
                server._onnx_head = None
                self.assertEqual(["angry", "happy"], server._onnx_labels())
                server._onnx_head = None
        finally:
            os.unlink(path)
