"""WAV bytes from a float clip. No torchaudio."""

import io
import struct
import wave


def _floats(samples):
    if isinstance(samples, dict):
        samples = samples["tts_speech"]
    if hasattr(samples, "detach"):
        samples = samples.detach()
        if hasattr(samples, "cpu"):
            samples = samples.cpu()
        if hasattr(samples, "float"):
            samples = samples.float()
        if hasattr(samples, "numpy"):
            samples = samples.numpy().reshape(-1).tolist()
            return samples
    if hasattr(samples, "reshape"):
        return samples.reshape(-1).tolist()
    return list(samples)


def wav_bytes(samples, sample_rate: int) -> bytes:
    pcm = bytearray()
    for value in _floats(samples):
        clipped = max(-1.0, min(1.0, float(value)))
        pcm += struct.pack("<h", int(clipped * 32767))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(bytes(pcm))
    return buf.getvalue()
