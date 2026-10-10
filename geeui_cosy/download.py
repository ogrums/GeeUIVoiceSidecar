"""Download Fun-CosyVoice3-0.5B-2512. Does not clone the CosyVoice git runtime."""

import argparse
import io
import struct
import wave
from pathlib import Path
from urllib.request import urlopen

from geeui_cosy.logic import MODEL_ID, MODEL_NAME

# GitHub LFS. raw.githubusercontent.com often returns the pointer text, which
# libsndfile then rejects with "Format not recognised". This URL follows to the blob.
PROMPT_URL = "https://github.com/QwenAudio/CosyVoice/raw/refs/heads/main/asset/zero_shot_prompt.wav"


def prompt_path(model_dir: Path) -> Path:
    return model_dir / "asset" / "zero_shot_prompt.wav"


def _chunks(data: bytes):
    if len(data) < 12 or not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
        raise ValueError("not a wav")
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        yield cid, data[pos + 8 : pos + 8 + size]
        pos += 8 + size + (size & 1)


def to_pcm16(data: bytes) -> bytes:
    """PCM 16-bit mono. Accepts PCM 16 or IEEE float 32. Rejects an LFS pointer."""
    if data.startswith(b"version https://git-lfs.github.com"):
        raise ValueError("git lfs pointer")
    fmt = channels = rate = bits = None
    payload = None
    for cid, body in _chunks(data):
        if cid == b"fmt " and len(body) >= 16:
            fmt, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", body)
        elif cid == b"data":
            payload = body
    if payload is None or fmt is None or channels != 1:
        raise ValueError(f"unsupported wav fmt={fmt} ch={channels}")
    if fmt == 1 and bits == 16:
        frames = payload
    elif fmt == 3 and bits == 32:
        count = len(payload) // 4
        samples = struct.unpack("<" + "f" * count, payload[: count * 4])
        frames = bytearray()
        for sample in samples:
            clipped = max(-1.0, min(1.0, sample))
            frames += struct.pack("<h", int(clipped * 32767))
        frames = bytes(frames)
    else:
        raise ValueError(f"unsupported wav fmt={fmt} bits={bits}")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buf.getvalue()


def ready(model_dir: Path) -> bool:
    wav = prompt_path(model_dir)
    if not (model_dir / "llm.pt").is_file() or not wav.is_file():
        return False
    try:
        to_pcm16(wav.read_bytes())
    except ValueError:
        return False
    return True


def fetch_prompt(url: str = PROMPT_URL) -> bytes:
    with urlopen(url, timeout=120) as resp:
        data = resp.read()
    try:
        return to_pcm16(data)
    except ValueError as exc:
        raise SystemExit(f"{url} is not a usable wav ({exc})") from exc


def ensure_prompt(model_dir: Path, fetch=fetch_prompt) -> Path:
    dest = prompt_path(model_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        try:
            pcm = to_pcm16(dest.read_bytes())
            if dest.read_bytes() != pcm:
                print(f"rewriting {dest.name} as pcm 16-bit")
                dest.write_bytes(pcm)
            return dest
        except ValueError:
            print(f"{dest.name} is not a wav libsndfile can open, fetching {PROMPT_URL}")
    else:
        print(f"prompt wav missing next to the weights, fetching {PROMPT_URL}")
    dest.write_bytes(fetch())
    return dest


def download(model_dir: Path) -> Path:
    model_dir = model_dir.resolve()
    if not (model_dir / "llm.pt").is_file():
        from huggingface_hub import snapshot_download

        print(f"downloading {MODEL_ID} -> {model_dir}")
        snapshot_download(MODEL_ID, local_dir=str(model_dir))
    if not (model_dir / "llm.pt").is_file():
        raise SystemExit(f"{MODEL_NAME} is missing llm.pt")
    ensure_prompt(model_dir)
    if not ready(model_dir):
        raise SystemExit(f"{MODEL_NAME} is missing llm.pt or asset/zero_shot_prompt.wav")
    return model_dir


def main():
    parser = argparse.ArgumentParser(description=f"Fetch {MODEL_ID}")
    parser.add_argument("--dir", default=f"models/{MODEL_NAME}")
    args = parser.parse_args()
    download(Path(args.dir))


if __name__ == "__main__":
    main()
