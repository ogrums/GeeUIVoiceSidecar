"""Download Fun-CosyVoice3-0.5B-2512. Does not clone the CosyVoice git runtime."""

import argparse
from pathlib import Path
from urllib.request import urlopen

from geeui_cosy.logic import MODEL_ID, MODEL_NAME

# The Hugging Face snapshot has llm.pt. The reference clip ships with the CosyVoice repo.
PROMPT_URL = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"


def prompt_path(model_dir: Path) -> Path:
    return model_dir / "asset" / "zero_shot_prompt.wav"


def ready(model_dir: Path) -> bool:
    wav = prompt_path(model_dir)
    return (model_dir / "llm.pt").is_file() and wav.is_file() and wav.stat().st_size > 44


def fetch_prompt(url: str = PROMPT_URL) -> bytes:
    with urlopen(url, timeout=60) as resp:
        data = resp.read()
    if data.startswith(b"version https://git-lfs.github.com"):
        raise SystemExit(f"{url} is a Git LFS pointer, not a wav")
    if len(data) < 44 or not data.startswith(b"RIFF"):
        raise SystemExit(f"{url} is not a wav")
    return data


def ensure_prompt(model_dir: Path, fetch=fetch_prompt) -> Path:
    dest = prompt_path(model_dir)
    if dest.is_file() and dest.stat().st_size > 44:
        return dest
    print(f"prompt wav missing next to the weights, fetching {PROMPT_URL}")
    dest.parent.mkdir(parents=True, exist_ok=True)
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
