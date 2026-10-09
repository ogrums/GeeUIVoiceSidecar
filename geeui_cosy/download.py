"""Download Fun-CosyVoice3-0.5B-2512. Does not clone the CosyVoice git runtime."""

import argparse
from pathlib import Path

from geeui_cosy.logic import MODEL_ID, MODEL_NAME


def ready(model_dir: Path) -> bool:
    return (model_dir / "llm.pt").is_file() and (model_dir / "asset" / "zero_shot_prompt.wav").is_file()


def download(model_dir: Path) -> Path:
    model_dir = model_dir.resolve()
    if ready(model_dir):
        print(f"weights already in {model_dir}")
        return model_dir
    from huggingface_hub import snapshot_download

    print(f"downloading {MODEL_ID} -> {model_dir}")
    snapshot_download(MODEL_ID, local_dir=str(model_dir))
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
