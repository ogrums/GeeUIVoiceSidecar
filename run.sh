#!/bin/sh
# CosyVoice3 sidecar for GeeUIVoice.
# Downloads FunAudioLLM/Fun-CosyVoice3-0.5B-2512, then serves FastAPI on port 13306.
#
# The speech runtime is the CosyVoice git repo (AutoModel), not this script.
# Install it into THIS venv, with Python 3.10. CosyVoice does not ship wheels for 3.14.
# This script does not need torchaudio: it writes the WAV itself.
# Without the runtime, /health stays up and POST /v1/audio/speech returns 503.
#
#   python3.10 -m venv .venv && . .venv/bin/activate
#   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
#   cd CosyVoice && pip install -r requirements.txt && cd ..
#   ./run.sh
# CosyVoice has no setup.py. Do not pip install -e . This script adds it to PYTHONPATH.
set -e
cd "$(dirname "$0")"
PORT="${PORT:-13306}"
MODEL_DIR="${COSY_MODEL_DIR:-models/Fun-CosyVoice3-0.5B-2512}"
export COSY_MODEL_DIR="$MODEL_DIR"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
pip install -q -r requirements.txt
python -m geeui_cosy.download --dir "$MODEL_DIR"
COSY_SRC="$(pwd)/CosyVoice"
if [ -d "$COSY_SRC/cosyvoice" ]; then
  export PYTHONPATH="$COSY_SRC:$COSY_SRC/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
else
  echo "CosyVoice source missing at $COSY_SRC. /v1/audio/speech will return 503."
fi
exec python -m uvicorn geeui_cosy.server:app --host 0.0.0.0 --port "$PORT"
