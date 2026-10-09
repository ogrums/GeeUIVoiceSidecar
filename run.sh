#!/bin/sh
# CosyVoice3 sidecar for GeeUIVoice.
# Downloads FunAudioLLM/Fun-CosyVoice3-0.5B-2512, then serves FastAPI on port 13306.
#
# The speech runtime is the CosyVoice git repo (AutoModel), not this script.
# Without it, /health stays up and POST /v1/audio/speech returns 503.
#
#   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
#   cd CosyVoice && pip install -r requirements.txt && pip install -e .
#   cd .. && ./run.sh
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
exec python -m uvicorn geeui_cosy.server:app --host 0.0.0.0 --port "$PORT"
