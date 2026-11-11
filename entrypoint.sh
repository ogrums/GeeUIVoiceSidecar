#!/bin/sh
set -e

MODELS_DIR="${MODELS_DIR:-/models}"
ONNX_NAME="emotion2vec_plus_base.onnx"
HEAD_NAME="emotion2vec_head.json"
ONNX_URL="${EMO_AUDIO_ONNX_URL:-https://huggingface.co/ziyu12345/emotion2vec_plus_base_onnx/resolve/main/emotion2vec_plus_base.onnx}"
HEAD_URL="${EMO_AUDIO_HEAD_URL:-https://huggingface.co/ziyu12345/emotion2vec_plus_base_onnx/resolve/main/emotion2vec_head.json}"

# Named volumes are root-owned on first create. Drop privileges after chown.
if [ "$(id -u)" = "0" ]; then
  mkdir -p "$MODELS_DIR"
  chown -R sidecar:sidecar "$MODELS_DIR" /app
  exec su -s /bin/sh sidecar -c 'exec /app/entrypoint.sh "$@"' -- "$@"
fi

download() {
  dest="$1"
  url="$2"
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    echo "présent : $dest"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  tmp="$dest.part"
  echo "téléchargement : $url -> $dest"
  python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$url" "$tmp"
  mv "$tmp" "$dest"
}

if [ "${EMO_AUDIO_DOWNLOAD:-1}" != "0" ]; then
  download "${EMO_AUDIO_ONNX:-$MODELS_DIR/$ONNX_NAME}" "$ONNX_URL"
  download "${EMO_AUDIO_HEAD:-$MODELS_DIR/$HEAD_NAME}" "$HEAD_URL"
else
  echo "EMO_AUDIO_DOWNLOAD=0 : pas de téléchargement des modèles"
fi

export EMO_AUDIO_ONNX="${EMO_AUDIO_ONNX:-$MODELS_DIR/$ONNX_NAME}"
export EMO_AUDIO_HEAD="${EMO_AUDIO_HEAD:-$MODELS_DIR/$HEAD_NAME}"

cd /app
exec python -u emo_server.py
