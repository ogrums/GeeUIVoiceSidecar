FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin sidecar \
    && mkdir -p /models \
    && chown sidecar:sidecar /models

COPY emo_server.py entrypoint.sh ./
RUN chmod 755 entrypoint.sh && chown sidecar:sidecar emo_server.py entrypoint.sh

EXPOSE 13306

ENV EMO_AUDIO_ONNX=/models/emotion2vec_plus_base.onnx \
    EMO_AUDIO_HEAD=/models/emotion2vec_head.json \
    EMO_HOST=http://0.0.0.0:13306

# entrypoint starts as root to chown the models volume, then drops to sidecar
ENTRYPOINT ["/app/entrypoint.sh"]
