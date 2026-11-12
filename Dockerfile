FROM python:3.12-slim

WORKDIR /app

# WITH_SPEAKER=0 pour une image sans sherpa-onnx (« qui parle » désactivé).
ARG WITH_SPEAKER=1

COPY requirements.txt requirements-speaker.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_SPEAKER" = "1" ]; then pip install --no-cache-dir -r requirements-speaker.txt; fi \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin sidecar \
    && mkdir -p /models /data \
    && chown sidecar:sidecar /models /data

COPY *.py entrypoint.sh ./
RUN rm -f test_*.py && chmod 755 entrypoint.sh && chown sidecar:sidecar *.py entrypoint.sh

EXPOSE 13306

ENV EMO_AUDIO_ONNX=/models/emotion2vec_plus_base.onnx \
    EMO_AUDIO_HEAD=/models/emotion2vec_head.json \
    EMO_HOST=http://0.0.0.0:13306 \
    SPEAKERS_FILE=/data/speakers.json

# entrypoint starts as root to chown the models volume, then drops to sidecar
ENTRYPOINT ["/app/entrypoint.sh"]
