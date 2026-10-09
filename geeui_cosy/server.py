"""One process for mood, route, and CosyVoice3 speech.

Kokoro stays on Lemonade. This server only synthesises when /route says cosyvoice.
Without the CosyVoice runtime, /health stays up and /v1/audio/speech returns 503.
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from geeui_cosy.download import ready
from geeui_cosy.logic import MODEL_NAME, instruct_for, parse_label, route

app = FastAPI(title="GeeUIVoice CosyVoice3")
MODEL_DIR = Path(os.environ.get("COSY_MODEL_DIR", f"models/{MODEL_NAME}"))
_runtime = None
mood = {"emotion": "neutral", "intensity": 0.0, "at": time.time()}


class SpeechIn(BaseModel):
    input: str
    voice: str = "neutral"
    model: str = MODEL_NAME
    response_format: str = "wav"


class RouteIn(BaseModel):
    text: str = ""
    emotion: str = ""
    lang: str = "fr"


class AffectIn(BaseModel):
    label: str = ""
    emotion: str = ""
    confidence: float = 0.7


def _decay(now):
    elapsed = max(0.0, now - mood["at"])
    if elapsed and mood["intensity"]:
        mood["intensity"] *= 0.5 ** (elapsed / 45.0)
        if mood["intensity"] < 0.08:
            mood["emotion"] = "neutral"
            mood["intensity"] = 0.0
    mood["at"] = now


def load_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    if not ready(MODEL_DIR):
        raise HTTPException(status_code=503, detail=f"weights missing in {MODEL_DIR}")
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="cosyvoice package is not installed. Clone FunAudioLLM/CosyVoice and pip install -e .",
        ) from exc
    _runtime = AutoModel(model_dir=str(MODEL_DIR))
    return _runtime


def synthesize(text: str, emotion: str) -> bytes:
    import io

    import torchaudio

    model = load_runtime()
    prompt = MODEL_DIR / "asset" / "zero_shot_prompt.wav"
    chunks = []
    for _, speech in enumerate(
        model.inference_instruct2(text, instruct_for(emotion), str(prompt), stream=False)
    ):
        buf = io.BytesIO()
        torchaudio.save(buf, speech, model.sample_rate, format="wav")
        chunks.append(buf.getvalue())
    if not chunks:
        raise HTTPException(status_code=500, detail="CosyVoice returned no audio")
    return chunks[-1]


@app.get("/health")
def health():
    runtime = False
    try:
        import cosyvoice.cli.cosyvoice  # noqa: F401

        runtime = True
    except ImportError:
        runtime = False
    return {
        "ok": True,
        "model": MODEL_NAME,
        "weights": ready(MODEL_DIR),
        "runtime": runtime,
    }


@app.get("/config")
def config():
    return {"model": MODEL_NAME, "model_dir": str(MODEL_DIR), "weights": ready(MODEL_DIR)}


@app.get("/mood")
def get_mood():
    _decay(time.time())
    return {"emotion": mood["emotion"], "intensity": round(mood["intensity"], 3)}


@app.post("/affect/text")
def affect_text(body: AffectIn):
    label = parse_label(body.label or body.emotion)
    now = time.time()
    _decay(now)
    weight = 0.6 * max(0.0, min(1.0, body.confidence))
    if weight > 0:
        mood["emotion"] = label
        mood["intensity"] = weight
        mood["at"] = now
    return {"mood": mood["emotion"], "intensity": round(mood["intensity"], 3)}


@app.post("/route")
def post_route(body: RouteIn):
    chosen = route(body.text, body.emotion or mood["emotion"], cosy_enabled=ready(MODEL_DIR))
    return chosen


@app.post("/v1/audio/speech")
def speech(body: SpeechIn):
    text = body.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")
    if len(text) > 180:
        raise HTTPException(status_code=400, detail="line longer than 180 characters")
    wav = synthesize(text, body.voice)
    from fastapi.responses import Response

    return Response(content=wav, media_type="audio/wav")
