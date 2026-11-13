#!/usr/bin/env python3
"""Mood, route, and host config for GeeUIVoiceEmo. Stdlib only.

Unset or blank variables fall back to the defaults below.
Lemonade stays the Kokoro path. CosyVoice is a second host, off unless set.
"""

import concurrent.futures
import io
import json
import re
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import vision

LABELS = {"neutral", "happy", "sad", "angry", "fear", "surprise"}
import speaker_id

LABELS = {"neutral", "happy", "sad", "angry", "fear", "surprise", "disgusted"}
HALF_LIFE = 45.0
BLEND = 0.6
MAX_CHARS = 180
ONNX_LABELS = [
    "angry", "disgusted", "fearful", "happy", "neutral", "other", "sad",
    "surprised", "unknown",
]
PROVIDER_ORDER = [
    "CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider",
    "DmlExecutionProvider", "VitisAIExecutionProvider", "CPUExecutionProvider",
]

DEFAULTS = {
    "EMO_HOST": "http://nimbus:13306",
    "LEMONADE_HOST": "http://nimbus:13305/api/v1",
    "KOKORO_MODEL": "kokoro-v1",
    "KOKORO_VOICE_FR": "ff_siwis",
    "KOKORO_VOICE_EN": "af_heart",
    "COSYVOICE_HOST": "",
    "COSYVOICE_MODEL": "cosyvoice2",
    "POCKET_TTS_URL": "",
    "POCKET_TTS_VOICE": "alba",
    "POCKET_TTS_TIMEOUT": "20",
    "EMO_AUDIO_CMD": "",
    "EMO_AUDIO_MODEL": "emotion2vec",
    "EMO_AUDIO_ONNX": "",
    "EMO_AUDIO_HEAD": "",
    "EMO_MIN_RMS": "0.003",
    "EMO_ORT_PROVIDERS": "",
    "EMO_AUDIO_LABELS": ",".join(ONNX_LABELS),
    "EMO_STT_MODEL": "Whisper-Small",
    "EMO_CHAT_MODEL": "gemma4e-flash-e2b-FLM",
    "NANOBOT_URL": "",
    "NANOBOT_API_KEY": "",
    "NANOBOT_SESSION": "rux",
    "NANOBOT_FAST": "rapide",
    "NANOBOT_HEAVY": "lourd",
    "NANOBOT_HEAVY_WORDS": (
        "explique,résume,planifie,calcule,compare,écris,traduis,"
        "pourquoi,recherche,réfléchis,prépare,apprends"
    ),
    "NANOBOT_HEAVY_CHARS": "200",
    "NANOBOT_TIMEOUT": "120",
    "SIDECAR_ADMIN_KEY": "",
}


def env(name):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return DEFAULTS[name]
    return raw.strip()


def config():
    cosy = env("COSYVOICE_HOST")
    emo = env("EMO_HOST").rstrip("/")
    parsed = urlparse(emo if "://" in emo else "http://" + emo)
    return {
        "emo_host": emo,
        "bind_port": parsed.port or 13306,
        "lemonade_host": env("LEMONADE_HOST").rstrip("/"),
        "kokoro_model": env("KOKORO_MODEL"),
        "kokoro_voice_fr": env("KOKORO_VOICE_FR"),
        "kokoro_voice_en": env("KOKORO_VOICE_EN"),
        "cosyvoice_host": cosy.rstrip("/") if cosy else "",
        "cosyvoice_model": env("COSYVOICE_MODEL"),
        "cosyvoice_enabled": bool(cosy),
        "pocket_tts_url": env("POCKET_TTS_URL").rstrip("/"),
        "pocket_tts_voice": env("POCKET_TTS_VOICE"),
        "pocket_tts_timeout": float(env("POCKET_TTS_TIMEOUT") or "20"),
        "pocket_tts_enabled": bool(env("POCKET_TTS_URL")),
        "audio_cmd": env("EMO_AUDIO_CMD"),
        "audio_model": env("EMO_AUDIO_MODEL"),
        "audio_onnx": env("EMO_AUDIO_ONNX"),
        "audio_head": env("EMO_AUDIO_HEAD"),
        "ort_providers": env("EMO_ORT_PROVIDERS"),
        "audio_labels": env("EMO_AUDIO_LABELS"),
        "stt_model": env("EMO_STT_MODEL"),
        "chat_model": env("EMO_CHAT_MODEL"),
        "nanobot_url": env("NANOBOT_URL").rstrip("/"),
        "nanobot_session": env("NANOBOT_SESSION"),
        "nanobot_fast": env("NANOBOT_FAST"),
        "nanobot_heavy": env("NANOBOT_HEAVY"),
        "nanobot_heavy_words": env("NANOBOT_HEAVY_WORDS"),
        "nanobot_heavy_chars": int(env("NANOBOT_HEAVY_CHARS") or "200"),
        "nanobot_timeout": float(env("NANOBOT_TIMEOUT") or "120"),
        "nanobot_enabled": bool(env("NANOBOT_URL")),
    }


mood = {"emotion": "neutral", "intensity": 0.0, "at": time.time()}



def _with_model_field(body, content_type, model):
    """Ajoute le champ multipart `model` (exigé par Lemonade) s'il manque."""
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match or not model:
        return body
    closing = b"--" + match.group(1).encode() + b"--"
    index = body.rfind(closing)
    if index < 0:
        return body
    field = (closing[:-2] + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
             + model.encode() + b"\r\n")
    return body[:index] + field + body[index:]


def parse_label(raw):
    """Map a model label onto LABELS. The ONNX class `disgusted` is kept as-is."""
    key = (raw or "").strip().lower()
    aliases = {"cheerful": "happy", "joy": "happy", "sadness": "sad",
               "anger": "angry", "afraid": "fear", "fearful": "fear",
               "surprised": "surprise", "disgust": "disgusted"}
    key = aliases.get(key, key)
    return key if key in LABELS else "neutral"


def record_vote(label, confidence, source, raw=None, raw_confidence=None):
    """Remember the raw top ONNX class (before parse_label) on the worker thread."""
    _onnx_local.source = source
    _onnx_local.raw = raw if raw is not None else label
    _onnx_local.raw_confidence = raw_confidence if raw_confidence is not None else confidence
    return label, confidence


def vote_fields(label, confidence, source, raw=None, raw_confidence=None):
    """Fields added to a transcription payload. `emotion` is parse_label; `emotion_raw` is the ONNX class."""
    return {
        "emotion": label,
        "emotion_confidence": confidence,
        "emotion_source": source,
        "emotion_raw": raw if raw is not None else label,
        "emotion_raw_confidence": raw_confidence if raw_confidence is not None else confidence,
    }


def decay(now):
    elapsed = max(0.0, now - mood["at"])
    if elapsed and mood["intensity"]:
        mood["intensity"] *= 0.5 ** (elapsed / HALF_LIFE)
        if mood["intensity"] < 0.08:
            mood["emotion"] = "neutral"
            mood["intensity"] = 0.0
    mood["at"] = now


def observe(label, confidence, now):
    decay(now)
    confidence = max(0.0, min(1.0, float(confidence)))
    if confidence <= 0:
        return
    weight = BLEND * confidence
    if label == mood["emotion"] or mood["emotion"] == "neutral":
        mood["emotion"] = label
        mood["intensity"] = max(0.0, min(1.0, mood["intensity"] * (1 - weight) + weight))
    elif weight > mood["intensity"]:
        mood["emotion"] = label
        mood["intensity"] = weight
    else:
        mood["intensity"] *= 1 - weight
    if mood["intensity"] < 0.08:
        mood["emotion"] = "neutral"
        mood["intensity"] = 0.0
    mood["at"] = now


def select_providers(available, override=""):
    """Prefer an accelerator, but only when the installed runtime provides it."""
    chosen = [p.strip() for p in override.split(",") if p.strip()]
    return chosen or [p for p in PROVIDER_ORDER if p in available]


def onnx_audio_vote(wav):
    """Classifie un WAV PCM 16 kHz ; le signal ONNX est toujours batché."""
    model = env("EMO_AUDIO_ONNX")
    if not model:
        return _fallback_audio(wav)
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError as exc:
        _record_onnx_error(exc)
        return _fallback_audio(wav)
    try:
        with wave.open(io.BytesIO(wav), "rb") as pcm:
            rate = pcm.getframerate()
            if rate != 16000 or pcm.getnchannels() != 1 or pcm.getsampwidth() != 2:
                raise ValueError("expected 16 kHz, mono, 16-bit PCM WAV")
            frames = pcm.getnframes()
            if frames < rate:
                return record_vote("neutral", 0.0, "fallback")
            samples = pcm.readframes(frames)
        signal = np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0
        # Silence ou quasi-silence : emotion2vec répond n'importe quoi (ex. surprise à 98 %).
        # On mesure la trame de 20 ms la plus forte : une voix douce avec des blancs reste classée.
        usable = signal[: len(signal) // 320 * 320].reshape(-1, 320)
        loudest = float(np.sqrt(np.mean(np.square(usable), axis=1)).max()) if usable.size else 0.0
        if loudest < float(env("EMO_MIN_RMS") or 0):
            return record_vote("neutral", 0.0, "silence")
        session, names = _onnx_session(model)
        inputs = {names[0]: signal.reshape(1, -1)}
        if len(names) == 2:
            inputs[names[1]] = np.array(16000, dtype=np.int64)
        features = session.run(None, inputs)[0]
        logits = classify_logits(features)
        confidence, index = softmax(logits)
        with _onnx_lock:
            _onnx_stats["classifications"] += 1
        raw_label = _onnx_labels()[index]
        return record_vote(parse_label(raw_label), confidence, "onnx", raw_label, confidence)
    except Exception as exc:
        _record_onnx_error(exc)
        return _fallback_audio(wav)


_onnx = None
_onnx_names = None
_onnx_head = None
_onnx_lock = threading.Lock()
_onnx_local = threading.local()
_onnx_errors = set()
_onnx_stats = {"classifications": 0, "failures": 0, "last_error": None}


def _record_onnx_error(exc):
    """Rend chaque erreur distincte visible sans saturer le journal."""
    message = f"{type(exc).__name__}: {exc}"
    with _onnx_lock:
        _onnx_stats["failures"] += 1
        _onnx_stats["last_error"] = message
        first = message not in _onnx_errors
        _onnx_errors.add(message)
    if first:
        log(f"Échec classification ONNX : {message}\n{traceback.format_exc()}")


def _fallback_audio(wav):
    label, confidence = audio_command_vote(wav)
    return record_vote(label, confidence, "fallback")


def classify_logits(output):
    """Logits natifs ou pooling NumPy [B,T,D], puis couche linéaire."""
    import numpy as np
    values = np.asarray(output.tolist() if hasattr(output, "tolist") else output)
    if values.ndim == 2:
        return values[0]
    if values.ndim != 3 or not values.shape[1] or not values.shape[2]:
        raise ValueError(f"forme de sortie ONNX inattendue : {values.shape}")
    if not env("EMO_AUDIO_HEAD"):
        raise ValueError("feature-only ONNX requires EMO_AUDIO_HEAD")
    pooled = values[0].mean(axis=0)
    head = _audio_head()
    return np.asarray(head[0]) @ pooled + np.asarray(head[1])


def _audio_head():
    global _onnx_head
    if _onnx_head is not None:
        return _onnx_head
    import numpy as np
    raw = json.loads(Path(env("EMO_AUDIO_HEAD")).read_text())
    weight = raw.get("weight", raw.get("weights", raw.get("Weight")))
    bias = raw.get("bias", raw.get("biases", raw.get("Bias")))
    if weight is None or bias is None:
        for value in raw.values():
            if isinstance(value, dict):
                weight = value.get("weight", value.get("weights", weight))
                bias = value.get("bias", value.get("biases", bias))
    if weight is None or bias is None:
        raise ValueError("EMO_AUDIO_HEAD must contain weight and bias arrays")
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if weight.shape[0] != len(bias):
        weight = weight.T
    if weight.shape[0] != len(bias):
        raise ValueError("dimensions incompatibles dans la tête ONNX")
    labels = raw.get("labels")
    _onnx_head = weight, bias, labels
    return _onnx_head


def _onnx_session(model):
    global _onnx, _onnx_names
    with _onnx_lock:
        if _onnx is None:
            import onnxruntime as ort
            providers = select_providers(ort.get_available_providers(), env("EMO_ORT_PROVIDERS"))
            session = ort.InferenceSession(model, providers=providers)
            names = [value.name for value in session.get_inputs()]
            if len(names) not in (1, 2):
                raise ValueError("expected an audio input and optional sampling-rate input")
            _onnx, _onnx_names = session, names
        return _onnx, _onnx_names


def _onnx_labels():
    override = os.environ.get("EMO_AUDIO_LABELS")
    if override is not None and override.strip():
        labels = override.split(",")
    elif env("EMO_AUDIO_HEAD"):
        labels = _audio_head()[2]
        if not labels:
            labels = ONNX_LABELS
    else:
        labels = ONNX_LABELS
    return [label.strip().rsplit("/", 1)[-1].lower() for label in labels if label.strip()]


def softmax(logits):
    import numpy as np
    weights = np.exp(np.asarray(logits) - np.max(logits))
    index = int(np.argmax(weights))
    return float(weights[index] / weights.sum()), index


def emotion_status():
    """État observable du classifieur, sans déclencher de chargement."""
    import importlib.util
    model = env("EMO_AUDIO_ONNX")
    head_path = env("EMO_AUDIO_HEAD")
    available = []
    version = None
    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort
        version = ort.__version__
        available = ort.get_available_providers()
    with _onnx_lock:
        loaded = _onnx is not None
        stats = dict(_onnx_stats)
    head_loaded = _onnx_head is not None
    labels = None
    if head_loaded:
        labels = _onnx_labels()
    return {
        "onnx_path": model,
        "onnx_configured": bool(model),
        "onnx_file_exists": bool(model) and Path(model).is_file(),
        "onnxruntime_importable": version is not None,
        "onnxruntime_version": version,
        "available_providers": available,
        "selected_providers": select_providers(available, env("EMO_ORT_PROVIDERS")),
        "head_path": head_path,
        "head_loaded": head_loaded,
        "labels": labels,
        "session_loaded": loaded,
        "last_error": stats["last_error"],
        "classifications": stats["classifications"],
        "failures": stats["failures"],
    }


def eager_load_onnx():
    """Charge le backbone au démarrage et journalise explicitement le résultat."""
    if not env("EMO_AUDIO_ONNX"):
        return
    try:
        session, names = _onnx_session(env("EMO_AUDIO_ONNX"))
        if env("EMO_AUDIO_HEAD"):
            _audio_head()
        log(f"ONNX prêt : {env('EMO_AUDIO_ONNX')}, entrées={names}, fournisseurs={session.get_providers()}")
    except Exception as exc:
        _record_onnx_error(exc)


def audio_command_vote(wav):
    cmd = env("EMO_AUDIO_CMD")
    if not cmd:
        return "neutral", 0.0
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav)
        path = tmp.name
    try:
        last = None
        for attempt in range(3):
            try:
                out = subprocess.check_output(cmd.split() + [path], timeout=3, text=True)
                parts = out.split()
                return parse_label(parts[0]), float(parts[1]) if len(parts) > 1 else 0.5
            except (subprocess.SubprocessError, ValueError, IndexError) as exc:
                last = exc
                log(f"audio hook try {attempt + 1} failed: {exc}")
                time.sleep(0.2 * (attempt + 1))
        log(f"audio hook gave up: {last}")
        return "neutral", 0.0
    finally:
        os.unlink(path)


def audio_vote(wav):
    # Kept as a small seam for callers and tests of the original audio hook.
    return onnx_audio_vote(wav)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --- nanobot (optional chat brain) ------------------------------------------

_session_locks = {}
_session_locks_guard = threading.Lock()


def _session_lock(session_id):
    """One lock per session so /model switches never interleave."""
    with _session_locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


def last_user_text(messages):
    """Return the content of the last OpenAI-style user message."""
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text") or "")
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        if content is None:
            return ""
        return str(content)
    return ""


def heard_emotion_prefix(now=None):
    """Prefix the heard mood when intensity is still meaningful after decay."""
    decay(now if now is not None else time.time())
    if mood["intensity"] >= 0.08:
        return f"(ton entendu : {mood['emotion']}) "
    return ""


_speakers = None
_speakers_guard = threading.Lock()


def speakers():
    """Service « qui parle » (désactivé sans SPEAKER_MODEL)."""
    global _speakers
    with _speakers_guard:
        if _speakers is None:
            _speakers = speaker_id.Speakers()
        return _speakers


def set_speakers(service):
    """Pour les tests : remplacer le service."""
    global _speakers
    with _speakers_guard:
        _speakers = service


def speaker_route(base_session, heard):
    """(session_id, préfixe, est_invité) selon la personne courante."""
    who = speakers().who()
    if who is None:
        return base_session, heard, False
    person_id, name = who
    tone = heard.strip()[1:-1] if heard else ""
    if person_id == speaker_id.GUEST:
        label = "un invité parle" + (", " + tone if tone else "")
        return f"{base_session}-{speaker_id.GUEST}", f"({label}) ", True
    label = f"{name} parle" + (", " + tone if tone else "")
    return f"{base_session}-{person_id}", f"({label}) ", False


def openai_reply(text, stream, model="rux-sidecar"):
    """Réponse de chat fabriquée par le sidecar, au format OpenAI (JSON ou SSE)."""
    created = int(time.time())
    if not stream:
        return "application/json; charset=utf-8", json.dumps({
            "id": f"rux-{created}", "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
        }, ensure_ascii=False).encode()
    chunk = {"id": f"rux-{created}", "object": "chat.completion.chunk", "created": created,
             "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                          "finish_reason": None}]}
    end = dict(chunk, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
    raw = ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
           + "data: " + json.dumps(end) + "\n\ndata: [DONE]\n\n").encode()
    return "text/event-stream; charset=utf-8", raw


def is_heavy(text, words=None, max_light=None):
    """Heuristic: French task keywords or a long phrase => Qwen3 preset."""
    cfg = config()
    if max_light is None:
        max_light = cfg["nanobot_heavy_chars"]
    if words is None:
        words = [w.strip().lower() for w in cfg["nanobot_heavy_words"].split(",") if w.strip()]
    body = (text or "").strip()
    if len(body) > max_light:
        return True
    lower = body.lower()
    return any(word in lower for word in words)


def _nanobot_headers():
    headers = {"Content-Type": "application/json"}
    key = env("NANOBOT_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


# nanobot answers HTTP 200 even when its LLM call fails, with the error as the reply
# text (providers/base.py, openai_compat_provider.py). Treat those as failures.
NANOBOT_ERROR_PREFIXES = ("Error calling LLM", "Error: ")


def nanobot_error_text(text):
    return bool(text) and text.lstrip().startswith(NANOBOT_ERROR_PREFIXES)


def _sse_text(buffer):
    """Concatenate delta contents seen so far in an SSE buffer; True if [DONE] seen."""
    text, done = "", False
    for line in buffer.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            done = True
            continue
        try:
            for choice in json.loads(data).get("choices", []):
                text += (choice.get("delta") or {}).get("content") or ""
        except (ValueError, AttributeError):
            pass
    return text, done


def _nanobot_post(payload, timeout=None):
    """POST JSON to nanobot /chat/completions. Returns (ok, response_or_exc)."""
    cfg = config()
    url = cfg["nanobot_url"] + "/chat/completions"
    if timeout is None:
        timeout = cfg["nanobot_timeout"]
    raw = json.dumps(payload).encode()
    request = Request(url, data=raw, headers=_nanobot_headers(), method="POST")
    try:
        response = urlopen(request, timeout=timeout)
        return True, response
    except HTTPError as exc:
        return False, exc
    except Exception as exc:
        return False, exc


def _nanobot_command(session_id, command, timeout=30):
    """Send a non-stream control phrase (/model …) and discard the body."""
    ok, response = _nanobot_post(
        {
            "session_id": session_id,
            "stream": False,
            "messages": [{"role": "user", "content": command}],
        },
        timeout=timeout,
    )
    if not ok:
        log(f"nanobot cmd fail {command!r}: {response}")
        if hasattr(response, "close"):
            try:
                response.close()
            except Exception:
                pass
        return False
    try:
        response.read()
        return 200 <= response.status < 300
    finally:
        response.close()




_BRACKET_ANN = re.compile(r"\[[^\[\]]*\]|\([^()]*\)")
_MUSIC_MARK = re.compile(r"[♪♫♬♩🎵🎶]+")
_NONSPEECH_PUNCT = ".,;:!?-–—•·\'\""
# Whisper credits, optional: "Sous-titres réalisés par la communauté d'Amara.org"
_SUBTITLE_CREDIT = re.compile(
    r"(?i)\bsous-titres?\s+r[ée]alis[ée]s?\s+par\b.*$",
)
# Leftover tokens after marks are stripped, e.g. "♪ musique ♪".
_NONSPEECH_WORD = re.compile(
    r"(?i)^(musique|music|m[uú]sica|applause|applaudissements?|"
    r"silence|blank[\s_-]?audio|inaudible|laughter|rires?|"
    r"bruit|noise|coughing?|tousse?)$"
)


def is_nonspeech_transcript(text):
    """True when the whole transcript is a Whisper non-speech annotation."""
    s = (text or "").strip()
    if not s:
        return False
    s = _SUBTITLE_CREDIT.sub("", s)
    # Drop every bracketed / parenthesised annotation, then music marks.
    while True:
        cleaned = _BRACKET_ANN.sub("", s)
        if cleaned == s:
            break
        s = cleaned
    s = _MUSIC_MARK.sub(" ", s)
    leftover = [t.strip(_NONSPEECH_PUNCT) for t in s.split()]
    leftover = [t for t in leftover if t]
    if not leftover:
        return True
    return all(_NONSPEECH_WORD.match(t) for t in leftover)

def clean_transcription(payload):
    """Whisper (Lemonade) returns ' texte\n': trim `text` and segment texts in place.

    A whole transcript that is only a non-speech annotation ([MUSIC], (musique),
    ♪, [BLANK_AUDIO], a subtitle credit…) becomes empty so the app skips the turn.
    """
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("text"), str):
        payload["text"] = payload["text"].strip()
        if is_nonspeech_transcript(payload["text"]):
            payload["text"] = ""
    for segment in payload.get("segments") or []:
        if isinstance(segment, dict) and isinstance(segment.get("text"), str):
            segment["text"] = segment["text"].strip()
            if is_nonspeech_transcript(segment["text"]):
                segment["text"] = ""
    return payload


KOKORO_VOICE = re.compile(r"^[a-z]{2}_[a-z0-9]+$")
KOKORO_ENGLISH = re.compile(r"^[ab][fm]_")
_pocket_voices = {"at": 0.0, "url": "", "names": set()}
_pocket_voices_lock = threading.Lock()


def pocket_voice_names(cfg):
    """Voice ids served by Pocket TTS (GET /v1/voices), cached 5 minutes. Empty on error."""
    url = cfg["pocket_tts_url"]
    with _pocket_voices_lock:
        cached = _pocket_voices
        if cached["url"] == url and time.time() - cached["at"] < 300:
            return cached["names"]
        names = set()
        try:
            with urlopen(url + "/v1/voices", timeout=3) as response:
                data = json.loads(response.read().decode())
            for item in data.get("data") or data.get("voices") or []:
                name = item.get("id") if isinstance(item, dict) else item
                if isinstance(name, str) and name:
                    names.add(name)
        except Exception as exc:
            log(f"pocket voices unavailable: {exc}")
        cached.update({"at": time.time(), "url": url, "names": names})
        return names


def pocket_voice(asked, cfg, known=None):
    """Pocket voice for a request voice, or None when the line must stay on Kokoro (English)."""
    asked = (asked or "").strip()
    if KOKORO_ENGLISH.match(asked):
        return None
    if not asked or KOKORO_VOICE.match(asked):
        return cfg["pocket_tts_voice"]
    names = pocket_voice_names(cfg) if known is None else known
    return asked if asked in names else cfg["pocket_tts_voice"]


def kokoro_body(data, cfg):
    """Request rewritten for Lemonade Kokoro: Kokoro model and a Kokoro voice."""
    out = dict(data)
    model = str(out.get("model") or "")
    if "kokoro" not in model.lower():
        out["model"] = cfg["kokoro_model"]
    voice = str(out.get("voice") or "")
    if not KOKORO_VOICE.match(voice):
        out["voice"] = cfg["kokoro_voice_fr"]
    out.pop("stream", None)
    return json.dumps(out, ensure_ascii=False).encode()


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        # UTF-8, not \u escapes: older app builds decode \u00e9 as "u00e9".
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        shown = raw.decode()
        if len(shown) > 300:
            shown = shown[:300] + "…"
        log(f"{self.command} {self.path} {code} {shown}")

    def _proxy(self, path, body=None, content_type=None):
        """Forward one OpenAI-shaped request without changing its body."""
        request = Request(
            config()["lemonade_host"] + path.removeprefix("/v1"),
            data=body,
            headers={"Content-Type": content_type} if content_type else {},
            method=self.command,
        )
        try:
            return urlopen(request, timeout=600)
        except HTTPError as exc:
            return exc

    def _stream(self, response, extra_headers=None):
        self.send_response(response.status)
        for name in ("Content-Type", "Cache-Control"):
            if response.headers.get(name):
                self.send_header(name, response.headers[name])
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = response.read1(16384)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            log(f"{self.command} {self.path} client disconnected")
        finally:
            response.close()

    def _transcribe(self, body):
        parts = {}
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            self._json(400, {"error": "expected multipart WAV"})
            return
        from email.parser import BytesParser
        from email.policy import default
        envelope = b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
        fields = set()
        for part in BytesParser(policy=default).parsebytes(envelope).iter_parts():
            fields.add(part.get_param("name", header="content-disposition"))
            if part.get_filename():
                parts[part.get_param("name", header="content-disposition")] = (
                    part.get_filename(), part.get_payload(decode=True),
                    part.get_content_type(),
                )
        if not parts.get("file"):
            self._json(400, {"error": "missing audio file"})
            return
        audio = parts["file"][1]
        if "model" not in fields:
            body = _with_model_field(body, content_type, config()["stt_model"])
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        try:
            upstream = executor.submit(self._proxy, self.path.split("?")[0], body, content_type)
            def classify():
                result = onnx_audio_vote(audio)
                return (
                    result,
                    getattr(_onnx_local, "source", "fallback"),
                    getattr(_onnx_local, "raw", result[0]),
                    getattr(_onnx_local, "raw_confidence", result[1]),
                )
            classified = executor.submit(classify)

            def who():
                try:
                    return speakers().observe(audio)
                except Exception as exc:
                    log(f"speaker id failed: {type(exc).__name__}: {exc}")
                    return None
            identified = executor.submit(who)
            response = upstream.result()
            with response:
                raw = response.read()
            if not 200 <= response.status < 300:
                self._raw(response.status, raw, response.headers.get("Content-Type"))
                return
            (label, confidence), source, raw_label, raw_confidence = classified.result()
            observe(label, confidence, time.time())
            payload = clean_transcription(json.loads(raw.decode()))
            payload.update(vote_fields(label, confidence, source, raw_label, raw_confidence))
            speaker = identified.result()
            if speaker:
                payload.update(speaker)
            self._json(response.status, payload)
        finally:
            executor.shutdown(wait=True)

    def _speech(self, body, content_type):
        """Pocket TTS first when POCKET_TTS_URL is set; Kokoro (Lemonade) on error or timeout."""
        cfg = config()
        try:
            data = json.loads(body.decode() or "{}")
        except (ValueError, UnicodeDecodeError):
            data = None
        if not isinstance(data, dict):
            self._stream(self._proxy("/v1/audio/speech", body, content_type), {"X-TTS-Engine": "kokoro"})
            return
        text = str(data.get("input") or "")
        voice = pocket_voice(data.get("voice"), cfg) if cfg["pocket_tts_enabled"] and text.strip() else None
        if voice:
            pocket = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": data.get("response_format") or "mp3",
            }
            if "stream" in data:
                pocket["stream"] = data["stream"]
            started = time.time()
            try:
                response = urlopen(
                    Request(
                        cfg["pocket_tts_url"] + "/v1/audio/speech",
                        data=json.dumps(pocket, ensure_ascii=False).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    ),
                    timeout=cfg["pocket_tts_timeout"],
                )
                log(
                    f"speech engine=pocket voice={voice} chars={len(text)} "
                    f"headers_after={time.time() - started:.2f}s"
                )
                self._stream(response, {"X-TTS-Engine": "pocket"})
                return
            except Exception as exc:
                log(f"speech engine=kokoro fallback reason=pocket:{exc}")
        self._stream(
            self._proxy("/v1/audio/speech", kokoro_body(data, cfg), "application/json"),
            {"X-TTS-Engine": "kokoro"},
        )

    def _raw(self, code, raw, content_type=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


    def _stream_nanobot(self, response, head=b""):
        """Relay nanobot SSE; append data: [DONE] if the upstream cut short."""
        self.send_response(response.status)
        for name in ("Content-Type", "Cache-Control"):
            if response.headers.get(name):
                self.send_header(name, response.headers[name])
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        seen_done = False
        try:
            if head:
                seen_done = b"data: [DONE]" in head
                self.wfile.write(f"{len(head):X}\r\n".encode() + head + b"\r\n")
                self.wfile.flush()
            while True:
                chunk = response.read1(16384)
                if not chunk:
                    break
                if b"data: [DONE]" in chunk:
                    seen_done = True
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            if not seen_done:
                done = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
                log("nanobot stream ended without [DONE]; appended")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            log(f"{self.command} {self.path} client disconnected")
            return True
        except Exception as exc:
            log(f"nanobot stream error after headers: {exc}")
            if not seen_done:
                try:
                    done = b"data: [DONE]\n\n"
                    self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
            return True
        finally:
            response.close()

    def _local_chat(self, body):
        """Phrases traitées par le sidecar lui-même (voix), avant nanobot ou Lemonade."""
        try:
            data = json.loads(body.decode() or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        text = last_user_text(data.get("messages"))
        try:
            reply = speakers().handle_chat(text)
        except Exception as exc:
            log(f"speaker intent failed: {type(exc).__name__}: {exc}")
            reply = None
        if not reply:
            return False
        log(f"chat route=local reply={reply[:80]!r}")
        content_type, raw = openai_reply(reply, bool(data.get("stream")))
        self._raw(200, raw, content_type)
        return True

    def _lemonade_chat(self, body, content_type, reason):
        log(f"chat route=lemonade fallback reason={reason}")
        self._stream(self._proxy("/v1/chat/completions", body, content_type))

    def _chat_via_nanobot(self, body, content_type):
        """Forward one user turn to nanobot; fall back to Lemonade on failure."""
        cfg = config()
        try:
            data = json.loads(body.decode() or "{}")
        except json.JSONDecodeError as exc:
            self._lemonade_chat(body, content_type, f"bad_json:{exc}")
            return
        user_text = last_user_text(data.get("messages"))
        if not user_text.strip():
            self._lemonade_chat(body, content_type, "no_user_message")
            return
        session_id, prefix, guest = speaker_route(cfg["nanobot_session"], heard_emotion_prefix())
        if guest and speaker_id.settings()["guest_route"] == "lemonade":
            self._lemonade_chat(body, content_type, "guest")
            return
        content = prefix + user_text
        stream = bool(data.get("stream"))
        heavy = is_heavy(user_text)
        payload = {
            "session_id": session_id,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        }
        lock = _session_lock(session_id)
        switched = False
        client_started = False
        with lock:
            try:
                if heavy:
                    if not _nanobot_command(session_id, f"/model {cfg['nanobot_heavy']}"):
                        raise RuntimeError("model_switch_heavy_failed")
                    switched = True
                log(
                    f"chat route=nanobot session={session_id} heavy={heavy} "
                    f"stream={stream} chars={len(content)}"
                )
                ok, response = _nanobot_post(payload)
                if not ok:
                    detail = response
                    if hasattr(response, "read"):
                        try:
                            detail = f"HTTP {response.status}: {response.read()[:200]!r}"
                        finally:
                            response.close()
                    raise RuntimeError(f"nanobot_post:{detail}")
                if not 200 <= response.status < 300:
                    try:
                        detail = response.read()[:200]
                    finally:
                        response.close()
                    raise RuntimeError(f"nanobot_status_{response.status}:{detail!r}")
                if stream:
                    # Hold the first bytes back until the reply text shows it is not an
                    # LLM error, so we can still fall back to Lemonade.
                    head = b""
                    while True:
                        chunk = response.read1(16384)
                        head += chunk
                        text, done = _sse_text(head)
                        if not chunk or done or len(text.strip()) >= 20:
                            break
                    if nanobot_error_text(text):
                        response.close()
                        raise RuntimeError(f"nanobot_llm_error:{text.strip()[:120]}")
                    client_started = True
                    self._stream_nanobot(response, head)
                else:
                    try:
                        raw = response.read()
                        try:
                            reply = json.loads(raw)["choices"][0]["message"].get("content") or ""
                        except (ValueError, KeyError, IndexError, TypeError):
                            reply = ""
                        if nanobot_error_text(reply):
                            raise RuntimeError(f"nanobot_llm_error:{reply.strip()[:120]}")
                        client_started = True
                        self._raw(
                            response.status,
                            raw,
                            response.headers.get("Content-Type") or "application/json",
                        )
                        log(f"POST {self.path} {response.status} nanobot non-stream {len(raw)}b")
                    finally:
                        response.close()
            except Exception as exc:
                if client_started:
                    log(f"nanobot chat failed mid-stream: {exc}")
                else:
                    self._lemonade_chat(body, content_type, str(exc))
            finally:
                if switched:
                    if not _nanobot_command(session_id, f"/model {cfg['nanobot_fast']}"):
                        log("nanobot failed to restore fast model")

    def _admin_skills(self, path):
        """Page /admin/skills (skills learned by voice). Returns True when handled."""
        if path != "/admin/skills" and not path.startswith("/admin/skills/"):
            return False
        admin = skills_admin.Skills()
        if not admin.enabled:
            self._json(404, {"error": "SIDECAR_ADMIN_KEY et NANOBOT_WORKSPACE requis"})
            return True
        if not admin.authorized(self.headers.get("Authorization")):
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="RUX", charset="UTF-8"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True
        message = ""
        if self.command == "POST":
            # Basic auth is sent by the browser on any site's form: same-origin only.
            origin = self.headers.get("Origin") or self.headers.get("Referer") or ""
            host = self.headers.get("Host") or ""
            if origin and urlparse(origin).netloc != host:
                self._json(403, {"error": "origine refusée"})
                return True
            parts = path.strip("/").split("/")
            if len(parts) != 4 or parts[3] not in ("approve", "reject"):
                self._json(404, {"error": "not found"})
                return True
            name, action = parts[2], parts[3]
            ok, message = admin.approve(name) if action == "approve" else admin.reject(name)
            log(f"admin skills {action} {name} ok={ok} {message}")
        elif path != "/admin/skills":
            self._json(404, {"error": "not found"})
            return True
        self._raw(200, admin.page(message), "text/html; charset=utf-8")
        return True

    def do_GET(self):
        try:
            self._get()
        except Exception as exc:
            log(f"GET {self.path} crash {exc}")
            traceback.print_exc()
            self._json(500, {"error": type(exc).__name__})

    def _get(self):
        path = self.path.split("?")[0]
        if self._admin_skills(path):
            return
        if path == "/v1/models":
            with self._proxy(path) as response:
                self._raw(response.status, response.read(), response.headers.get("Content-Type"))
            return
        if path == "/v1/emotion/status":
            self._json(200, emotion_status())
            return
        if path == "/v1/speakers":
            if self._admin_ok():
                self._json(200, speakers().listing())
            return
        if path == "/config":
            self._json(200, config())
            return
        if path != "/mood":
            self._json(404, {"error": "not found"})
            return
        now = time.time()
        decay(now)
        self._json(200, {"emotion": mood["emotion"], "intensity": round(mood["intensity"], 3)})

    def _admin_ok(self):
        key = env("SIDECAR_ADMIN_KEY")
        given = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        given = given or (self.headers.get("X-Admin-Key") or "").strip()
        if not key:
            self._json(403, {"error": "SIDECAR_ADMIN_KEY non défini"})
            return False
        if given != key:
            self._json(401, {"error": "clé d'administration invalide"})
            return False
        return True

    def do_DELETE(self):
        try:
            path = self.path.split("?")[0]
            if not path.startswith("/v1/speakers/"):
                self._json(404, {"error": "not found"})
                return
            if not self._admin_ok():
                return
            person = path.rsplit("/", 1)[1]
            removed = speakers().store.delete(person)
            self._json(200 if removed else 404, {"deleted": removed, "id": person})
        except Exception as exc:
            log(f"DELETE {self.path} crash {exc}")
            self._json(500, {"error": type(exc).__name__})

    def do_POST(self):
        try:
            self._post()
        except Exception as exc:
            log(f"POST {self.path} crash {exc}")
            traceback.print_exc()
            self._json(500, {"error": type(exc).__name__})

    def _post(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        path = self.path.split("?")[0]
        preview = "(image)" if b'"image_url"' in body[:65536] else body[:200].decode(errors="replace")
        log(f"POST {path} from {self.client_address[0]} {length}b {preview!r}")
        now = time.time()
        cfg = config()
        if self._admin_skills(path):
            return
        if path == "/v1/audio/transcriptions":
            self._transcribe(body)
            return
        if path == "/v1/chat/completions":
            content_type = self.headers.get("Content-Type") or "application/json"
            # A photo becomes "(RUX voit : ...)" text. Seeing questions never go to
            # nanobot (its file tools used to list a folder when asked "tu vois ?").
            had_image = b'"image_url"' in body[:65536]
            body = vision.rewrite(body, cfg["lemonade_host"], log=log)
            try:
                data = json.loads(body.decode() or "{}")
            except (ValueError, UnicodeDecodeError):
                data = {}
            if vision.is_vision_turn(data, had_image):
                log(f"chat route=lemonade vision skip_nanobot had_image={had_image}")
                self._stream(self._proxy(path, body, content_type))
            elif cfg.get("nanobot_enabled"):
                self._chat_via_nanobot(body, content_type)
            else:
                self._stream(self._proxy(path, body, content_type))
            return

        if path == "/v1/audio/speech":
            self._speech(body, self.headers.get("Content-Type"))
            return
        if path == "/affect/audio":
            label, conf = audio_vote(body)
            observe(label, conf, now)
            self._json(200, {"audio": label, "confidence": conf, "mood": mood["emotion"],
                             "audio_model": cfg["audio_model"]})
            return
        try:
            data = json.loads(body.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json", "body": body.decode(errors="replace")[:200]})
            return
        if path == "/affect/text":
            label = parse_label(data.get("label") or data.get("emotion"))
            observe(label, data.get("confidence", 0.7), now)
            self._json(200, {"mood": mood["emotion"], "intensity": round(mood["intensity"], 3)})
            return
        if path == "/route":
            text = data.get("text") or ""
            emotion = parse_label(data.get("emotion") or mood["emotion"])
            lang = (data.get("lang") or "fr").lower()
            voice = cfg["kokoro_voice_en"] if lang.startswith("en") else cfg["kokoro_voice_fr"]
            engine = "kokoro"
            host = cfg["lemonade_host"]
            model = cfg["kokoro_model"]
            if cfg["cosyvoice_enabled"] and emotion != "neutral" and 0 < len(text.strip()) <= MAX_CHARS:
                engine = "cosyvoice"
                host = cfg["cosyvoice_host"]
                model = cfg["cosyvoice_model"]
                voice = emotion
            self._json(200, {
                "engine": engine,
                "host": host,
                "model": model,
                "voice": voice,
                "chat_model": cfg["chat_model"],
                "lemonade_host": cfg["lemonade_host"],
            })
            return
        self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        log(f"{self.address_string()} {fmt % args}")


if __name__ == "__main__":
    cfg = config()
    log(f"listen 0.0.0.0:{cfg['bind_port']} advertised {cfg['emo_host']}")
    log(json.dumps(cfg))
    eager_load_onnx()
    server = ThreadingHTTPServer(("0.0.0.0", cfg["bind_port"]), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopped")
    finally:
        server.server_close()
