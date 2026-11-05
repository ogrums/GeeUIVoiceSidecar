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

LABELS = {"neutral", "happy", "sad", "angry", "fear", "surprise"}
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
    "EMO_AUDIO_CMD": "",
    "EMO_AUDIO_MODEL": "emotion2vec",
    "EMO_AUDIO_ONNX": "",
    "EMO_AUDIO_HEAD": "",
    "EMO_MIN_RMS": "0.003",
    "EMO_ORT_PROVIDERS": "",
    "EMO_AUDIO_LABELS": ",".join(ONNX_LABELS),
    "EMO_STT_MODEL": "Whisper-Small",
    "EMO_CHAT_MODEL": "gemma4e-flash-e2b-FLM",
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
        "audio_cmd": env("EMO_AUDIO_CMD"),
        "audio_model": env("EMO_AUDIO_MODEL"),
        "audio_onnx": env("EMO_AUDIO_ONNX"),
        "audio_head": env("EMO_AUDIO_HEAD"),
        "ort_providers": env("EMO_ORT_PROVIDERS"),
        "audio_labels": env("EMO_AUDIO_LABELS"),
        "stt_model": env("EMO_STT_MODEL"),
        "chat_model": env("EMO_CHAT_MODEL"),
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
    key = (raw or "").strip().lower()
    aliases = {"cheerful": "happy", "joy": "happy", "sadness": "sad",
               "anger": "angry", "afraid": "fear", "fearful": "fear",
               "surprised": "surprise"}
    key = aliases.get(key, key)
    return key if key in LABELS else "neutral"


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
                _onnx_local.source = "fallback"
                return "neutral", 0.0
            samples = pcm.readframes(frames)
        signal = np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0
        # Silence ou quasi-silence : emotion2vec répond n'importe quoi (ex. surprise à 98 %).
        # On mesure la trame de 20 ms la plus forte : une voix douce avec des blancs reste classée.
        usable = signal[: len(signal) // 320 * 320].reshape(-1, 320)
        loudest = float(np.sqrt(np.mean(np.square(usable), axis=1)).max()) if usable.size else 0.0
        if loudest < float(env("EMO_MIN_RMS") or 0):
            _onnx_local.source = "silence"
            return "neutral", 0.0
        session, names = _onnx_session(model)
        inputs = {names[0]: signal.reshape(1, -1)}
        if len(names) == 2:
            inputs[names[1]] = np.array(16000, dtype=np.int64)
        features = session.run(None, inputs)[0]
        logits = classify_logits(features)
        confidence, index = softmax(logits)
        with _onnx_lock:
            _onnx_stats["classifications"] += 1
        _onnx_local.source = "onnx"
        return parse_label(_onnx_labels()[index]), confidence
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
    _onnx_local.source = "fallback"
    return audio_command_vote(wav)


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


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
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

    def _stream(self, response):
        self.send_response(response.status)
        for name in ("Content-Type", "Cache-Control"):
            if response.headers.get(name):
                self.send_header(name, response.headers[name])
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
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            upstream = executor.submit(self._proxy, self.path.split("?")[0], body, content_type)
            def classify():
                result = onnx_audio_vote(audio)
                return result, getattr(_onnx_local, "source", "fallback")
            classified = executor.submit(classify)
            response = upstream.result()
            with response:
                raw = response.read()
            if not 200 <= response.status < 300:
                self._raw(response.status, raw, response.headers.get("Content-Type"))
                return
            (label, confidence), source = classified.result()
            observe(label, confidence, time.time())
            payload = json.loads(raw.decode())
            payload["emotion"] = label
            payload["emotion_confidence"] = confidence
            payload["emotion_source"] = source
            self._json(response.status, payload)
        finally:
            executor.shutdown(wait=True)

    def _raw(self, code, raw, content_type=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            self._get()
        except Exception as exc:
            log(f"GET {self.path} crash {exc}")
            traceback.print_exc()
            self._json(500, {"error": type(exc).__name__})

    def _get(self):
        path = self.path.split("?")[0]
        if path == "/v1/models":
            with self._proxy(path) as response:
                self._raw(response.status, response.read(), response.headers.get("Content-Type"))
            return
        if path == "/v1/emotion/status":
            self._json(200, emotion_status())
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
        preview = body[:200].decode(errors="replace")
        log(f"POST {path} from {self.client_address[0]} {length}b {preview!r}")
        now = time.time()
        cfg = config()
        if path == "/v1/audio/transcriptions":
            self._transcribe(body)
            return
        if path in ("/v1/chat/completions", "/v1/audio/speech"):
            self._stream(self._proxy(path, body, self.headers.get("Content-Type")))
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
