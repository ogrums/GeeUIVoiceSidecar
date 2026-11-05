#!/usr/bin/env python3
"""Campagne de test du sidecar, hors ligne (mock) ou contre un serveur réel.

Dépendances : Python 3.9+ uniquement (bibliothèque standard). Aucun ONNX,
Lemonade, requests ou numpy n'est nécessaire. Les WAV sont générés avec wave/array.
"""
from __future__ import annotations

import argparse
import array
import concurrent.futures
import csv
import html
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import sys
import tempfile
import threading
import time
import traceback
import uuid
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
EMOTIONS = ("happy", "sad", "angry", "fear", "surprise", "neutral")
RAW_NEUTRAL = ("disgusted", "other", "unknown")
MOCK_CLASSIFIER = os.environ.get("GEEUI_MOCK_CLASSIFIER", "1") != "0"
RAVDESS_LABELS = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised")


def wav_bytes(seconds=1.2, kind="tone"):
    """WAV PCM 16 kHz mono, sans dépendance externe."""
    count = int(16000 * seconds)
    samples = array.array("h")
    rng = random.Random(42)
    for i in range(count):
        if kind == "silence":
            value = 0
        elif kind == "noise":
            value = rng.randint(-10000, 10000)
        else:
            value = int(7000 * math.sin(2 * math.pi * 220 * i / 16000))
        samples.append(value)
    if sys.byteorder != "little":
        samples.byteswap()
    out = io.BytesIO()
    with wave.open(out, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())
    return out.getvalue()


STT_MODEL = os.environ.get("EMO_STT_MODEL", "Whisper-Small")


def multipart(filename=None, content=b"", content_type="audio/wav", model=None):
    boundary = "----GeeUI" + uuid.uuid4().hex
    body = b""
    if model:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n"
                 f"{model}\r\n").encode()
    if filename is not None:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n").encode()
        body += content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def request(url, method="GET", body=None, content_type=None, timeout=15):
    started = time.perf_counter()
    req = Request(url, data=body, method=method,
                  headers={"Content-Type": content_type} if content_type else {})
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
            status, headers = response.status, dict(response.headers)
    except HTTPError as exc:
        with exc:
            raw = exc.read()
            status, headers = exc.code, dict(exc.headers)
    elapsed = (time.perf_counter() - started) * 1000
    return status, headers, raw, elapsed


def transcription(url, filename, audio, content_type="audio/wav", model=True):
    body, mime = multipart(filename, audio, content_type, STT_MODEL if model is True else model)
    return request(url + "/v1/audio/transcriptions", "POST", body, mime, timeout=20)


def json_body(raw):
    try:
        return json.loads(raw.decode())
    except (ValueError, UnicodeDecodeError):
        return {}


class StubState:
    delay = 0.0
    last_model = None


class FakeLemonade(BaseHTTPRequestHandler):
    """Amont local déterministe reproduisant les formes OpenAI attendues."""
    def log_message(self, *_):
        pass

    def _reply(self, status, body, mime="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._reply(200, json.dumps({"object": "list", "data": [{"id": "Whisper-Small"}]}).encode())
        else:
            self._reply(404, b'{"error":"not found"}')

    def do_POST(self):
        if StubState.delay:
            time.sleep(StubState.delay)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/v1/audio/transcriptions":
            try:
                from email.parser import BytesParser
                from email.policy import default
                envelope = (b"Content-Type: " + self.headers["Content-Type"].encode()
                            + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
                parts = list(BytesParser(policy=default).parsebytes(envelope).iter_parts())
                fields = {p.get_param("name", header="content-disposition"): p for p in parts}
                if "model" not in fields:
                    # Même réponse que Lemonade quand le champ model manque.
                    self._reply(400, json.dumps({"error": {"code": "missing_argument",
                        "message": "Missing required argument 'model'",
                        "type": "invalid_request_error"}}).encode())
                    return
                StubState.last_model = fields["model"].get_content().strip()
                audio = next(p.get_payload(decode=True) for p in parts if p.get_filename())
                with wave.open(io.BytesIO(audio), "rb") as check:
                    check.getparams()
            except Exception:
                self._reply(400, b'{"error":{"message":"invalid audio","type":"invalid_request_error"}}')
                return
            # Des champs additionnels permettent de vérifier le passthrough strict.
            self._reply(200, json.dumps({"text": "bonjour le robot", "language": "fr",
                                         "duration": 1.2, "segments": [{"id": 0, "text": "bonjour"}],
                                         "usage": {"type": "duration", "seconds": 1}}).encode())
        elif self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in (b'data: {"choices":[{"delta":{"content":"Bon"}}]}\n\n',
                          b'data: {"choices":[{"delta":{"content":"jour"}}]}\n\n', b'data: [DONE]\n\n'):
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(0.02)
        elif self.path == "/v1/audio/speech":
            self._reply(200, wav_bytes(0.1), "audio/wav")
        else:
            self._reply(404, b'{"error":"not found"}')


class MockHarness:
    def __init__(self):
        spec = importlib.util.spec_from_file_location("emo_server_under_test", ROOT.parent / "emo_server.py")
        self.server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.server)
        self.original_vote = self.server.onnx_audio_vote
        self.original_urlopen = self.server.urlopen
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeLemonade)
        self.sidecar = None
        self.threads = []

    def start(self):
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        os.environ["LEMONADE_HOST"] = f"http://127.0.0.1:{self.upstream.server_port}/v1"
        os.environ["EMO_AUDIO_ONNX"] = ""
        os.environ["EMO_AUDIO_CMD"] = ""
        self.start_sidecar()

    def start_sidecar(self):
        if self.sidecar:
            self.sidecar.shutdown()
            self.sidecar.server_close()
        self.sidecar = ThreadingHTTPServer(("127.0.0.1", 0), self.server.Handler)
        thread = threading.Thread(target=self.sidecar.serve_forever, daemon=True)
        thread.start()
        self.threads.append(thread)
        self.url = f"http://127.0.0.1:{self.sidecar.server_port}"

    def stop(self):
        if self.sidecar:
            self.sidecar.shutdown()
            self.sidecar.server_close()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.server.onnx_audio_vote = self.original_vote
        self.server.urlopen = self.original_urlopen
        StubState.delay = 0

    def force_labels(self):
        # La queue RIFF ignorée par wave associe chaque requête à un label,
        # sans modifier le code de production ni réseau externe.
        original = self.server.onnx_audio_vote
        def classifier(audio):
            for label in RAVDESS_LABELS + EMOTIONS + RAW_NEUTRAL:
                marker = ("GEEUI-" + label).encode()
                if marker in audio:
                    # Les labels bruts non robot sont convertis exactement comme le
                    # classifieur ONNX : parse_label les ramène à neutral.
                    return self.server.parse_label(label), 0.93
            return original(audio)
        self.server.onnx_audio_vote = classifier


class Campaign:
    def __init__(self):
        self.cases = []

    def add(self, case_id, objective, passed, expected, observed, latency_ms, details=None):
        self.cases.append({"id": case_id, "objectif": objective, "statut": "PASS" if passed else "FAIL",
                           "attendu": expected, "observe": observed, "latence_ms": round(latency_ms, 2),
                           "details": details or {}})


def check_transcription(campaign, url, case_id, filename, audio, expected, passthrough=False):
    status, _, raw, elapsed = transcription(url, filename, audio)
    data = json_body(raw)
    ok = status == 200 and data.get("emotion") == expected and isinstance(data.get("emotion_confidence"), (float, int))
    if case_id == "EDGE-01":
        ok = ok and data.get("emotion_confidence") == 0.0
    if case_id.startswith("LIVE-") and case_id != "LIVE-SHORT":
        inactive = (data.get("emotion_source") == "fallback"
                    or (data.get("emotion_confidence") == 0.0 and data.get("emotion_source") != "silence"))
        if inactive:
            ok = False
            observation = "classifieur inactif (repli)"
        else:
            observation = f"HTTP {status}, emotion={data.get('emotion')}, confiance={data.get('emotion_confidence')}"
    else:
        observation = f"HTTP {status}, emotion={data.get('emotion')}, confiance={data.get('emotion_confidence')}"
    if passthrough:
        ok = ok and data.get("text") == "bonjour le robot" and data.get("language") == "fr" and data.get("segments") == [{"id": 0, "text": "bonjour"}]
    campaign.add(case_id, f"Transcription et émotion {expected}", ok,
                 f"HTTP 200, emotion={expected}, confiance numérique",
                 observation, elapsed,
                 {"reponse": data})


def run_mock():
    harness = MockHarness()
    campaign = Campaign()
    harness.start()
    if MOCK_CLASSIFIER:
        harness.force_labels()
    audio = wav_bytes()
    try:
        status, _, raw, elapsed = request(harness.url + "/v1/emotion/status")
        status_body = json_body(raw)
        campaign.add("STATUS-01", "Diagnostic du classifieur ONNX",
                     status == 200 and status_body.get("session_loaded") is False
                     and status_body.get("onnx_configured") is False
                     and "classifications" in status_body and "failures" in status_body,
                     "HTTP 200, configuration, session et compteurs exposés",
                     f"HTTP {status}, {raw!r}", elapsed, {"statut_onnx": status_body})
        for index, label in enumerate(RAVDESS_LABELS + RAW_NEUTRAL[1:], 1):
            marked = audio + b"GEEUI-" + label.encode()  # queue RIFF ignorée par wave
            expected = harness.server.parse_label(label)
            check_transcription(campaign, harness.url, f"EMO-{index:02d}", label + ".wav", marked, expected,
                                passthrough=(label == "happy"))
            campaign.cases[-1]["details"]["ravdess_label"] = label
        harness.server.onnx_audio_vote = harness.original_vote
        check_transcription(campaign, harness.url, "EDGE-01", "court.wav", wav_bytes(0.5), "neutral")
        if MOCK_CLASSIFIER:
            harness.force_labels()
        check_transcription(campaign, harness.url, "EDGE-02", "silence.wav", wav_bytes(kind="silence") + b"GEEUI-neutral", "neutral")
        check_transcription(campaign, harness.url, "EDGE-03", "bruit.wav", wav_bytes(kind="noise") + b"GEEUI-neutral", "neutral")
        status, _, raw, elapsed = transcription(harness.url, "corrompu.wav", b"ceci n'est pas un WAV")
        campaign.add("EDGE-04", "Rejet d'un fichier non audio", status == 400 and "emotion" not in json_body(raw),
                     "HTTP 400 amont conservé, aucun champ émotion", f"HTTP {status}, {raw[:120]!r}", elapsed)
        body, mime = multipart()
        status, _, raw, elapsed = request(harness.url + "/v1/audio/transcriptions", "POST", body, mime)
        campaign.add("EDGE-05", "Champ file absent", status == 400 and json_body(raw).get("error") == "missing audio file",
                     "HTTP 400 missing audio file", f"HTTP {status}, {raw!r}", elapsed)
        large = wav_bytes(3.0) + b"0" * (2 * 1024 * 1024) + b"GEEUI-neutral"
        check_transcription(campaign, harness.url, "EDGE-06", "grand.wav", large, "neutral")
        StubState.last_model = None
        status, _, raw, elapsed = transcription(harness.url, "sans-model.wav", audio + b"GEEUI-neutral", model=None)
        expected_model = os.environ.get("EMO_STT_MODEL") or "Whisper-Small"
        campaign.add("EDGE-07", "Requête client sans champ model",
                     status == 200 and StubState.last_model == expected_model,
                     f"HTTP 200, le sidecar ajoute model={expected_model}",
                     f"HTTP {status}, model reçu par Lemonade={StubState.last_model}", elapsed)

        # Modèle absent et modèle invalide : repli sûr de la vraie fonction de production.
        harness.server.onnx_audio_vote = harness.original_vote
        os.environ["EMO_AUDIO_ONNX"] = ""
        check_transcription(campaign, harness.url, "CONF-01", "sans-modele.wav", audio, "neutral")
        os.environ["EMO_AUDIO_ONNX"] = "/chemin/inexistant.onnx"
        check_transcription(campaign, harness.url, "CONF-02", "modele-invalide.wav", audio, "neutral")
        os.environ["EMO_AUDIO_ONNX"] = ""
        if MOCK_CLASSIFIER:
            harness.force_labels()

        # Panne amont : port local volontairement fermé.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            dead_port = sock.getsockname()[1]
        os.environ["LEMONADE_HOST"] = f"http://127.0.0.1:{dead_port}/v1"
        status, _, raw, elapsed = transcription(harness.url, "happy.wav", audio + b"GEEUI-happy")
        campaign.add("UP-01", "Lemonade indisponible", status == 500,
                     "Erreur contrôlée HTTP 500 (comportement actuel)", f"HTTP {status}, {raw!r}", elapsed)
        os.environ["LEMONADE_HOST"] = f"http://127.0.0.1:{harness.upstream.server_port}/v1"
        StubState.delay = 0.3
        original_urlopen = harness.server.urlopen
        def short_timeout(*args, **kwargs):
            kwargs["timeout"] = 0.1
            return original_urlopen(*args, **kwargs)
        harness.server.urlopen = short_timeout
        status, _, raw, elapsed = transcription(harness.url, "happy.wav", audio + b"GEEUI-happy")
        campaign.add("UP-02", "Dépassement de délai Lemonade", status == 500,
                     "Erreur contrôlée HTTP 500 (comportement actuel)", f"HTTP {status}, {raw!r}", elapsed)
        harness.server.urlopen = original_urlopen
        StubState.delay = 0

        payload = json.dumps({"model": "test", "stream": True, "messages": []}).encode()
        status, headers, raw, elapsed = request(harness.url + "/v1/chat/completions", "POST", payload, "application/json")
        campaign.add("PROXY-01", "Proxy chat SSE et chunks", status == 200 and "Bon" in raw.decode() and "jour" in raw.decode() and raw.endswith(b"data: [DONE]\n\n") and "text/event-stream" in headers.get("Content-Type", ""),
                     "SSE complet Bon+jour+[DONE]", f"HTTP {status}, {raw!r}", elapsed)
        status, headers, raw, elapsed = request(harness.url + "/v1/audio/speech", "POST", b'{"input":"bonjour"}', "application/json")
        campaign.add("PROXY-02", "Proxy synthèse vocale WAV", status == 200 and raw.startswith(b"RIFF") and headers.get("Content-Type") == "audio/wav",
                     "HTTP 200 audio/wav RIFF", f"HTTP {status}, {headers.get('Content-Type')}, {len(raw)} octets", elapsed)
        status, _, raw, elapsed = request(harness.url + "/v1/models")
        campaign.add("PROXY-03", "Proxy liste des modèles", status == 200 and json_body(raw)["data"][0]["id"] == "Whisper-Small",
                     "HTTP 200 et liste Whisper inchangée", f"HTTP {status}, {raw!r}", elapsed)
        # Mesure dédiée : 5 requêtes, moyenne et p95 calculées dans le rapport.
        latencies = []
        for _ in range(5):
            status, _, _, elapsed = transcription(harness.url, "neutral.wav", audio + b"GEEUI-neutral")
            if status == 200:
                latencies.append(elapsed)
        campaign.add("PERF-01", "Mesure de latence transcription", len(latencies) == 5,
                     "5 mesures valides", f"{len(latencies)}/5 mesures", statistics.mean(latencies) if latencies else 0,
                     {"serie_ms": [round(v, 2) for v in latencies]})
    finally:
        harness.stop()
    return campaign, []


def expected_from_name(path):
    name = path.stem.lower()
    aliases = {"fearful": "fear", "surprised": "surprise", "disgusted": "neutral", "other": "neutral", "unknown": "neutral"}
    for raw, emotion in aliases.items():
        if raw in name:
            return emotion
    for emotion in EMOTIONS:
        if emotion in name:
            return emotion
    return None


def percentile95(values):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]


def run_live(url, audio_dir):
    campaign = Campaign()
    manifest = audio_dir / "manifest.csv"
    rows = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = [(audio_dir / row["file"], row["expected_emotion"].strip().lower()) for row in csv.DictReader(stream)]
    else:
        rows = [(p, expected_from_name(p)) for p in sorted(audio_dir.glob("*.wav"))]
    if not rows:
        raise RuntimeError(f"Aucun WAV ni manifest.csv exploitable dans {audio_dir}")
    observations = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as stream:
            raw_by_file = {audio_dir / row["file"]: row.get("ravdess_label", "non_precise") for row in csv.DictReader(stream)}
    else:
        raw_by_file = {}
    try:
        status, _, raw, elapsed = request(url + "/v1/emotion/status")
        preflight = json_body(raw)
        preflight_ok = status == 200 and preflight.get("session_loaded") is True
        observed = f"HTTP {status}, session_loaded={preflight.get('session_loaded')}, dernière_erreur={preflight.get('last_error')}"
    except Exception as exc:
        preflight, elapsed, preflight_ok = {}, 0.0, False
        observed = f"{type(exc).__name__}: {exc}"
    campaign.add("PRE-01", "Préflight du classifieur ONNX", preflight_ok,
                 "GET /v1/emotion/status : session_loaded=true", observed, elapsed,
                 {"statut_onnx": preflight})
    for index, (path, expected) in enumerate(rows, 1):
        if not path.is_file() or expected not in EMOTIONS:
            campaign.add(f"LIVE-{index:03d}", str(path), False, "Fichier existant et émotion valide", "Entrée invalide", 0)
            continue
        check_transcription(campaign, url, f"LIVE-{index:03d}", path.name, path.read_bytes(), expected)
        case = campaign.cases[-1]
        observed = case["details"].get("reponse", {}).get("emotion")
        raw_label = raw_by_file.get(path, "non_precise")
        campaign.cases[-1]["details"]["ravdess_label"] = raw_label
        observations.append((expected, observed, raw_label))
    # Cas systématiques générés localement, sans écrire de WAV dans le dépôt.
    for case_id, name, data in (("LIVE-SHORT", "court.wav", wav_bytes(0.5)),
                                ("LIVE-SILENCE", "silence.wav", wav_bytes(kind="silence")),
                                ("LIVE-NOISE", "bruit.wav", wav_bytes(kind="noise"))):
        check_transcription(campaign, url, case_id, name, data, "neutral")
    return campaign, observations


def write_reports(campaign, observations, mode, target, destination):
    destination.mkdir(parents=True, exist_ok=True)
    passed = sum(c["statut"] == "PASS" for c in campaign.cases)
    failed = len(campaign.cases) - passed
    latencies = [c["latence_ms"] for c in campaign.cases if c["latence_ms"] > 0]
    matrix = {expected: {observed: 0 for observed in EMOTIONS + ("autre/erreur",)} for expected in EMOTIONS}
    raw_accuracy = {}
    for expected, observed, raw_label in observations:
        matrix[expected][observed if observed in EMOTIONS else "autre/erreur"] += 1
        total, correct = raw_accuracy.get(raw_label, (0, 0))
        raw_accuracy[raw_label] = (total + 1, correct + (expected == observed))
    raw_accuracy = {label: correct / total for label, (total, correct) in raw_accuracy.items()}
    accuracy = None
    if observations:
        accuracy = {e: (sum(1 for a, b, _ in observations if a == e and b == e) / max(1, sum(1 for a, b, _ in observations if a == e))) for e in EMOTIONS}
        accuracy = {e: v for e, v in accuracy.items() if any(a == e for a, _, _ in observations)}
    preflight_case = next((c for c in campaign.cases if c["id"] == "PRE-01"), None)
    data = {"mode": mode, "cible": target, "date": datetime.now().astimezone().isoformat(),
            "preflight_onnx": preflight_case.get("details", {}).get("statut_onnx") if preflight_case else None,
            "resume": {"total": len(campaign.cases), "reussis": passed, "echoues": failed},
            "latence_ms": {"moyenne": statistics.mean(latencies) if latencies else None, "p95": percentile95(latencies)},
            "exactitude_par_emotion_robot": accuracy, "exactitude_par_label_ravdess": raw_accuracy or None,
            "matrice_de_confusion": matrix if observations else None,
            "cas": campaign.cases}
    (destination / "resultats.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Rapport de test sidecar — {mode}", "", f"- Date : {data['date']}", f"- Cible : `{target}`",
             f"- **Résultat : {passed}/{len(campaign.cases)} réussis, {failed} échoué(s)**",
             f"- Latence moyenne : {data['latence_ms']['moyenne'] and round(data['latence_ms']['moyenne'], 2)} ms ; p95 : {data['latence_ms']['p95'] and round(data['latence_ms']['p95'], 2)} ms", ""]
    if preflight_case:
        lines += ["## Préflight ONNX", "", "```json", json.dumps(data["preflight_onnx"], ensure_ascii=False, indent=2), "```", ""]
    if accuracy is not None:
        lines += ["## Exactitude par émotion robot", ""] + [f"- {e} : {v:.1%}" for e, v in accuracy.items()]
        lines += ["", "## Exactitude par label brut RAVDESS", ""] + [f"- {e} : {v:.1%}" for e, v in raw_accuracy.items()]
        lines += ["", "## Matrice de confusion robot (lignes = attendu)", ""]
        columns = list(next(iter(matrix.values())))
        lines += ["| Attendu | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        lines += ["| " + e + " | " + " | ".join(str(matrix[e][c]) for c in columns) + " |" for e in matrix] + [""]
    lines += ["## Détail des cas", "", "| ID | Objectif | Statut | Attendu | Observé | Latence (ms) |", "|---|---|---|---|---|---:|"]
    for c in campaign.cases:
        clean = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {c['id']} | {clean(c['objectif'])} | **{c['statut']}** | {clean(c['attendu'])} | {clean(c['observe'])} | {c['latence_ms']} |")
    lines += ["", "Les détails JSON complets de chaque réponse sont dans `resultats.json`."]
    (destination / "RAPPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # HTML autonome : le Markdown reste la source lisible, aucune ressource distante.
    style = "body{font:14px sans-serif;max-width:1200px;margin:2em auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}.PASS{color:green}.FAIL{color:red}"
    rows_html = "".join(f"<tr><td>{html.escape(c['id'])}</td><td>{html.escape(c['objectif'])}</td><td class='{c['statut']}'>{c['statut']}</td><td>{html.escape(c['attendu'])}</td><td>{html.escape(c['observe'])}</td><td>{c['latence_ms']}</td></tr>" for c in campaign.cases)
    (destination / "rapport.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Rapport sidecar</title><style>{style}</style><h1>Rapport sidecar — {mode}</h1><p>{passed} réussis / {failed} échoués</p><table><tr><th>ID</th><th>Objectif</th><th>Statut</th><th>Attendu</th><th>Observé</th><th>ms</th></tr>{rows_html}</table>", encoding="utf-8")
    return failed == 0


def main():
    global STT_MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--url", default="http://127.0.0.1:13306", help="URL du sidecar en mode live")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "samples/ravdess", help="WAV et manifest.csv (défaut : corpus RAVDESS commité)")
    parser.add_argument("--stt-model", default=STT_MODEL, help="Modèle Whisper envoyé dans le champ model (défaut : EMO_STT_MODEL ou Whisper-Small)")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    STT_MODEL = args.stt_model
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = args.reports_dir / stamp
    try:
        if args.mode == "mock":
            campaign, observations = run_mock()
            target = "amont Lemonade simulé localement"
        else:
            campaign, observations = run_live(args.url.rstrip("/"), args.audio_dir)
            target = args.url
    except Exception as exc:
        campaign = Campaign()
        campaign.add("RUN-01", "Exécution de la campagne", False, "Campagne exécutée", f"{type(exc).__name__}: {exc}", 0,
                     {"traceback": traceback.format_exc()})
        observations, target = [], args.url
    success = write_reports(campaign, observations, args.mode, target, destination)
    print(f"Rapports : {destination}")
    print(f"Résultat : {sum(c['statut'] == 'PASS' for c in campaign.cases)}/{len(campaign.cases)} réussis")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
