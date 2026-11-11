"""Qui parle ? Empreinte vocale (WeSpeaker ResNet34-LM via sherpa-onnx), optionnelle.

Sans `SPEAKER_MODEL`, ou si sherpa-onnx manque, tout est désactivé et le sidecar se
comporte comme avant. Aucun audio n'est gardé : seulement une empreinte moyenne par
personne dans `SPEAKERS_FILE` (droits 600).
"""

import io
import json
import math
import os
import re
import tempfile
import threading
import time
import unicodedata
import wave
from pathlib import Path

DEFAULTS = {
    "SPEAKER_MODEL": "",
    "SPEAKERS_FILE": "~/.local/share/geeui/speakers.json",
    "SPEAKER_THRESHOLD": "0.5",
    "SPEAKER_MARGIN": "0.05",
    "SPEAKER_DOUBT": "0.35",
    "SPEAKER_STICKY_S": "60",
    "SPEAKER_MIN_S": "1.2",
    "SPEAKER_ENROLL_MIN_S": "2.0",
    "SPEAKER_ENROLL_SAMPLES": "3",
    "SPEAKER_ENROLL_AGREE": "0.5",
    "SPEAKER_ENROLL_TIMEOUT_S": "120",
    "SPEAKER_PROVIDER": "cpu",
    "SPEAKER_THREADS": "1",
    "SPEAKER_GUEST_ROUTE": "nanobot",
}

GUEST = "invite"


def _env(name):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return DEFAULTS[name]
    return raw.strip()


def settings():
    return {
        "model": _env("SPEAKER_MODEL"),
        "file": os.path.expanduser(_env("SPEAKERS_FILE")),
        "threshold": float(_env("SPEAKER_THRESHOLD")),
        "margin": float(_env("SPEAKER_MARGIN")),
        "doubt": float(_env("SPEAKER_DOUBT")),
        "sticky_s": float(_env("SPEAKER_STICKY_S")),
        "min_s": float(_env("SPEAKER_MIN_S")),
        "enroll_min_s": float(_env("SPEAKER_ENROLL_MIN_S")),
        "enroll_samples": int(_env("SPEAKER_ENROLL_SAMPLES")),
        "enroll_agree": float(_env("SPEAKER_ENROLL_AGREE")),
        "enroll_timeout_s": float(_env("SPEAKER_ENROLL_TIMEOUT_S")),
        "provider": _env("SPEAKER_PROVIDER"),
        "threads": int(_env("SPEAKER_THREADS")),
        "guest_route": _env("SPEAKER_GUEST_ROUTE").lower(),
    }


# --- vecteurs ---------------------------------------------------------------

def normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a, b):
    return sum(x * y for x, y in zip(normalize(a), normalize(b)))


def mean_vector(vectors):
    vectors = [normalize(v) for v in vectors]
    return normalize([sum(col) / len(vectors) for col in zip(*vectors)])


def slug(name):
    """« Marie-Hélène » -> « marie-helene » : identifiant de session nanobot."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def display_name(name):
    return "-".join(p[:1].upper() + p[1:].lower() for p in name.strip().split("-"))


def wav_duration(wav):
    try:
        with wave.open(io.BytesIO(wav), "rb") as pcm:
            return pcm.getnframes() / float(pcm.getframerate() or 16000)
    except Exception:
        return 0.0


# --- extracteur (sherpa-onnx, chargé à la demande) ----------------------------

class SherpaExtractor:
    """Empreinte d'un WAV PCM 16 bits mono. `embed` renvoie None si indisponible."""

    def __init__(self, model, provider="cpu", threads=1):
        self.model = model
        self.provider = provider
        self.threads = threads
        self._extractor = None
        self._lock = threading.Lock()
        self.error = None

    @property
    def name(self):
        return Path(self.model).name if self.model else ""

    def _load(self):
        if self._extractor is not None or self.error:
            return self._extractor
        try:
            import sherpa_onnx
            if not Path(self.model).is_file():
                raise FileNotFoundError(self.model)
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.model, num_threads=self.threads, provider=self.provider)
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        except Exception as exc:  # ImportError, modèle absent, provider inconnu
            self.error = f"{type(exc).__name__}: {exc}"
        return self._extractor

    def available(self):
        return self._load() is not None

    def embed(self, wav):
        extractor = self._load()
        if extractor is None:
            return None
        import numpy as np
        with wave.open(io.BytesIO(wav), "rb") as pcm:
            rate = pcm.getframerate()
            channels = pcm.getnchannels()
            raw = pcm.readframes(pcm.getnframes())
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels)[:, 0].copy()
        if samples.size < rate // 2:
            return None
        with self._lock:
            stream = extractor.create_stream()
            stream.accept_waveform(rate, samples)
            stream.input_finished()
            return list(extractor.compute(stream))


# --- stockage -------------------------------------------------------------------

class Store:
    """speakers.json : {"model": …, "people": {id: {name, embedding, …}}}. Écriture atomique."""

    def __init__(self, path, model_name):
        self.path = Path(path)
        self.model_name = model_name
        self._lock = threading.Lock()

    def _read(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"model": self.model_name, "people": {}}
        if data.get("model") and self.model_name and data["model"] != self.model_name:
            # Une empreinte d'un autre modèle ne se compare pas : il faut réenregistrer.
            return {"model": self.model_name, "people": {}, "stale": data.get("model")}
        data.setdefault("people", {})
        return data

    def people(self):
        with self._lock:
            return self._read()["people"]

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"model": self.model_name, "people": data["people"]}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".speakers-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=1)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def put(self, person_id, record):
        with self._lock:
            data = self._read()
            data["people"][person_id] = record
            self._write(data)

    def delete(self, person_id):
        with self._lock:
            data = self._read()
            if person_id not in data["people"]:
                return False
            del data["people"][person_id]
            self._write(data)
            return True


# --- phrases reconnues par le sidecar ------------------------------------------

_NAME = r"([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]{1,30})"
_INTENTS = [
    ("enroll", re.compile(r"(?:retiens|apprends|enregistre|m[ée]morise)\s+ma\s+voix\W+(?:je\s+suis|je\s+m'appelle|c'est)\s+" + _NAME, re.I)),
    ("enroll", re.compile(r"(?:je\s+suis|je\s+m'appelle|c'est)\s+" + _NAME + r"\W+(?:retiens|apprends|enregistre|m[ée]morise)\s+ma\s+voix", re.I)),
    ("forget", re.compile(r"oublie\s+ma\s+voix", re.I)),
    ("whoami", re.compile(r"qui\s+suis[- ]je|tu\s+me\s+reconnais|sais[- ]tu\s+qui\s+je\s+suis", re.I)),
    ("cancel", re.compile(r"^\W*(annule|laisse tomber|stop|arr[êe]te)\b", re.I)),
]


def parse_intent(text):
    for kind, pattern in _INTENTS:
        match = pattern.search(text or "")
        if match:
            return kind, (match.group(1) if match.groups() else None)
    return None, None


# --- le service ------------------------------------------------------------------

class Speakers:
    """Identification, enrôlement à la voix et personne courante. Une instance par sidecar."""

    def __init__(self, extractor=None, store=None, cfg=None, clock=time.time):
        self.cfg = cfg or settings()
        self.extractor = extractor or SherpaExtractor(
            self.cfg["model"], self.cfg["provider"], self.cfg["threads"])
        self.store = store or Store(self.cfg["file"], getattr(self.extractor, "name", ""))
        self.clock = clock
        self.current = None  # {"id", "name", "score", "at"}
        self.enroll = None   # {"id", "name", "samples", "at"}
        self.reply = None    # texte que le prochain tour de chat doit dire
        self._lock = threading.RLock()

    def enabled(self):
        return bool(self.cfg["model"]) and self.extractor.available()

    def known(self):
        return bool(self.store.people())

    # --- STT -------------------------------------------------------------------
    def observe(self, wav):
        """Appelé sur chaque WAV de /v1/audio/transcriptions. Renvoie les champs speaker*."""
        if not self.enabled():
            return None
        now = self.clock()
        vec = self.extractor.embed(wav)
        duration = wav_duration(wav)
        with self._lock:
            self._expire(now)
            if self.enroll is not None:
                self._enroll_sample(vec, duration, now)
                person = self.enroll or self.current
                if person:
                    return {"speaker": person["id"], "speaker_name": person["name"], "speaker_score": None}
            if vec is None:
                return None
            return self.identify(vec, duration, now)

    def identify(self, vec, duration, now):
        with self._lock:
            people = self.store.people()
            if not people:
                return None
            recent = self.current is not None and now - self.current["at"] <= self.cfg["sticky_s"]
            scores = sorted(
                ((cosine(vec, p["embedding"]), pid) for pid, p in people.items()), reverse=True)
            best, best_id = scores[0]
            second = scores[1][0] if len(scores) > 1 else -1.0
            if duration < self.cfg["min_s"] and recent:
                chosen = self.current["id"]
            elif best >= self.cfg["threshold"] and best - second >= self.cfg["margin"]:
                chosen = best_id
            elif best >= self.cfg["doubt"] and recent and self.current["id"] != GUEST:
                chosen = self.current["id"]
            else:
                chosen = GUEST
            name = people[chosen]["name"] if chosen in people else "Invité"
            score = round(dict((pid, s) for s, pid in scores).get(chosen, best), 3)
            self.current = {"id": chosen, "name": name, "score": score, "at": now}
            return {"speaker": chosen, "speaker_name": name, "speaker_score": score}

    # --- enrôlement --------------------------------------------------------------
    def _expire(self, now):
        if self.enroll and now - self.enroll["at"] > self.cfg["enroll_timeout_s"]:
            self.enroll = None
            self.reply = None

    def _enroll_sample(self, vec, duration, now):
        need = self.cfg["enroll_samples"]
        self.enroll["at"] = now
        if vec is None or duration < self.cfg["enroll_min_s"]:
            self.reply = "[neutral] Celle-là était trop courte. Dis-moi une phrase un peu plus longue."
            return
        self.enroll["samples"].append(vec)
        got = len(self.enroll["samples"])
        if got < need:
            self.reply = "[happy] Encore une." if got < need - 1 else "[happy] Une dernière."
            return
        samples = self.enroll["samples"]
        worst = min(cosine(a, b) for i, a in enumerate(samples) for b in samples[i + 1:]) if len(samples) > 1 else 1.0
        name, pid = self.enroll["name"], self.enroll["id"]
        self.enroll = None
        if worst < self.cfg["enroll_agree"]:
            self.reply = ("[sad] Tes phrases ne se ressemblent pas assez, il y avait peut-être du bruit "
                          "ou une autre voix. Redis-moi « retiens ma voix, je suis " + name + " » pour recommencer.")
            return
        self.store.put(pid, {
            "name": name,
            "embedding": [round(x, 6) for x in mean_vector(samples)],
            "samples": len(samples),
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "consent": "voice",
            "role": "adulte",
        })
        self.current = {"id": pid, "name": name, "score": 1.0, "at": now}
        self.reply = f"[happy] C'est noté, {name}. Je te reconnaîtrai maintenant."

    # --- chat ---------------------------------------------------------------------
    def handle_chat(self, text):
        """Réponse locale du sidecar (enrôlement, « qui suis-je »…), ou None pour nanobot."""
        if not self.enabled():
            return None
        now = self.clock()
        kind, arg = parse_intent(text)
        with self._lock:
            self._expire(now)
            if self.enroll is not None:
                if kind == "cancel":
                    self.enroll = None
                    self.reply = None
                    return "[neutral] D'accord, j'arrête d'apprendre ta voix."
                reply = self.reply or "[neutral] Dis-moi une phrase, je t'écoute."
                self.reply = None
                return reply
            if self.reply:  # le dernier échantillon vient d'être traité
                reply, self.reply = self.reply, None
                return reply
            if kind == "enroll":
                return self._start_enroll(arg, now)
            if kind == "forget":
                return self._forget()
            if kind == "whoami":
                return self._whoami()
        return None

    def _start_enroll(self, raw_name, now):
        name = display_name(raw_name)
        pid = slug(name)
        if not pid or pid == GUEST:
            return "[neutral] Je n'ai pas compris ton prénom. Redis « retiens ma voix, je suis » puis ton prénom."
        people = self.store.people()
        current = self.current["id"] if self.current else None
        if pid in people and current != pid:
            return (f"[neutral] Je connais déjà la voix de {people[pid]['name']}. "
                    f"Pour la remplacer, il faut que {people[pid]['name']} me le demande.")
        self.enroll = {"id": pid, "name": name, "samples": [], "at": now}
        n = self.cfg["enroll_samples"]
        return (f"[happy] D'accord {name}. Je garde juste une empreinte de ta voix, pas l'enregistrement. "
                f"Dis-moi {n} phrases, une à la fois, d'au moins deux secondes.")

    def _forget(self):
        cur = self.current
        if not cur or cur["id"] == GUEST or not self.store.delete(cur["id"]):
            return "[neutral] Je ne reconnais pas ta voix, je n'ai donc rien à oublier."
        self.current = None
        return f"[neutral] J'ai oublié ta voix, {cur['name']}."

    def _whoami(self):
        cur = self.current
        if not self.known():
            return "[neutral] Je ne connais encore aucune voix. Dis « retiens ma voix, je suis » puis ton prénom."
        if not cur or cur["id"] == GUEST:
            return "[neutral] Je ne te reconnais pas. Tu peux dire « retiens ma voix, je suis » puis ton prénom."
        return f"[happy] Tu es {cur['name']}, j'en suis sûr à {int(round((cur['score'] or 0) * 100))} %."

    def who(self):
        """(id, nom) de la personne courante, ou None si la fonction est éteinte / personne d'inscrit."""
        if not self.enabled() or not self.known():
            return None
        if self.current is None:
            return (GUEST, "Invité")
        return (self.current["id"], self.current["name"])

    def listing(self):
        return {
            "enabled": self.enabled(),
            "model": getattr(self.extractor, "name", ""),
            "error": getattr(self.extractor, "error", None),
            "current": self.current,
            "people": [
                {"id": pid, **{k: v for k, v in p.items() if k != "embedding"}}
                for pid, p in self.store.people().items()
            ],
        }
