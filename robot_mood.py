"""L'humeur de RUX (point 4 de la roadmap) : deux valeurs qui changent lentement.

- contentement (-1..1) : monte quand on lui parle ou le complimente, baisse si on le rudoie
  ou si personne ne lui parle pendant des heures ;
- énergie (0..1) : basse la nuit, remonte le jour et pendant une conversation.

Les deux reviennent vers leur valeur de repos avec une demi-vie de quelques heures.
Le sidecar l'expose (GET /v1/robot/mood, et dans /v1/robot/pending) et l'ajoute au préfixe
envoyé à nanobot quand elle n'est pas ordinaire. L'app choisit le visage au repos selon `label`.
Les règles sont ici, dans le code, pas dans le modèle. ROBOT_MOOD_ENABLED=0 : rien ne change.
"""

import datetime as dt
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

DEFAULTS = {
    "ROBOT_MOOD_ENABLED": "0",
    "ROBOT_MOOD_HALF_LIFE_H": "3",
    "ROBOT_MOOD_BORED_H": "4",
    "ROBOT_MOOD_MISS_H": "8",
    "ROBOT_MOOD_FILE": "~/.local/share/geeui/robot_mood.json",
}

REST_CONTENT = 0.2
DAY_ENERGY = 0.7
NIGHT_ENERGY = 0.1

COMPLIMENT = re.compile(
    r"\b(merci|bravo|bien jou[ée]|super|g[ée]nial|je t'aime|je t'adore|t'es (?:trop |vraiment )?"
    r"(?:mignon|gentil|g[ée]nial|dr[ôo]le|fort|beau|intelligent)|tu es (?:trop |vraiment )?"
    r"(?:mignon|gentil|g[ée]nial|dr[ôo]le|fort|beau|intelligent))", re.I)
RUDE = re.compile(r"\b(t'es nul|tu es nul|idiot|stupide|b[êe]te|ferme[- ]la|d[ée]bile|nul)\b", re.I)

# Libellés et ce qu'ils deviennent pour nanobot (préfixe) ; None = rien à dire.
PHRASES = {
    "joyeux": "de très bonne humeur",
    "content": None,
    "calme": None,
    "fatigue": "un peu fatigué",
    "ennuye": "un peu ennuyé, personne ne lui a parlé depuis longtemps",
    "boudeur": "un peu boudeur",
    "triste": "un peu triste",
}


def _env(name):
    raw = os.environ.get(name)
    return DEFAULTS[name] if raw is None or not raw.strip() else raw.strip()


def settings():
    return {
        "enabled": _env("ROBOT_MOOD_ENABLED").lower() in ("1", "true", "yes", "oui"),
        "half_life_s": float(_env("ROBOT_MOOD_HALF_LIFE_H")) * 3600,
        "bored_s": float(_env("ROBOT_MOOD_BORED_H")) * 3600,
        "miss_s": float(_env("ROBOT_MOOD_MISS_H")) * 3600,
        "file": os.path.expanduser(_env("ROBOT_MOOD_FILE")),
    }


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class RobotMood:
    def __init__(self, cfg=None, clock=time.time, night=lambda stamp: False):
        self.cfg = cfg or settings()
        self.clock = clock
        self.night = night  # heures calmes de l'initiative
        self._lock = threading.Lock()
        self.state = self._load()

    # --- persistance (survit à un redémarrage du sidecar) ---------------------------
    def _load(self):
        try:
            data = json.loads(Path(self.cfg["file"]).read_text(encoding="utf-8"))
            return {"content": float(data["content"]), "energy": float(data["energy"]),
                    "at": float(data["at"]), "contact": float(data.get("contact", data["at"]))}
        except (OSError, ValueError, KeyError, TypeError):
            now = self.clock()
            return {"content": REST_CONTENT, "energy": DAY_ENERGY, "at": now, "contact": now}

    def _save(self):
        if not self.cfg["enabled"]:
            return
        path = Path(self.cfg["file"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".robot_mood-", suffix=".json")
            with os.fdopen(fd, "w") as handle:
                json.dump(self.state, handle)
            os.replace(tmp, path)
        except OSError:
            pass

    # --- dynamique --------------------------------------------------------------------
    def _relax(self, now):
        elapsed = max(0.0, now - self.state["at"])
        if elapsed:
            keep = 0.5 ** (elapsed / self.cfg["half_life_s"])
            energy_rest = NIGHT_ENERGY if self.night(now) else DAY_ENERGY
            self.state["content"] = REST_CONTENT + (self.state["content"] - REST_CONTENT) * keep
            self.state["energy"] = energy_rest + (self.state["energy"] - energy_rest) * keep
        self.state["at"] = now

    def user_turn(self, text, heard="neutral", now=None):
        """Quelqu'un parle au robot. Renvoie True si c'est un retour après une longue absence."""
        if not self.cfg["enabled"]:
            return False
        now = self.clock() if now is None else now
        with self._lock:
            self._relax(now)
            missed = now - self.state["contact"] >= self.cfg["miss_s"] and not self.night(now)
            delta = 0.08 + (0.3 if missed else 0.0)
            if COMPLIMENT.search(text or ""):
                delta += 0.25
            if RUDE.search(text or ""):
                delta -= 0.35
            delta += {"happy": 0.05, "sad": -0.05, "angry": -0.05, "fear": -0.03}.get(heard, 0.0)
            self.state["content"] = _clamp(self.state["content"] + delta, -1.0, 1.0)
            self.state["energy"] = _clamp(self.state["energy"] + 0.05, 0.0, 1.0)
            self.state["contact"] = now
            self._save()
            return missed

    def snapshot(self, now=None):
        now = self.clock() if now is None else now
        with self._lock:
            self._relax(now)
            content, energy = self.state["content"], self.state["energy"]
            alone = now - self.state["contact"]
        # Longtemps seul (en journée) : l'ennui tire le contentement vers le bas.
        if alone > self.cfg["bored_s"] and not self.night(now):
            content -= min(0.5, (alone - self.cfg["bored_s"]) / 3600 * 0.1)
        if energy < 0.25:
            label = "fatigue"
        elif content <= -0.35:
            label = "triste"
        elif content <= -0.1:
            label = "boudeur" if alone > self.cfg["bored_s"] else "triste"
        elif alone > self.cfg["bored_s"] and not self.night(now):
            label = "ennuye"
        elif content >= 0.55:
            label = "joyeux"
        elif content >= 0.25:
            label = "content"
        else:
            label = "calme"
        return {"label": label, "content": round(content, 3), "energy": round(energy, 3)}

    def prefix(self, missed=False, now=None):
        if not self.cfg["enabled"]:
            return ""
        if missed:
            return "(humeur de RUX : content de revoir quelqu'un après une longue absence) "
        phrase = PHRASES.get(self.snapshot(now)["label"])
        return f"(humeur de RUX : {phrase}) " if phrase else ""
