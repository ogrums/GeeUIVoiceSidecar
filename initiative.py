"""Prendre l'initiative : rappels, suivis « À suivre », attention si tendu, tâche du soir.

Le sidecar décide QUAND parler (garde-fous dans le code), nanobot écrit QUOI dire.
L'app au repos lit GET /v1/robot/pending, dit la phrase, puis POST /v1/robot/ack.
INITIATIVE_ENABLED=0 (défaut) : rien ne change, `pending` reste vide.
"""

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    ZoneInfo = None

DEFAULTS = {
    "INITIATIVE_ENABLED": "0",
    "INITIATIVE_MAX_PER_DAY": "4",
    "INITIATIVE_MIN_GAP_MIN": "90",
    "INITIATIVE_TICK_S": "30",
    "INITIATIVE_TTL_MIN": "30",
    "INITIATIVE_STATE_DIR": "~/.local/share/geeui",
    "QUIET_HOURS": "21:30-08:00",
    "GREETING_FROM": "06:00",
    "NIGHTLY_AT": "03:00",
    "NANOBOT_WORKSPACE": "",
    "REMINDER_GRACE_MIN": "30",
    "FOLLOWUP_LATE_DAYS": "2",
    "TENSION_DAYS": "3",
    "TENSION_MIN_TURNS": "10",
    "TENSION_SHARE": "0.4",
    "TENSION_CONFIDENCE": "0.6",
    "TZ": "Europe/Paris",
}

NEGATIVE = {"sad", "angry", "fear"}


def _env(name):
    raw = os.environ.get(name)
    return DEFAULTS[name] if raw is None or not raw.strip() else raw.strip()


def settings():
    return {
        "enabled": _env("INITIATIVE_ENABLED").lower() in ("1", "true", "yes", "oui"),
        "max_per_day": int(_env("INITIATIVE_MAX_PER_DAY")),
        "min_gap_s": float(_env("INITIATIVE_MIN_GAP_MIN")) * 60,
        "tick_s": float(_env("INITIATIVE_TICK_S")),
        "ttl_s": float(_env("INITIATIVE_TTL_MIN")) * 60,
        "state_dir": os.path.expanduser(_env("INITIATIVE_STATE_DIR")),
        "quiet": _env("QUIET_HOURS"),
        "greeting_from": _env("GREETING_FROM"),
        "nightly_at": _env("NIGHTLY_AT"),
        "workspace": os.path.expanduser(_env("NANOBOT_WORKSPACE")),
        "reminder_grace_s": float(_env("REMINDER_GRACE_MIN")) * 60,
        "followup_late_days": int(_env("FOLLOWUP_LATE_DAYS")),
        "tension_days": int(_env("TENSION_DAYS")),
        "tension_min_turns": int(_env("TENSION_MIN_TURNS")),
        "tension_share": float(_env("TENSION_SHARE")),
        "tension_confidence": float(_env("TENSION_CONFIDENCE")),
        "tz": _env("TZ"),
    }


def _zone(name):
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def _hm(text):
    h, m = text.strip().split(":")
    return int(h) * 60 + int(m)


def slug(name):
    plain = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.stem + "-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default


# --- « À suivre » dans memory/MEMORY.md -----------------------------------------

_FOLLOW = re.compile(r"^\s*-\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$")


def parse_followups(markdown):
    """Lignes « - AAAA-MM-JJ | Prénom | sujet | quoi faire » de la section « ## À suivre »."""
    items, inside = [], False
    for line in (markdown or "").splitlines():
        if line.startswith("## "):
            inside = line[3:].strip().lower() in ("à suivre", "a suivre")
            continue
        if not inside:
            continue
        match = _FOLLOW.match(line)
        if not match:
            continue
        try:
            day = dt.date.fromisoformat(match.group(1))
        except ValueError:
            continue
        key = hashlib.sha1(line.strip().encode()).hexdigest()[:12]
        items.append({"id": key, "date": day, "name": match.group(2), "person": slug(match.group(2)),
                      "subject": match.group(3), "todo": match.group(4)})
    return items


# --- balises [rappel:…] écrites par nanobot ----------------------------------------

REMINDER_TAG = re.compile(r"\[rappel:\s*([^|\]]+?)\s*\|\s*([^\]]+?)\s*\]", re.I)


def parse_reminders(text, now_local):
    """[rappel:2026-11-10T19:30|sortir le linge] ou [rappel:+60|…] (minutes). Datetimes conscients."""
    found = []
    for when, what in REMINDER_TAG.findall(text or ""):
        when = when.strip()
        try:
            if when.startswith("+"):
                at = now_local + dt.timedelta(minutes=float(when[1:]))
            else:
                at = dt.datetime.fromisoformat(when)
                if at.tzinfo is None:
                    at = at.replace(tzinfo=now_local.tzinfo)
        except ValueError:
            continue
        if at <= now_local:
            continue
        found.append((at, what.strip()))
    return found


# --- « ne me dérange pas » ----------------------------------------------------------

_NUM = {"une": 1, "un": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "dix": 10,
        "quinze": 15, "vingt": 20, "trente": 30, "quarante-cinq": 45}
_DND = re.compile(r"ne me d[ée]range (?:pas|plus)|tais[- ]toi|silence|laisse[- ]moi tranquille|chut\b", re.I)
_WAKE = re.compile(r"tu peux (?:re)?parler|tu peux reprendre|reparle[- ]moi|fin du silence", re.I)
_FOR = re.compile(r"pendant\s+(\d+|" + "|".join(_NUM) + r")\s*(heure|h\b|minute|min\b)", re.I)
_UNTIL = re.compile(r"jusqu'?\s*[àa]\s*(\d{1,2})\s*(?:h|heures?)\s*(\d{2})?", re.I)


def parse_dnd(text, now_local, wake_at="08:00"):
    """None, ("wake", None) ou ("dnd", datetime de fin)."""
    text = text or ""
    if _WAKE.search(text):
        return "wake", None
    if not _DND.search(text):
        return None
    span = _FOR.search(text)
    if span:
        raw = span.group(1).lower()
        amount = int(raw) if raw.isdigit() else _NUM[raw]
        unit = 60 if span.group(2).lower().startswith("h") else 1
        return "dnd", now_local + dt.timedelta(minutes=amount * unit)
    until = _UNTIL.search(text)
    if until:
        end = now_local.replace(hour=int(until.group(1)) % 24, minute=int(until.group(2) or 0),
                                second=0, microsecond=0)
        if end <= now_local:
            end += dt.timedelta(days=1)
        return "dnd", end
    h, m = (int(x) for x in wake_at.split(":"))
    end = now_local.replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= now_local:
        end += dt.timedelta(days=1)
    return "dnd", end


def _say_time(moment):
    return f"{moment.hour} h" + (f" {moment.minute:02d}" if moment.minute else "")


# --- le planificateur -------------------------------------------------------------------

class Initiative:
    """État et règles. `generate(person_id, instruction) -> str|None` écrit la phrase (nanobot)."""

    def __init__(self, cfg=None, clock=time.time, generate=None, people=None, command=None):
        self.cfg = cfg or settings()
        self.clock = clock
        self.generate = generate or (lambda person, instruction: None)
        self.people = people or (lambda: {})        # {id: nom} des personnes enregistrées
        self.command = command or (lambda session, text: False)  # /compact, /dream
        self.zone = _zone(self.cfg["tz"])
        self.dir = Path(self.cfg["state_dir"])
        self.pending = []
        self._lock = threading.RLock()
        self._thread = None
        self._stop = threading.Event()

    # --- temps ---------------------------------------------------------------
    def local(self, now=None):
        stamp = self.clock() if now is None else now
        return dt.datetime.fromtimestamp(stamp, self.zone) if self.zone else dt.datetime.fromtimestamp(stamp).astimezone()

    def quiet(self, local):
        try:
            start, end = (_hm(x) for x in self.cfg["quiet"].split("-"))
        except ValueError:
            return False
        minute = local.hour * 60 + local.minute
        return (start <= minute or minute < end) if start > end else (start <= minute < end)

    # --- fichiers ----------------------------------------------------------------
    def _state(self):
        return _read_json(self.dir / "initiative_state.json", {})

    def _save_state(self, state):
        _atomic_json(self.dir / "initiative_state.json", state)

    def _person(self, state, person, today):
        entry = state.setdefault("people", {}).setdefault(person, {})
        if entry.get("day") != today:
            entry.update(day=today, count=0)
        return entry

    # --- entrées venant du sidecar --------------------------------------------------
    def record_emotion(self, person, emotion, confidence, now=None):
        """Une ligne par phrase entendue d'une personne connue (pour « tendu plusieurs jours »)."""
        if not self.cfg["enabled"] or not person or person == "invite":
            return
        local = self.local(now)
        line = json.dumps({"t": local.isoformat(timespec="seconds"), "day": local.date().isoformat(),
                           "person": person, "emotion": emotion, "confidence": round(float(confidence), 3)})
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "emotions.jsonl", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def user_turn(self, person, now=None):
        """Premier tour du jour d'une personne connue : consigne de bonjour (préfixe), sinon ''."""
        if not self.cfg["enabled"] or not person or person == "invite":
            return ""
        local = self.local(now)
        today = local.date().isoformat()
        with self._lock:
            state = self._state()
            entry = self._person(state, person, today)
            entry["seen"] = today
            first = entry.get("greeted") != today and local.hour * 60 + local.minute >= _hm(self.cfg["greeting_from"])
            if first:
                entry["greeted"] = today
            self._save_state(state)
        if not first:
            return ""
        name = self.people().get(person, person.capitalize())
        return f"(c'est la première fois aujourd'hui que {name} te parle : commence par lui dire bonjour) "

    def presence(self, person, now=None):
        """Toucher, capteur : la personne est là (déclenche ses suivis du jour), sans bonjour."""
        if not self.cfg["enabled"] or not person or person == "invite":
            return
        today = self.local(now).date().isoformat()
        with self._lock:
            state = self._state()
            self._person(state, person, today)["seen"] = today
            self._save_state(state)

    def phrase(self, text, now=None):
        """« Ne me dérange pas », « tu peux reparler » : réponse locale, sinon None."""
        if not self.cfg["enabled"]:
            return None
        local = self.local(now)
        parsed = parse_dnd(text, local, self.cfg["quiet"].split("-")[-1] if "-" in self.cfg["quiet"] else "08:00")
        if not parsed:
            return None
        kind, end = parsed
        with self._lock:
            state = self._state()
            if kind == "wake":
                state.pop("dnd_until", None)
                self._save_state(state)
                return "[happy] D'accord, je pourrai de nouveau te parler de moi-même."
            state["dnd_until"] = end.timestamp()
            self._save_state(state)
            self.pending = [p for p in self.pending if p["kind"] == "reminder"]
        when = "demain " + _say_time(end) if end.date() > local.date() else _say_time(end)
        return f"[neutral] D'accord, je ne parle plus de moi-même jusqu'à {when}. Les rappels restent actifs."

    def reminders_from_reply(self, text, person, now=None):
        """Enregistre les balises [rappel:…] d'une réponse de nanobot. Renvoie le nombre ajouté."""
        if not self.cfg["enabled"]:
            return 0
        found = parse_reminders(text, self.local(now))
        if not found:
            return 0
        with self._lock:
            path = self.dir / "reminders.json"
            items = _read_json(path, [])
            for at, what in found:
                key = hashlib.sha1(f"{at.isoformat()}|{what}|{person}".encode()).hexdigest()[:12]
                if not any(r["id"] == key for r in items):
                    items.append({"id": key, "at": at.isoformat(), "text": what, "person": person or ""})
            _atomic_json(path, items)
        return len(found)

    # --- file d'attente pour l'app --------------------------------------------------------
    def items(self, now=None):
        if not self.cfg["enabled"]:
            return []
        stamp = self.clock() if now is None else now
        with self._lock:
            self.pending = [p for p in self.pending if p["expires"] > stamp]
            return [{k: v for k, v in p.items() if k != "expires"} for p in self.pending]

    def ack(self, item_id, status, now=None):
        stamp = self.clock() if now is None else now
        with self._lock:
            item = next((p for p in self.pending if p["id"] == item_id), None)
            if item is None:
                return None
            self.pending.remove(item)
            if status == "spoken" and item["kind"] != "reminder" and item.get("person"):
                state = self._state()
                entry = self._person(state, item["person"], self.local(stamp).date().isoformat())
                entry["count"] = entry.get("count", 0) + 1
                entry["last_at"] = stamp
                state["last_at"] = stamp
                self._save_state(state)
            if status != "spoken" and item["kind"] == "followup":
                # Pas dit (occupé, veille) : on pourra le reposer plus tard.
                state = self._state()
                done = state.get("followups_done", [])
                if item["source"] in done:
                    done.remove(item["source"])
                self._save_state(state)
            return item

    def _queue(self, kind, person, text, stamp, source=""):
        key = hashlib.sha1(f"{kind}|{person}|{source}|{stamp}".encode()).hexdigest()[:10]
        expires = stamp + self.cfg["ttl_s"]
        item = {"id": key, "kind": kind, "person": person or "", "text": text,
                "expires_at": self.local(expires).isoformat(timespec="seconds"), "expires": expires,
                "source": source}
        self.pending.append(item)
        return item

    # --- un passage du planificateur --------------------------------------------------------
    def tick(self, now=None):
        if not self.cfg["enabled"]:
            return []
        stamp = self.clock() if now is None else now
        local = self.local(stamp)
        made = []
        made += self._due_reminders(stamp, local)
        self._nightly(local)
        with self._lock:
            state = self._state()
            if self.quiet(local) or state.get("dnd_until", 0) > stamp:
                return made
            if any(p["kind"] != "reminder" for p in self.pending):
                return made  # une initiative à la fois
        for person, instruction, fallback, kind, source in self._candidates(local):
            if not self._allowed(person, stamp, local):
                continue
            text = self.generate(person, instruction) or fallback
            with self._lock:
                made.append(self._queue(kind, person, text, stamp, source))
                state = self._state()
                if kind == "followup":
                    state.setdefault("followups_done", []).append(source)
                if kind == "tension":
                    self._person(state, person, local.date().isoformat())["tension_at"] = stamp
                self._save_state(state)
            break
        return made

    def _allowed(self, person, stamp, local):
        state = self._state()
        entry = state.get("people", {}).get(person, {})
        count = entry.get("count", 0) if entry.get("day") == local.date().isoformat() else 0
        if count >= self.cfg["max_per_day"]:
            return False
        return stamp - state.get("last_at", 0) >= self.cfg["min_gap_s"]

    def _due_reminders(self, stamp, local):
        made = []
        with self._lock:
            path = self.dir / "reminders.json"
            items = _read_json(path, [])
            if not items:
                return made
            keep = []
            for item in items:
                try:
                    at = dt.datetime.fromisoformat(item["at"])
                except (KeyError, ValueError):
                    continue
                if at > local:
                    keep.append(item)
                    continue
                if (local - at).total_seconds() > self.cfg["reminder_grace_s"]:
                    continue  # trop tard : jamais dit en retard
                name = self.people().get(item.get("person"), "")
                text = f"[neutral] {name + ', ' if name else ''}petit rappel : {item['text']}."
                made.append(self._queue("reminder", item.get("person"), text, stamp, item["id"]))
            _atomic_json(path, keep)
        return made

    def _candidates(self, local):
        state = self._state()
        today = local.date()
        people = self.people()
        seen = {pid for pid, e in state.get("people", {}).items() if e.get("seen") == today.isoformat()}
        done = set(state.get("followups_done", []))
        for item in self.followups():
            late = (today - item["date"]).days
            if item["id"] in done or item["person"] not in people or item["person"] not in seen:
                continue
            if 0 <= late <= self.cfg["followup_late_days"]:
                name = people[item["person"]]
                yield (item["person"],
                       f"Prends des nouvelles de {name} en une phrase, au sujet de : {item['subject']} ({item['todo']}).",
                       f"[happy] Dis-moi {name}, {item['subject']}, ça s'est passé comment ?",
                       "followup", item["id"])
        for person in sorted(seen & set(people)):
            entry = state.get("people", {}).get(person, {})
            if self._tense(person, today) and local.timestamp() - entry.get("tension_at", 0) > 7 * 86400:
                name = people[person]
                yield (person,
                       f"{name} semble tendu ces derniers jours. Une phrase douce pour lui proposer d'en parler, sans diagnostic.",
                       f"[sad] {name}, tu as l'air un peu sous pression ces jours-ci. Tu veux en parler ?",
                       "tension", f"tension-{today.isoformat()}")

    def followups(self):
        if not self.cfg["workspace"]:
            return []
        try:
            text = (Path(self.cfg["workspace"]) / "memory" / "MEMORY.md").read_text(encoding="utf-8")
        except OSError:
            return []
        return parse_followups(text)

    def _tense(self, person, today):
        stats = {}
        try:
            with open(self.dir / "emotions.jsonl", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("person") != person:
                        continue
                    day = stats.setdefault(row.get("day"), [0, 0])
                    day[0] += 1
                    if row.get("emotion") in NEGATIVE and row.get("confidence", 0) >= self.cfg["tension_confidence"]:
                        day[1] += 1
        except OSError:
            return False
        for back in range(self.cfg["tension_days"]):
            total, bad = stats.get((today - dt.timedelta(days=back)).isoformat(), [0, 0])
            if total < self.cfg["tension_min_turns"] or bad / total <= self.cfg["tension_share"]:
                return False
        return True

    # --- le soir : /compact puis /dream ------------------------------------------------------
    def _nightly(self, local):
        if not self.cfg["nightly_at"]:
            return False
        minute = local.hour * 60 + local.minute
        target = _hm(self.cfg["nightly_at"])
        today = local.date().isoformat()
        with self._lock:
            state = self._state()
            if state.get("nightly") == today or not target <= minute < target + 60:
                return False
            state["nightly"] = today
            self._save_state(state)
        for person in sorted(self.people()):
            self.command(person, "/compact")
        self.command(None, "/dream")
        return True

    # --- fil de fond ------------------------------------------------------------------------------
    def start(self):
        if not self.cfg["enabled"] or self._thread:
            return
        def run():
            while not self._stop.wait(self.cfg["tick_s"]):
                try:
                    self.tick()
                except Exception as exc:  # le planificateur ne doit jamais tomber
                    print(time.strftime("%H:%M:%S"), f"initiative tick failed: {type(exc).__name__}: {exc}", flush=True)
        self._thread = threading.Thread(target=run, name="initiative", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
