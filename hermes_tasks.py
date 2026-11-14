"""Tâches confiées à Hermes Agent depuis RUX. Stdlib only.

L'appli détecte « confie à Hermes … » après la STT et appelle POST /hermes/tasks.
Le sidecar crée un run Hermes (API Runs, compatible OpenAI), garde la liste des
tâches dans un fichier JSON, suit leur statut, et fabrique un court résumé à dire
à voix haute (modèle de conversation via Lemonade) quand on demande le résultat.

Désactivé tant que HERMES_URL est vide. La clé vient de HERMES_API_KEY (jamais
écrite dans les journaux ni renvoyée par l'API).
"""

import json
import os
import re
import tempfile
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULTS = {
    "HERMES_URL": "",
    "HERMES_API_KEY": "",
    "HERMES_TASKS_FILE": "~/.local/share/geeui/hermes_tasks.json",
    "HERMES_TASKS_TOKEN": "",
    "HERMES_POLL_SECONDS": "10",
    "HERMES_TIMEOUT": "15",
    "HERMES_MAX_TASKS": "50",
    "HERMES_SUMMARY_MODEL": "",
    "HERMES_SUMMARY_TIMEOUT": "60",
}

DONE = {"completed", "succeeded"}
FAILED = {"failed", "cancelled", "canceled", "stopped", "error", "expired"}

PROMPT_TEMPLATE = (
    "Tâche confiée par RUX (le robot) à la demande de son utilisateur :\n"
    "{prompt}\n\n"
    "Consignes :\n"
    "- Utilise l'outil terminal (et les outils fichiers) pour obtenir les faits. "
    "Exécute réellement les commandes, ne devine pas leur sortie.\n"
    "- N'invente rien. Si tu ne peux pas vérifier une information, dis-le clairement.\n"
    "- Dossiers : /workspace/data (lecture et écriture), /workspace/projects (lecture seule, s'il existe).\n"
    "- Termine par un résultat clair et concis, en français."
)

SUMMARY_SYSTEM = (
    "Tu es RUX, un petit robot de bureau. On te donne le résultat d'une tâche faite par "
    "l'agent Hermes. Résume-le en une à trois phrases courtes, en français, faciles à dire "
    "à voix haute : pas de markdown, pas de liste, pas de code, pas de chemin de fichier long. "
    "N'ajoute aucune information absente du résultat. Si la tâche a échoué, dis-le simplement."
)


def env(name):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return DEFAULTS[name]
    return raw.strip()


def _int(name):
    try:
        return max(1, int(float(env(name))))
    except ValueError:
        return int(DEFAULTS[name])


def enabled():
    return bool(env("HERMES_URL"))


# --- Intention vocale -------------------------------------------------------

def _fold(text):
    """Minuscules sans accents, pour comparer des transcriptions Whisper."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().replace("’", "'")


_HERMES = r"(?:hermes|herm[eè]s|air ?mess|erm[eè]s|hermesse)"
_CREATE = re.compile(
    r"^\W*(?:(?:rux|dobby|hey dobby|hi rux)\W+)?"
    r"(?:confie|confier|demande|demander|delegue|deleguer|passe|donne)\s+"
    r"(?:(?:ca|cela|une tache|la tache)\s+)?(?:a|au)\s+" + _HERMES + r"\b\W*"
    r"(?:(?:de|d'|qu'il|qu'elle|que|pour)\s*)?(?P<prompt>.*)$"
)
# Repli tolérant aux erreurs de transcription : « Koffee a Hermès de compter… »,
# « … à Hermès, qu'il liste … ». Le verbe peut être mal entendu, « à Hermès de » suffit.
_CREATE_LOOSE = re.compile(
    r"\b(?:a|au)\s+" + _HERMES + r"\b\W*(?:de\s+|d'\s*|qu'il\s+|qu'elle\s+|pour\s+)(?P<prompt>.+)$"
)
_READ_HERMES = re.compile(r"\bresultats?\b.*\b" + _HERMES + r"\b|\b" + _HERMES + r"\b.*\bresultats?\b")
_READ_SHORT = re.compile(
    r"^\W*(?:(?:rux|dobby)\W+)?(?:lis|lire|dis|donne|raconte)[- ]?(?:moi\s+)?"
    r"(?:le|les|son|ses)?\s*resultats?\W*$"
)
_LEAD = re.compile(r"^(?:d['’]\s*|de\s+|qu['’]il\s+|qu['’]elle\s+|que\s+|pour\s+)", re.I)
_STATUS = re.compile(
    r"\bou en est " + _HERMES + r"\b|\b" + _HERMES + r"\s+a\s+(?:fini|termine)\b"
    r"|\btaches?\s+(?:de\s+|d'\s*)?" + _HERMES + r"\b"
)


def parse_intent(text):
    """{"intent": "create"|"read"|"status"|None, "prompt": str|None} depuis une phrase."""
    folded = _fold(text).strip()
    match = _CREATE.match(folded)
    if match:
        # Garder la casse et les accents d'origine : couper la phrase au même nombre de mots.
        rest = match.group("prompt").strip()
        prompt = _tail_original(text, rest) if rest else ""
        prompt = _LEAD.sub("", prompt.strip(" ,.;:!?"))
        return {"intent": "create", "prompt": prompt or None}
    if _READ_HERMES.search(folded) and not _STATUS.search(folded) or _READ_SHORT.match(folded):
        return {"intent": "read", "prompt": None}
    if _STATUS.search(folded):
        return {"intent": "status", "prompt": None}
    loose = _CREATE_LOOSE.search(folded)
    if loose:
        rest = loose.group("prompt").strip()
        prompt = _LEAD.sub("", _tail_original(text, rest).strip(" ,.;:!?…"))
        return {"intent": "create", "prompt": prompt or None}
    # Phrase très courte qui nomme Hermes (« Hermès ? », « ou un Hermès ») : son état.
    if re.search(r"\b" + _HERMES + r"\b", folded) and len(re.findall(r"\w+", folded)) <= 4:
        return {"intent": "status", "prompt": None}
    return {"intent": None, "prompt": None}


def _tail_original(original, folded_tail):
    words = (original or "").split()
    count = len(folded_tail.split())
    return " ".join(words[-count:]) if count else ""


# --- Stockage ---------------------------------------------------------------

class TaskStore:
    """Liste persistante des tâches (JSON, écriture atomique)."""

    def __init__(self, path=None):
        self.path = Path(os.path.expanduser(path or env("HERMES_TASKS_FILE")))
        self.lock = threading.RLock()
        self.tasks = {}
        self._load()

    def _load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.tasks = {t["id"]: t for t in data.get("tasks", []) if "id" in t}
        except (OSError, ValueError):
            self.tasks = {}

    def save(self):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            ordered = sorted(self.tasks.values(), key=lambda t: t.get("created", 0))
            ordered = ordered[-_int("HERMES_MAX_TASKS"):]
            self.tasks = {t["id"]: t for t in ordered}
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".hermes-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"tasks": ordered}, handle, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)

    def add(self, prompt):
        task = {
            "id": "t" + uuid.uuid4().hex[:10],
            "run_id": None,
            "prompt": prompt,
            "status": "queued",
            "created": time.time(),
            "finished": None,
            "result": None,
            "error": None,
            "read": False,
            "summary": None,
        }
        with self.lock:
            self.tasks[task["id"]] = task
            self.save()
        return task

    def get(self, task_id):
        with self.lock:
            if task_id == "latest":
                return self.latest()
            task = self.tasks.get(task_id)
            return dict(task) if task else None

    def update(self, task_id, **fields):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            task.update(fields)
            self.save()
            return dict(task)

    def latest(self):
        """Dernière tâche terminée non lue, sinon la dernière terminée, sinon la dernière."""
        with self.lock:
            items = sorted(self.tasks.values(), key=lambda t: t.get("created", 0), reverse=True)
            for wanted in (
                lambda t: is_finished(t) and not t.get("read"),
                is_finished,
                lambda t: True,
            ):
                for task in items:
                    if wanted(task):
                        return dict(task)
        return None

    def listing(self):
        with self.lock:
            items = sorted(self.tasks.values(), key=lambda t: t.get("created", 0), reverse=True)
            tasks = [brief(t) for t in items]
        return {
            "enabled": enabled(),
            "unread": sum(1 for t in tasks if is_finished(t) and not t["read"]),
            "running": sum(1 for t in tasks if not is_finished(t)),
            "tasks": tasks,
        }


def is_finished(task):
    return task.get("status") in ("completed", "failed")


def brief(task):
    out = dict(task)
    if out.get("result") and len(out["result"]) > 500:
        out["result"] = out["result"][:500] + "…"
    return out


# --- Client Hermes ----------------------------------------------------------

class HermesError(Exception):
    pass


def _hermes(method, path, payload=None, timeout=None):
    base = env("HERMES_URL").rstrip("/")
    if not base:
        raise HermesError("HERMES_URL non défini")
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    key = env("HERMES_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = Request(base + path, data=data, method=method, headers=headers)
    try:
        with urlopen(request, timeout=timeout or _int("HERMES_TIMEOUT")) as response:
            return json.loads(response.read() or b"{}")
    except HTTPError as exc:
        raise HermesError(f"HTTP {exc.code}") from None
    except (URLError, OSError, ValueError) as exc:
        raise HermesError(type(exc).__name__ + ": " + str(getattr(exc, "reason", exc))[:120]) from None


def hermes_prompt(prompt):
    return PROMPT_TEMPLATE.format(prompt=prompt.strip())


def submit(store, prompt, log=print):
    task = store.add(prompt)
    try:
        answer = _hermes("POST", "/runs", {"input": hermes_prompt(prompt)})
    except HermesError as exc:
        log(f"hermes create failed {exc}")
        return store.update(task["id"], status="failed", error=str(exc), finished=time.time())
    run_id = answer.get("run_id") or answer.get("id")
    if not run_id:
        return store.update(task["id"], status="failed", error="réponse Hermes sans run_id",
                            finished=time.time())
    log(f"hermes task {task['id']} run={run_id}")
    task = store.update(task["id"], run_id=run_id, status="running")
    _apply(store, task, answer)
    return store.get(task["id"])


def _output_text(output):
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def _apply(store, task, run):
    status = (run.get("status") or "").lower()
    if status in DONE:
        return store.update(task["id"], status="completed", result=_output_text(run.get("output")),
                            finished=time.time())
    if status in FAILED:
        error = run.get("error") or run.get("last_error") or status
        return store.update(task["id"], status="failed", error=_output_text(error)[:300],
                            result=_output_text(run.get("output")) or None, finished=time.time())
    return task


def refresh(store, log=print):
    """Interroge Hermes pour les tâches en cours. Renvoie le nombre de tâches terminées."""
    with store.lock:
        pending = [dict(t) for t in store.tasks.values() if t.get("status") == "running" and t.get("run_id")]
    finished = 0
    for task in pending:
        try:
            run = _hermes("GET", "/runs/" + task["run_id"])
        except HermesError as exc:
            if "HTTP 404" in str(exc):
                store.update(task["id"], status="failed", error="run inconnu de Hermes",
                             finished=time.time())
                finished += 1
            continue
        updated = _apply(store, task, run)
        if updated and is_finished(updated):
            finished += 1
            log(f"hermes task {task['id']} {updated['status']}")
    return finished


def stop(store, task_id):
    task = store.get(task_id)
    if not task:
        return None
    if task.get("run_id") and not is_finished(task):
        try:
            _hermes("POST", "/runs/" + task["run_id"] + "/stop", {})
        except HermesError:
            pass
        task = store.update(task["id"], status="failed", error="arrêtée", finished=time.time())
    return task


# --- Résumé parlé -----------------------------------------------------------

_MARKDOWN = re.compile(r"[`*_#>|]+")


def plain(text, limit=300):
    text = _MARKDOWN.sub("", text or "")
    # Une ligne = une phrase à l'oral (listes, résultats ligne par ligne).
    text = re.sub(r"([^.!?:;,\s])[ \t]*\n+", r"\1. ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: end + 1] if end > limit // 3 else cut.rstrip() + "…")


def spoken_summary(task, lemonade_host, chat_model, post=None):
    """Phrase courte à lire pour une tâche (résumé par le modèle de conversation)."""
    status = task.get("status")
    if status not in ("completed", "failed"):
        return "Hermes travaille encore sur : " + plain(task.get("prompt"), 120) + "."
    if status == "failed" and not task.get("result"):
        return "La tâche confiée à Hermes n'a pas abouti."
    result = task.get("result") or ""
    if not result.strip():
        return "Hermes a terminé, mais sans rien répondre."
    payload = {
        "model": env("HERMES_SUMMARY_MODEL") or chat_model,
        "stream": False,
        "max_tokens": 160,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": (
                f"Tâche : {task.get('prompt')}\nStatut : {status}\n"
                f"Résultat de Hermes :\n{result[:4000]}"
            )},
        ],
    }
    post = post or _lemonade_post
    try:
        answer = post(lemonade_host.rstrip("/") + "/chat/completions", payload)
        text = answer["choices"][0]["message"]["content"]
        text = plain(re.sub(r"<think>.*?</think>", "", text or "", flags=re.S), 400)
        if text:
            return text
    except Exception:
        pass
    return plain(result, 300)


def _lemonade_post(url, payload):
    request = Request(url, data=json.dumps(payload).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=_int("HERMES_SUMMARY_TIMEOUT")) as response:
        return json.loads(response.read())


# --- Poller ------------------------------------------------------------------

_store = None
_store_lock = threading.Lock()
_last_refresh = [0.0]


def store():
    global _store
    with _store_lock:
        wanted = Path(os.path.expanduser(env("HERMES_TASKS_FILE")))
        if _store is None or _store.path != wanted:
            _store = TaskStore(wanted)
        return _store


def maybe_refresh(log=print, force=False):
    if not enabled():
        return
    now = time.time()
    if force or now - _last_refresh[0] >= _int("HERMES_POLL_SECONDS"):
        _last_refresh[0] = now
        refresh(store(), log=log)


def start_poller(log=print):
    """Fil de fond : suit les tâches en cours même si l'appli ne demande rien."""
    if not enabled():
        return None

    def loop():
        while True:
            try:
                maybe_refresh(log=log, force=True)
            except Exception as exc:  # garder le fil vivant
                log(f"hermes poll error {type(exc).__name__}")
            time.sleep(_int("HERMES_POLL_SECONDS"))

    thread = threading.Thread(target=loop, name="hermes-poll", daemon=True)
    thread.start()
    log(f"hermes tasks on, poll {_int('HERMES_POLL_SECONDS')}s, file {store().path}")
    return thread


def public_config():
    return {"hermes_enabled": enabled(), "hermes_url": env("HERMES_URL"),
            "hermes_key_set": bool(env("HERMES_API_KEY")),
            "hermes_token_required": bool(env("HERMES_TASKS_TOKEN"))}


# --- Routes HTTP ------------------------------------------------------------

def _authorized(handler):
    token = env("HERMES_TASKS_TOKEN")
    if not token:
        return True
    given = (handler.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
    given = given or (handler.headers.get("X-Hermes-Token") or "").strip()
    if given == token:
        return True
    handler._json(401, {"error": "jeton Hermes invalide"})
    return False


def route(handler, method, path, body=b"", lemonade_host="", chat_model="", log=print):
    """True si la requête concernait /hermes/… (réponse déjà envoyée)."""
    if path != "/hermes" and not path.startswith("/hermes/"):
        return False
    if not _authorized(handler):
        return True
    parts = [p for p in path.split("/") if p][1:]
    data = {}
    if method == "POST" and body:
        try:
            data = json.loads(body.decode() or "{}")
        except (ValueError, UnicodeDecodeError):
            handler._json(400, {"error": "bad json"})
            return True
    if parts == ["intent"] and method == "POST":
        handler._json(200, parse_intent(data.get("text") or ""))
        return True
    if not enabled():
        handler._json(503, {"error": "Hermes désactivé (HERMES_URL vide)", "enabled": False})
        return True
    tasks = store()
    if parts == ["tasks"] and method == "GET":
        maybe_refresh(log=log)
        handler._json(200, tasks.listing())
        return True
    if parts == ["tasks"] and method == "POST":
        prompt = (data.get("prompt") or "").strip()
        if not prompt and data.get("text"):
            intent = parse_intent(data["text"])
            prompt = (intent["prompt"] or "") if intent["intent"] == "create" else ""
        if not prompt:
            handler._json(400, {"error": "prompt vide"})
            return True
        task = submit(tasks, prompt[:2000], log=log)
        code = 201 if task["status"] != "failed" else 502
        handler._json(code, task)
        return True
    if len(parts) >= 2 and parts[0] == "tasks":
        if parts[1] != "latest":
            maybe_refresh(log=log)
        task = tasks.get(parts[1])
        if parts[1] == "latest":
            maybe_refresh(log=log, force=True)
            task = tasks.get("latest")
        if not task:
            handler._json(404, {"error": "tâche inconnue"})
            return True
        action = parts[2] if len(parts) > 2 else ""
        if method == "GET" and not action:
            handler._json(200, task)
            return True
        if method == "POST" and action == "read":
            handler._json(200, tasks.update(task["id"], read=True))
            return True
        if method == "POST" and action == "stop":
            handler._json(200, stop(tasks, task["id"]))
            return True
        if method == "POST" and action == "speak":
            text = task.get("summary") if is_finished(task) else None
            if not text:
                text = spoken_summary(task, lemonade_host, chat_model)
                if is_finished(task):
                    tasks.update(task["id"], summary=text)
            if is_finished(task) and data.get("mark_read", True):
                tasks.update(task["id"], read=True)
            handler._json(200, {"id": task["id"], "status": task["status"], "text": text})
            return True
    handler._json(404, {"error": "not found"})
    return True
