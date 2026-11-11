"""Skills learned by voice: nanobot writes drafts, a human approves them on a small page.

nanobot writes `brouillons/<nom>/SKILL.md` in its workspace (not loaded as a skill). The
page /admin/skills lists the drafts; "Valider" moves the folder to `skills/<nom>/` (nanobot
reads it on the next turn), "Refuser" deletes it. A draft that uses an action tag outside
the allowed list cannot be approved. Protected by SIDECAR_ADMIN_KEY (HTTP Basic password
or Bearer); no key = page off.
"""

import base64
import hmac
import html
import os
import re
import shutil
from pathlib import Path

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
TAG = re.compile(r"\[([a-zA-Z_]+)(?::([^\]\n]{0,120}))?\](?!\()")

EMOTIONS = {"happy", "sad", "angry", "fear", "surprise", "neutral"}
NUANCES = {"love", "heart", "laugh", "wink", "cry", "thinking", "sleepy", "cold", "hot"}
LIGHTS = {"red", "green", "blue", "orange", "white", "yellow", "purple", "cyan", "off"}
ACTIONS = {
    "move": {"forward", "back"},
    "volume": {"up", "down"},
    "light": LIGHTS,
}
PLAIN = EMOTIONS | NUANCES | {"photo", "home"}
# Tags the sidecar itself reads in replies (reminders, forget voice) are fine in a skill.
SIDECAR = {"rappel", "oubli"}


def settings():
    root = os.environ.get("NANOBOT_WORKSPACE", "").strip()
    return {
        "workspace": Path(root) if root else None,
        "key": os.environ.get("SIDECAR_ADMIN_KEY", "").strip(),
        "drafts": os.environ.get("SKILL_DRAFTS_DIR", "brouillons").strip() or "brouillons",
    }


def bad_tags(text):
    """Tags a skill may not use: anything outside AGENTS.md's action list."""
    bad = []
    for match in TAG.finditer(text):
        name, arg = match.group(1).lower(), match.group(2)
        if arg is None and name in PLAIN:
            continue
        if arg is not None and name in ACTIONS and arg.strip().lower() in ACTIONS[name]:
            continue
        if arg is not None and name in SIDECAR:
            continue
        bad.append(match.group(0))
    return sorted(set(bad))


def problems(text):
    out = []
    if not re.search(r"^---\s*\n(?:.*\n)*?name:\s*\S+", text, re.M):
        out.append("en-tête manquant (--- name: … description: … ---)")
    tags = bad_tags(text)
    if tags:
        out.append("balises interdites : " + ", ".join(tags))
    return out


class Skills:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings()

    @property
    def enabled(self):
        return bool(self.cfg["key"]) and self.cfg["workspace"] is not None

    def authorized(self, header):
        key = self.cfg["key"]
        if not key or not header:
            return False
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer":
            given = value.strip()
        elif scheme.lower() == "basic":
            try:
                given = base64.b64decode(value.strip()).decode().partition(":")[2]
            except Exception:  # noqa: BLE001
                return False
        else:
            return False
        return hmac.compare_digest(given.encode(), key.encode())

    def _dir(self, kind):
        base = self.cfg["workspace"]
        return base / (self.cfg["drafts"] if kind == "draft" else "skills")

    def drafts(self):
        root = self._dir("draft")
        out = []
        if not root.is_dir():
            return out
        for folder in sorted(root.iterdir()):
            skill = folder / "SKILL.md"
            if folder.is_dir() and NAME.match(folder.name) and skill.is_file():
                text = skill.read_text(encoding="utf-8", errors="replace")[:20000]
                out.append({
                    "name": folder.name,
                    "text": text,
                    "problems": problems(text),
                    "exists": (self._dir("skill") / folder.name).exists(),
                })
        return out

    def approve(self, name):
        """(ok, message). Moves brouillons/<name> to skills/<name>."""
        if not NAME.match(name or ""):
            return False, "nom invalide"
        src = self._dir("draft") / name
        dst = self._dir("skill") / name
        skill = src / "SKILL.md"
        if not skill.is_file():
            return False, "brouillon introuvable"
        issues = problems(skill.read_text(encoding="utf-8", errors="replace"))
        if issues:
            return False, "; ".join(issues)
        if dst.exists():
            return False, f"un skill « {name} » existe déjà"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True, f"« {name} » est actif"

    def reject(self, name):
        if not NAME.match(name or ""):
            return False, "nom invalide"
        src = self._dir("draft") / name
        if not src.is_dir():
            return False, "brouillon introuvable"
        shutil.rmtree(src)
        return True, f"« {name} » supprimé"

    def page(self, message=""):
        rows = []
        for d in self.drafts():
            name = html.escape(d["name"])
            issues = "".join(f"<li>{html.escape(p)}</li>" for p in d["problems"])
            if d["exists"]:
                issues += "<li>un skill de ce nom existe déjà</li>"
            blocked = bool(d["problems"]) or d["exists"]
            approve = (
                "" if blocked else
                f'<form method="post" action="/admin/skills/{name}/approve"><button>Valider</button></form>'
            )
            rows.append(
                f"<section><h2>{name}</h2>"
                + (f'<ul class="ko">{issues}</ul>' if issues else "")
                + f"<pre>{html.escape(d['text'])}</pre>{approve}"
                f'<form method="post" action="/admin/skills/{name}/reject"><button>Refuser</button></form>'
                "</section>"
            )
        body = "".join(rows) or "<p>Aucun brouillon.</p>"
        note = f'<p class="msg">{html.escape(message)}</p>' if message else ""
        return (
            "<!doctype html><html lang=fr><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width'>"
            "<title>RUX : skills à valider</title><style>"
            "body{font-family:sans-serif;max-width:50rem;margin:1rem auto;padding:0 1rem}"
            "pre{white-space:pre-wrap;background:#f4f4f4;padding:.5rem}form{display:inline}"
            "button{margin:.2rem;padding:.4rem 1rem}.ko{color:#b00}.msg{background:#eef;padding:.5rem}"
            "</style><h1>Skills appris à la voix</h1>" + note + body + "</html>"
        ).encode()
