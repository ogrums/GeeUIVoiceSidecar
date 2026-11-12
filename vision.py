"""Vision on demand: a photo sent with a chat turn becomes a short text description.

The app adds one JPEG to the user message, OpenAI style:
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
The sidecar asks a vision model on Lemonade (VISION_MODEL) for a description, then
replaces the image with text: "(RUX voit : …) qu'est-ce que tu vois ?". nanobot and the
fast chat model stay text-only. The photo is never written to disk and never logged.
"""

import json
import os
from urllib.request import Request, urlopen

DEFAULTS = {
    # Empty = no vision: the image is dropped and nanobot is told the eyes are off.
    "VISION_MODEL": "",
    # Empty = LEMONADE_HOST. Any OpenAI-compatible server that accepts image_url works.
    "VISION_URL": "",
    "VISION_API_KEY": "",
    "VISION_TIMEOUT": "60",
    "VISION_MAX_TOKENS": "200",
    "VISION_PROMPT": (
        "Tu es les yeux d'un petit robot de bureau. Décris en français, en deux ou trois "
        "phrases simples, ce que montre la photo : les objets, les personnes (sans deviner "
        "qui c'est), les couleurs. S'il y a du texte lisible, recopie-le exactement. "
        "Réponds aussi à la question si elle porte sur l'image. Question : {question}"
    ),
}

BLIND = "(RUX a voulu regarder mais n'a pas pu : {why}) "
SEEN = "(RUX voit : {what}) "


def env(name):
    raw = os.environ.get(name)
    return DEFAULTS[name] if raw is None or not raw.strip() else raw.strip()


def settings(lemonade_host=""):
    return {
        "model": env("VISION_MODEL"),
        "url": (env("VISION_URL") or lemonade_host).rstrip("/"),
        "key": env("VISION_API_KEY"),
        "timeout": float(env("VISION_TIMEOUT")),
        "max_tokens": int(env("VISION_MAX_TOKENS")),
        "prompt": env("VISION_PROMPT"),
    }


def split_user(message):
    """(text, [image urls]) of one OpenAI user message."""
    content = message.get("content")
    if not isinstance(content, list):
        return (content if isinstance(content, str) else ""), []
    texts, images = [], []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(part.get("text") or "")
        elif part.get("type") == "image_url":
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if isinstance(url, str) and url:
                images.append(url)
    return "".join(texts), images


def last_user(messages):
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return None


def has_image(data):
    message = last_user(data.get("messages"))
    return bool(message) and bool(split_user(message)[1])


def describe(image_url, question, cfg, post=None):
    """Text description from the vision model. Raises on any error."""
    payload = {
        "model": cfg["model"],
        "stream": False,
        "max_tokens": cfg["max_tokens"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": cfg["prompt"].replace("{question}", question or "qu'est-ce que tu vois ?")},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
    }
    raw = (post or _post)(cfg["url"] + "/chat/completions", payload, cfg)
    text = json.loads(raw)["choices"][0]["message"].get("content") or ""
    text = " ".join(text.replace("(", "").replace(")", "").split())
    if not text:
        raise ValueError("empty description")
    return text[:800]


def _post(url, payload, cfg):
    headers = {"Content-Type": "application/json"}
    if cfg["key"]:
        headers["Authorization"] = "Bearer " + cfg["key"]
    request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=cfg["timeout"]) as response:
        return response.read()


def rewrite(body, lemonade_host="", post=None, log=print):
    """Return the request body with the last user image replaced by its description.

    Bodies without an image come back unchanged (same bytes)."""
    try:
        data = json.loads(body.decode() or "{}")
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(data, dict) or not has_image(data):
        return body
    message = last_user(data["messages"])
    question, images = split_user(message)
    cfg = settings(lemonade_host)
    if not cfg["model"]:
        prefix = BLIND.format(why="la vue n'est pas branchée")
        log("vision off: VISION_MODEL empty, image dropped")
    else:
        try:
            what = describe(images[-1], question.strip(), cfg, post)
            prefix = SEEN.format(what=what)
            log(f"vision model={cfg['model']} chars={len(what)}")
        except Exception as exc:  # noqa: BLE001 - any failure means "could not see"
            prefix = BLIND.format(why="ma caméra ou mon modèle de vision ne répond pas")
            log(f"vision failed: {type(exc).__name__}: {str(exc)[:120]}")
    for msg in data["messages"]:
        if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), list):
            # Older images too: nothing downstream needs the pixels.
            msg["content"] = split_user(msg)[0]
    message["content"] = prefix + (question.strip() or "Qu'est-ce que tu vois ?")
    return json.dumps(data, ensure_ascii=False).encode()
