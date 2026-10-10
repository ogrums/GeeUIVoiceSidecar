"""Routing and CosyVoice3 instruct strings. No torch, no network."""

MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODEL_NAME = "Fun-CosyVoice3-0.5B-2512"
MAX_CHARS = 180
LABELS = {"neutral", "happy", "sad", "angry", "fear", "surprise"}

# CosyVoice3 inference_instruct2 prefix. The clip after <|endofprompt|> is the line to speak.
INSTRUCT = {
    "happy": "You are a helpful assistant. 请用尽可能开心的语气说话。<|endofprompt|>",
    "sad": "You are a helpful assistant. 请用悲伤、低落的语气说话。<|endofprompt|>",
    "angry": "You are a helpful assistant. 请用愤怒的语气说话。<|endofprompt|>",
    "fear": "You are a helpful assistant. 请用害怕、不安的语气说话。<|endofprompt|>",
    "surprise": "You are a helpful assistant. 请用惊讶的语气说话。<|endofprompt|>",
    "neutral": "You are a helpful assistant. 请用平静的语气说话。<|endofprompt|>",
}

ALIASES = {
    "cheerful": "happy",
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "afraid": "fear",
    "fearful": "fear",
    "surprised": "surprise",
}


def parse_label(raw):
    key = ALIASES.get((raw or "").strip().lower(), (raw or "").strip().lower())
    return key if key in LABELS else "neutral"


def instruct_for(emotion):
    return INSTRUCT[parse_label(emotion)]


def route(text, emotion, cosy_enabled):
    """Same rule as TtsRoute. A non-neutral line uses CosyVoice when the sidecar is up.

    Length does not pick the engine. The client clips to MAX_CHARS before POST.
    """
    label = parse_label(emotion)
    if cosy_enabled and label != "neutral" and text.strip():
        return {
            "engine": "cosyvoice",
            "model": MODEL_NAME,
            "voice": label,
        }
    return {"engine": "kokoro", "model": "kokoro-v1", "voice": ""}
