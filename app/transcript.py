"""Small pure helpers for turning a list of transcript segments into the
two download formats, plus a light text-cleanup helper used on Whisper
output. Kept separate from main.py so formatting logic isn't tangled up
with WebSocket/session handling."""

import json
import re


def format_txt(segments: list) -> str:
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{seg.get('speaker', 'Speaker')}:")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def format_json(segments: list) -> str:
    clean = [s for s in segments if s.get("text", "").strip()]
    return json.dumps(clean, ensure_ascii=False, indent=2)


_REPEAT_PHRASE_RE = re.compile(r"\b(\w+(?:\s+\w+){0,3})(\s+\1\b)+", re.IGNORECASE)


def dedupe_repeated_phrases(text: str) -> str:
    """Collapses immediate repeated word/phrase runs Whisper occasionally
    hallucinates on noisy or silent-ish audio, e.g. 'thank you thank you
    thank you' -> 'thank you'. Only collapses back-to-back repeats, never
    touches legitimately repeated words spoken further apart."""
    if not text:
        return text
    previous = None
    current = text
    # repeat until stable in case of nested repeats (rare, but cheap to handle)
    while previous != current:
        previous = current
        current = _REPEAT_PHRASE_RE.sub(r"\1", current)
    return current
