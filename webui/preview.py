"""Single-line audition clips (voice + pitch + rate)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from btlib import tts


def synth(text: str, voice: str, out_dir: Path, *, pitch: str = "+0Hz",
          rate: str = "+0%", engine: str = "edge") -> Path:
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to speak")
    if not voice:
        raise ValueError("no voice selected")

    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{engine}|{voice}|{pitch}|{rate}|{text}".encode()).hexdigest()[:16]
    out = out_dir / f"preview_{key}.mp3"
    if out.exists() and out.stat().st_size > 0:
        return out

    if engine == "kokoro":
        tts.synth_chunk_kokoro(text, voice, out, rate=rate, pitch=pitch)
    elif engine == "pyttsx3":
        tts.synth_chunk_pyttsx3(text, voice, out)
    else:
        tts.synth_chunk_edge(text, voice, out, rate=rate, pitch=pitch)
    return out
