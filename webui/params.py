"""Parameter specification for the web UI.

One entry per CLI flag.  ``build_argv`` turns the widget values back into the
exact command line the CLI would receive, so the UI never re-implements any
pipeline logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Param:
    key: str
    label: str
    kind: str                       # bool | choice | int | float | text
    default: Any = None
    flag: Optional[str] = None      # CLI flag, None for UI-only fields
    choices: List[Any] = field(default_factory=list)
    help: str = ""
    group: str = "General"
    step: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


PARAMS: List[Param] = [
    # ---------------------------------------------------------------- engine
    Param("engine", "TTS engine", "choice", "edge", "--engine",
          ["edge", "kokoro", "pyttsx3"],
          "edge = online, most natural; kokoro = offline neural; pyttsx3 = offline robotic.",
          "Engine"),
    Param("voice_profile", "Voice profile", "text", "",
          "--voice-profile", [],
          "Profile name inside voice_priority.json (novel, news, conversation).", "Engine"),
    Param("voice_priority", "Voice priority file", "text", "",
          "--voice-priority", [],
          "Optional path to a custom voice_priority.json.", "Engine"),
    Param("kokoro_model", "Kokoro model", "text", "",
          "--kokoro-model", [], "Path to kokoro-v1.0.onnx (downloads if missing).", "Engine"),
    Param("kokoro_voices", "Kokoro voices", "text", "",
          "--kokoro-voices", [], "Path to voices-v1.0.bin.", "Engine"),
    Param("kokoro_fast", "Kokoro int8 (faster)", "bool", False,
          "--kokoro-fast", [], "Smaller int8 model: faster on CPU, slightly worse quality.", "Engine"),
    Param("kokoro_threads", "Kokoro threads", "int", 0,
          "--kokoro-threads", [], "CPU threads; 0 = all cores.", "Engine"),
    Param("kokoro_device", "Kokoro device", "choice", "auto",
          "--kokoro-device", ["auto", "cpu", "cuda"], "", "Engine"),

    # ----------------------------------------------------------------- voice
    Param("voice", "Narrator voice", "text", "", "--voice", [],
          "Blank = automatic. e.g. en-US-AndrewMultilingualNeural.", "Voice"),
    Param("rate", "Rate", "text", "+0%", "--rate", [],
          "Speaking rate, e.g. +15% or -10%.", "Voice"),
    Param("dialogue_voice", "Dialogue voice (single)", "text", "",
          "--dialogue-voice", [], "Only used when multi-voice is off.", "Voice"),
    Param("dialogue_rate", "Dialogue rate", "text", "", "--dialogue-rate", [], "Edge only.", "Voice"),
    Param("dialogue_pitch", "Dialogue pitch", "text", "", "--dialogue-pitch", [], "Edge only, e.g. +10Hz.", "Voice"),
    Param("dialogue_pause_ms", "Dialogue pause (ms)", "int", 200,
          "--dialogue-pause-ms", [], "Extra silence between narration and dialogue.", "Voice"),
    Param("gap_ms", "Chunk gap (ms)", "int", 350, "--gap-ms", [], "", "Voice"),
    Param("chapter_pause_ms", "Chapter pause (ms)", "int", 900,
          "--chapter-pause-ms", [], "", "Voice"),
    Param("multi_voice", "Multi-voice (per character)", "bool", True,
          "--multi-voice", [], "Give every detected speaker their own voice.", "Voice"),
    Param("no_dialogue_voice", "Disable dialogue/narration split", "bool", False,
          "--no-dialogue-voice", [], "Read everything in one voice.", "Voice"),

    # ----------------------------------------------------------- attribution
    Param("attribution_mode", "Attribution engine", "choice", "scene", None,
          ["scene", "whole", "heuristic", "light", "heavy"],
          "scene = best on public corpora; whole = HMM; heuristic = legacy; "
          "light/heavy = model-based.", "Attribution"),
    Param("max_characters", "Max characters", "int", 12, "--max-characters", [],
          "Characters beyond this are folded into the Unknown bucket.", "Attribution"),
    Param("unknown_below", "Unknown threshold", "float", None,
          "--unknown-below", [], "Leave blank for the engine default (0.30).", "Attribution",
          step=0.05, minimum=0.0, maximum=1.0),
    Param("narrator_gender", "Narrator gender", "choice", "",
          "--narrator-gender", ["", "male", "female"],
          "Gender of a first-person narrator; blank = infer.", "Attribution"),
    Param("llm_model", "Heavy model tag", "text", "llama3:8b",
          "--llm-model", [], "Only for the 'heavy' attribution engine.", "Attribution"),
    Param("llm_endpoint", "Heavy endpoint", "text", "http://localhost:11434/api/generate",
          "--llm-endpoint", [], "Ollama / OpenAI-compatible URL.", "Attribution"),

    # ---------------------------------------------------------------- output
    Param("per_chapter", "One MP3 per chapter", "bool", False,
          "--per-chapter", [], "Writes a directory instead of a single file.", "Output"),
    Param("concurrency", "TTS concurrency", "int", 6, "--concurrency", [],
          "Parallel edge-tts requests. 6 is stable; 8-10 is often faster but can hit 403s.",
          "Output", minimum=1, maximum=16),
    Param("on_chunk_failure", "On chunk failure", "choice", "stop",
          "--on-chunk-failure", ["stop", "skip"], "", "Output"),
    Param("fresh", "Ignore saved progress", "bool", False,
          "--fresh", [], "Re-synthesize everything (deletes cached chunks).", "Output"),
    Param("keep_workdir", "Keep workdir after run", "bool", False,
          "--keep-workdir", [], "", "Output"),
    Param("min_chapter_words", "Merge chapters below (words)", "int", 150,
          "--min-chapter-words", [], "", "Output"),
]

BY_KEY: Dict[str, Param] = {p.key: p for p in PARAMS}
GROUPS = ["Engine", "Voice", "Attribution", "Output"]

_MODE_FLAGS = {
    "scene": "--scene-context",
    "whole": "--whole-book-context",
    "light": "--light-model",
    "heavy": "--heavy-model",
}

# Flags the UI owns and never exposes as free text.
_RESERVED = {"--cast-file", "--workdir", "--chapters", "-o", "--output"}


def build_argv(values: Dict[str, Any], *, extra: str = "") -> List[str]:
    """Turn widget values into CLI tokens (without the program name)."""
    argv: List[str] = []
    mode = values.get("attribution_mode") or "scene"
    if mode in _MODE_FLAGS:
        argv.append(_MODE_FLAGS[mode])

    for p in PARAMS:
        if p.key == "attribution_mode" or p.flag is None:
            continue
        if p.flag in _RESERVED:
            continue
        v = values.get(p.key, p.default)
        if p.kind == "bool":
            if v:
                argv.append(p.flag)
            continue
        if v is None or v == "":
            continue
        if p.kind == "float":
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
        argv.extend([p.flag, str(v)])

    if extra.strip():
        import shlex
        argv.extend(shlex.split(extra))
    return argv


def default_values() -> Dict[str, Any]:
    """A fresh set of UI values for one book."""
    vals = {p.key: p.default for p in PARAMS}
    vals["narrator_gender"] = ""
    vals["voice_profile"] = ""
    vals["fresh"] = False
    return vals


def _coerce(p: Param, value: Any) -> Any:
    if p.kind == "bool":
        return str(value).lower() not in ("0", "false", "no", "off", "")
    if p.kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return p.default
    if p.kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return value


def parse_argv(argv: List[str]) -> Dict[str, Any]:
    """Best-effort inverse of :func:`build_argv` for saved jobs.

    This lets a book reopened from an old ``meta.json`` keep the settings that
    produced its saved chunks (especially ``max_characters`` and ``rate``).
    """
    vals = default_values()
    by_flag = {p.flag: p for p in PARAMS if p.flag}
    mode_flags = {v: k for k, v in _MODE_FLAGS.items()}
    skip = _RESERVED | {"--analyze-only", "--list-chapters", "--list-voices", "--sample"}
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        if tok in mode_flags:
            vals["attribution_mode"] = mode_flags[tok]
            i += 1
            continue
        if tok == "--fresh":
            i += 1
            continue
        if tok.startswith("--") and "=" in tok:
            flag, value = tok.split("=", 1)
            p = by_flag.get(flag)
            if p:
                vals[p.key] = _coerce(p, value)
            i += 1
            continue
        if tok in by_flag:
            p = by_flag[tok]
            if p.kind == "bool":
                vals[p.key] = True
                i += 1
            elif i + 1 < len(argv):
                vals[p.key] = _coerce(p, argv[i + 1])
                i += 2
            else:
                i += 1
            continue
        if tok in skip:
            i += 2 if i + 1 < len(argv) else 1
            continue
        i += 1

    vals["fresh"] = False
    if vals.get("narrator_gender") in (None, "auto"):
        vals["narrator_gender"] = ""
    return vals
