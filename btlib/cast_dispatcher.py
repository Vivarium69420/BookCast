"""Cast dispatcher: picks the attribution engine, caches the result, and tells
the pipeline where the cast file lives.

This module was reconstructed (its source was lost). The two public functions
are called from ``pipeline.py``::

    generate_cast(in_path, book, args) -> dict
    resolve_cast_file(in_path, args)   -> Path

Engine selection mirrors the CLI:

    --heavy-model          -> btlib.cast_llm.analyze_book_llm (scene + LLM)
    --light-model          -> btlib.cast_light.analyze_book_light  (HF QA model)
    --whole-book-context   -> btlib.cast_whole.analyze_book_whole  (turn-taking HMM)
    (default)              -> btlib.cast.analyze_book              (regex heuristic)

The heavy/light modules are imported lazily so a plain heuristic run never pays
for torch or a network dependency.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from . import tts
from .cast import (analyze_book, assign_voices, cast_path_for,
                   ensure_narrator_character)

# Bumped whenever the cast schema or the cache-key inputs change. v17 adds the
# LLM reconciliation / weak-line adjudication engine and stores its usage;
# v18 adds the first-appearance/gender fallbacks in the scene engine; v19 adds
# paragraph-scoped tags and the expanded speech-verb/contraction handling;
# v20 adds an editable, voice-reserved narrator to generated casts.
CAST_VERSION = 20


# --------------------------------------------------------------------------
# Engine selection
# --------------------------------------------------------------------------

def select_mode(args) -> str:
    """Return the short engine tag: 'heavy', 'light', 'whole', 'scene' or
    'heuristic'."""
    if getattr(args, "heavy_model", False):
        return "heavy"
    if getattr(args, "light_model", False):
        return "light"
    if getattr(args, "whole_book_context", False):
        return "whole"
    if getattr(args, "scene_context", False):
        return "scene"
    return "heuristic"


def _mode_label(args, mode: str) -> str:
    """The human-readable ``mode`` string stored in the cast file."""
    if mode == "heavy":
        strategy = getattr(args, "llm_strategy", None) or "weak"
        return f"heavy-llm ({getattr(args, 'llm_model', 'unknown')}, {strategy})"
    if mode == "light":
        return "lightweight-transformer"
    if mode == "whole":
        return "whole-book"
    if mode == "scene":
        return "scene"
    return "heuristic"


def _heavy_strategy(args) -> str:
    strategy = getattr(args, "llm_strategy", None) or "weak"
    return strategy if strategy in {"cast", "weak", "all"} else "weak"


def _cache_tag(args, mode: str) -> str:
    """The short tag that also names the on-disk cast file."""
    if mode == "heavy":
        try:
            from .cast_llm import LLM_PROMPT_VERSION
        except Exception:  # pragma: no cover - import should always work
            LLM_PROMPT_VERSION = 1
        max_calls = getattr(args, "llm_max_calls", None)
        return (f"heavy:{getattr(args, 'llm_model', 'unknown')}:"
                f"{_heavy_strategy(args)}:p{LLM_PROMPT_VERSION}:"
                f"b{int(getattr(args, 'llm_batch_size', None) or 8)}:"
                f"c{float(getattr(args, 'llm_confidence', None) or 0.6):.2f}:"
                f"n{max_calls or 'all'}")
    return {"light": "light", "whole": "whole", "scene": "scene", "heuristic": "heuristic"}[mode]


def _input_signature(in_path: Path) -> dict:
    stat = in_path.stat()
    return {
        "path": str(in_path.resolve()),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def _cache_key(in_path: Path, args, mode: str) -> dict:
    return {
        "input": _input_signature(in_path),
        "max_characters": getattr(args, "max_characters", None),
        "narrator_gender": getattr(args, "narrator_gender", None),
        "voice_bank": tts.voice_bank_signature(),
        "version": CAST_VERSION,
        "tag": _cache_tag(args, mode),
    }


# --------------------------------------------------------------------------
# Building the cast
# --------------------------------------------------------------------------

def _build_cast(book, args, mode: str) -> dict:
    max_characters = getattr(args, "max_characters", 12)
    narrator_gender = getattr(args, "narrator_gender", None)

    if mode == "heavy":
        from .cast_llm import analyze_book_llm

        max_calls = getattr(args, "llm_max_calls", None)
        data = analyze_book_llm(
            book,
            max_characters=max_characters,
            model=getattr(args, "llm_model", None) or "llama3:8b",
            endpoint=getattr(args, "llm_endpoint", None) or "http://localhost:11434/api/generate",
            narrator_gender=narrator_gender,
            strategy=_heavy_strategy(args),
            batch_size=int(getattr(args, "llm_batch_size", None) or 8),
            confidence_threshold=float(getattr(args, "llm_confidence", None) or 0.6),
            cache_dir=getattr(args, "llm_cache_dir", None) or None,
            max_calls=int(max_calls) if max_calls else None,
            strict=bool(getattr(args, "analyze_only", False)),
            progress=bool(getattr(args, "analyze_only", False)),
        )
    elif mode == "light":
        from .cast_light import analyze_book_light

        data = analyze_book_light(
            book,
            max_characters=max_characters,
            narrator_gender=narrator_gender,
        )
    elif mode == "whole":
        from .cast_whole import analyze_book_whole

        data = analyze_book_whole(
            book,
            max_characters=max_characters,
            unknown_below=getattr(args, "unknown_below", None),
            narrator_gender=narrator_gender,
        )
    elif mode == "scene":
        from .cast_scene import analyze_book_scene

        data = analyze_book_scene(
            book,
            max_characters=max_characters,
            unknown_below=getattr(args, "unknown_below", None),
            narrator_gender=narrator_gender,
        )
    else:
        data = analyze_book(book, max_characters=max_characters)
        data["mode"] = "heuristic"

    # Add a dedicated narrator row (for first-person books) and reserve the
    # top-priority voice for it.  Reassigning here is safe: this is a fresh
    # analysis, so no manual voice edits exist yet.  It also covers engines
    # that assigned voices before we injected the narrator.
    narrator = ensure_narrator_character(data.get("characters", []), narrator_gender)
    if narrator is not None:
        assign_voices(data["characters"])
        data["narrator"] = narrator.get("voice")
    return data


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def resolve_cast_file(in_path: Path, args) -> Path:
    """Where the cast JSON for this input (and engine) lives / is written."""
    if getattr(args, "cast_out", None):
        return Path(args.cast_out)
    if getattr(args, "cast_file", None):
        return Path(args.cast_file)

    mode = select_mode(args)
    if mode == "heuristic":
        return cast_path_for(in_path)

    if mode == "heavy":
        tag = f"heavy-{_heavy_strategy(args)}"
    else:
        tag = {"light": "light", "whole": "whole", "scene": "scene"}[mode]
    return in_path.parent / f".{in_path.stem}.cast.{tag}.json"


def generate_cast(in_path: Path, book, args) -> dict:
    """Return the cast for ``book``: an explicit --cast-file, a cached analysis,
    or a freshly built one (which is then saved)."""
    if getattr(args, "cast_file", None) and not getattr(args, "cast_out", None):
        print(f"Using cast file: {args.cast_file}", file=sys.stderr)
        return json.loads(Path(args.cast_file).read_text(encoding="utf-8"))

    mode = select_mode(args)
    path = resolve_cast_file(in_path, args)
    key = _cache_key(in_path, args, mode)

    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if cached and cached.get("_cache_key") == key:
            print(f"Using cached cast analysis: {path}", file=sys.stderr)
            return cached

    print("Analyzing characters and dialogue (no cast.json found or settings changed)...",
          file=sys.stderr)
    data = _build_cast(book, args, mode)
    data.setdefault("mode", _mode_label(args, mode))
    data["version"] = CAST_VERSION
    data["_cache_key"] = key

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

    print(f"Cast analysis saved to {path} — edit it by hand and re-run with "
          f"--cast-file {path} to override.", file=sys.stderr)
    return data
