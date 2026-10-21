"""Cast dispatcher: picks the attribution engine, caches the result, and tells
the pipeline where the cast file lives.

This module was reconstructed (its source was lost). The two public functions
are called from ``pipeline.py``::

    generate_cast(in_path, book, args) -> dict
    resolve_cast_file(in_path, args)   -> Path

Engine selection mirrors the CLI:

    --heavy-model          -> btlib.cast_heavy.analyze_book_heavy  (local LLM)
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
from .cast import analyze_book, cast_path_for

# Bumped whenever the cast schema or the cache-key inputs change. v8 adds the
# first-person narrator fields (`person`/`narrator`) to each character.
CAST_VERSION = 16


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
        return f"heavy-llm ({getattr(args, 'llm_model', 'unknown')})"
    if mode == "light":
        return "lightweight-transformer"
    if mode == "whole":
        return "whole-book"
    if mode == "scene":
        return "scene"
    return "heuristic"


def _cache_tag(args, mode: str) -> str:
    """The short tag that also names the on-disk cast file."""
    if mode == "heavy":
        return f"heavy:{getattr(args, 'llm_model', 'unknown')}"
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
        from .cast_heavy import analyze_book_heavy

        return analyze_book_heavy(
            book,
            max_characters=max_characters,
            model=getattr(args, "llm_model", None) or "llama3:8b",
            endpoint=getattr(args, "llm_endpoint", None) or "http://localhost:11434/api/generate",
            narrator_gender=narrator_gender,
        )

    if mode == "light":
        from .cast_light import analyze_book_light

        return analyze_book_light(
            book,
            max_characters=max_characters,
            narrator_gender=narrator_gender,
        )

    if mode == "whole":
        from .cast_whole import analyze_book_whole

        return analyze_book_whole(
            book,
            max_characters=max_characters,
            unknown_below=getattr(args, "unknown_below", None),
            narrator_gender=narrator_gender,
        )

    if mode == "scene":
        from .cast_scene import analyze_book_scene

        return analyze_book_scene(
            book,
            max_characters=max_characters,
            unknown_below=getattr(args, "unknown_below", None),
            narrator_gender=narrator_gender,
        )

    data = analyze_book(book, max_characters=max_characters)
    data["mode"] = "heuristic"
    return data


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def resolve_cast_file(in_path: Path, args) -> Path:
    """Where the cast JSON for this input (and engine) lives / is written."""
    if getattr(args, "cast_file", None):
        return Path(args.cast_file)

    mode = select_mode(args)
    if mode == "heuristic":
        return cast_path_for(in_path)

    tag = {"heavy": "heavy", "light": "light", "whole": "whole", "scene": "scene"}[mode]
    return in_path.parent / f".{in_path.stem}.cast.{tag}.json"


def generate_cast(in_path: Path, book, args) -> dict:
    """Return the cast for ``book``: an explicit --cast-file, a cached analysis,
    or a freshly built one (which is then saved)."""
    if getattr(args, "cast_file", None):
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
