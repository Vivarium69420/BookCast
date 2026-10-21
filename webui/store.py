"""Per-book workspace and upload handling for the web UI.

Layout::

    runs/<slug>/
        cast.json     editable cast (always passed to the CLI as --cast-file)
        work/         explicit --workdir (chunks + progress.json)
        out.mp3       combined output (or out/ for --per-chapter)
        log.txt       last run log
        meta.json     last parameters / timestamps
        preview/      audition clips
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"
RUNS = ROOT / "runs"
PRESETS = ROOT / "webui" / "presets"


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return (s or "book")[:60]


def sha256(path: Path, limit: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_upload(tmp_path: str, original_name: Optional[str] = None) -> Path:
    """Copy an uploaded file into books/ (never overwrite; reuse identical)."""
    BOOKS.mkdir(parents=True, exist_ok=True)
    src = Path(tmp_path)
    name = original_name or src.name
    target = BOOKS / name
    if target.exists():
        if sha256(target) == sha256(src):
            return target
        stem, suffix = target.stem, target.suffix
        n = 2
        while (BOOKS / f"{stem} ({n}){suffix}").exists():
            n += 1
        target = BOOKS / f"{stem} ({n}){suffix}"
    shutil.copy2(src, target)
    return target


def workspace_for(book: Path) -> Path:
    ws = RUNS / slugify(book.stem)
    ensure(ws)
    return ws


def ensure(ws: Path) -> Path:
    (ws / "work").mkdir(parents=True, exist_ok=True)
    (ws / "preview").mkdir(parents=True, exist_ok=True)
    return ws


def cast_path(ws: Path) -> Path:
    return ws / "cast.json"


def out_path(ws: Path, per_chapter: bool) -> Path:
    return ws / "out" if per_chapter else ws / "out.mp3"


def load_meta(ws: Path) -> Dict[str, Any]:
    p = ws / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_meta(ws: Path, **kw) -> None:
    meta = load_meta(ws)
    meta.update(kw)
    meta["updated"] = time.time()
    (ws / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def list_outputs(ws: Path) -> List[Path]:
    files: List[Path] = []
    single = ws / "out.mp3"
    if single.exists() and single.stat().st_size > 0:
        files.append(single)
    outdir = ws / "out"
    if outdir.is_dir():
        files.extend(sorted(p for p in outdir.glob("*.mp3") if p.stat().st_size > 0))
    return files


def invalidate_chapters(ws: Path, chapters: Iterable[int]) -> List[int]:
    """Forget cached audio for the given 1-based chapter indices.

    The CLI's progress signature does not hash the cast *content*, only the
    cast file path, so edited voices would otherwise resume with stale audio.
    """
    idxs = sorted({int(c) for c in chapters})
    prog = ws / "work" / "progress.json"
    if not idxs or not prog.exists():
        return []
    try:
        data = json.loads(prog.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    done = []
    for ch in data.get("chapters", []):
        i = ch.get("index")
        if i not in idxs:
            continue
        ch["num_chunks"] = None
        ch["chunk_done"] = []
        ch["chapter_combined_done"] = False
        ch["chapter_output_done"] = False
        done.append(i)
        shutil.rmtree(ws / "work" / "chunks" / f"ch_{i:04d}", ignore_errors=True)
        combined = ws / "work" / "chapters" / f"ch_{i:04d}_combined.mp3"
        combined.unlink(missing_ok=True)
        outdir = ws / "out"
        if outdir.is_dir():
            for p in outdir.glob(f"*{i:02d}*.mp3"):
                p.unlink(missing_ok=True)
    if done:
        data["final_done"] = False
        tmp = prog.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(prog)
        # the combined file is now stale
        (ws / "out.mp3").unlink(missing_ok=True)
    return done


# --------------------------------------------------------------------------
# Voice bank / presets
# --------------------------------------------------------------------------

def voice_profiles() -> Dict[str, Dict[str, List[str]]]:
    p = ROOT / "voice_priority.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("profiles") or {}


def all_voices(profile: Optional[str] = None) -> List[str]:
    profiles = voice_profiles()
    chosen = profiles.get(profile) if profile else None
    if chosen is None:
        default = None
        p = ROOT / "voice_priority.json"
        if p.exists():
            try:
                default = json.loads(p.read_text(encoding="utf-8")).get("default_profile")
            except (OSError, json.JSONDecodeError):
                default = None
        chosen = profiles.get(default) or next(iter(profiles.values()), {})
    seen: List[str] = []
    for pool in ("female", "male", "neutral"):
        for v in chosen.get(pool, []):
            if v not in seen:
                seen.append(v)
    return seen


def voices_for_gender(gender: Optional[str], profile: Optional[str] = None) -> List[str]:
    profiles = voice_profiles()
    if not profiles:
        return all_voices(profile)
    chosen = profiles.get(profile) if profile else None
    if chosen is None:
        p = ROOT / "voice_priority.json"
        default = None
        if p.exists():
            try:
                default = json.loads(p.read_text(encoding="utf-8")).get("default_profile")
            except (OSError, json.JSONDecodeError):
                default = None
        chosen = profiles.get(default) or next(iter(profiles.values()), {})
    key = {"male": "male", "female": "female"}.get(gender or "", "neutral")
    return list(chosen.get(key) or chosen.get("neutral") or [])


def preset_names() -> List[str]:
    if not PRESETS.exists():
        return []
    return sorted(p.stem for p in PRESETS.glob("*.json"))


def load_preset(name: str) -> Dict[str, Any]:
    p = PRESETS / f"{name}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_preset(name: str, values: Dict[str, Any]) -> Path:
    PRESETS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9 _-]+", "", name).strip() or "preset"
    p = PRESETS / f"{safe}.json"
    p.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
