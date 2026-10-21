"""Cast JSON editing: characters, merges, and per-quote reassignment."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from btlib.extract import split_dialogue_narration

from . import store

UNKNOWN_IDS = {"unknown", "minor_f", "minor_m", "narrator"}


# ---------------------------------------------------------------- load / save
def load(ws: Path) -> Optional[Dict[str, Any]]:
    p = store.cast_path(ws)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save(ws: Path, data: Dict[str, Any]) -> Path:
    p = store.cast_path(ws)
    if p.exists():
        shutil.copy2(p, p.with_suffix(".json.bak"))
    recount(data)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def recount(data: Dict[str, Any]) -> None:
    counts: Counter = Counter()
    for entries in (data.get("attribution") or {}).values():
        for e in entries:
            cid = e.get("character_id")
            if cid:
                counts[cid] += 1
    for ch in data.get("characters", []):
        ch["line_count"] = counts.get(ch["id"], 0)


# ---------------------------------------------------------------- characters
def characters(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(data.get("characters") or [])


def find(data: Dict[str, Any], cid: str) -> Optional[Dict[str, Any]]:
    return next((c for c in characters(data) if c["id"] == cid), None)


def characters_rows(data: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for c in sorted(characters(data), key=lambda x: -(x.get("line_count") or 0)):
        rows.append([
            c["id"],
            c.get("name") or "",
            int(c.get("line_count") or 0),
            c.get("gender") or "",
            c.get("voice") or "",
            c.get("pitch") or "+0Hz",
        ])
    return rows


def character_names(data: Dict[str, Any], include_buckets: bool = False) -> List[str]:
    out = []
    for c in sorted(characters(data), key=lambda x: -(x.get("line_count") or 0)):
        if not include_buckets and c["id"] in UNKNOWN_IDS:
            continue
        out.append(f"{c['id']}\t{c.get('name') or c['id']} ({c.get('line_count', 0)})")
    return out


def apply(
    data: Dict[str, Any],
    cid: str,
    *,
    name: Optional[str] = None,
    gender: Optional[str] = None,
    voice: Optional[str] = None,
    pitch: Optional[str] = None,
) -> Tuple[bool, str]:
    c = find(data, cid)
    if not c:
        return False, f"unknown character id: {cid}"
    if name is not None and name.strip():
        c["name"] = name.strip()
    if pitch is not None and pitch.strip():
        c["pitch"] = pitch.strip()
    if gender is not None:
        c["gender"] = gender or None
    if voice is not None and voice.strip():
        c["voice"] = voice.strip()
    return True, "applied"


def auto_voice(data: Dict[str, Any], gender: str, exclude_cid: Optional[str] = None,
               profile: Optional[str] = None) -> Optional[str]:
    pool = store.voices_for_gender(gender, profile)
    used = {c.get("voice") for c in characters(data) if c["id"] != exclude_cid}
    for v in pool:
        if v not in used:
            return v
    return pool[0] if pool else None


def merge(data: Dict[str, Any], src: str, dst: str) -> Tuple[bool, str]:
    if src == dst:
        return False, "cannot merge a character into itself"
    if src in UNKNOWN_IDS or dst in UNKNOWN_IDS:
        return False, "cannot merge bucket ids"
    if not find(data, src) or not find(data, dst):
        return False, "unknown character id"
    n = 0
    for entries in (data.get("attribution") or {}).values():
        for e in entries:
            if e.get("character_id") == src:
                e["character_id"] = dst
                n += 1
    data["characters"] = [c for c in data["characters"] if c["id"] != src]
    recount(data)
    return True, f"moved {n} lines from {src} to {dst}"


def delete(data: Dict[str, Any], cid: str) -> Tuple[bool, str]:
    if cid in UNKNOWN_IDS:
        return False, "cannot delete a bucket"
    if not find(data, cid):
        return False, "unknown character id"
    n = 0
    for entries in (data.get("attribution") or {}).values():
        for e in entries:
            if e.get("character_id") == cid:
                e["character_id"] = "unknown"
                n += 1
    data["characters"] = [c for c in data["characters"] if c["id"] != cid]
    recount(data)
    return True, f"{cid} removed; {n} lines moved to Unknown"


# -------------------------------------------------------------------- quotes
def chapter_keys(data: Dict[str, Any]) -> List[str]:
    keys = list((data.get("attribution") or {}).keys())
    return sorted(keys, key=lambda k: int(k) if str(k).isdigit() else 0)


def quote_rows(data: Dict[str, Any], chapter_key: str, text: str,
               only_uncertain: bool = False) -> List[List[Any]]:
    segs = split_dialogue_narration(text)
    names = {c["id"]: (c.get("name") or c["id"]) for c in characters(data)}
    rows = []
    for e in (data.get("attribution") or {}).get(str(chapter_key), []):
        seg_idx = e.get("seg_idx")
        if seg_idx is None or seg_idx >= len(segs):
            continue
        kind, seg_text = segs[seg_idx]
        if kind != "dialogue":
            continue
        cid = e.get("character_id")
        conf = e.get("confidence") or ""
        if only_uncertain and not (conf == "low" or cid in UNKNOWN_IDS):
            continue
        rows.append([
            seg_idx,
            names.get(cid, cid or ""),
            cid or "",
            e.get("method") or "",
            conf,
            " ".join(seg_text.split())[:300],
        ])
    rows.sort(key=lambda r: r[0])
    return rows


def assign(data: Dict[str, Any], chapter_key: str, seg_idx: int, cid: str) -> Tuple[bool, str]:
    entries = (data.get("attribution") or {}).get(str(chapter_key), [])
    for e in entries:
        if e.get("seg_idx") == seg_idx:
            e["character_id"] = cid
            m = e.get("method") or ""
            if "manual" not in m:
                e["method"] = m + "+manual"
            e["confidence"] = "high"
            recount(data)
            return True, f"segment {seg_idx} -> {cid}"
    return False, f"segment {seg_idx} not found"


def segment_text(chapter_text: str, seg_idx: int) -> str:
    segs = split_dialogue_narration(chapter_text)
    if 0 <= seg_idx < len(segs):
        return segs[seg_idx][1]
    return ""


def character_sample(data: Dict[str, Any], chapter_texts: Dict[int, str],
                     cid: str, min_words: int = 3) -> str:
    """Longest quote spoken by ``cid`` — a good audition line."""
    best = ""
    for key, entries in (data.get("attribution") or {}).items():
        try:
            text = chapter_texts.get(int(key), "")
        except (TypeError, ValueError):
            continue
        if not text:
            continue
        for e in entries:
            if e.get("character_id") != cid:
                continue
            seg = segment_text(text, e.get("seg_idx", -1))
            seg = seg.strip().strip('"“”')
            if len(seg.split()) >= min_words and len(seg) > len(best):
                best = seg
    return best[:240]
