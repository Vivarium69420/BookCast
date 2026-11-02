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
# Buckets may share a fallback voice; the narrator's voice is a real
# assignment and must stay unique so narration never sounds like a character.
VOICE_LOCKED_IDS = {"narrator"}


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
    if voice is not None and voice.strip():
        voice = voice.strip()
        # Two named characters sharing a voice is almost never intentional:
        # the narrator and a major character become indistinguishable, and
        # every attribution mistake sounds like a voice bug.  Buckets may
        # share a fallback voice, and pitch can still distinguish them.
        # Validate before mutating anything, so a rejected edit cannot leave
        # a half-applied name/gender change in the in-memory cast.
        if cid not in UNKNOWN_IDS or cid in VOICE_LOCKED_IDS:
            owner = next((o for o in characters(data)
                          if o["id"] != cid
                          and (o["id"] not in UNKNOWN_IDS
                               or o["id"] in VOICE_LOCKED_IDS)
                          and (o.get("voice") or "") == voice), None)
            if owner:
                return False, (f"voice `{voice}` is already used by "
                               f"{owner.get('name') or owner['id']} ({owner['id']}); "
                               "pick another or use Auto")
    if name is not None and name.strip():
        c["name"] = name.strip()
    if pitch is not None and pitch.strip():
        c["pitch"] = pitch.strip()
    if gender is not None:
        c["gender"] = gender or None
    if voice is not None and voice.strip():
        c["voice"] = voice
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


# ---------------------------------------------------------------- import
def source_label(data: Optional[Dict[str, Any]]) -> str:
    if not data:
        return "none"
    mode = str(data.get("mode") or "cast")
    llm = data.get("llm") or {}
    model = llm.get("model")
    if model and model not in mode:
        return f"{mode} · {model}"
    return mode


def _chapter_lengths(book) -> Dict[str, int]:
    out = {}
    for i, ch in enumerate(book.chapters, start=1):
        out[str(i)] = len(split_dialogue_narration(ch.text))
    return out


def validate_import(data: Any, book=None, current: Optional[Dict[str, Any]] = None
                    ) -> Tuple[bool, List[str], str]:
    """Validate a cast JSON before it replaces the workspace cast.

    Returns ``(ok, problems, summary_markdown)``.  Problems are fatal; the
    summary always describes what would change.
    """
    problems: List[str] = []
    warnings: List[str] = []
    if not isinstance(data, dict):
        return False, ["The file is not a JSON object."], ""

    chars = data.get("characters")
    if not isinstance(chars, list) or not chars:
        problems.append("`characters` is missing or empty.")
        chars = []

    ids: List[str] = []
    names: Dict[str, str] = {}
    genders: Dict[str, str] = {}
    for i, c in enumerate(chars):
        if not isinstance(c, dict):
            problems.append(f"Character #{i + 1} is not an object.")
            continue
        cid = c.get("id")
        if not cid or not isinstance(cid, str):
            problems.append(f"Character #{i + 1} has no string `id`.")
            continue
        if cid in names:
            problems.append(f"Duplicate character id `{cid}`.")
            continue
        ids.append(cid)
        names[cid] = str(c.get("name") or cid)
        genders[cid] = str(c.get("gender") or "")
    if "unknown" not in names:
        problems.append("No `unknown` bucket; every attribution needs somewhere to fall back to.")

    att = data.get("attribution")
    if not isinstance(att, dict):
        problems.append("`attribution` is missing or is not an object.")
        att = {}

    lengths = _chapter_lengths(book) if book is not None else {}
    lines = unknown = 0
    methods: Counter = Counter()
    for key, entries in att.items():
        if not isinstance(entries, list):
            problems.append(f"Chapter `{key}` is not a list.")
            continue
        if lengths and str(key) not in lengths:
            problems.append(f"Chapter `{key}` is outside the opened book ({len(lengths)} chapters).")
            continue
        for e in entries:
            if not isinstance(e, dict):
                problems.append(f"Chapter `{key}` has a non-object attribution row.")
                continue
            seg = e.get("seg_idx")
            if not isinstance(seg, int):
                problems.append(f"Chapter `{key}` has a non-integer `seg_idx`.")
                continue
            if lengths and not (0 <= seg < lengths[str(key)]):
                problems.append(f"Chapter `{key}` segment {seg} is out of range "
                                f"(0-{lengths[str(key)] - 1}).")
            cid = e.get("character_id")
            if cid in (None, ""):
                problems.append(f"Chapter `{key}` segment {seg} has no `character_id`.")
            elif cid not in names:
                problems.append(f"Chapter `{key}` segment {seg} refers to unknown id `{cid}`.")
            lines += 1
            unknown += int(cid in (None, "", "unknown", "minor_m", "minor_f"))
            methods[str(e.get("method") or "")] += 1

    if book is not None and data.get("book_title"):
        if str(data["book_title"]).strip().lower() != str(book.title).strip().lower():
            warnings.append(f"Book title differs: cast has `{data['book_title']}`, "
                            f"open book is `{book.title}`. Check the chapter/segment "
                            "references before applying.")

    # Diff summary (not fatal).
    old_lines: Dict[Tuple[str, int], Optional[str]] = {}
    if current:
        for key, entries in (current.get("attribution") or {}).items():
            for e in entries:
                if isinstance(e, dict) and isinstance(e.get("seg_idx"), int):
                    old_lines[(str(key), e["seg_idx"])] = e.get("character_id")
    changed = 0
    for key, entries in att.items():
        for e in entries:
            if (isinstance(e, dict) and isinstance(e.get("seg_idx"), int)
                    and old_lines.get((str(key), e["seg_idx"]), e.get("character_id")) != e.get("character_id")):
                changed += 1

    old_ids = {c.get("id") for c in (current or {}).get("characters", [])}
    new_ids = set(ids)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    renamed = []
    if current:
        old_names = {c.get("id"): c.get("name") for c in current.get("characters", [])}
        old_genders = {c.get("id"): c.get("gender") for c in current.get("characters", [])}
        renamed = [f"`{cid}`: {old_names.get(cid)} → {names[cid]}"
                   for cid in new_ids & old_ids if old_names.get(cid) != names[cid]]
    else:
        old_genders = {}
    gender_changes = [f"`{cid}`: {old_genders.get(cid) or '—'} → {genders.get(cid) or '—'}"
                      for cid in sorted(new_ids & set(old_genders))
                      if (old_genders.get(cid) or "") != (genders.get(cid) or "")]

    lines_md = [
        f"**Source:** {source_label(data)}",
        f"**Cast:** {len(ids)} characters · **attributed lines:** {lines} "
        f"· **unknown/minor:** {unknown} ({unknown / max(lines, 1):.0%})",
    ]
    if current:
        lines_md.append(f"**Diff:** {changed} line(s) changed · {len(added)} added · "
                        f"{len(removed)} removed · {len(renamed)} renamed · "
                        f"{len(gender_changes)} gender change(s)")
        if renamed:
            lines_md.append("Renamed: " + "; ".join(renamed[:12])
                            + (" …" if len(renamed) > 12 else ""))
        if gender_changes:
            lines_md.append("Gender: " + "; ".join(gender_changes[:12])
                            + (" …" if len(gender_changes) > 12 else ""))
    if methods:
        top = ", ".join(f"`{m}`={n}" for m, n in methods.most_common(6))
        lines_md.append(f"**Methods:** {top}")
    if warnings:
        lines_md.append("**Warnings:** " + " ".join(warnings))

    if problems:
        lines_md.append("\n**Cannot import:**\n" + "\n".join(f"- {p}" for p in problems[:16]))
    else:
        lines_md.append("\n**Ready to apply.** This replaces the current cast; the old file "
                        "is kept as `cast.json.bak`.")
    return not problems, problems, "\n\n".join(lines_md)
