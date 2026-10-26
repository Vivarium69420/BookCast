"""Scene attribution with an LLM second opinion.

The deterministic scene engine is the base.  The model is used only where it is
worth the tokens:

1. **Cast reconciliation** — one compact pass over the candidate name table.
   It merges split identities, drops places/objects that look like people,
   fixes gender, and identifies the first-person narrator.
2. **Weak-line adjudication** — the scene engine's fallback buckets
   (``scene-single``, ``scene-core-alt``, ``pronoun``, ``action-beat``, ...)
   are resolved again with the full scene text and a *closed* cast list.

Explicit ``"X said"`` tags and confident same-paragraph anchors are never
overridden.  All responses are cached by :mod:`btlib.llm`, so the same analysis
can run on a CPU/Colab GPU and resume after a disconnect.

The output is the normal cast.json schema, so the pipeline and the evaluation
harness can use it without changes.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import cast as base
from .cast_scene import _scene_bounds, analyze_book_scene
from .extract import split_dialogue_narration
from .llm import DEFAULT_ENDPOINT, DEFAULT_MODEL, LLMClient, LLMError
from .models import Book

LLM_PROMPT_VERSION = 1

# Methods the scene engine uses when it has no explicit tag.  These are the
# buckets measured at 0.20-0.47 on PDNC; everything else is left alone.
WEAK_METHODS = {
    "scene-single",
    "scene-recent",
    "scene-roundrobin",
    "scene-presence-alt",
    "scene-core-alt",
    "scene-alternate",
    "scene-participant",
    "scene-carry",
    "scene-addressee",
    "pronoun",
    "action-beat",
    "unresolved",
}

BUCKET_IDS = {"unknown", "minor_m", "minor_f"}
_EXPLICIT_METHODS = {"tag", "narrator", "self-intro"}

# Cast edits are cheap to get wrong and expensive to hear.  A merge or a
# deletion needs stronger evidence than a rename/gender change.
CAST_EDIT_CONFIDENCE = 0.5
CAST_DESTRUCTIVE_CONFIDENCE = 0.7


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _short(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _scene_text(text: str, n: int) -> str:
    """Collapse whitespace but keep paragraph boundaries visible as ¶."""
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"\n\s*\n+", " ¶ ", text)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _entry_confidence(e: dict) -> float:
    try:
        return max(0.0, min(1.0, float(e.get("confidence", 1.0))))
    except (TypeError, ValueError):
        return 1.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "yes", "y", "1"}


def _recount(data: Dict[str, Any]) -> None:
    counts: Counter = Counter()
    for entries in (data.get("attribution") or {}).values():
        for e in entries:
            cid = e.get("character_id")
            if cid:
                counts[cid] += 1
    for ch in data.get("characters", []):
        ch["line_count"] = counts.get(ch["id"], 0)


def _named_chars(data: Dict[str, Any]) -> List[dict]:
    return [c for c in data.get("characters", []) if c.get("id") not in BUCKET_IDS]


def _character_name(data: Dict[str, Any], cid: Optional[str]) -> str:
    if not cid:
        return "Unknown"
    for c in data.get("characters", []):
        if c.get("id") == cid:
            return c.get("name") or cid
    return cid


def _narrator_id(data: Dict[str, Any]) -> Optional[str]:
    for c in data.get("characters", []):
        if c.get("narrator") or c.get("person") == "first":
            return c.get("id")
    return None


def _snippets(book: Book, character: dict, limit: int = 1) -> List[str]:
    """A couple of short in-book mentions, so the model can judge gender/identity."""
    names = [character.get("name") or ""] + list(character.get("aliases") or [])
    names = [n for n in names if n and len(n) >= 2]
    names.sort(key=len, reverse=True)
    out: List[str] = []
    full = "\n".join(ch.text for ch in book.chapters)
    for name in names[:4]:
        m = re.search(r"\b" + re.escape(name) + r"\b", full)
        if not m:
            continue
        start = max(0, m.start() - 90)
        end = min(len(full), m.end() + 110)
        snippet = _short(full[start:end], 220)
        if snippet and snippet not in out:
            out.append(snippet)
        if len(out) >= limit:
            break
    return out


def _entry_is_weak(e: dict) -> bool:
    method = str(e.get("method") or "")
    cid = e.get("character_id")
    confidence = str(e.get("confidence") or "").lower()
    if method in _EXPLICIT_METHODS:
        return False
    if method.startswith("tag-"):  # a named tag the cast did not cover
        return True
    if method == "scene-same-par" and confidence == "high":
        return False
    if method.startswith("llm-"):
        return False
    if cid in (None, "", "unknown", "minor_m", "minor_f"):
        return True
    if method in WEAK_METHODS:
        return True
    if method.endswith("-minor"):
        return True
    return confidence in {"low", "medium"} and method not in {"tag", "narrator"}


def _cast_table(data: Dict[str, Any]) -> List[dict]:
    rows = []
    for c in _named_chars(data):
        rows.append({
            "id": c.get("id"),
            "name": c.get("name") or "",
            "aliases": list(c.get("aliases") or []),
            "gender": c.get("gender"),
            "narrator": bool(c.get("narrator") or c.get("person") == "first"),
            "line_count": int(c.get("line_count") or 0),
        })
    for cid, name in (("minor_f", "Unnamed woman (minor)"),
                      ("minor_m", "Unnamed man (minor)"),
                      ("unknown", "Unknown speaker")):
        rows.append({"id": cid, "name": name, "aliases": [], "gender": None,
                     "narrator": False, "line_count": 0})
    return rows


# --------------------------------------------------------------------------
# Cast reconciliation
# --------------------------------------------------------------------------

def _cast_prompt(book: Book, data: Dict[str, Any]) -> str:
    rows = []
    for c in _named_chars(data):
        rows.append({
            "id": c.get("id"),
            "name": c.get("name") or "",
            "aliases": list(c.get("aliases") or []),
            "gender": c.get("gender"),
            "narrator": bool(c.get("narrator") or c.get("person") == "first"),
            "lines": int(c.get("line_count") or 0),
            "snippets": _snippets(book, c, limit=2),
        })
    return f"""You are reconciling the character list for the novel "{book.title}".

Each entry is a candidate character produced by a rule-based extractor. Decide:
- Is it a real person? Set is_person=false for places, organisations, objects,
  titles, or words that are not people.
- Are two entries the same person? Set merge_into to the other id.
- Fix the canonical name and aliases (keep the form used by the narration).
- Set gender to "male", "female", or "neutral" using honorifics, pronouns, name,
  and the snippets. Do not guess from a surname alone.
- Set is_narrator=true for exactly one character only if the book has a
  first-person narrator ("I said"). Otherwise false.

Rules: never invent ids; only use ids from the input. One entry per id.
Return one JSON object:
{{"characters": [
  {{"id": "c1", "name": "Canonical Name", "aliases": ["alias"],
    "gender": "male|female|neutral", "is_person": true,
    "is_narrator": false, "merge_into": null,
    "confidence": 0.0, "reason": "short evidence"}}
]}}

Candidates:
{json.dumps(rows, indent=2, ensure_ascii=False)}
"""


def _apply_cast_updates(data: Dict[str, Any], result: Dict[str, Any],
                        narrator_gender: Optional[str] = None) -> Counter:
    stats: Counter = Counter()
    raw = result.get("characters")
    if not isinstance(raw, list):
        return stats

    before = copy.deepcopy(data)
    chars = {c.get("id"): c for c in data.get("characters", [])}
    valid = set(chars)
    entries = [e for e in raw if isinstance(e, dict) and isinstance(e.get("id"), str)
               and e.get("id") in valid]
    if not entries:
        return stats

    # Normalise merged / narrator flags before touching attribution.
    merge: Dict[str, str] = {}
    for e in entries:
        src = str(e.get("id"))
        dst = e.get("merge_into")
        if (dst and str(dst) in valid and str(dst) != src
                and _entry_confidence(e) >= CAST_DESTRUCTIVE_CONFIDENCE):
            merge[src] = str(dst)

    # Resolve chains c3 -> c2 -> c1.
    def final(cid: str) -> str:
        seen = set()
        while cid in merge and cid not in seen:
            seen.add(cid)
            cid = merge[cid]
        return cid

    for src in list(merge):
        merge[src] = final(merge[src])

    # Apply merges first: keep the target's id, absorb names/aliases.
    merge_narrator: Optional[str] = None
    for src, dst in merge.items():
        if src == dst or src not in chars or dst not in chars:
            continue
        target, source = chars[dst], chars[src]
        names = {target.get("name") or "", source.get("name") or ""}
        aliases = set(target.get("aliases") or []) | set(source.get("aliases") or [])
        aliases |= {n for n in names if n}
        target["aliases"] = sorted(aliases - {target.get("name") or ""})
        target["line_count"] = int(target.get("line_count") or 0) + int(source.get("line_count") or 0)
        if source.get("narrator") or source.get("person") == "first":
            target["narrator"] = True
            target["person"] = "first"
            merge_narrator = dst
        chars.pop(src, None)
        stats["merged"] += 1

    # Rewrite attribution through the merge map.
    for entries_ in (data.get("attribution") or {}).values():
        for e in entries_:
            cid = e.get("character_id")
            if cid in merge:
                e["character_id"] = merge[cid]

    # Drop non-people, and apply names/gender/narrator flags to survivors.
    narrator_seen: Optional[str] = None
    for e in entries:
        original = str(e.get("id"))
        cid = final(original)
        ch = chars.get(cid)
        if not ch or cid in BUCKET_IDS:
            continue
        # A merged source's name/aliases/gender are stale; the target's own
        # entry is authoritative.  (Otherwise the last entry in the list could
        # rename a merged character to the discarded alias.)
        if original in merge:
            continue
        conf = _entry_confidence(e)
        person = e.get("is_person")
        if person is not None and not _truthy(person):
            if conf >= CAST_DESTRUCTIVE_CONFIDENCE:
                ch["_drop"] = True
            else:
                stats["rejected_low_confidence"] += 1
            continue
        if conf < CAST_EDIT_CONFIDENCE:
            stats["rejected_low_confidence"] += 1
            continue

        name = str(e.get("name") or "").strip()
        if name and name != ch.get("name"):
            ch["name"] = name
            stats["renamed"] += 1
        aliases = e.get("aliases")
        if isinstance(aliases, list):
            clean = [str(a).strip() for a in aliases if str(a).strip()]
            ch["aliases"] = sorted(set(clean) | {ch.get("name") or ""})
        gender = str(e.get("gender") or "").lower().strip()
        if gender in {"male", "female", "neutral"}:
            gender = None if gender == "neutral" else gender
            if gender != ch.get("gender"):
                ch["gender"] = gender
                stats["gender_changed"] += 1
        if _truthy(e.get("is_narrator")) and narrator_seen is None:
            narrator_seen = cid
            stats["narrator"] += 1

    if narrator_seen is None and merge_narrator and merge_narrator in chars:
        narrator_seen = merge_narrator

    # Remove non-people and send their lines to Unknown.
    dropped = {cid for cid, ch in chars.items() if ch.pop("_drop", False)}
    if dropped:
        for entries_ in (data.get("attribution") or {}).values():
            for e in entries_:
                if e.get("character_id") in dropped:
                    e["character_id"] = "unknown"
        for cid in dropped:
            chars.pop(cid, None)
        stats["dropped"] += len(dropped)

    # Exactly one narrator if the model identified one (or the caller forced a
    # gender, which only makes sense for a first-person book).
    if narrator_seen and narrator_seen in chars:
        for cid, ch in chars.items():
            ch["narrator"] = cid == narrator_seen
            ch["person"] = "first" if cid == narrator_seen else "third"

    data["characters"] = [c for c in data.get("characters", []) if c.get("id") in chars]

    if narrator_gender in {"male", "female"}:
        nid = _narrator_id(data)
        if nid and nid in chars:
            chars[nid]["gender"] = narrator_gender
            stats["narrator_gender"] += 1

    # A model that decides most of the cast is junk/duplicates would make the
    # book worse than the deterministic baseline.  Reject that whole pass.  The
    # line-share check also catches deleting the protagonist while keeping
    # enough minor names to satisfy the character-count check.
    _recount(data)
    before_named = len(_named_chars(before))
    after_named = len(_named_chars(data))
    before_lines = sum(int(c.get("line_count") or 0) for c in _named_chars(before))
    after_lines = sum(int(c.get("line_count") or 0) for c in _named_chars(data))
    if ((before_named >= 6 and after_named < before_named * 0.5)
            or (before_lines >= 20 and after_lines < before_lines * 0.6)):
        data.clear()
        data.update(before)
        return Counter({"rejected_cast_pass": 1})

    return stats


# --------------------------------------------------------------------------
# Weak-line adjudication
# --------------------------------------------------------------------------

def _scene_window(segments: Sequence[tuple], start: int, end: int,
                  targets: Iterable[int], max_chars: int = 12000) -> Tuple[int, int]:
    """Keep the whole scene when it is small, otherwise a window around targets."""
    text_len = sum(len(t) for _k, t in segments[start:end])
    if text_len <= max_chars and end - start <= 90:
        return start, end
    t = list(targets)
    lo = max(start, min(t) - 10)
    hi = min(end, max(t) + 11)
    return lo, hi


def _scene_prompt(book: Book, chapter_title: str, data: Dict[str, Any],
                  segments: Sequence[tuple], start: int, end: int,
                  targets: List[int], by_idx: Dict[int, dict]) -> str:
    cast_lines = []
    for c in _cast_table(data):
        aliases = ", ".join(c["aliases"][:6])
        extra = f" | aliases: {aliases}" if aliases else ""
        gender = f" | {c['gender']}" if c["gender"] else ""
        narrator = " | first-person narrator" if c["narrator"] else ""
        cast_lines.append(f"- {c['id']} | {c['name']}{extra}{gender}{narrator}")

    lo, hi = _scene_window(segments, start, end, targets)
    scene_lines = []
    for i in range(lo, hi):
        kind, text = segments[i]
        tag = "D" if kind == "dialogue" else "N"
        scene_lines.append(f"[{i}] {tag}: {_scene_text(text, 700)}")

    target_rows = []
    for idx in targets:
        e = by_idx.get(idx) or {}
        target_rows.append({
            "seg_idx": idx,
            "text": _scene_text(segments[idx][1], 450),
            "guess_id": e.get("character_id"),
            "guess_name": _character_name(data, e.get("character_id")),
            "method": e.get("method"),
            "confidence": e.get("confidence"),
        })

    return f"""You are adjudicating who speaks each target line in a scene from the novel "{book.title}".
Chapter: {chapter_title}

Cast — use ONLY these ids, never invent a name:
{chr(10).join(cast_lines)}

Scene (segment ids; D=dialogue, N=narration; ¶ marks a paragraph break):
{chr(10).join(scene_lines)}

Targets:
{json.dumps(target_rows, indent=2, ensure_ascii=False)}

Rules:
- Choose speaker_id from the cast ids above, or "unknown" if the evidence is
  truly insufficient.
- Use narration tags, action beats, vocatives, turn-taking, and who is present
  in the scene. A quote sharing a paragraph with a tag usually has that tag's
  speaker; a blank line usually starts a new speaker.
- Do not treat a name used as a vocative ("Come here, Mary") as the speaker.
- The narrator is the "I" voice when the book is first-person.
- Prefer a named character over minor_m/minor_f; use a minor bucket only for a
  genuinely unnamed man/woman.
- Confidence is 0.0-1.0. Use <0.5 if you are guessing.

Return one JSON object:
{{"results": [
  {{"seg_idx": 12, "speaker_id": "c1", "confidence": 0.0,
    "evidence": "short quote from the narration", "addressee_id": null}}
]}}
"""


def _normalise_speaker_id(raw: Any, data: Dict[str, Any]) -> Optional[str]:
    if raw is None:
        return None
    sid = str(raw).strip()
    if not sid:
        return None
    ids = {c.get("id") for c in data.get("characters", [])}
    if sid in ids:
        return sid
    low = sid.lower().replace(" ", "_").replace("-", "_")
    if low in {"unknown", "none", "n_a", "na"}:
        return "unknown"
    if low in {"minor_m", "minor_male", "unnamed_man", "man"}:
        return "minor_m"
    if low in {"minor_f", "minor_female", "unnamed_woman", "woman"}:
        return "minor_f"
    if low in {"narrator", "the_narrator", "i"}:
        return _narrator_id(data)
    # Tolerate c01 / C1 spellings.
    m = re.fullmatch(r"c0*(\d+)", low)
    if m:
        candidate = f"c{int(m.group(1))}"
        if candidate in ids:
            return candidate
    # Some models return a name/alias instead of an id.  Accept a unique match.
    needle = sid.lower()
    matches = [c.get("id") for c in data.get("characters", [])
               if (c.get("name") or "").lower() == needle
               or needle in {str(a).lower() for a in (c.get("aliases") or [])}]
    return matches[0] if len(matches) == 1 else None


def _apply_results(data: Dict[str, Any], chapter_key: str,
                   results: List[dict], target_set: set,
                   threshold: float, allow_unknown: bool) -> Counter:
    stats: Counter = Counter()
    entries = (data.get("attribution") or {}).get(str(chapter_key)) or []
    by_idx = {e.get("seg_idx"): e for e in entries}
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("seg_idx"))
        except (TypeError, ValueError):
            continue
        if idx not in target_set or idx not in by_idx:
            continue
        try:
            conf = float(r.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        if conf < threshold:
            stats["rejected_low_confidence"] += 1
            continue
        sid = _normalise_speaker_id(r.get("speaker_id"), data)
        if sid is None:
            stats["rejected_unknown_id"] += 1
            continue
        e = by_idx[idx]
        if sid == "unknown" and not allow_unknown:
            stats["rejected_abstain"] += 1
            continue
        if sid == "unknown" and conf < max(threshold, 0.8):
            stats["rejected_abstain"] += 1
            continue
        old = e.get("character_id")
        if sid == old:
            stats["unchanged"] += 1
            continue
        e["character_id"] = sid
        e["confidence"] = "high" if conf >= 0.8 else "medium"
        e["method"] = "llm-unknown" if sid == "unknown" else "llm-adjudicated"
        e["llm_confidence"] = round(conf, 3)
        evidence = _short(str(r.get("evidence") or ""), 180)
        if evidence:
            e["evidence"] = evidence
        addressee = _normalise_speaker_id(r.get("addressee_id"), data)
        if addressee and addressee != "unknown":
            e["addressee_id"] = addressee
        stats["changed"] += 1
    return stats


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def analyze_book_llm(
    book: Book,
    max_characters: int = 12,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    narrator_gender: Optional[str] = None,
    *,
    strategy: str = "weak",
    batch_size: int = 8,
    confidence_threshold: float = 0.6,
    allow_unknown: bool = True,
    cache_dir: Optional[Any] = None,
    max_calls: Optional[int] = None,
    client: Optional[LLMClient] = None,
    verbose: bool = True,
    strict: bool = False,
) -> dict:
    """Run scene attribution, reconcile the cast, then adjudicate weak lines.

    ``strategy`` is ``"cast"`` (reconciliation only), ``"weak"`` (default), or
    ``"all"`` (every dialogue line; useful for an ablation).  On any LLM
    failure the scene result is kept, so a render never dies because a model
    server is down.
    """
    strategy = strategy if strategy in {"cast", "weak", "all"} else "weak"
    batch_size = max(1, int(batch_size or 8))
    confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
    data = analyze_book_scene(book, max_characters=max_characters,
                              narrator_gender=narrator_gender)
    data["mode"] = f"scene+llm ({model}, {strategy})"
    data["llm"] = {
        "model": model,
        "endpoint": endpoint,
        "strategy": strategy,
        "prompt_version": LLM_PROMPT_VERSION,
        "batch_size": batch_size,
        "confidence_threshold": confidence_threshold,
    }

    client = client or LLMClient(model=model, endpoint=endpoint,
                                 cache_dir=cache_dir, verbose=verbose)
    stats: Counter = Counter()
    errors = 0

    # ---- 1. cast reconciliation -----------------------------------------
    try:
        result = client.query_json(_cast_prompt(book, data))
        s = _apply_cast_updates(data, result, narrator_gender=narrator_gender)
        stats.update(s)
        if verbose:
            print(f"  [LLM] cast: {dict(s)}", file=sys.stderr)
    except LLMError as exc:
        errors += 1
        print(f"  ! LLM cast reconciliation failed: {exc}", file=sys.stderr)
        if strict:
            raise

    # Voices must follow any gender/narrator changes.  This is still a fresh
    # analysis, so no manual voice assignment exists yet.
    base.assign_voices(data.get("characters", []))
    _recount(data)

    if strategy == "cast":
        data["llm"].update({"usage": client.usage.summary(), "stats": dict(stats),
                            "errors": errors})
        return data

    # ---- 2. weak-line adjudication --------------------------------------
    total_targets = 0
    consecutive_errors = 0
    llm_down = False
    for ch_idx, chapter in enumerate(book.chapters, start=1):
        if llm_down:
            break
        segments = split_dialogue_narration(chapter.text)
        entries = (data.get("attribution") or {}).get(str(ch_idx)) or []
        if not entries:
            continue
        by_idx = {e.get("seg_idx"): e for e in entries}
        targets_all = [
            e.get("seg_idx") for e in entries
            if isinstance(e.get("seg_idx"), int) and (
                strategy == "all" or _entry_is_weak(e))
        ]
        targets_all = [i for i in targets_all if 0 <= i < len(segments)
                       and segments[i][0] == "dialogue"]
        targets_all.sort()
        if not targets_all:
            continue
        total_targets += len(targets_all)

        # Group targets by scene so the prompt gets coherent local context.
        for (sc_start, sc_end) in _scene_bounds(segments):
            scene_targets = [i for i in targets_all if sc_start <= i < sc_end]
            # Split a scene into local batches.  A single prompt that spans
            # 40 segments and 30k characters would blow past an 8k context.
            batches: List[List[int]] = []
            current: List[int] = []
            for idx in scene_targets:
                if current and (len(current) >= batch_size or idx - current[-1] > 12):
                    batches.append(current)
                    current = []
                current.append(idx)
            if current:
                batches.append(current)

            for pos, batch in enumerate(batches):
                if max_calls is not None and client.usage.calls >= max_calls:
                    print(f"  ! LLM call budget reached ({max_calls}); keeping scene "
                          f"for the rest of the book.", file=sys.stderr)
                    max_calls = 0
                    llm_down = True
                    break
                prompt = _scene_prompt(book, chapter.title, data, segments,
                                       sc_start, sc_end, batch, by_idx)
                try:
                    result = client.query_json(prompt)
                    rows = result.get("results")
                    if not isinstance(rows, list):
                        raise LLMError("missing 'results' list")
                    s = _apply_results(data, str(ch_idx), rows, set(batch),
                                       confidence_threshold, allow_unknown)
                    stats.update(s)
                    consecutive_errors = 0
                    if verbose and sum(s.values()):
                        print(f"  [LLM] ch{ch_idx} batch {pos + 1}: "
                              f"{dict(s)}", file=sys.stderr)
                except LLMError as exc:
                    errors += 1
                    consecutive_errors += 1
                    print(f"  ! LLM adjudication failed (chapter {ch_idx}, "
                          f"segments {batch[0]}-{batch[-1]}): {exc}", file=sys.stderr)
                    if consecutive_errors >= 3:
                        llm_down = True
                        print("  ! three consecutive LLM failures; keeping the scene "
                              "result for the rest of the book.", file=sys.stderr)
                        break
            if llm_down:
                break

    _recount(data)
    data["llm"].update({
        "usage": client.usage.summary(),
        "stats": dict(stats),
        "errors": errors,
        "targets": total_targets,
    })
    if verbose:
        print(f"  [LLM] adjudication: {data['llm']['usage']} stats={dict(stats)} "
              f"targets={total_targets}", file=sys.stderr)
    return data
