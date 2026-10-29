"""Scene-level structured attribution engine.

The heuristic (``cast.py``) and whole-book (``cast_whole.py``) engines both guess
the speaker of an untagged line by *turn-taking over the whole chapter*: whoever
spoke two lines ago says this one. That is why back-and-forth dialogue is the
weakest slice of the evaluation.

This engine adds explicit structure instead:

1. **Scene segmentation** — a chapter is cut into scenes at long narration
   gaps and typographic scene breaks.
2. **Presence tracking** — which characters are actually in each scene, inferred
   from the names mentioned in that scene's narration (and from vocatives).
3. **Constrained turn-taking** — untagged lines are resolved only among the
   characters present in the scene:
     * two present  -> strict A/B alternation, seeded by the last tagged speaker;
     * many present -> prefer a character named in the immediately preceding
       narration, then the least-recently-used one;
     * a vocative ("Come here, Mary") marks the *addressee*, so the speaker is
       someone else present.

The output uses the same cast.json schema as the other engines, so it drops into
the pipeline and the evaluation harness unchanged.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from . import cast as base
from .extract import split_dialogue_narration
from .models import Book

SCENE_VERSION = 11

# A narration gap longer than this (characters) is treated as a scene break.
SCENE_GAP = 500
# Explicit scene dividers ("***", "---", "* * *", "# # #").
SCENE_BREAK_RE = re.compile(r"(?m)^\s*(?:[*#~=\u2014\u2013]\s*){3,}\s*$")

NAME_RE = re.compile(r"\b[A-Z][a-z]+\b")
VOC_START_RE = re.compile(r'^\s*[\u201c"]?\s*([A-Z][a-z]+)\s*[,!]')
VOC_END_RE = re.compile(r',\s*([A-Z][a-z]+)\s*[.!?]?\s*[\u201d"]?\s*$')


# --------------------------------------------------------------------------
# Scene structure
# --------------------------------------------------------------------------

def _scene_bounds(segments: List[tuple]) -> List[Tuple[int, int]]:
    """Half-open [start, end) segment index ranges, one per scene."""
    bounds = []
    start = 0
    for i, (kind, text) in enumerate(segments):
        if i == 0:
            continue
        if kind == "narration" and (len(text) > SCENE_GAP or SCENE_BREAK_RE.search(text)):
            if i > start:
                bounds.append((start, i))
            start = i
    bounds.append((start, len(segments)))
    return bounds


def _scene_presence(scene: List[tuple], token_char: Dict[str, object],
                    midsentence_caps: set) -> List[object]:
    """Character ids mentioned in a scene, most-mentioned first."""
    counts: Counter = Counter()
    for kind, text in scene:
        if kind == "narration":
            for word in NAME_RE.findall(text):
                if word in base.STOP_FIRST_WORDS:
                    continue
                cid = token_char.get(word.lower())
                if cid is not None:
                    counts[cid] += 1
        else:
            head = text.strip()
            for rx in (VOC_START_RE, VOC_END_RE):
                m = rx.search(head)
                if m:
                    cid = token_char.get(m.group(1).lower())
                    if cid is not None:
                        counts[cid] += 1
    return [cid for cid, _ in counts.most_common()]


def _same_paragraph(segments: List[tuple], seg_idx: int) -> bool:
    """True if this dialogue segment continues the paragraph of the previous
    one — i.e. the narration between them has no blank-line break. Measured on
    PDNC: quotes sharing a paragraph have the same speaker 96.8% of the time,
    quotes separated by a blank line only 23%, and directly adjacent quotes
    (no narration between) just 0.7% — so back-to-back quotes must NOT be
    treated as a continuation."""
    if seg_idx <= 0:
        return False
    kind, text = segments[seg_idx - 1]
    if kind == "narration":
        return "\n\n" not in text
    return False


def _vocative(text: str, token_char: Dict[str, object]):
    head = text.strip()
    for rx in (VOC_START_RE, VOC_END_RE):
        m = rx.search(head)
        if m:
            cid = token_char.get(m.group(1).lower())
            if cid is not None:
                return cid
    return None


def _token_char_map(raw_name_counts: Counter, name_to_root: dict, root_to_id: dict):
    """Map a lower-cased single token ('mary') to the character id it most
    often belongs to across the book's raw tagged names."""
    scores: Dict[str, Counter] = defaultdict(Counter)
    for raw, count in raw_name_counts.items():
        cid = root_to_id.get(name_to_root.get(raw))
        if not cid:
            continue
        for tok in base._name_tokens(raw):
            tok = tok.lower()
            if len(tok) >= 2:
                scores[tok][cid] += count
    return {tok: cnt.most_common(1)[0][0] for tok, cnt in scores.items()}


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def analyze_book_scene(book: Book, max_characters: int = 12,
                       unknown_below: Optional[float] = None,
                       narrator_gender: Optional[str] = None) -> dict:
    midsentence_caps = base._collect_midsentence_caps(book)

    # --- Pass A: tags + name counts ---
    chapter_scans = []
    raw_name_counts: Counter = Counter()
    tagged_raw_names = set()
    for chapter in book.chapters:
        segments = split_dialogue_narration(chapter.text)
        # Broad tags discover the cast; strict tags attribute lines.  Keeping
        # the two separate avoids losing a character whose name only appears
        # in a narrator aside while still refusing that mention as a speaker.
        tags = base._tag_dialogue_segments(
            segments, midsentence_caps, unique_pronoun_narration=True,
            paragraph_scoped=True)
        strict_tags = base._tag_dialogue_segments(
            segments, midsentence_caps, strict_action_beats=True,
            unique_pronoun_narration=True, paragraph_scoped=True)
        for seg_idx, (kind, text) in enumerate(segments):
            if kind != "dialogue":
                continue
            name, _snip, method = tags.get(seg_idx, (None, "", "unresolved"))
            if name and name not in base.PRONOUN_SENTINELS:
                raw_name_counts[name] += 1
                if method == "tag":
                    tagged_raw_names.add(name)
        chapter_scans.append((segments, strict_tags))

    # --- Cluster + rank characters (same identity model as the heuristic) ---
    clusters, name_to_root = base._cluster_names(raw_name_counts)
    characters, root_to_id = base._build_cast(
        book, clusters, name_to_root, raw_name_counts, tagged_raw_names,
        midsentence_caps, max_characters)
    base.add_minor_characters(characters)  # Unknown stays the final state
    base.assign_voices(characters)
    char_by_id = {c["id"]: c for c in characters}
    narrator_id = next((c["id"] for c in characters if c.get("narrator")), None)

    token_char = _token_char_map(raw_name_counts, name_to_root, root_to_id)

    # --- Pass B: per-scene speaker assignment ---
    attribution: Dict[str, List[dict]] = {}
    minor_mentions: Counter = Counter()

    for chapter_num, (segments, tags) in enumerate(chapter_scans, start=1):
        chapter_attr: List[dict] = []
        carry_presence: List[object] = []
        for (s_start, s_end) in _scene_bounds(segments):
            scene = segments[s_start:s_end]
            presence = _scene_presence(scene, token_char, midsentence_caps) or carry_presence
            carry_presence = presence
            # Participants = characters actually tagged as speaking somewhere in
            # this scene. Far more reliable than "mentioned somewhere in the
            # scene": in a two-hander, the tagged tag-team *is* the dialogue.
            participants: List[str] = []
            part_counts: Counter = Counter()
            first_tagged: Optional[str] = None
            for seg_idx in range(s_start, s_end):
                if segments[seg_idx][0] != "dialogue":
                    continue
                raw_name, _snip, method = tags.get(seg_idx, (None, "", ""))
                if raw_name:
                    cid = root_to_id.get(name_to_root.get(raw_name))
                    if cid:
                        part_counts[cid] += 1
                        if first_tagged is None and method in ("tag", "narrator"):
                            first_tagged = cid
                # A self-introduction ("Hi, I'm Willem") is a speaker
                # announcement, not just a mention.  It is often the first
                # time a character speaks, before narration has tagged them.
                cid = base.self_introduced_name(
                    segments[seg_idx][1], name_to_root, root_to_id)
                if cid:
                    part_counts[cid] += 1
                    if first_tagged is None:
                        first_tagged = cid
            # A first-person narrator is a participant even when the scene
            # only names one other person.  Without this, an untagged reply
            # such as "No. It's already all arranged." inherits the other
            # character's voice, which is especially confusing when both are
            # women.  Only do it for a one-person scene: a scene with two
            # tagged others is usually a conversation the narrator is
            # observing, not part of.
            if narrator_id and narrator_id not in part_counts:
                named_present = [c for c in presence if c != "unknown"]
                if len(part_counts) == 1:
                    part_counts[narrator_id] = 0
                elif not part_counts and len(named_present) == 1:
                    part_counts[named_present[0]] = 1
                    part_counts[narrator_id] = 0
            participants = [cid for cid, _ in part_counts.most_common()]
            # In a two-hander whose first quote is untagged, the first
            # *tagged* line is usually the other speaker's reply.  Seed the
            # alternation with it so the opening line does not inherit the
            # most frequent tag by default.
            last: Optional[str] = first_tagged if len(participants) == 2 else None
            recent: List[str] = []  # most recent first

            # Same-paragraph quotes are one speaker 96.8% of the time, so treat
            # each maximal chain of them as a single speaker turn: resolve the
            # group once (anchored on any real tag in it) instead of letting a
            # single mistake cascade down the paragraph.
            dial_idx = [i for i in range(s_start, s_end) if segments[i][0] == "dialogue"]
            groups: List[List[int]] = []
            for idx in dial_idx:
                if groups and _same_paragraph(segments, idx):
                    groups[-1].append(idx)
                else:
                    groups.append([idx])

            for group in groups:
                anchor = None
                for idx in group:
                    raw_name, _snip, found_method = tags.get(idx, (None, "", "unresolved"))
                    if raw_name and found_method in ("tag", "narrator"):
                        cid = root_to_id.get(name_to_root.get(raw_name))
                        if cid:
                            anchor = cid
                            break

                group_guess = None
                for seg_idx in group:
                    kind, text = segments[seg_idx]
                    raw_name, _snippet, found_method = tags.get(seg_idx, (None, "", "unresolved"))
                    character_id = confidence = method = None
                    pronoun_gender = None

                    if raw_name in base.PRONOUN_SENTINELS:
                        pronoun_gender = ("male" if raw_name == base.PRONOUN_TAG_M
                                          else "female")
                        cid = base._same_gender_candidate(
                            pronoun_gender, recent + participants, char_by_id)
                        if cid:
                            character_id, confidence, method = cid, "medium", "pronoun"
                        else:
                            # No named speaker of that gender is active.  Do
                            # not hand a "he says" line to a female character
                            # just because she is the only participant; use a
                            # male minor voice instead.  This is the common
                            # first-appearance case ("... he says" before the
                            # character is named).
                            same_gender = [
                                c for c in list(participants) + list(presence)
                                if (char_by_id.get(c) or {}).get("gender") == pronoun_gender
                            ]
                            if same_gender:
                                character_id = same_gender[0]
                                confidence, method = "medium", "pronoun-presence"
                            else:
                                character_id = ("minor_m" if pronoun_gender == "male"
                                                else "minor_f")
                                confidence, method = "low", f"pronoun-{pronoun_gender}"

                    elif raw_name:
                        cid = root_to_id.get(name_to_root.get(raw_name))
                        if cid:
                            character_id, method = cid, found_method
                            confidence = "high" if found_method in ("tag", "narrator") else "medium"
                        else:
                            character_id, confidence = "unknown", "low"
                            method = f"{found_method}-minor"
                            if raw_name != base.NARRATOR_SENTINEL:
                                minor_mentions[raw_name] += 1

                    if character_id in (None, "unknown", "minor_m", "minor_f"):
                        # '"Amanda Thomas," she says.' — a quote that is just a
                        # full name identifies the speaker even without a name tag.
                        # The same applies when a nearby object mention already
                        # put this line in the Unknown bucket: the quote itself
                        # is stronger evidence than the mention.
                        cid = base.self_introduced_name(text, name_to_root, root_to_id)
                        if cid:
                            character_id, confidence, method = cid, "medium", "self-intro"

                    if character_id is None:
                        if anchor is not None:
                            character_id, confidence, method = anchor, "high", "scene-same-par"
                        elif group_guess is not None:
                            character_id, confidence, method = group_guess
                        else:
                            character_id, confidence, method = _infer_speaker(
                                text, presence, participants, last, recent, token_char)

                    if not raw_name and character_id not in (
                            None, "unknown", "minor_m", "minor_f"):
                        # Continue a recently-heard unnamed speaker when the
                        # narration still says "he/she" and no named character
                        # of that gender is in the scene.  This is the first
                        # appearance of a character before the book names him
                        # ("I think she's right," he says).  Restricting it to
                        # an already-active minor keeps it from overriding
                        # ordinary named dialogue elsewhere.
                        resolved = char_by_id.get(character_id) or {}
                        adjacent_gender = base.minor_gender(segments, seg_idx)
                        if (adjacent_gender and resolved.get("gender")
                                and resolved["gender"] != adjacent_gender):
                            minor_id = ("minor_m" if adjacent_gender == "male"
                                        else "minor_f")
                            named_same_gender = any(
                                (char_by_id.get(c) or {}).get("gender") == adjacent_gender
                                and c not in ("minor_m", "minor_f", "unknown")
                                for c in list(participants) + list(presence))
                            if (not named_same_gender
                                    and (last == minor_id or minor_id in recent)):
                                character_id = minor_id
                                confidence = "low"
                                method = f"scene-gender-{adjacent_gender}"

                    if group_guess is None and character_id not in (None, "unknown"):
                        # The next line(s) in this paragraph almost certainly
                        # belong to the same speaker.  A pronoun-tagged minor
                        # is still a real speaker; only a completely
                        # unresolved/minor line should not seed the group.
                        if (character_id not in ("minor_m", "minor_f")
                                or (method or "").startswith("pronoun")):
                            group_guess = (character_id, confidence, "scene-same-par")

                    if character_id != "unknown":
                        if last != character_id:
                            recent.insert(0, character_id)
                            del recent[4:]
                        last = character_id
                    else:
                        # No name for this speaker: at least give them a voice
                        # of the right gender, using the narration around the
                        # quote ("he says", "the man repeats").
                        gender = pronoun_gender or base.minor_gender(segments, seg_idx)
                        if gender:
                            character_id = "minor_m" if gender == "male" else "minor_f"
                            method = f"{method}-{gender}"

                    chapter_attr.append({
                        "seg_idx": seg_idx,
                        "character_id": character_id,
                        "confidence": confidence,
                        "method": method,
                    })
        attribution[str(chapter_num)] = chapter_attr

    final_counts: Counter = Counter()
    for entries in attribution.values():
        for e in entries:
            final_counts[e["character_id"]] += 1
    for ch in characters:
        ch["line_count"] = final_counts.get(ch["id"], 0)

    return {
        "version": base.HEURISTIC_VERSION,
        "mode": "scene",
        "scene_version": SCENE_VERSION,
        "book_title": book.title,
        "max_characters": max_characters,
        "characters": characters,
        "attribution": attribution,
        "minor_mentions": dict(minor_mentions.most_common(30)),
    }


def _infer_speaker(text: str, presence: List[object], participants: List[str],
                   last: Optional[str], recent: List[str], token_char: Dict[str, object]):
    """Resolve an untagged dialogue line. Returns (character_id, confidence,
    method).

    The participant set (characters tagged as speaking in this scene) is
    trusted first: when a scene has exactly two participants, untagged lines
    alternate strictly between them. Only when a scene has no/none/many
    participants do we fall back to everyone *mentioned* in the scene."""
    voc = _vocative(text, token_char)

    def alternate(pool: List[str], method: str):
        a, b = pool[0], pool[1]
        if voc in (a, b):
            return (b if voc == a else a), "medium", method
        if last == a:
            return b, "medium", method
        if last == b:
            return a, "medium", method
        for cid in recent:            # seed from the more recently used one
            if cid == a:
                return b, "medium", method
            if cid == b:
                return a, "medium", method
        return a, "medium", method

    if len(participants) == 2:
        return alternate(participants, "scene-alternate")
    if len(participants) > 2:
        # More than two tagged speakers in the scene, but the vast majority of
        # scenes are really a two-hander; alternate between the two that are
        # actually tagged most, rather than round-robining among everyone.
        return alternate(participants[:2], "scene-core-alt")

    # Zero or one participant: use who is mentioned in the scene.
    present = [c for c in presence if c != "unknown"]
    if len(present) == 2:
        return alternate(present, "scene-presence-alt")
    if len(present) > 2:
        pool = participants or present
        for cid in reversed(recent):
            if cid in pool and cid != voc:
                return cid, "low", "scene-recent"
        for cid in pool:
            if cid != voc:
                return cid, "low", "scene-roundrobin"
    if len(present) == 1:
        if voc == present[0]:
            return "unknown", "low", "scene-addressee"
        return present[0], "medium", "scene-single"

    if participants:
        return participants[0], "low", "scene-participant"
    if last:
        return last, "low", "scene-carry"
    return "unknown", "low", "unresolved"
