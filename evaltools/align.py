"""Turn chapter text into (kind, text, start, end) segments whose ``kind``/``text``
match ``btlib.extract.split_dialogue_narration`` exactly, but which also carry
character offsets. This lets us line the pipeline's segmentation up against the
gold quote spans from PDNC / LitBank.
"""

import re
from typing import List, Tuple

# Must stay identical to btlib.extract.DIALOGUE_RE.
DIALOGUE_RE = re.compile(r'("[^"]*"|\u201c[^\u201d]*\u201d)', re.DOTALL)

Span = Tuple[str, str, int, int]  # (kind, text, start, end)


def segment_spans(text: str) -> List[Span]:
    """``split_dialogue_narration`` with offsets bolted on."""
    segments: List[Span] = []
    pos = 0
    for m in DIALOGUE_RE.finditer(text):
        if m.start() > pos:
            narration = text[pos:m.start()]
            if narration.strip():
                segments.append(("narration", narration, pos, m.start()))
        segments.append(("dialogue", m.group(0), m.start(), m.end()))
        pos = m.end()
    if pos < len(text):
        tail = text[pos:]
        if tail.strip():
            segments.append(("narration", tail, pos, len(text)))
    if not segments:
        segments = [("narration", text, 0, len(text))]
    return segments


def verify_segmentation_matches_extract(text: str) -> bool:
    """Sanity check: our offset version must be byte-for-byte the same split the
    runtime pipeline performs."""
    from btlib.extract import split_dialogue_narration

    theirs = split_dialogue_narration(text)
    ours = [(kind, seg) for kind, seg, _, _ in segment_spans(text)]
    return theirs == ours


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_spans(gold: List[Tuple[int, int]],
                pred: List[Tuple[int, int]],
                threshold: float = 0.5):
    """Greedy one-to-one matching by best IoU.

    Returns (pairs, unmatched_gold_idx, unmatched_pred_idx) where ``pairs`` is a
    list of ``(gold_index, pred_index, score)`` sorted by gold index.
    """
    scored = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            s = iou(g, p)
            if s >= threshold:
                scored.append((s, gi, pi))
    scored.sort(reverse=True)

    used_g, used_p = set(), set()
    pairs = []
    for s, gi, pi in scored:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, s))

    pairs.sort()
    unmatched_gold = [i for i in range(len(gold)) if i not in used_g]
    unmatched_pred = [i for i in range(len(pred)) if i not in used_p]
    return pairs, unmatched_gold, unmatched_pred
