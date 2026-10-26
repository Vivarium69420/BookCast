"""Run an attribution engine over an eval corpus and score it against gold."""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .align import match_spans, segment_spans
from .corpora import Doc, _norm
from .metrics import Metrics

ENGINES = ("heuristic", "scene", "whole", "light", "heavy")

_UNKNOWN = {"", "unknown", "narrator", "narrator (i)", "none", "n/a"}
_UNKNOWN_IDS = {"unknown", "minor_f", "minor_m", "narrator"}
_GENDER_MAP = {"male": "M", "female": "F", "m": "M", "f": "F", "a": "A", "u": "U"}
# A gold speaker needs at least this many quotes to count as a "major character"
# for the character-recall metric (filters one-line walk-ons).
MAJOR_MIN_QUOTES = 5


def _is_unknown_id(cid) -> bool:
    return cid in _UNKNOWN_IDS


def _is_unknown_name(name: Optional[str]) -> bool:
    if not name:
        return True
    n = _norm(name)
    return n in _UNKNOWN or n.startswith("unknown") or n.startswith("unnamed")


# --------------------------------------------------------------------------
# Engine drivers
# --------------------------------------------------------------------------

def run_engine(doc: Doc, engine: str, max_characters: int = 12,
               narrator_gender: Optional[str] = None,
               unknown_below: Optional[float] = None,
               llm_model: Optional[str] = None,
               llm_endpoint: Optional[str] = None,
               llm_strategy: str = "weak",
               llm_batch_size: int = 8,
               llm_confidence: float = 0.6,
               llm_cache_dir: Optional[str] = None,
               llm_max_calls: Optional[int] = None) -> Tuple[dict, float]:
    book = doc.to_book()
    t0 = time.time()
    if engine == "heuristic":
        from btlib.cast import analyze_book
        data = analyze_book(book, max_characters=max_characters)
    elif engine == "scene":
        from btlib.cast_scene import analyze_book_scene
        data = analyze_book_scene(book, max_characters=max_characters,
                                  unknown_below=unknown_below,
                                  narrator_gender=narrator_gender)
    elif engine == "whole":
        from btlib.cast_whole import analyze_book_whole
        data = analyze_book_whole(book, max_characters=max_characters,
                                  unknown_below=unknown_below,
                                  narrator_gender=narrator_gender)
    elif engine == "light":
        from btlib.cast_light import analyze_book_light
        data = analyze_book_light(book, max_characters=max_characters,
                                  narrator_gender=narrator_gender)
    elif engine == "heavy":
        from btlib.cast_llm import analyze_book_llm
        data = analyze_book_llm(book, max_characters=max_characters,
                                model=llm_model or "llama3:8b",
                                endpoint=llm_endpoint or "http://localhost:11434/api/generate",
                                narrator_gender=narrator_gender,
                                strategy=llm_strategy,
                                batch_size=llm_batch_size,
                                confidence_threshold=llm_confidence,
                                cache_dir=llm_cache_dir,
                                max_calls=llm_max_calls,
                                # An eval must never silently report scene numbers
                                # under the "heavy" label when Ollama is absent.
                                strict=True)
    else:
        raise ValueError(f"unknown engine {engine!r}")
    return data, time.time() - t0


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _cast_tables(cast_data: dict):
    chars = {c.get("id"): c for c in cast_data.get("characters", [])}
    att: Dict[int, Dict[int, Optional[str]]] = {}
    for k, v in (cast_data.get("attribution") or {}).items():
        att[int(k)] = {int(e["seg_idx"]): e.get("character_id") for e in v}
    # Engines number chapters from either 0 or 1; detect which.
    base = 0 if 0 in att else (1 if 1 in att else 0)
    return chars, att, base


def _alias_to_main(doc: Doc) -> Dict[str, str]:
    """Map every normalised alias to the normalised main name.

    Values are normalised so callers can compare them directly to
    ``_norm(gold_speaker)``; returning the raw main name made gender lookup
    (``doc.genders`` is keyed by ``_norm(main)``) and gold-speaker counts fail
    for multi-word names.
    """
    alias_to_main: Dict[str, str] = {}
    for main, als in doc.aliases.items():
        main_n = _norm(main)
        for a in als:
            alias_to_main.setdefault(_norm(a), main_n)
        alias_to_main.setdefault(main_n, main_n)
    return alias_to_main


def _make_resolver(doc: Doc):
    alias_to_main = _alias_to_main(doc)

    def resolve(pred_name: Optional[str], gold_name: str) -> bool:
        if not pred_name:
            return False
        pn, gn = _norm(pred_name), _norm(gold_name)
        if pn in _UNKNOWN:
            return False
        # Canonicalise *both* sides.  The old code mapped the prediction but
        # compared it to the raw gold string, so a correct prediction failed
        # whenever PDNC labelled the quote with an alias ("Lucy" vs
        # "Miss Honeychurch").
        pn = alias_to_main.get(pn, pn)
        gn = alias_to_main.get(gn, gn)
        if pn == gn:
            return True
        # Token-subset fallback: PDNC's alias tables omit common short forms
        # ("Jake" for "Jake Barnes"), so without this 4.5% of quotes cannot be
        # scored correctly no matter what the engine predicts.
        pt, gt = set(pn.split()), set(gn.split())
        return bool(pt) and bool(gt) and (pt <= gt or gt <= pt)

    return resolve


def _cluster_identity_metrics(doc: Doc, chars: dict, exclude_ids: set, gold_counts: Dict[str, int]):
    """Clustering + gender health measured from the cast itself (names + aliases
    mapped to gold characters), independent of attribution accuracy:

    * merge  — one predicted character covers >1 gold character
    * split  — one gold character is covered by >1 predicted character
    * gender — a predicted character's gender matches its gold character's
    """
    alias_to_main = _alias_to_main(doc)

    cid_to_gold: Dict[object, set] = defaultdict(set)
    for cid, ch in chars.items():
        if cid in exclude_ids:
            continue
        names = [ch.get("name") or ""] + list(ch.get("aliases") or [])
        for nm in names:
            g = alias_to_main.get(_norm(nm))
            if g:
                cid_to_gold[cid].add(g)

    merges = sum(1 for s in cid_to_gold.values() if len(s) > 1)
    gold_to_cid: Dict[str, set] = defaultdict(set)
    for cid, golds in cid_to_gold.items():
        for g in golds:
            gold_to_cid[g].add(cid)
    splits = sum(1 for s in gold_to_cid.values() if len(s) > 1)

    gender_correct = gender_total = 0
    for cid, golds in cid_to_gold.items():
        pg = _GENDER_MAP.get(str(chars.get(cid, {}).get("gender") or "").lower())
        for g in golds:
            gg = doc.genders.get(g)
            if pg in ("M", "F", "A") and gg in ("M", "F", "A"):
                gender_total += 1
                gender_correct += int(pg == gg)

    # Character detection: can a named cast member be mapped to a real gold
    # character (precision), and do we have a name for every major gold speaker
    # (recall)? Unlike merge/split this counts *spurious* characters, which
    # never appear in cid_to_gold. Only meaningful when the corpus supplies an
    # alias table (PDNC); without one nothing maps, so report "not measured".
    if alias_to_main:
        named = [cid for cid in chars if cid not in exclude_ids]
        cast_named = len(named)
        cast_real = sum(1 for cid in named if cid in cid_to_gold)
        gold_major = sum(1 for g, n in gold_counts.items() if n >= MAJOR_MIN_QUOTES)
        gold_major_covered = sum(1 for g, n in gold_counts.items()
                                 if n >= MAJOR_MIN_QUOTES and g in gold_to_cid)
    else:
        cast_named = cast_real = gold_major = gold_major_covered = 0

    return (merges, len(cid_to_gold), splits, len(gold_to_cid), gender_correct,
            gender_total, cast_named, cast_real, gold_major, gold_major_covered)


def score_doc(doc: Doc, cast_data: dict) -> Metrics:
    m = Metrics()
    chars, att, base = _cast_tables(cast_data)
    resolve = _make_resolver(doc)
    # Alias -> main name, so alternate surface forms of one character are not
    # mistaken for a speaker change (matters for back-and-forth detection and
    # for matching gold speakers to predicted clusters).
    a2m = _alias_to_main(doc)

    ordered: List[Tuple[str, Optional[int], bool]] = []  # (gold speaker, cid, correct)

    for ci, (title, text) in enumerate(doc.chapters):
        segs = segment_spans(text)
        dial_idx = [i for i, s in enumerate(segs) if s[0] == "dialogue"]
        dial_spans = [(segs[i][2], segs[i][3]) for i in dial_idx]

        golds = [(q.start, q.end, q) for (c, q) in doc.quotes if c == ci]
        if not golds and not dial_spans:
            continue
        golds.sort(key=lambda g: (g[0], g[1]))
        gold_spans = [(g[0], g[1]) for g in golds]

        pairs, unmatched_gold, unmatched_pred = match_spans(gold_spans, dial_spans)
        m.det_tp += len(pairs)
        m.det_fn += len(unmatched_gold)
        m.det_fp += len(unmatched_pred)

        gi_to_pi = {gi: pi for gi, pi, _ in pairs}
        att_ch = att.get(ci + base, {})

        for gi, (gs, ge, q) in enumerate(golds):
            m.attr_e2e_total += 1
            cid = None
            pred_name = None
            if gi in gi_to_pi:
                seg_idx = dial_idx[gi_to_pi[gi]]
                cid = att_ch.get(seg_idx)
                pred_name = chars.get(cid, {}).get("name") if cid is not None else None
            if cid is None or _is_unknown_id(cid) or _is_unknown_name(pred_name):
                m.unknown += 1

            correct = (not _is_unknown_id(cid)) and resolve(pred_name, q.speaker)
            if gi in gi_to_pi:
                m.attr_given_total += 1
                if correct:
                    m.attr_given_correct += 1
            if correct:
                m.attr_e2e_correct += 1

            if cid is not None and not _is_unknown_id(cid) and not _is_unknown_name(pred_name):
                pass  # gender is scored per-cluster below, not per attributed quote

            ordered.append((a2m.get(_norm(q.speaker), _norm(q.speaker)), cid, correct))

    # back-and-forth subset: A B A patterns across the doc's gold quote sequence
    for i in range(2, len(ordered)):
        if ordered[i][0] != ordered[i - 1][0] and ordered[i][0] == ordered[i - 2][0]:
            m.bf_total += 1
            m.bf_correct += int(ordered[i][2])

    # Gold quote speakers may be *aliases* of a character (e.g. PDNC labels
    # Dorian Gray's lines with the alias while the main name is "Prince
    # Charming"), so resolve them to the main name before counting.
    gold_counts: Dict[str, int] = defaultdict(int)
    for _c, q in doc.quotes:
        s = _norm(q.speaker)
        if s and s not in _UNKNOWN:
            gold_counts[a2m.get(s, s)] += 1

    merges, clusters, splits, speakers, gc, gt, cn, cr, gm, gmc = _cluster_identity_metrics(
        doc, chars, _UNKNOWN_IDS, gold_counts)
    m.merge_clusters, m.clusters = merges, clusters
    m.split_speakers, m.speakers = splits, speakers
    m.gender_correct, m.gender_total = gc, gt
    m.cast_named, m.cast_real = cn, cr
    m.gold_major, m.gold_major_covered = gm, gmc
    return m


def evaluate(docs: List[Doc], engine: str, **kwargs) -> Tuple[Metrics, List[tuple]]:
    total = Metrics()
    per_doc = []
    for doc in docs:
        cast_data, secs = run_engine(doc, engine, **kwargs)
        m = score_doc(doc, cast_data)
        total.add(m)
        per_doc.append((doc.doc_id, m, secs))
        total.per_doc = per_doc
    return total, per_doc
