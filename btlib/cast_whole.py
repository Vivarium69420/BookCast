import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from . import cast as base
from .extract import split_dialogue_narration
from .models import Book

WHOLE_BOOK_VERSION = 2
UNKNOWN_BELOW = 0.3
HIGH_AT = 0.85
MEDIUM_AT = 0.55
DEFAULT_PARAMS = {"cont": 0.9, "alt": 0.6, "self": 0.2, "alt_q": 0.7, "self_q": 0.05}
EM_ROUNDS = 4
SCENE_BREAK_MIX = 0.5
LONG_GAP_CHARS = 450
E_TAG_OTHER = 0.04
E_BEAT_NAME_OTHER = 0.3
E_PRONOUN_VERB_MISMATCH = 0.1
E_PRONOUN_BEAT_MISMATCH = 0.3
E_FIRST_PERSON_OTHER = 0.3
E_VOCATIVE_SELF = 0.15

_SPEECH = "|".join(base.SPEECH_VERBS)
_LEAD = r"^[\s,;:—–\-]*"
RE_TAIL_PRON = re.compile(_LEAD + r"(he|she)\b", re.IGNORECASE)
RE_TAIL_PRON_VERB = re.compile(_LEAD + rf"(he|she)\s+(?:{_SPEECH})\b", re.IGNORECASE)
RE_FIRST = re.compile(_LEAD + r"I(?:\b|['’](?:m|ll|d|ve))")
RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
RE_VOC_START = re.compile(r'^[“"]?\s*([A-Z][a-z]+)\s*[,.!?—]')
RE_VOC_END = re.compile(r'[,\s]\s*([A-Z][a-z]+)\s*[.!?]?[”"]?$')


def _gender_of_pron(word: str) -> str:
    return "male" if word.lower() == "he" else "female"


def _line_features(chapter_num: int, segments: List[tuple], seg_idx: int) -> dict:
    f = {
        "same_par": False, "after_question": False, "gap": 0, "pron_tail": None, "pron_tail_verb": False,
        "pron_head": None, "first_tail": False, "first_head": False, "vocatives": set(),
    }
    prev = segments[seg_idx - 1] if seg_idx >= 1 else None
    nxt = segments[seg_idx + 1] if seg_idx + 1 < len(segments) else None
    # narration immediately before the quote: gender/pronoun of the previous beat
    if prev and prev[0] == "narration":
        prev_text = prev[1]
        f["gap"] = len(prev_text)
        if seg_idx >= 2 and segments[seg_idx - 2][0] == "dialogue" and "\n" not in prev_text:
            f["same_par"] = True
        head = prev_text.rsplit("\n", 1)[-1]
        parts = [p for p in RE_SENT_SPLIT.split(head.strip()) if p]
        last = parts[-1] if parts else ""
        m = re.match(r"^\W*(he|she)\b", last, re.IGNORECASE)
        if m:
            f["pron_head"] = _gender_of_pron(m.group(1))
        if re.match(r"^\W*I(?:\b|['’](?:m|ll|d|ve))", last):
            f["first_head"] = True
    # narration immediately after the quote
    if nxt and nxt[0] == "narration":
        tail = nxt[1].split("\n", 1)[0]
        m = RE_TAIL_PRON.match(tail)
        if m:
            f["pron_tail"] = _gender_of_pron(m.group(1))
            f["pron_tail_verb"] = bool(RE_TAIL_PRON_VERB.match(tail))
        if RE_FIRST.match(tail):
            f["first_tail"] = True
    # was the nearest previous quote a question?
    j = seg_idx - 1
    while j >= 0 and segments[j][0] != "dialogue":
        j -= 1
    if j >= 0 and re.search(r'\?[\u201d\"]?\s*$', segments[j][1]):
        f["after_question"] = True
    # names the speaker is addressing
    quote = segments[seg_idx][1]
    if len(quote.split()) > 2:
        for rx in (RE_VOC_START, RE_VOC_END):
            m = rx.search(quote.strip())
            if not m: continue
            f["vocatives"].add(m.group(1))
    return f


def _normalize(vec: List[float]) -> List[float]:
    s = sum(vec)
    if s <= 0:
        return [1.0 / len(vec)] * len(vec)
    return [v / s for v in vec]


def _build_transitions(S: int, pop: List[float], partner: List[List[float]], params: Optional[dict] = None):
    prm = dict(DEFAULT_PARAMS)
    prm.update(params or {})
    p_cont = prm["cont"]

    def others(j, exclude):
        w = [partner[j][k] + 0.5 * pop[k] for k in range(S)]
        for e in exclude:
            if e is not None and 0 <= e < S:
                w[e] = 0.0
        return _normalize(w) if sum(w) > 0 else _normalize(list(pop))

    T = [[[None] * S for _ in range(S + 1)] for _ in range(4)]
    for i in range(S + 1):
        for j in range(S):
            for kind, (a_key, s_key) in ((0, ("alt", "self")), (3, ("alt_q", "self_q"))):
                p_alt, p_self = prm[a_key], prm[s_key]
                if i < S and i != j:
                    rest = others(j, [i, j])
                    row = [(1 - p_alt - p_self) * r for r in rest]
                    row[i] += p_alt
                    row[j] += p_self
                else:
                    rest = others(j, [j])
                    row = [(1 - p_self) * r for r in rest]
                    row[j] += p_self
                T[kind][i][j] = _normalize(row)
            rest = others(j, [j])
            row = [(1 - p_cont) * r for r in rest]
            row[j] += p_cont
            T[1][i][j] = _normalize(row)
            fresh = _normalize(list(pop))
            row = [SCENE_BREAK_MIX * fresh[k] + (1 - SCENE_BREAK_MIX) * T[0][i][j][k] for k in range(S)]
            T[2][i][j] = _normalize(row)
    return T


def _forward_backward(S: int, emis: List[List[float]], kinds: List[int], T, pop: List[float]):
    n = len(emis)
    if n == 0:
        return [], {}

    def idx(i, j):
        return i * S + j

    P = (S + 1) * S
    alphas = []

    a = [0.0] * P
    for j in range(S):
        a[idx(S, j)] = pop[j] * emis[0][j]
    z = sum(a) or 1.0
    alphas.append([v / z for v in a])

    for t in range(1, n):
        Tk = T[kinds[t]]
        nxt = [0.0] * P
        e = emis[t]
        prev = alphas[-1]
        for i in range(S + 1):
            for j in range(S):
                av = prev[idx(i, j)]
                if av < 1e-12:
                    continue
                row = Tk[i][j]
                base_idx = j * S
                for k in range(S):
                    nxt[base_idx + k] += av * row[k] * e[k]
        z = sum(nxt) or 1.0
        alphas.append([v / z for v in nxt])

    betas = [None] * n
    betas[n - 1] = [1.0] * P
    for t in range(n - 2, -1, -1):
        Tk = T[kinds[t + 1]]
        e = emis[t + 1]
        bnext = betas[t + 1]
        b = [0.0] * P
        for i in range(S + 1):
            for j in range(S):
                row = Tk[i][j]
                tot = 0.0
                base_idx = j * S
                for k in range(S):
                    tot += row[k] * e[k] * bnext[base_idx + k]
                b[idx(i, j)] = tot
        z = max(b) or 1.0
        betas[t] = [v / z for v in b]

    post = []
    for t in range(n):
        marg = [0.0] * S
        a_t, b_t = alphas[t], betas[t]
        for i in range(S + 1):
            for j in range(S):
                marg[j] += a_t[idx(i, j)] * b_t[idx(i, j)]
        post.append(_normalize(marg))
    return post, {}


def _desc_re(nouns: str):
    return re.compile(rf"\b{_ART}(?:{nouns})\s+(?:{_SPEECH})\b|\b(?:{_SPEECH})\s+{_ART}(?:{nouns})\b", re.IGNORECASE)


_MALE_NOUNS = "man|boy|gentleman|guy|fellow|lad|father|son|husband|brother|priest|king"
_FEMALE_NOUNS = "woman|girl|lady|mother|daughter|wife|sister|queen|maid"
_ART = r"(?:the|a|an|his|her|their)\s+(?:[A-Za-z-]+\s+){0,2}"
_DESC_M = _desc_re(_MALE_NOUNS)
_DESC_F = _desc_re(_FEMALE_NOUNS)
_PRON_SUBJ = re.compile(r"\b(he|she)\b", re.IGNORECASE)
MINOR_F, MINOR_M = "minor_f", "minor_m"
E_MODEL_OTHER = 0.15
SCENE_ABSENT = 0.5
MINOR_EXPLAINED = 0.4
UNKNOWN_IN_SCENE = 0.5
E_DESCRIPTOR_OTHER = 0.15
_MALE_TITLES = {"Mr", "Sir", "Uncle"}
_FEMALE_TITLES = {"Ms", "Mrs", "Aunt", "Lady", "Miss"}


def _descriptor_gender(segments: List[tuple], seg_idx: int) -> Optional[str]:
    """'the woman said' / 'said the old man' next to an untagged quote -> an unnamed speaker of that gender."""
    snippets = []
    if seg_idx + 1 < len(segments) and segments[seg_idx + 1][0] == "narration":
        snippets.append(segments[seg_idx + 1][1][:120])
    if seg_idx - 1 >= 0 and segments[seg_idx - 1][0] == "narration":
        snippets.append(segments[seg_idx - 1][1][-120:])
    for sn in snippets:
        m, f = _DESC_M.search(sn), _DESC_F.search(sn)
        if m and not f:
            return "male"
        if f and not m:
            return "female"
    return None


def _pronoun_after_name(name: Optional[str], snippet: str) -> Optional[str]:
    """In 'Anna said, her voice low.' the he/she that follows the name (same sentence) tells Anna's gender."""
    if not name or name == base.NARRATOR_SENTINEL or not snippet:
        return None
    i = snippet.find(name)
    if i < 0:
        return None
    rest = snippet[i + len(name):]
    end = re.search(r"[.!?]", rest)
    m = _PRON_SUBJ.search(rest[:end.start()] if end else rest)
    if not m:
        return None
    return "male" if m.group(1).lower() == "he" else "female"


def _resolve_gender(variants, votes: Counter) -> Optional[str]:
    from .names import name_gender
    score = 0
    for n in variants:
        toks = n.split()
        title = toks[0].strip(".")
        if title in _MALE_TITLES:
            score += 3
        elif title in _FEMALE_TITLES:
            score -= 3
        for t in toks:
            g = name_gender(t)
            if not g: continue
            score += 3 if g == "male" else -3
    score += votes["male"] - votes["female"]
    if score >= 2:
        return "male"
    if score <= -2:
        return "female"
    return None


def analyze_book_whole(book: Book, max_characters: int = 12, unknown_below: Optional[float] = None,
                       narrator_gender: Optional[str] = None, resolver=None, gender_hints=None) -> dict:
    """Whole-book speaker attribution with a turn-taking HMM.

    Evidence per line: explicit names (regex or an AI `resolver`), he/she clues, "the woman said" descriptors,
    first-person beats, and names the speaker is addressing. Structure: same paragraph = same speaker,
    new paragraph = the other person, long narration = new scene. Unnamed people get gender-matched
    "minor" voices instead of one generic Unknown.
    """
    cutoff = UNKNOWN_BELOW if unknown_below is None else unknown_below
    midsentence_caps = base._collect_midsentence_caps(book)

    scans = []
    raw_name_counts = Counter()
    pron_votes = defaultdict(Counter)
    for ch_num, chapter in enumerate(book.chapters, start=1):
        segments = split_dialogue_narration(chapter.text)
        # The HMM has no notion of the pronoun sentinels, so keep its tag stream
        # to the classic name-only passes.
        tags = base._tag_dialogue_segments(segments, midsentence_caps, pronouns=False)
        first_in_chapter = True
        scene_text = ""
        for seg_idx, (kind, _text) in enumerate(segments):
            if kind != "dialogue":
                scene_text = _text[-500:] if len(_text) > LONG_GAP_CHARS else (scene_text + " " + _text)[-1500:]
                continue
            raw_name, snippet, method = tags.get(seg_idx, (None, "", "unresolved"))
            if resolver is not None:
                raw_name, snippet, method = resolver(ch_num, segments, seg_idx, raw_name, snippet, method)
            feats = _line_features(ch_num, segments, seg_idx)
            if raw_name:
                raw_name_counts[raw_name] += 1
                g = _pronoun_after_name(raw_name, snippet)
                if g:
                    pron_votes[raw_name][g] += 1
            scans.append({
                "chapter": ch_num, "seg_idx": seg_idx, "raw_name": raw_name, "method": method,
                "feats": feats, "chapter_start": first_in_chapter,
                "desc_gender": None if raw_name else _descriptor_gender(segments, seg_idx),
                "scene_text": scene_text,
            })
            scene_text = (scene_text + " " + _text)[-1500:]
            first_in_chapter = False

    clusters, name_to_root = base._cluster_names(raw_name_counts)
    tagged_raw_names = {s["raw_name"] for s in scans
                        if s["raw_name"] and s["method"] == "tag"}

    def whole_gender(root, variants, is_narrator):
        if is_narrator and narrator_gender:
            return narrator_gender
        votes = Counter()
        for n in variants:
            votes.update(pron_votes.get(n, {}))
            for g, c in (gender_hints or {}).get(n, {}).items():
                votes[g] += 2 * c
        return _resolve_gender(variants, votes)

    characters, root_to_id = base._build_cast(
        book, clusters, name_to_root, raw_name_counts, tagged_raw_names,
        midsentence_caps, max_characters, gender_resolver=whole_gender)
    unknown_char = characters.pop()  # Unknown must stay the final state
    n_named = len(characters)
    characters.append({"id": MINOR_F, "name": "Unnamed woman (minor)", "aliases": [], "gender": "female", "line_count": 0})
    characters.append({"id": MINOR_M, "name": "Unnamed man (minor)", "aliases": [], "gender": "male", "line_count": 0})
    characters.append(unknown_char)

    ids = [c["id"] for c in characters]
    S = len(ids)
    U = S - 1
    state_of = {cid: s for s, cid in enumerate(ids)}
    state_gender = [c["gender"] for c in characters]
    minor_state = {"female": state_of[MINOR_F], "male": state_of[MINOR_M]}
    narrator_state = next((state_of[c["id"]] for c in characters if c.get("narrator")), None)

    # first owner of a name token wins; tokens shared by two characters are dropped
    token_owner = {}
    for c in characters[:n_named]:
        for alias in c["aliases"]:
            for tok in base._name_tokens(alias):
                s_i = state_of[c["id"]]
                token_owner[tok] = s_i if token_owner.get(tok, s_i) == s_i else None

    def strong_state(line):
        raw = line["raw_name"]
        if not raw or line["method"] not in ("tag", "narrator"):
            return None
        if raw == base.NARRATOR_SENTINEL:
            return narrator_state
        return state_of.get(root_to_id.get(name_to_root.get(raw)))

    emis, kinds = [], []
    for line in scans:
        f = line["feats"]
        e = [1.0] * S
        raw_name, method = line["raw_name"], line["method"]
        if raw_name:
            st = narrator_state if raw_name == base.NARRATOR_SENTINEL else state_of.get(root_to_id.get(name_to_root.get(raw_name)), U)
            if method in ("tag", "narrator"):
                other = E_TAG_OTHER
            elif method in ("lightweight-qa", "heavy-llm"):
                other = E_MODEL_OTHER
            else:
                other = E_BEAT_NAME_OTHER
            if st is not None:
                for s_i in range(S):
                    if s_i != st:
                        e[s_i] *= other
        for pron, mism in ((f["pron_tail"], E_PRONOUN_VERB_MISMATCH if f["pron_tail_verb"] else E_PRONOUN_BEAT_MISMATCH),
                           (f["pron_head"], E_PRONOUN_BEAT_MISMATCH)):
            if not pron: continue
            for s_i in range(S):
                if state_gender[s_i] and state_gender[s_i] != pron:
                    e[s_i] *= mism
        if line["desc_gender"]:
            target = minor_state[line["desc_gender"]]
            for s_i in range(S):
                if s_i != target:
                    e[s_i] *= E_DESCRIPTOR_OTHER
        if narrator_state is not None and (f["first_tail"] or f["first_head"]) and not raw_name:
            for s_i in range(S):
                if s_i != narrator_state:
                    e[s_i] *= E_FIRST_PERSON_OTHER
        mentioned = {token_owner[w] for w in set(re.findall(r"[A-Z][a-z]+", line["scene_text"])) if token_owner.get(w) is not None}
        if mentioned:
            for s_i in range(n_named):
                if s_i not in mentioned and s_i != narrator_state:
                    e[s_i] *= SCENE_ABSENT
            present = [s_i for s_i in mentioned if s_i != narrator_state]
            for g, ms in minor_state.items():
                if any(state_gender[s_i] in (g, None) for s_i in present):
                    e[ms] *= MINOR_EXPLAINED
            if not raw_name:
                e[U] *= UNKNOWN_IN_SCENE
        for voc in f["vocatives"]:
            owner = token_owner.get(voc)
            if owner is None: continue
            e[owner] *= E_VOCATIVE_SELF
        emis.append(e)
        if f["same_par"]:
            kind = 1
        elif line["chapter_start"] or f["gap"] > LONG_GAP_CHARS:
            kind = 2
        elif f["after_question"]:
            kind = 3
        else:
            kind = 0
        kinds.append(kind)

    strong = [strong_state(line) for line in scans]
    counts = Counter(st for st in strong if st is not None)
    total = sum(counts.values())
    avg = total / max(1, n_named)
    pop = [counts[s_i] + 1.0 for s_i in range(S)]
    pop[minor_state["female"]] = pop[minor_state["male"]] = 1.0 + 0.2 * avg
    pop[U] = 0.3
    pop = _normalize(pop)
    partner = [[0.1] * S for _ in range(S)]
    for i in range(len(scans) - 1):
        a, b = scans[i], scans[i + 1]
        sa, sb = strong[i], strong[i + 1]
        if a["chapter"] != b["chapter"] or b["feats"]["same_par"] or sa is None or sb is None or sa == sb: continue
        partner[sa][sb] += 1
        partner[sb][sa] += 1

    pairs = same = 0
    for i in range(len(scans) - 1):
        a, b = scans[i], scans[i + 1]
        if a["chapter"] != b["chapter"] or b["feats"]["same_par"] or b["feats"]["gap"] > LONG_GAP_CHARS: continue
        if strong[i] is None or strong[i + 1] is None: continue
        pairs += 1
        same += strong[i] == strong[i + 1]
    p_self = min(0.4, max(0.03, (same + 1.2) / (pairs + 8)))
    p_alt = 0.88 - p_self
    params = {"alt": p_alt, "self": p_self, "alt_q": min(0.9, p_alt + 0.1), "self_q": max(0.02, p_self / 4)}

    T = _build_transitions(S, pop, partner, params)
    post, _ = _forward_backward(S, emis, kinds, T, pop)

    attribution = {str(i): [] for i in range(1, len(book.chapters) + 1)}
    for line, p in zip(scans, post):
        best = max(range(S - 1), key=lambda s_i: p[s_i])
        if p[U] > max(p[best], 0.5):
            best = U
        conf_p = p[best]
        if best != U and conf_p >= cutoff:
            cid = ids[best]
        else:
            pg = line["feats"]["pron_tail"] or line["feats"]["pron_head"]
            cid = ids[minor_state[pg]] if pg else "unknown"
        if cid == "unknown":
            method = "unresolved"
        else:
            method = line["method"] if line["raw_name"] else "turn-taking"
        attribution[str(line["chapter"])].append({
            "seg_idx": line["seg_idx"], "character_id": cid,
            "confidence": "high" if conf_p >= HIGH_AT else "medium" if conf_p >= MEDIUM_AT else "low",
            "method": method, "p": round(conf_p, 3),
        })

    final_counts = Counter(a["character_id"] for lst in attribution.values() for a in lst)
    characters = [c for c in characters if c["id"] not in (MINOR_F, MINOR_M) or final_counts.get(c["id"], 0) > 0]
    for ch in characters:
        ch["line_count"] = final_counts.get(ch["id"], 0)
    base.assign_voices(characters)
    return {
        "version": base.HEURISTIC_VERSION,
        "mode": "whole-book",
        "whole_book_version": WHOLE_BOOK_VERSION,
        "unknown_below": cutoff,
        "narrator_gender": narrator_gender,
        "book_title": book.title,
        "max_characters": max_characters,
        "characters": characters,
        "attribution": attribution,
        "minor_mentions": {},
        "turn_params": {k: round(v, 3) for k, v in params.items()},
    }
