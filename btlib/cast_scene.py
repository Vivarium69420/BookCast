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

SCENE_VERSION = 12

# A narration gap longer than this (characters) is treated as a scene break.
SCENE_GAP = 500
# Explicit scene dividers ("***", "---", "* * *", "# # #").
SCENE_BREAK_RE = re.compile(r"(?m)^\s*(?:[*#~=\u2014\u2013]\s*){3,}\s*$")

NAME_RE = re.compile(r"\b[A-Z][a-z]+\b")
VOC_START_RE = re.compile(r'^\s*[\u201c"]?\s*([A-Z][a-z]+)\s*[,!]')
VOC_END_RE = re.compile(r',\s*([A-Z][a-z]+)\s*[.!?]?\s*[\u201d"]?\s*$')

# Kinship terms are named roles in a cast ("Mom", "Dad", "Grandma"), but the
# narration often uses the generic lowercase form ("my mother says").  Mapping
# the variants onto the cast member lets those tags reach the right voice.
ROLE_GROUPS = {
    "mother": ("mother", "mom", "mommy", "mama", "mum", "mummy"),
    "father": ("father", "dad", "daddy", "papa", "pop"),
    "grandmother": ("grandmother", "grandma", "granny", "grandmamma", "nana"),
    "grandfather": ("grandfather", "grandpa", "grandpapa", "granddad"),
}

# A first-person action beat immediately next to a quote usually introduces the
# narrator's own speech ("I turn to Grandma. "..." / "..." I quickly lie.).
# Reaction beats ("I shake my head.") deliberately do not match: there the other
# person is still the one talking.
_NARRATOR_REACTION_RE = re.compile(
    r"\bI(?:['’]m| am| was| felt| feel)\b[^.]{0,50}\b(?:startled|surprised|"
    r"confused|stunned|shocked|relieved|horrified|embarrassed|angry|furious|"
    r"hurt|sad|disappointed|mortified|flustered|taken\s+aback)\b",
    re.IGNORECASE)

# A first-person thought/reaction ("I don't entirely believe her.") usually
# belongs to the narrator, and the quote after it is the narrator's reply.  A
# physical beat ("I shake my head.") can still be a listener reaction, so only
# cognition verbs get this treatment.
_NARRATOR_VOICE_RE = re.compile(
    r"\bmy\s+voice\b|\bI\s+hear\s+(?:a|my)\s+voice\b",
    re.IGNORECASE)

_NARRATOR_THOUGHT_RE = re.compile(
    r"\bI\s+(?:[A-Za-z]+(?:['’][A-Za-z]+)?\s+){0,2}"
    r"(?:think|believe|wonder|realize|realise|know|"
    r"remember|guess|suppose|doubt|understand|decide|hope|fear|suspect|"
    r"assume)\b",
    re.IGNORECASE)

_NARRATOR_ACTION_RE = re.compile(
    r"\bI\s+(?:\w+\s+){0,2}(?:say|ask|tell|reply|answer|whisper|add|continue|"
    r"call|mutter|mumble|snap|yell|shout|lie|joke|admit|explain|repeat|"
    r"try|turn\s+to)\b",
    re.IGNORECASE)

# Small stopword/character inventories for the languages this book actually
# mixes in.  The goal is not full language identification; it is to stop a
# French/Spanish line from falling through to an arbitrary English-speaking
# character just because that character happened to be mentioned nearby.
LANGUAGE_WORDS = {
    "fr": {
        "français", "francais", "je", "tu", "vous", "nous", "il",
        "elle", "le", "la", "les", "un", "une", "des", "du", "de", "et",
        "est", "que", "qui", "pour", "parce", "avec", "dans", "mon", "ma",
        "mes", "votre", "voulez", "apprendre", "bonjour", "merci", "oui",
        "non", "pourquoi", "comment", "où", "très", "bien", "suis", "appelle",
        "veux", "tache", "tâche", "bravo", "enchantée", "divorce", "même",
        "courageux", "inconnu", "pâté", "beauté", "enchanteresse",
    },
    "es": {
        "español", "el", "los", "las", "una", "que", "por",
        "para", "con", "hola", "gracias", "amor", "año", "nuevo", "deseo",
        "borrón", "cuenta", "mi", "tu", "su", "es", "está", "soy", "quiero",
        "porque", "pero", "más", "muy", "aquí", "donde", "cuando", "como",
        "qué", "sí",
    },
    "de": {
        "deutsch", "der", "die", "das", "und", "ist", "nicht",
        "ich", "du", "er", "sie", "wir", "ihr", "mit", "für", "auf", "ein",
        "eine", "einen", "dem", "den", "des", "zu", "von", "sich", "mir",
        "mich", "mein", "dein", "sprechen", "danke", "bitte", "ja", "nein",
    },
    "it": {
        "italiano", "il", "lo", "gli", "le", "una", "che", "per",
        "con", "sono", "sei", "è", "non", "sì", "grazie", "ciao", "amore",
        "anno", "nuovo", "desiderio", "mio", "tua", "sua", "perché", "come",
        "dove", "quando",
    },
}
_LANGUAGE_MARKERS = {
    "fr": ("french", "français", "francais"),
    "es": ("spanish", "español", "espanol"),
    "de": ("german", "deutsch"),
    "it": ("italian", "italiano"),
}
_LANGUAGE_ACCENTS = {
    "fr": set("àâçéèêëîïôùûüÿœæ"),
    "es": set("áéíóúñü¿¡"),
    "de": set("äöüß"),
    "it": set("àèéìòù"),
}
_LANGUAGE_VERB_RE = re.compile(
    r"\b(?:speak|say|ask|reply|tell|answer|learn|study|take|practice|read|write)"
    r"[a-z]*\b", re.IGNORECASE)


def _role_char_map(characters: List[dict]) -> Dict[str, str]:
    """Map lowercase kinship terms ('mother', 'mom', ...) to the cast member.

    A book that has both 'Mom' and 'Mother' as separate characters will keep
    the first match rather than silently fuse them; the cast editor still
    exposes them separately.
    """
    out: Dict[str, str] = {}
    for c in characters:
        if c.get("id") in ("unknown", "minor_m", "minor_f", "narrator"):
            continue
        toks = set()
        for value in [c.get("name") or ""] + list(c.get("aliases") or []):
            toks.update(re.findall(r"[a-z]+", value.lower()))
        for _canonical, variants in ROLE_GROUPS.items():
            if toks.intersection(variants):
                for term in variants:
                    out.setdefault(term, c["id"])
    return out


def _cid_for_raw(raw_name: Optional[str], name_to_root: dict, root_to_id: dict,
                 role_char: Dict[str, str]) -> Optional[str]:
    if not raw_name:
        return None
    cid = root_to_id.get(name_to_root.get(raw_name))
    if cid:
        return cid
    return role_char.get(raw_name.lower())


def _role_tag(segments: List[tuple], seg_idx: int,
              role_char: Dict[str, str]) -> Optional[Tuple[str, str, str]]:
    """Catch lowercase kinship tags such as 'my mother says' / 'her father asked'."""
    if not role_char:
        return None
    terms = "|".join(sorted((re.escape(t) for t in role_char), key=len, reverse=True))
    verb_alt = "|".join(base.SPEECH_VERBS)
    pattern = re.compile(
        rf"\b(?:my|his|her|our|their|the)\s+({terms})\s+"
        rf"(?:{verb_alt})\b",
        re.IGNORECASE)
    # The simple form without a possessive ("Mother says") is normally caught by
    # the capitalized name-tag pass, but keeping it here makes the role pass
    # robust when the tag text is lowercased by a publisher.
    bare = re.compile(
        rf"\b({terms})\s+(?:{'|'.join(base.SPEECH_VERBS)})\b",
        re.IGNORECASE)
    for _narr_idx, snippet, _prefer_last in base._context_snippets(
            segments, seg_idx, set(), paragraph_scoped=True):
        m = pattern.search(snippet) or bare.search(snippet)
        if m:
            term = m.group(1).lower()
            cid = role_char.get(term)
            if cid:
                return cid, snippet, "role-tag"
    return None


def _first_person_action(segments: List[tuple], seg_idx: int) -> bool:
    """A first-person speech beat *after* the quote ("..." I quickly lie).

    Narration *before* the next quote is handled by the name-tag pass or by
    _narration_subject(); treating it here as well made a tag for the previous
    line leak onto the following reply.
    """
    idx = seg_idx + 1
    if idx >= len(segments) or segments[idx][0] != "narration":
        return False
    text = re.split(r"\n[ \t]*\n", segments[idx][1], maxsplit=1)[0]
    return bool(_NARRATOR_ACTION_RE.search(text))


def _detect_language(text: str) -> Optional[str]:
    """Return 'fr'/'es'/'de'/'it' when a quote is clearly not English.

    A couple of accented words in an English sentence ("résumé", "café") must
    not trigger this, so require at least two stopword hits or one stopword plus
    several accent characters.
    """
    low = text.lower()
    words = re.findall(r"[a-zà-ÿ']+", low)
    if not words:
        return None
    best = None
    best_score = 0.0
    for lang, stops in LANGUAGE_WORDS.items():
        hits = sum(1 for w in words if w in stops)
        if not hits:
            continue
        accents = sum(1 for ch in low if ch in _LANGUAGE_ACCENTS.get(lang, set()))
        score = hits + (0.5 if accents >= 2 else 0.0)
        if score > best_score:
            best, best_score = lang, score
    if best is None:
        return None
    if best_score >= 2.0:
        return best
    return None


def _in_language_scene(segments: List[tuple], seg_idx: int,
                      window: int = 10) -> bool:
    """True when this quote sits inside a run of non-English dialogue.

    Used only to keep an unnamed student/teacher as the current speaker across
    an English aside in a language class; applying it in ordinary English
    scenes would over-produce minor voices.
    """
    hits = 0
    lo = max(0, seg_idx - window)
    hi = min(len(segments), seg_idx + window + 1)
    for i in range(lo, hi):
        if i == seg_idx or segments[i][0] != "dialogue":
            continue
        if _detect_language(segments[i][1]):
            hits += 1
            if hits >= 2:
                return True
    return False


def _language_speakers(book: Book, name_to_root: dict, root_to_id: dict,
                       narrator_id: Optional[str]) -> Dict[str, set]:
    """Characters the narration explicitly associates with a language.

    Only narration is examined, and the name and the language word must occur
    in the same sentence with a speech/language verb between them.  A name
    merely mentioned near the word "French" is not a French speaker: in JOD,
    Mom talks *about* Allyson's French class, which previously made every line
    in the class look like something Mom could plausibly say.
    """
    out: Dict[str, set] = defaultdict(set)
    language_re = re.compile(r"\b(french|spanish|german|italian)\b", re.IGNORECASE)
    speak_verb_re = re.compile(
        r"\b(?:speak|speaks|spoke|speaking|say|says|said|ask|asks|asked|"
        r"reply|replies|replied|answer|answers|answered|tell|tells|told|"
        r"talk|talks|talked|learn|learns|learned|learning|study|studies|"
        r"studied|studying|take|takes|took|taking|practice|practices|"
        r"practiced|know|knows|knew|understand|"
        r"understands|understood|translate|translates|translated|confer|"
        r"confers|conferred|order|orders|ordered)\b",
        re.IGNORECASE)
    for chapter in book.chapters:
        for kind, part in split_dialogue_narration(chapter.text):
            if kind != "narration":
                continue
            for sentence in base._SENT_SPLIT_RE.split(part):
                for m in language_re.finditer(sentence):
                    lang = m.group(1).lower()[:2]
                    before = sentence[:m.start()]
                    # First-person narration ("I'm learning French", "my
                    # French class") belongs to the narrator/protagonist.
                    if narrator_id and (
                            re.search(r"\bI(?:'m| am| was| have| had| started|"
                                      r" took| take| study| studied| learn| learned|"
                                      r" learning| speak| spoke| speaking| studying|"
                                      r" taking)\b[^.!?]{0,120}$", before,
                                      re.IGNORECASE)
                            or re.search(r"\bmy\s+(?:French|Spanish|German|"
                                         r"Italian)\b", before, re.IGNORECASE)):
                        out[lang].add(narrator_id)
                    # Named characters: the verb must sit between the name
                    # and the language word in this same sentence.
                    for nm in NAME_RE.finditer(sentence):
                        name = nm.group(0)
                        if name in base.STOP_FIRST_WORDS:
                            continue
                        cid = root_to_id.get(name_to_root.get(name))
                        if not cid:
                            continue
                        if nm.start() < m.start():
                            span = sentence[nm.end():m.start()]
                        else:
                            span = sentence[m.end():nm.start()]
                        if speak_verb_re.search(span):
                            out[lang].add(cid)
    return out

def _language_fallback(cid: Optional[str], text: str, method: Optional[str],
                       confidence: Optional[str], last: Optional[str],
                       language_speakers: Dict[str, set],
                       char_by_id: Dict[str, dict],
                       local_presence: List[object],
                       narrator_id: Optional[str] = None) -> Optional[str]:
    """Keep a non-English line away from a character with no language link."""
    lang = _detect_language(text)
    if not lang:
        return cid
    known = language_speakers.get(lang) or set()
    if not known or cid in known or cid in ("minor_f", "minor_m", "unknown", None):
        return cid
    if method in ("tag", "narrator") and confidence == "high":
        return cid
    for present in local_presence:
        if present in known and present != narrator_id:
            return present
    gender = (char_by_id.get(cid) or {}).get("gender")
    if gender == "male":
        return "minor_m"
    if gender == "female":
        return "minor_f"
    return cid


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


def _resolve_pronoun(gender: str, last: Optional[str], recent: List[str],
                     participants: List[str], local_presence: List[object],
                     same_paragraph: bool, char_by_id: Dict[str, dict],
                     narrator_id: Optional[str] = None
                     ) -> Tuple[str, str, str]:
    """Resolve a bare 'she says' / 'he says' tag.

    In the same paragraph the previous speaker is usually continuing.  At a
    paragraph boundary a same-gender *other* speaker is the likely reply; this
    is the Mom/Allyson and Madame/Carol failure mode.  If no named candidate
    exists, fall back to a gendered minor bucket rather than an arbitrary
    nearby character.
    """
    candidates: List[object] = []
    for cid in [last] + recent + local_presence + participants:
        if cid and cid not in candidates:
            candidates.append(cid)
    # In a first-person book "she says" refers to another woman, never to the
    # narrating "I".  Excluding the narrator stops a classroom teacher from
    # inheriting Allyson's voice merely because Allyson is the only named
    # woman in the scene.
    same = [c for c in candidates
            if c != narrator_id
            and (char_by_id.get(c) or {}).get("gender") == gender]
    if same_paragraph and last in same:
        return last, "medium", "pronoun"
    # An active unnamed speaker continues until a named person of that gender
    # actually enters the exchange.  This keeps a teacher/waiter monologue from
    # jumping to a relative mentioned earlier in the chapter.
    if last in ("minor_m", "minor_f") and last in same:
        return last, "medium", "pronoun"
    others = [c for c in same if c != last]
    if others:
        return others[0], "medium", "pronoun"
    if same:
        return same[0], "medium", "pronoun"
    return ("minor_m" if gender == "male" else "minor_f",
            "low", f"pronoun-{gender}")


def _active_pair(last: Optional[str], recent: List[str],
                 participants: List[str], pending: Optional[str]) -> List[str]:
    """The two people currently talking, rather than the two most frequent in
    the whole scene.  A four-person family table can still have a two-person
    exchange in the middle of it."""
    if not last:
        pool = ([pending] if pending else []) + participants
        return pool[:2]
    others: List[str] = []
    for cid in ([pending] if pending else []) + recent + participants:
        if cid and cid != last and cid not in others:
            others.append(cid)
    return [last, others[0]] if others else [last]


def _first_subject(sentence: str, token_char: Dict[str, object],
                   char_by_id: Dict[str, dict], last: Optional[str],
                   recent: List[str], narrator_id: Optional[str],
                   pending: Optional[str] = None,
                   prefer_other: bool = False) -> Optional[str]:
    """First grammatical subject of a narration sentence, when it is a cast
    member or a pronoun whose referent we can resolve."""
    text = sentence.lstrip(" \t\r\n\"'“”‘’—-")
    m = re.match(r"(I|She|He|They|We|You|[A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", text)
    if not m:
        return None
    word = m.group(1)
    if word == "I":
        return narrator_id
    if word in ("She", "He"):
        gender = "female" if word == "She" else "male"
        # In "'You wanted to be a doctor.' She nods." the pronoun refers to
        # the addressee, not to the previous (female) speaker.  The pending
        # addressee is the better antecedent when its gender agrees.
        if (pending and (char_by_id.get(pending) or {}).get("gender") == gender):
            return pending
        pool = ([last] if last else []) + recent
        if prefer_other:
            pool = [c for c in pool if c != last] + ([last] if last else [])
        for cid in pool:
            if (char_by_id.get(cid) or {}).get("gender") == gender:
                return cid
        return narrator_id
    if word in ("They", "We", "You"):
        return None
    cid = token_char.get(word.lower())
    if cid:
        return cid
    return None


def _between_narration(segments: List[tuple], prev_idx: Optional[int],
                       cur_idx: int, previous_tag_snippet: str = "") -> str:
    start = (prev_idx + 1) if prev_idx is not None else cur_idx
    text = "".join(t for kind, t in segments[start:cur_idx] if kind == "narration")
    if previous_tag_snippet:
        text = text.replace(previous_tag_snippet, " ", 1)
    return text


def _named_other_actor(text: str, token_char: Dict[str, object],
                      narrator_id: Optional[str]) -> bool:
    for m in NAME_RE.finditer(text):
        cid = token_char.get(m.group(0).lower())
        if cid and cid != narrator_id:
            return True
    return False


def _narration_subject(segments: List[tuple], prev_idx: Optional[int],
                       cur_idx: int, token_char: Dict[str, object],
                       char_by_id: Dict[str, dict], last: Optional[str],
                       recent: List[str], narrator_id: Optional[str],
                       previous_tag_snippet: str = "",
                       pending: Optional[str] = None) -> Optional[str]:
    """Who the narration between two quotes says is acting.

    Used only as a weak fallback.  We deliberately look at the first subject of
    each sentence, not every capitalized word: in "She winks at me, then turns
    to Mom" the subject is still She, while Mom is only the addressee.
    """
    text = _between_narration(segments, prev_idx, cur_idx, previous_tag_snippet)
    if not text.strip():
        return None
    # "I hear a voice--my voice--announcing this" is an unmistakable cue that
    # the next quote belongs to the narrator, even when the previous sentence's
    # grammatical subject was Mom.
    if _NARRATOR_VOICE_RE.search(text):
        return narrator_id
    subject = None
    local_last = last
    first_sentence = True
    for sentence in base._SENT_SPLIT_RE.split(text):
        cid = _first_subject(
            sentence, token_char, char_by_id, local_last, recent,
            narrator_id, pending,
            # At a paragraph break, "She takes the check back" can refer to
            # the other woman in the conversation rather than to the line
            # immediately before it.  Keep the prior speaker inside the same
            # paragraph, where the action usually continues their turn.
            prefer_other=first_sentence and text.startswith("\n\n"))
        first_sentence = False
        if cid:
            subject = cid
            # Within a narration passage, a later "She ..." usually refers to
            # the most recently named subject ("Mom comes in. She glances ...").
            local_last = cid
        # "I go into the spare room and find Dad hiding" introduces the next
        # speaker even though the grammatical subject of the sentence was I.
        m = re.search(r"\b(?:find|finds|see|sees|notice|notices|spot|spots)\s+"
                      r"([A-Z][a-z]+)\b", sentence)
        if m:
            found = token_char.get(m.group(1).lower())
            if found and found != narrator_id:
                subject = found
                local_last = found
    if subject is None:
        return None
    # A first-person reaction beat ("I shake my head") is the listener, so the
    # previous speaker keeps talking.  A first-person speech/direct address
    # beat ("I turn to Grandma", "I quickly lie") belongs to the narrator.
    if subject == narrator_id and last and last != narrator_id:
        between = "".join(
            t for kind, t in
            segments[(prev_idx + 1) if prev_idx is not None else cur_idx:cur_idx]
            if kind == "narration")
        if (_NARRATOR_THOUGHT_RE.search(between)
                or _NARRATOR_VOICE_RE.search(between)):
            return narrator_id
        if not _NARRATOR_ACTION_RE.search(between):
            # A pronoun tag belongs to the previous speaker ("She declares."
            # after Mom's line -> Mom continues).  A named actor in the beat
            # does the same.  Otherwise this is the narrator's own reaction
            # ("I spit out my lemonade.") and the next quote is the narrator's.
            if (base._PRONOUN_TAG_RE.search(between)
                    or _named_other_actor(between, token_char, narrator_id)):
                return last
            return narrator_id
    return subject


def _addressee_from_context(text: str, segments: List[tuple], seg_idx: int,
                            token_char: Dict[str, object],
                            narrator_id: Optional[str], last: Optional[str],
                            speaker: Optional[str] = None
                            ) -> Optional[str]:
    """Infer who a quote is aimed at from a vocative or the adjacent narration."""
    voc = _vocative(text, token_char)
    if voc:
        return voc
    for idx in (seg_idx + 1, seg_idx - 1):
        if idx < 0 or idx >= len(segments) or segments[idx][0] != "narration":
            continue
        part = segments[idx][1]
        if idx > seg_idx:
            part = re.split(r"\n[ \t]*\n", part, maxsplit=1)[0]
        else:
            part = re.split(r"\n[ \t]*\n", part)[-1]
        if narrator_id and re.search(r"\b(?:me|us)\b", part, re.IGNORECASE):
            # Before the quote, "I'm startled by ... me" is the narrator's
            # reaction, not an indication that the next line addresses her.
            # Only a beat whose subject is the speaker ("She pours me ...")
            # makes the narrator the addressee.
            if idx > seg_idx:
                return narrator_id
            head = re.split(r"\n[ \t]*\n", part)[0].lstrip(" \t\r\n")
            if re.match(r"(?:She|He|[A-Z][a-z]+)\b", head):
                if speaker is None or re.match(r"(?:She|He)\b", head):
                    return narrator_id
                first = re.match(r"([A-Z][a-z]+)\b", head).group(1).lower()
                if token_char.get(first) == speaker:
                    return narrator_id
        m = re.search(r"\b(?:to|at|with)\s+([A-Z][a-z]+)\b", part)
        if m:
            cid = token_char.get(m.group(1).lower())
            if cid:
                return cid
    return None

_ACTION_BEAT_VERBS = set(base.SPEECH_VERBS) | {
    "nod", "nods", "smile", "smiles", "laugh", "laughs", "shrug", "shrugs",
    "sigh", "sighs", "look", "looks", "turn", "turns", "glance", "glances",
    "stare", "stares", "frown", "frowns", "grin", "grins", "wink", "winks",
    "shake", "shakes", "point", "points", "gesture", "gestures", "pause",
    "pauses", "beam", "beams", "blush", "blushes", "scowl", "scowls",
    "sniff", "sniffs", "cry", "cries", "sob", "sobs", "giggle", "giggles",
    "chuckle", "chuckles", "mutter", "mutters", "mumble", "mumbles",
}


def _valid_action_beat(raw_name: str, snippet: str) -> bool:
    """Reject a mention like 'Kali blasts her radio' as a speech beat.

    The broad tag pass is useful for cast discovery, but a subordinate-clause
    mention is not evidence that Kali is speaking the quote next to it.
    """
    if not snippet or not raw_name:
        return True
    if re.match(r"\s*(?:when|while|as|after|before|because|if|though|although|"
                r"since|until|unless)\b", snippet, re.IGNORECASE):
        return False
    token = raw_name.split()[-1]
    for m in re.finditer(r"\b" + re.escape(token) + r"\b", snippet):
        tail = snippet[m.end():m.end() + 80]
        if re.search(r"\b(?:" + "|".join(
                re.escape(v) for v in _ACTION_BEAT_VERBS) + r")\b",
                tail, re.IGNORECASE):
            return True
    return False


def _valid_narrator_tag(snippet: str, token_char: Dict[str, object]) -> bool:
    """Reject a narrator tag produced by an 'I' later in a beat about someone else.

    Example: 'Mom looks up ... a pair of pants I haven't even tried on.' is a
    Mom action beat, not evidence that Allyson is speaking.
    """
    if not snippet:
        return True
    first = base._SENT_SPLIT_RE.split(snippet.strip(), maxsplit=1)[0]
    m = re.match(r"([A-Z][a-z]+)\b", first)
    return not (m and m.group(1).lower() in token_char)


def _mentioned_char_ids(text: str, role_char: Dict[str, str]) -> set:
    """Characters named or referred to by kinship in a quote.

    A person being talked *about* ("Your mom is having a hard time") is a poor
    guess for the addressee when the narrator is also present.
    """
    out = set()
    lower = text.lower()
    for role, cid in role_char.items():
        if re.search(rf"\b{re.escape(role)}\b", lower):
            out.add(cid)
    return out


def _second_person_addressee(text: str, speaker: Optional[str],
                             last: Optional[str], recent: List[str],
                             participants: List[str],
                             local_presence: List[object],
                             exclude: Optional[set] = None) -> Optional[str]:
    """When a quote says 'you', the next reply is usually the other person."""
    if not re.search(r"\b(?:you|your|yours|yourself)\b", text, re.IGNORECASE):
        return None
    pool: List[object] = []
    # Only established speakers count here.  A nearby narration mention can be
    # the topic of "your dorm" without being the addressee; using local
    # presence made "Where's your collection?" point at Kali merely because the
    # next sentence mentioned her radio.
    for cid in ([last] if last else []) + recent + participants:
        if cid and cid != speaker and cid not in pool:
            pool.append(cid)
    if exclude:
        preferred = [c for c in pool if c not in exclude]
        if preferred:
            return preferred[0]
        return None
    return pool[0] if pool else None


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
    role_char = _role_char_map(characters)
    language_speakers = _language_speakers(
        book, name_to_root, root_to_id, narrator_id)

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
                if raw_name and raw_name not in base.PRONOUN_SENTINELS:
                    cid = _cid_for_raw(raw_name, name_to_root, root_to_id, role_char)
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

            # Point at the scene's opening narration.  A quote that opens a
            # scene still needs the beat before it ("Madame smiles. "..."");
            # with None, _between_narration() starts at the quote itself and
            # sees nothing.
            prev_dialogue = s_start - 1
            pending_addressee = None
            for group in groups:
                anchor = None
                for idx in group:
                    raw_name, _snip, found_method = tags.get(idx, (None, "", "unresolved"))
                    if raw_name and found_method in ("tag", "narrator"):
                        cid = _cid_for_raw(raw_name, name_to_root, root_to_id, role_char)
                        if cid:
                            anchor = cid
                            break

                group_guess = None
                for seg_idx in group:
                    kind, text = segments[seg_idx]
                    raw_name, _snippet, found_method = tags.get(seg_idx, (None, "", "unresolved"))
                    character_id = confidence = method = None
                    pronoun_gender = None
                    same_paragraph = _same_paragraph(segments, seg_idx)
                    # Presence is deliberately local to this quote.  A relative
                    # mentioned at the start of a long chapter must not become
                    # the default voice for a classroom/restaurant scene later.
                    local_presence = _scene_presence(
                        segments[max(s_start, seg_idx - 6):min(s_end, seg_idx + 7)],
                        token_char, midsentence_caps) or presence
                    prev_snippet = ""
                    if prev_dialogue is not None and prev_dialogue >= s_start:
                        prev_raw, prev_snip, _ = tags.get(
                            prev_dialogue, (None, "", ""))
                        # Strip a previous *named* tag from the narration
                        # window so it cannot be mistaken for the next speaker.
                        # Unknown names ("Carol whispers") carry the only
                        # action beat we have; keep those sentences in view.
                        if (prev_raw is None
                                or prev_raw in base.PRONOUN_SENTINELS
                                or prev_raw == base.NARRATOR_SENTINEL
                                or _cid_for_raw(prev_raw, name_to_root,
                                                root_to_id, role_char)):
                            prev_snippet = prev_snip or ""

                    if raw_name in base.PRONOUN_SENTINELS:
                        pronoun_gender = ("male" if raw_name == base.PRONOUN_TAG_M
                                          else "female")
                        character_id, confidence, method = _resolve_pronoun(
                            pronoun_gender, last, recent, participants,
                            local_presence, same_paragraph, char_by_id,
                            narrator_id)

                    elif raw_name:
                        if (found_method == "action-beat"
                                and not _valid_action_beat(raw_name, _snippet)):
                            raw_name = None
                            found_method = "unresolved"
                        elif (raw_name == base.NARRATOR_SENTINEL
                              and not _valid_narrator_tag(_snippet, token_char)):
                            # The tagger saw an "I" somewhere in the beat and
                            # called it the narrator, but the sentence actually
                            # starts with someone else: "Mom looks up ... a pair
                            # of pants I haven't tried on."  Recover the real
                            # actor instead of throwing the only tag away.
                            cid = _first_subject(
                                _snippet, token_char, char_by_id, last,
                                recent, narrator_id, pending_addressee)
                            if cid and cid != narrator_id:
                                character_id = cid
                                confidence = "medium"
                                method = "action-beat"
                                raw_name = None
                            else:
                                raw_name = None
                                found_method = "unresolved"
                        else:
                            cid = _cid_for_raw(raw_name, name_to_root, root_to_id, role_char)
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
                        role_tag = _role_tag(segments, seg_idx, role_char)
                        if role_tag:
                            character_id, _role_snippet, method = role_tag
                            confidence = "medium"

                    if character_id is None and narrator_id and _first_person_action(segments, seg_idx):
                        character_id, confidence, method = narrator_id, "medium", "narrator-action"

                    if character_id is None:
                        if anchor is not None:
                            character_id, confidence, method = anchor, "high", "scene-same-par"
                        elif group_guess is not None:
                            character_id, confidence, method = group_guess
                        else:
                            pending = pending_addressee
                            if (pending and pending not in participants
                                    and pending not in local_presence
                                    and pending != narrator_id and pending != last):
                                pending = None
                            subject = _narration_subject(
                                segments, prev_dialogue, seg_idx, token_char,
                                char_by_id, last, recent, narrator_id,
                                prev_snippet, pending)
                            valid_subject = (
                                subject is not None
                                and (subject == last or subject == narrator_id
                                     or subject in participants
                                     or subject in local_presence))
                            reaction = _between_narration(
                                segments, prev_dialogue, seg_idx, prev_snippet)
                            if (len(reaction) > 120
                                    and reaction.startswith("\n\n")):
                                pending = None
                            if (pending == narrator_id and not same_paragraph
                                    and _NARRATOR_REACTION_RE.search(reaction)):
                                # The narrator's emotional reaction is followed
                                # by the narrator speaking, not by the previous
                                # speaker continuing.
                                character_id, confidence, method = pending, "medium", "narrator-reaction"
                            elif (valid_subject and subject == last
                                    and pending and pending != last
                                    and not same_paragraph):
                                # The previous speaker is still going; a
                                # listener's "you" does not make them the
                                # next speaker.
                                character_id, confidence, method = subject, "medium", "narration-subject"
                            elif (last in ("minor_m", "minor_f")
                                  and subject is None and reaction.strip()):
                                character_id, confidence, method = last, "low", "scene-minor-continue"
                            elif pending and pending != last and not same_paragraph:
                                character_id, confidence, method = pending, "medium", "addressee"
                            elif valid_subject:
                                character_id, confidence, method = subject, "medium", "narration-subject"
                            else:
                                character_id, confidence, method = _infer_speaker(
                                    text, local_presence, participants, last,
                                    recent, token_char, pending)

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
                                for c in list(participants) + list(local_presence))
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
                            if _in_language_scene(segments, seg_idx):
                                if last != character_id:
                                    recent.insert(0, character_id)
                                    del recent[4:]
                                last = character_id

                    # A vocative names the addressee, not the speaker.  This
                    # catches "Mother, can you feed her ...?" being tagged to
                    # Grandma by the surrounding action beat, or a teacher
                    # saying "Bravo, Allyson" being given to Allyson.
                    voc = _vocative(text, token_char)
                    if voc and character_id == voc:
                        pair = _active_pair(last, recent, participants,
                                            pending_addressee)
                        other = next((c for c in pair if c != voc), None)
                        if other is None:
                            other = next((c for c in local_presence
                                          if c != voc), None)
                        if other is None and last and last != voc:
                            other = last
                        if other is None and narrator_id and narrator_id != voc:
                            other = narrator_id
                        if other is None:
                            gender = (char_by_id.get(voc) or {}).get("gender")
                            other = ("minor_m" if gender == "male"
                                     else "minor_f")
                        character_id = other
                        confidence = "medium"
                        method = f"{method}-vocative"

                    # A French/Spanish line should not be handed to a nearby
                    # relative who has no link to that language.  This is what
                    # kept pulling Madame Lambert's French class onto Mom.
                    if character_id:
                        language_cid = _language_fallback(
                            character_id, text, method, confidence, last,
                            language_speakers, char_by_id, local_presence,
                            narrator_id)
                        if language_cid != character_id:
                            character_id = language_cid
                            method = f"{method}-lang"

                    # Track who the line was aimed at.  This lets the next
                    # untagged paragraph resolve as the reply ("Are you ready,
                    # Allyson?" -> Allyson), which is the main failure mode in
                    # four-person family scenes.
                    addressee = _addressee_from_context(
                        text, segments, seg_idx, token_char, narrator_id, last,
                        character_id)
                    if not addressee:
                        mentioned = None
                        if (narrator_id
                                and (narrator_id in participants
                                     or narrator_id in local_presence)):
                            # Only first-person scenes have the "your mom"
                            # problem; third-person dialogue often addresses a
                            # person who is also being discussed.
                            mentioned = _mentioned_char_ids(text, role_char)
                        addressee = _second_person_addressee(
                            text, character_id, last, recent, participants,
                            local_presence, exclude=mentioned)
                    if pending_addressee == character_id:
                        pending_addressee = None
                    if addressee and addressee != character_id:
                        pending_addressee = addressee
                    else:
                        # A pending addressee is only good for the immediate
                        # reply.  If this line gives no "you"/vocative evidence,
                        # carrying it further pulled the next untagged family
                        # line back to a stale speaker.
                        pending_addressee = None

                    chapter_attr.append({
                        "seg_idx": seg_idx,
                        "character_id": character_id,
                        "confidence": confidence,
                        "method": method,
                    })
                    prev_dialogue = seg_idx
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
                   last: Optional[str], recent: List[str],
                   token_char: Dict[str, object],
                   pending: Optional[str] = None):
    """Resolve an untagged dialogue line. Returns (character_id, confidence,
    method).

    The participant set (characters tagged as speaking in this scene) is
    trusted first, but the active pair is the last speaker and the most recent
    *other* speaker, not the two highest-frequency characters in the whole
    scene.  That distinction matters at a four-person family table where Mom
    and Dad may dominate the chapter but a two-person Allyson/Kali exchange is
    happening right now.
    """
    voc = _vocative(text, token_char)

    def choose(pool: List[str], method: str):
        if not pool:
            return "unknown", "low", "unresolved"
        if len(pool) == 1:
            if voc == pool[0]:
                other = next((c for c in presence if c != pool[0]), None)
                if other:
                    return other, "low", method + "-addressee"
                return "unknown", "low", method + "-addressee"
            return pool[0], "medium", method
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

    pair = _active_pair(last, recent, participants, pending)
    if len(pair) >= 2:
        method = "scene-core-alt" if len(participants) > 2 else "scene-alternate"
        return choose(pair[:2], method)
    if len(pair) == 1:
        return choose(pair, "scene-active")

    # Zero participants: use who is mentioned locally in the scene.
    present = [c for c in presence if c != "unknown"]
    if len(present) == 2:
        return choose(present, "scene-presence-alt")
    if len(present) > 2:
        if pending and pending in present and pending != last:
            return pending, "medium", "scene-pending"
        for cid in reversed(recent):
            if cid in present and cid != voc:
                return cid, "low", "scene-recent"
        for cid in present:
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
