"""Heuristic "who's in this book, and who's talking" pass.

This runs once per book, independent of any synthesis settings, and
produces a cast.json: a list of characters (with an assigned voice) plus a
per-chapter, per-dialogue-segment guess at who's speaking. It's a pure
regex/rules heuristic — no ML model, no network call — so it's fast, free,
and fully deterministic, at the cost of accuracy on books with light or
unusual dialogue tagging (roughly 70-85% attribution accuracy is a
reasonable expectation, not a guarantee).

The output is deliberately a plain, inspectable JSON file: every
attribution carries a confidence and a method, so a human (or, down the
line, a trained model) can tell which lines are solid tag-matches versus
guesses, and correct the guesses. That makes this a reasonable starting
point for building a labeled dataset later, not just a one-off feature.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .extract import split_dialogue_narration
from .models import Book
from .names import name_gender
from .tts import VOICE_BANK

HEURISTIC_VERSION = 16  # bump whenever the attribution logic changes meaningfully

# Past tense AND present tense (base + third-person -s) — a book that
# mixes tenses (present-tense scenes are common for flashbacks, dream
# sequences, or a manuscript-within-the-book, as in "Verity") would
# otherwise have every "I say"/"I admit"/"Jeremy says"-style tag be
# invisible to a past-tense-only verb list.
_SPEECH_VERB_STEMS = [
    "said", "asked", "replied", "whispered", "shouted", "muttered", "cried",
    "exclaimed", "murmured", "called", "answered", "continued", "added",
    "interrupted", "yelled", "snapped", "laughed", "sighed", "began",
    "declared", "insisted", "protested", "demanded", "repeated", "returned",
    "observed", "remarked", "retorted", "growled", "breathed", "admitted",
    "chuckled", "grumbled", "scoffed", "sneered", "stated", "announced",
    "warned", "teased", "joked", "gasped", "hissed", "wondered", "mused",
    "countered", "snorted", "drawled", "stammered", "groaned", "moaned",
    "confessed", "agreed", "noted", "offered", "suggested", "urged",
    "pleaded", "begged", "ordered", "commanded", "scolded", "chided",
    "breathed", "panted", "wheezed", "finished", "concluded", "quipped",
]
_PRESENT_TENSE_VERBS = [
    "say", "says", "ask", "asks", "reply", "replies", "whisper", "whispers",
    "shout", "shouts", "mutter", "mutters", "cry", "cries", "exclaim",
    "exclaims", "murmur", "murmurs", "call", "calls", "answer", "answers",
    "continue", "continues", "add", "adds", "interrupt", "interrupts",
    "yell", "yells", "snap", "snaps", "laugh", "laughs", "sigh", "sighs",
    "begin", "begins", "declare", "declares", "insist", "insists",
    "protest", "protests", "demand", "demands", "repeat", "repeats",
    "return", "returns", "observe", "observes", "remark", "remarks",
    "retort", "retorts", "growl", "growls", "breathe", "breathes",
    "admit", "admits", "chuckle", "chuckles", "grumble", "grumbles",
    "scoff", "scoffs", "sneer", "sneers", "state", "states", "announce",
    "announces", "warn", "warns", "tease", "teases", "joke", "jokes",
    "gasp", "gasps", "hiss", "hisses", "wonder", "wonders", "muse",
    "muses", "counter", "counters", "snort", "snorts", "drawl", "drawls",
    "stammer", "stammers", "groan", "groans", "moan", "moans", "confess",
    "confesses", "agree", "agrees", "note", "notes", "offer", "offers",
    "suggest", "suggests", "urge", "urges", "plead", "pleads", "beg",
    "begs", "order", "orders", "command", "commands", "scold", "scolds",
    "chide", "chides", "pant", "pants", "wheeze", "wheezes", "finish",
    "finishes", "conclude", "concludes", "quip", "quips",
]
SPEECH_VERBS = sorted(set(_SPEECH_VERB_STEMS + _PRESENT_TENSE_VERBS), key=len, reverse=True)
_SPEECH_VERB_SET = {v.lower() for v in SPEECH_VERBS}

HONORIFICS = ["Mr", "Mrs", "Ms", "Dr", "Miss", "Aunt", "Uncle", "Sir", "Lady", "Captain", "Professor", "Master", "Mistress"]
_HONORIFICS_LOWER = {h.lower() for h in HONORIFICS}

# Honorifics that imply a gender (Dr/Captain/Professor do not). Used both to
# break clustering ties and as a gender vote.
_TITLE_GENDER = {
    "mr": "male", "mister": "male", "sir": "male", "lord": "male", "master": "male",
    "uncle": "male", "father": "male", "king": "male", "prince": "male",
    "mrs": "female", "miss": "female", "ms": "female", "madam": "female",
    "madame": "female", "lady": "female", "aunt": "female", "mother": "female",
    "queen": "female", "princess": "female", "mademoiselle": "female",
    "mistress": "female",
}

STOP_FIRST_WORDS = {
    "The", "A", "An", "He", "She", "It", "They", "I", "We", "You", "This",
    "That", "These", "Those", "There", "Then", "Now", "But", "And", "Or",
    "Nor", "Yet", "When", "After", "Before", "While", "Since", "Until",
    "Although", "Though", "Because", "Unless", "Whenever", "Whereas",
    "If", "So", "Yes", "No", "Oh", "Well", "Okay", "Perhaps", "Maybe",
    "Probably", "Certainly", "Obviously", "Clearly", "Actually", "Still",
    "Already", "Soon", "Later", "Finally", "Suddenly", "Eventually",
    "Instead", "Otherwise", "Meanwhile", "Even", "Plus", "Besides",
    "Also", "Likewise", "Similarly", "Regardless", "Nevertheless",
    "Nonetheless", "Here", "What", "Why", "How",
    "Who", "Which", "Chapter",
    # Capitalized only because they happen to start a sentence — never
    # real names, but easily mistaken for one by a bare-name/action-beat
    # match with no speech-verb anchor to protect it.
    "His", "Her", "Him", "My", "Our", "Your", "Their", "Its", "Mine",
    "Yours", "Theirs", "Hers", "Some", "Any", "Each", "Every", "Most",
    "All", "No", "Such", "Only", "Just", "One", "Something", "Someone",
    "Somebody", "Nothing", "Nobody", "Everyone", "Everybody", "Anybody",
    "Anything", "Anyone", "As", "For",
    # Exclamations/interjections — these are legitimately capitalized
    # mid-sentence too ("Oh God,", "Jesus Christ"), so the midsentence-caps
    # corroboration check can't rule them out the way it does "His"/"My".
    "God", "Jesus", "Christ", "Lord", "Hell", "Heaven", "Goddammit",
    "Goddamn", "Dammit", "Damn", "Geez", "Gosh", "Golly", "Hallelujah",
}

# Capitalised words that routinely get captured as a speaker tag or a bare
# action-beat mention but never name a person we want a voice for. Anything
# containing one of these is rejected outright; the second, smaller set only
# rejects the *final* token, so a surname like "Hill" survives while a place
# name like "Milsom Street" does not.
_NON_PERSON_WORDS = {
    # Sentence-initial function words STOP_FIRST_WORDS happens to miss.
    "in", "to", "at", "of", "on", "by", "with", "from", "into", "over",
    "under", "against", "between", "during", "without", "within", "upon",
    "toward", "towards", "across", "along", "around", "behind", "beyond",
    "beside", "despite", "except", "inside", "outside", "through",
    "throughout", "till", "up", "down", "off", "out", "past", "than",
    "unlike", "versus", "via", "never", "presently", "st", "th", "nd", "rd",
    # Nationalities / languages / demonyms.
    "english", "french", "german", "spanish", "italian", "russian",
    "polish", "irish", "scottish", "welsh", "dutch", "swedish", "danish",
    "norwegian", "american", "chinese", "japanese", "greek", "turkish",
    "egyptian", "persian", "jewish", "englishman", "englishmen",
    "frenchman", "frenchmen", "irishman", "scotsman", "dutchman",
    "gentleman", "gentlemen", "gentlewoman",
    "yorker", "yorkers", "londoner", "londoners", "parisian", "parisians",
    "venetian", "venetians",
    # Cities / countries / regions.
    "london", "paris", "rome", "venice", "oxford", "eton", "madrid",
    "vienna", "berlin", "moscow", "america", "england", "france", "italy",
    "spain", "germany", "russia", "europe", "india", "china", "japan",
    "greece", "egypt", "scotland", "ireland", "wales", "boston", "chicago",
    "dublin", "yorkshire", "cornwall", "devon", "kent", "essex", "sussex",
    "norfolk", "new york",
    # Calendar words.
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # Spelled-out numbers (chapter headings: "Chapter Two" -> "Two").
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
}

_NON_PERSON_HEADS = {
    "street", "road", "avenue", "boulevard", "square", "lane", "court",
    "terrace", "crescent", "arch", "statue", "hotel", "cafe", "café",
    "restaurant", "park", "gardens", "bridge", "castle", "abbey",
    "cathedral", "church", "chapel", "station", "museum", "theatre",
    "theater", "market", "wharf", "docks", "beach", "valley", "woods",
    "forest", "mount", "palace", "tower", "college", "university",
    "library", "montparnasse", "manor", "grange", "cottage", "villa",
    "chateau", "estate", "rectory", "parsonage", "vicarage", "asylum",
    "saloon", "tavern", "press", "gazette",
}

# A character must be mentioned at least this many times to earn a slot unless
# a real speech-verb tag vouches for it — one-off and two-off names are
# overwhelmingly the action-beat pass mis-firing on a place, an adjective, a
# brand, or a typo.
MIN_CHARACTER_EVIDENCE = 4
# Only prune one-off candidates once a book has at least this many distinct
# candidate names — below it, one-off names are more likely real minor
# characters than action-beat noise (short texts / samples).
PRUNE_MIN_CANDIDATES = 6

_NAME = r'(?:(?:' + "|".join(HONORIFICS) + r')\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}'
_VERBS = "|".join(SPEECH_VERBS)
TAG_RE = re.compile(
    rf'(?P<name1>{_NAME})\s+(?P<verb1>{_VERBS})\b'
    rf'|'
    rf'\b(?P<verb2>{_VERBS})\s+(?P<name2>{_NAME})'
)

# First-person narrators ("I said", not "Lowen said") would otherwise be
# invisible to TAG_RE, since "I" is deliberately excluded from _NAME to
# avoid noise everywhere else. Matched separately and mapped to a dedicated
# "narrator" pseudo-character rather than folded into Unknown.
NARRATOR_TAG_RE = re.compile(rf'\bI\s+(?:{_VERBS})\b|\b(?:{_VERBS})\s+I\b')
NARRATOR_SENTINEL = "\u0000NARRATOR"  # never collides with a real captured name

# Pronoun-only tags: '"..." she says.' / '"..." said he.'  Extremely common,
# and previously invisible — the line then fell through to alternation and
# could land on a speaker of the wrong gender.  Kept as sentinels and resolved
# in the attribution pass against the characters present/in play.
PRONOUN_TAG_M = "\u0000HE"
PRONOUN_TAG_F = "\u0000SHE"
PRONOUN_SENTINELS = (PRONOUN_TAG_M, PRONOUN_TAG_F)
_PRONOUN_TAG_RE = re.compile(
    rf'\b(he|she)\s+(?:{_VERBS})\b'                       # he said
    rf'|\b(?:{_VERBS})\s+(he|she)\b'                      # said she
    rf'|\b(?:hear|heard|listening to|listened to)\s+(him|her)\b'  # I hear her mumble
    rf'|\b(?:his|her)\s+(?:mumble|murmur|reply|response|retort|whisper|answer|voice)\b',
    re.IGNORECASE)

# The tag regex happily glues a leading interjection onto a name
# ("Goddammit Barron answers this question." -> "Goddammit Barron"), which then
# looks like an unknown two-word person.  Drop such lead-ins.
_NAME_LEAD_INTERJECTIONS = {
    "goddammit", "goddamn", "dammit", "damn", "oh", "well", "hey", "god",
    "good", "my", "so", "but", "and", "then", "yes", "no", "please",
    "thanks", "thank", "sorry", "okay", "ok",
}


def _strip_interjection(name: str) -> str:
    toks = name.split()
    while len(toks) > 1 and toks[0].lower() in _NAME_LEAD_INTERJECTIONS:
        toks.pop(0)
    return " ".join(toks)


# A quote that is nothing but a full name ("Amanda Thomas," she says.) is the
# speaker introducing themselves.  Require two tokens: a one-word quote is far
# more often the *addressee* ("Jeremy," she said.).
_SELF_INTRO_RE = re.compile(
    r'^[\s"“”\'‘’]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[\s,.;:!?"“”\'‘’]*$')
# "Hi, Lulu, I'm Willem." / "My name is Willem." — the same signal as a
# name-only quote, but embedded in an ordinary sentence.  Only names already
# present in the cast are accepted, so "I'm fine" cannot create a speaker.
_SELF_INTRO_SAY_RE = re.compile(
    r"\b(?:I['’]?m|I am|my name['’]?s|my name is)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")

# Fallback for action-beat attribution with no speech verb at all
# ('Lowen crossed her arms. "Fine."') — extremely common in contemporary
# prose. Weaker evidence than a verb-tag, so it's kept as its own method/
# confidence level rather than silently treated as equal to a real tag.
# With no verb to anchor it, this is the pass most exposed to capturing an
# ordinary word that only happens to be capitalized because it starts a
# sentence ("His", "My", "As"...) — STOP_FIRST_WORDS catches the common
# ones by name, but can never be exhaustive. _collect_midsentence_caps
# below is the general-purpose backstop: a real name gets used capitalized
# in the middle of sentences constantly ("she looked at Jeremy and..."); a
# word that's only ever capitalized at a sentence's own start essentially
# never does, which is a strong, book-agnostic signal either way.
BARE_NAME_RE = re.compile(_NAME)
MIDSENTENCE_CAP_RE = re.compile(r'(?<=[a-z,;:])\s+([A-Z][a-z]+)')

PRONOUN_MALE_RE = re.compile(r"\b(he|him|his)\b", re.IGNORECASE)
PRONOUN_FEMALE_RE = re.compile(r"\b(she|her|hers)\b", re.IGNORECASE)


def _valid_name(name: str) -> bool:
    toks = [t.rstrip(".") for t in name.split()]
    if not toks:
        return False
    lowered = [t.lower() for t in toks]
    # A real name never contains a speech verb. This rejects chapter headings
    # ("CHAPTER VII. Anne Says Her Prayers" -> "Anne Says Her") that the
    # bare-name/action-beat pass would otherwise adopt as a character.
    if any(t in _SPEECH_VERB_SET for t in lowered):
        return False
    # A bare honorific ("Mr", "Mrs", "Miss") is not a character.
    if all(t in _HONORIFICS_LOWER for t in lowered):
        return False
    # Never adopt a place, nationality, calendar word, or sentence-initial
    # function word as a speaker (see _NON_PERSON_WORDS).
    if any(t in _NON_PERSON_WORDS for t in lowered):
        return False
    if lowered[-1] in _NON_PERSON_HEADS:
        return False
    first = toks[0]
    if first in HONORIFICS:
        return True
    return first not in STOP_FIRST_WORDS


def _collect_midsentence_caps(book: Book) -> set:
    """Every capitalized word that appears somewhere NOT at the start of a
    sentence/quote (i.e. immediately after a lowercase letter, comma,
    semicolon, or colon plus whitespace) — see BARE_NAME_RE's comment."""
    found = set()
    for chapter in book.chapters:
        found.update(MIDSENTENCE_CAP_RE.findall(chapter.text))
    return found


def _iter_name_tags(text: str):
    for m in TAG_RE.finditer(text):
        name = (m.group("name1") or m.group("name2") or "").strip()
        # Strip before validating: a lead-in interjection otherwise makes the
        # whole capture fail ("Goddammit Barron" -> rejected, name lost).
        name = _strip_interjection(name)
        if name and _valid_name(name):
            yield name


def self_introduced_name(text: str, name_to_root: dict, root_to_id: dict) -> Optional[str]:
    """A quote that names the speaker -> that speaker.

    Handles both the name-only form (``"Amanda Thomas," she says.``) and the
    ordinary self-introduction (``"Hi, I'm Willem."``).  The latter matters at
    a character's first appearance, before any narration has tagged them: the
    scene engine would otherwise alternate the line onto whoever else happens
    to be present, often a character of the wrong gender.
    """
    for m in (_SELF_INTRO_RE.match(text), _SELF_INTRO_SAY_RE.search(text)):
        if not m:
            continue
        name = m.group(1).strip().rstrip(".,;:!?")
        cid = root_to_id.get(name_to_root.get(name))
        if cid:
            return cid
    return None


def _gender_hint(text: str) -> Optional[str]:
    male = len(PRONOUN_MALE_RE.findall(text))
    female = len(PRONOUN_FEMALE_RE.findall(text))
    if male and male > female:
        return "male"
    if female and female > male:
        return "female"
    return None


# "the man said" / "said the old woman" — a gendered descriptor next to an
# untagged quote tells us the gender of an *unnamed* speaker.
_MALE_NOUNS = r"man|boy|gentleman|guy|fellow|lad|father|son|husband|brother|priest|king"
_FEMALE_NOUNS = r"woman|girl|lady|mother|daughter|wife|sister|queen|maid"
_DESC_ART = r"(?:the|a|an|his|her|their)\s+(?:[A-Za-z-]+\s+){0,2}"
_MALE_DESC_RE = re.compile(rf"\b{_DESC_ART}(?:{_MALE_NOUNS})\b", re.IGNORECASE)
_FEMALE_DESC_RE = re.compile(rf"\b{_DESC_ART}(?:{_FEMALE_NOUNS})\b", re.IGNORECASE)


def minor_gender(segments: List[tuple], seg_idx: int) -> Optional[str]:
    """Gender of an unnamed speaker, from the narration hugging the quote.

    Used to route an unattributed line to a gendered minor voice instead of a
    single generic one — otherwise a man described as "he" can end up sounding
    like the female narrator.  Descriptors ('the man said') win over bare
    pronouns, and the narration *after* the quote is trusted first because the
    tag-follows-quote convention dominates.
    """
    snips = []
    if seg_idx + 1 < len(segments) and segments[seg_idx + 1][0] == "narration":
        snips.append(_clip_head(segments[seg_idx + 1][1], SNIPPET_CHARS))
    if seg_idx - 1 >= 0 and segments[seg_idx - 1][0] == "narration":
        snips.append(_clip_tail(segments[seg_idx - 1][1], SNIPPET_CHARS))
    for sn in snips:
        male, female = bool(_MALE_DESC_RE.search(sn)), bool(_FEMALE_DESC_RE.search(sn))
        if male != female:
            return "male" if male else "female"
    for sn in snips:
        g = _gender_hint(sn)
        if g:
            return g
    return None


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-z]+\b")
# A name after one of these is an object/mention, not the subject of an action
# beat ("... hit on Melanie?").  The action-beat pass has no verb to anchor it,
# so this is the cheapest way to keep nearby mentions from becoming speakers.
_ACTION_BEAT_PREP_RE = re.compile(
    r"\b(?:on|at|to|with|for|about|from|of|in|into|like|as|by|near|behind|"
    r"beside|around|toward|towards|over|under|after|before|during|without|"
    r"against|among|between)\s+$",
    re.IGNORECASE)
# "Mary said, her voice low." / "Mary looked up and she smiled." — a subject
# pronoun *after* a name in the same sentence is a much sharper gender signal
# than pronouns anywhere in the sentence.
_NAME_SUBJ_PRON_RE = re.compile(r"\b([A-Z][a-z]+)\b[^.!?]*?\b(he|she)\b")


def _gender_evidence(book: Book) -> Dict[str, Counter]:
    """For every capitalized word, count the male/female pronouns that appear in
    the *narration* sentences it occurs in, plus a stronger vote when a subject
    pronoun follows the name in the same sentence. One pass over the book, so a
    character with hundreds of mentions gets hundreds of pronoun votes instead
    of the single 120-char snippet the old ``_gender_hint`` looked at."""
    evidence: Dict[str, Counter] = defaultdict(Counter)
    for chapter in book.chapters:
        for kind, text in split_dialogue_narration(chapter.text):
            if kind != "narration":
                continue
            for sentence in _SENT_SPLIT_RE.split(text):
                m = len(PRONOUN_MALE_RE.findall(sentence))
                f = len(PRONOUN_FEMALE_RE.findall(sentence))
                if m or f:
                    for word in _CAP_WORD_RE.findall(sentence):
                        evidence[word.lower()]["male"] += m
                        evidence[word.lower()]["female"] += f
                for mt in _NAME_SUBJ_PRON_RE.finditer(sentence):
                    g = "male" if mt.group(2) == "he" else "female"
                    evidence[mt.group(1).lower()][g] += 3
    return evidence


def _cluster_gender(variants, evidence: Dict[str, Counter]) -> Optional[str]:
    """Decide a cluster's gender, strongest signal first:

    1. a gendered honorific (Mr/Mrs/Miss/Lady/... — Dr/Captain/Professor are
       neutral, so they fall through), since it's essentially never wrong;
    2. the first-name lexicon (``names.name_gender``);
    3. only if neither exists, book-wide pronoun co-occurrence — noisy, because
       a narration sentence names two people and its pronouns are then credited
       to both, so it must not override the two signals above.
    """
    honor_votes: Counter = Counter()
    for v in variants:
        if v == NARRATOR_SENTINEL:
            continue
        _, honors = _name_parts(v)
        for h in honors:
            if h in _TITLE_GENDER:
                honor_votes[_TITLE_GENDER[h]] += 1
    if honor_votes:
        ranked = honor_votes.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            return ranked[0][0]

    for v in variants:
        if v == NARRATOR_SENTINEL:
            continue
        core, _ = _name_parts(v)
        if core:
            g = name_gender(core[0])
            if g:
                return g

    votes: Counter = Counter()
    for v in variants:
        if v == NARRATOR_SENTINEL:
            continue
        for t in (w.strip(".").lower() for w in v.split() if w.strip(".")):
            c = evidence.get(t)
            if c:
                votes["male"] += c.get("male", 0)
                votes["female"] += c.get("female", 0)
    if not votes:
        return None
    ranked = votes.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _clip_head(text: str, n: int) -> str:
    """First n chars, extended so a word is never cut in half."""
    if len(text) <= n:
        return text
    b = n
    while b < len(text) and (text[b].isalnum() or text[b] in "'-"):
        b += 1
    return text[:b]


def _clip_tail(text: str, n: int) -> str:
    """Last n chars, extended so a word is never cut in half."""
    if len(text) <= n:
        return text
    a = len(text) - n
    while a > 0 and (text[a - 1].isalnum() or text[a - 1] in "'-"):
        a -= 1
    return text[a:]


SNIPPET_CHARS = 160


def _context_snippets(segments: List[tuple], seg_idx: int,
                      consumed_narration: set) -> List[Tuple[int, str, bool]]:
    """(narration_index, snippet, prefer_last) triples for the narration right
    after, then right before, a dialogue segment — 'after' preferred since the
    tag-follows-quote convention dominates. ``prefer_last`` is True for the
    preceding narration, where the name *nearest* the quote (the last one) is
    the speaker. Windows are snapped to word boundaries so a name is never
    truncated ('Marilla Cu')."""
    snippets = []
    if seg_idx + 1 < len(segments) and segments[seg_idx + 1][0] == "narration" and (seg_idx + 1) not in consumed_narration:
        after = segments[seg_idx + 1][1]
        # A narration that *starts* a new paragraph is the next speaker's cue
        # ("... ?"  /blank line/  "Barron answers this question."), not a tag
        # for the quote before the break.  It still serves as the preceding
        # context of the next quote.
        if not after.lstrip(" \t").startswith("\n\n"):
            snippets.append((seg_idx + 1, _clip_head(after, SNIPPET_CHARS), False))
    if seg_idx - 1 >= 0 and segments[seg_idx - 1][0] == "narration" and (seg_idx - 1) not in consumed_narration:
        snippets.append((seg_idx - 1, _clip_tail(segments[seg_idx - 1][1], SNIPPET_CHARS), True))
    return snippets


def _tag_dialogue_segments(segments: List[tuple], midsentence_caps: set,
                           pronouns: bool = True,
                           strict_action_beats: bool = False,
                           unique_pronoun_narration: bool = False) -> Dict[int, Tuple[Optional[str], str, str]]:
    """Find who's speaking for every dialogue segment in a chapter's
    segment list: {seg_idx: (raw_name_or_None, snippet, method)}.

    Three passes, strongest evidence first — once a dialogue segment is
    resolved it's not reconsidered by a later, weaker pass, and a
    narration segment already used to resolve one dialogue line isn't
    reused for its other neighbor:
      1. Explicit speech-verb + name tag ('"..." Tom said.') — method 'tag'.
      2. First-person narrator tag ('"..." I said.') — method 'narrator'.
      3. Bare name mention with no speech verb at all, e.g. an action beat
         ('Lowen crossed her arms. "Fine."') — much weaker evidence than a
         real tag, so kept as its own method 'action-beat' rather than
         conflated with a true tag match.
    """
    results: Dict[int, Tuple[Optional[str], str, str]] = {}
    consumed_narration = set()
    dialogue_idxs = [i for i, (kind, _t) in enumerate(segments) if kind == "dialogue"]

    def pick(cands, prefer_last):
        if not cands:
            return None
        return cands[-1] if prefer_last else cands[0]

    def tag_matcher(s, prefer_last):
        return pick([n for n in _iter_name_tags(s)], prefer_last)

    def narrator_matcher(s, prefer_last):
        return NARRATOR_SENTINEL if NARRATOR_TAG_RE.search(s) else None

    def pronoun_matcher(s, prefer_last):
        m = _PRONOUN_TAG_RE.search(s)
        if not m:
            return None
        word = next((g for g in m.groups() if g), "").lower()
        return PRONOUN_TAG_F if word in ("she", "her") else PRONOUN_TAG_M

    def bare_name_matcher(s, prefer_last):
        # With no verb to anchor it, this is the pass most exposed to capturing
        # an ordinary capitalized word, so it keeps the mid-sentence-cap
        # corroboration check and rejects anything containing a speech verb
        # (which kills chapter headings like "Anne Says Her Prayers").
        #
        # Cast discovery deliberately stays liberal: a name mentioned in the
        # narrator's thoughts still helps us learn the cast.  Attribution uses
        # the strict form, where only the sentence touching the quote and a
        # plausible subject count (a name several sentences away is usually a
        # mention, not an action beat).
        if not strict_action_beats:
            cands = [n for n in BARE_NAME_RE.findall(s)
                     if _valid_name(n) and n.split()[0] in midsentence_caps]
            return pick(cands, prefer_last)

        parts = [p.strip() for p in _SENT_SPLIT_RE.split(s.strip()) if p.strip()]
        if not parts:
            return None
        sentence = parts[-1] if prefer_last else parts[0]
        # A question immediately before/after a quote is narration addressed
        # to the reader, not an action beat naming the speaker.
        if sentence.endswith("?"):
            return None
        cands = []
        for n in BARE_NAME_RE.findall(sentence):
            if not _valid_name(n) or n.split()[0] not in midsentence_caps:
                continue
            idx = sentence.find(n)
            if idx < 0:
                continue
            # "hit on Melanie" / "looked at Mary" -> object, not speaker.
            if _ACTION_BEAT_PREP_RE.search(sentence[:idx].rstrip()):
                continue
            # A sentence that is just the name ("Louise. I like that.") is a
            # quoted word, not an action beat.  Require some other prose.
            rest = sentence[:idx] + " " + sentence[idx + len(n):]
            if not re.search(r"[a-z]", rest):
                continue
            cands.append(n)
        return pick(cands, prefer_last)

    def run_pass(matcher, method, consume=True, skip=None):
        for seg_idx in dialogue_idxs:
            if seg_idx in results:
                continue
            blocked = consumed_narration | (skip or set())
            for narr_idx, snippet, prefer_last in _context_snippets(segments, seg_idx, blocked):
                match = matcher(snippet, prefer_last)
                if match:
                    results[seg_idx] = (match, snippet, method)
                    if consume:
                        consumed_narration.add(narr_idx)
                    elif skip is not None:
                        skip.add(narr_idx)
                    break

    run_pass(tag_matcher, "tag")
    run_pass(narrator_matcher, "narrator")
    if pronouns:
        # Runs before the bare-name pass (a pronoun tag is stronger evidence
        # than a name merely being nearby) but does not consume the narration:
        # it must not starve the pass that discovers characters.  It still
        # tracks the narration it used, so "she moans" cannot tag both the
        # quote before it and an unrelated quote after it.
        pronoun_used = set() if unique_pronoun_narration else None
        run_pass(pronoun_matcher, "pronoun", consume=False, skip=pronoun_used)
    run_pass(bare_name_matcher, "action-beat")

    for seg_idx in dialogue_idxs:
        if seg_idx not in results:
            results[seg_idx] = (None, "", "unresolved")

    return results


class _UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _name_tokens(name: str) -> set:
    return {w.strip(".") for w in name.split() if w.strip(".") not in HONORIFICS}


def _name_parts(name: str):
    """Split a mention into (core_tokens, honorific_tokens), both lower-cased and
    dot-stripped, e.g. 'Dr. Craven' -> (('craven',), ('dr',))."""
    toks = [w.strip(".") for w in name.split() if w.strip(".")]
    honors = tuple(sorted(t.lower() for t in toks if t in HONORIFICS))
    core = tuple(t.lower() for t in toks if t not in HONORIFICS)
    return core, honors


def _is_prefix(a: tuple, b: tuple) -> bool:
    return len(a) < len(b) and b[:len(a)] == a


# Bare-surname bridging ("Sowerby" + "Susan Sowerby"). It also merges
# "Mrs. Miller" with "Daisy Miller" (same surname, both female), which is wrong;
# keep it off by default and only enable it if it measurably helps.
SURNAME_BRIDGE = False


def _same_person(a: str, b: str, core_honors: Optional[Dict[tuple, set]] = None) -> bool:
    """Conservative identity test between two raw name strings.

    Replaces the old "share any token" rule, which transitively fused
    'Colin' / 'Colin Craven' / 'Dr. Craven' / 'Mr. Craven' and every name
    sharing a surname (all the 'Beavers'). Two mentions merge only when:
      * their non-honorific tokens are identical and their honorifics don't
        conflict ('Craven' ~ 'Dr. Craven'; NOT 'Mr. Craven' ~ 'Dr. Craven'), or
      * one is an ordered prefix of the other ('Colin' ~ 'Colin Craven').

    A bare surname ('Weston') must NOT bridge a 'Mr. X' and a 'Mrs. X' into one
    cluster, so when the book uses a surname with two or more different
    honorifics, that bare surname is left unattached (``core_honors``).
    """
    ca, ha = _name_parts(a)
    cb, hb = _name_parts(b)
    if not ca or not cb:
        return False
    if ca == cb:
        if ha == hb:
            return True
        if not ha or not hb:
            if core_honors is not None and len(core_honors.get(ca, set())) >= 2:
                return False
            return True
        return False
    if _is_prefix(ca, cb) or _is_prefix(cb, ca):
        return True
    if SURNAME_BRIDGE:
        for xc, xh, yc in ((ca, ha, cb), (cb, hb, ca)):
            if len(xc) == 1 and len(yc) >= 2 and yc[-1] == xc[0]:
                g = next((_TITLE_GENDER[t] for t in xh if t in _TITLE_GENDER), None)
                if g:
                    yg = name_gender(yc[0])
                    if yg and yg != g:
                        return False
                    return True
    return False


# --- first-person narrator ------------------------------------------------

_VOCATIVE_RE = re.compile(r"(?:^[\s\"'“‘]*|[,;!?]\s+)([A-Z][a-z]{2,})\s*[,!?]")
# Words that read as vocatives but are really titles/terms of address.
_TITLE_LED = _HONORIFICS_LOWER | {
    "madame", "mme", "mademoiselle", "mister", "general", "colonel", "major",
    "doctor", "father", "mother", "brother", "sister", "count", "countess",
    "prince", "princess", "duke", "duchess", "king", "queen", "lord", "lady",
    "mamma", "mama", "papa", "uncle", "aunt", "granny", "grandmother",
    "grandfather", "darling", "dear", "sir", "madam", "grandmamma",
    "grandpapa",
}
# A book counts as first-person only once "I said"/"said I" recurs this often —
# a lone occurrence is an aside, not a narrative voice.
NARRATOR_MIN_LINES = 8
# The narrator is addressed throughout the book; a minor character only
# locally. Require both a minimum number of direct addresses and a minimum
# number of chapters they are spread across.
NARRATOR_MIN_VOCATIVES = 5
NARRATOR_MIN_CHAPTERS = 3


def _vocative_counts(book: Book):
    """(counts, chapters) for names used as direct address inside dialogue."""
    counts: Counter = Counter()
    chapters: Counter = Counter()
    for chapter in book.chapters:
        seen = set()
        for kind, text in split_dialogue_narration(chapter.text):
            if kind != "dialogue":
                continue
            for m in _VOCATIVE_RE.findall(text):
                counts[m] += 1
                seen.add(m)
        for m in seen:
            chapters[m] += 1
    return counts, chapters


def _discover_narrator(book: Book, speaker_counts: Counter,
                       midsentence_caps: set) -> Optional[str]:
    """Best guess at the first-person narrator's name.

    In a first-person novel other characters *address* the narrator by name
    while the narrator's own speech is tagged "I said" — so the narrator's name
    shows up as a vocative, spread across many chapters, yet (almost) never as
    a speech-tag subject. Pick the most widely-addressed such name, requiring
    it to be a genuine proper noun (used mid-sentence, which rules out
    interjections like "Indeed") that is clearly an addressee rather than a
    speaker.
    """
    voc, voc_chapters = _vocative_counts(book)
    best = None
    best_key = None
    for name, vcount in voc.items():
        if vcount < NARRATOR_MIN_VOCATIVES:
            continue
        if voc_chapters[name] < NARRATOR_MIN_CHAPTERS:
            continue
        toks = name.split()
        if toks[0].lower() in _TITLE_LED or not _valid_name(name):
            continue
        if name not in midsentence_caps:  # interjection/sentence-initial noise
            continue
        scount = 0
        for raw, n in speaker_counts.items():
            if raw == name or raw.split()[0] == name:
                scount += n
        if scount * 2 > vcount:  # mostly a speaker, not the narrator
            continue
        key = (voc_chapters[name], vcount, -scount)
        if best_key is None or key > best_key:
            best, best_key = name, key
    return best


def _merge_narrator(clusters: dict, name_to_root: dict, narrator_root,
                    narrator_name: str) -> None:
    """Fold the cluster that owns the narrator's name into the narrator root, so
    the two are one character rather than a spurious split."""
    target = narrator_name.split()[0].lower()
    for root in [r for r in clusters if r != narrator_root]:
        if any(v == narrator_name or v.split()[0].lower() == target
               for v in clusters[root]["names"]):
            clusters[narrator_root]["count"] += clusters[root]["count"]
            clusters[narrator_root]["names"] |= clusters[root]["names"]
            for v in clusters[root]["names"]:
                name_to_root[v] = narrator_root
            del clusters[root]


def _build_cast(book: Book, clusters: dict, name_to_root: dict,
                raw_name_counts: Counter, tagged_raw_names: set,
                midsentence_caps: set, max_characters: int,
                gender_resolver=None):
    """Shared character roster for both engines: pick the roots worth a voice,
    name the first-person narrator, and resolve each one's canonical name and
    gender. Returns (characters, root_to_id).

    ``gender_resolver(root, variants, is_narrator)`` overrides the default
    book-wide pronoun resolver (the whole-book engine has a sharper one).
    """
    # Name the first-person narrator before ranking: the discovered name can
    # absorb a separate low-count cluster ('Jake' tagged a couple of times).
    narrator_root = next((r for r in clusters
                          if clusters[r]["names"] == {NARRATOR_SENTINEL}), None)
    narrator_name = None
    if (narrator_root is not None
            and clusters[narrator_root]["count"] >= NARRATOR_MIN_LINES):
        speaker_counts: Counter = Counter()
        for raw, n in raw_name_counts.items():
            if raw != NARRATOR_SENTINEL:
                speaker_counts[raw] += n
        narrator_name = _discover_narrator(book, speaker_counts, midsentence_caps)
        if narrator_name:
            _merge_narrator(clusters, name_to_root, narrator_root, narrator_name)

    tagged_roots = {name_to_root[r] for r in tagged_raw_names
                    if r in name_to_root}
    ranked_roots = sorted(clusters, key=lambda r: clusters[r]["count"], reverse=True)
    # Cut one-off dubious names (places, adjectives, typos the action-beat pass
    # picked up) once the roster is big enough that such names are noise rather
    # than the cast — a short text with three characters keeps all three.
    if len(ranked_roots) >= PRUNE_MIN_CANDIDATES:
        candidates = [r for r in ranked_roots
                      if clusters[r]["count"] >= MIN_CHARACTER_EVIDENCE
                      or r in tagged_roots
                      or r == narrator_root]
    else:
        candidates = ranked_roots
    kept_roots = candidates[:max_characters]
    root_to_id = {root: f"c{i}" for i, root in enumerate(kept_roots, start=1)}

    gender_evidence = None if gender_resolver else _gender_evidence(book)
    characters = []
    for root in kept_roots:
        variants = clusters[root]["names"]
        is_narrator = root == narrator_root
        if is_narrator:
            canonical = narrator_name or "Narrator (I)"
            aliases = sorted(v for v in variants if v != NARRATOR_SENTINEL)
            if narrator_name:
                aliases = sorted(set(aliases) | {narrator_name})
        else:
            canonical = max(variants, key=lambda n: (len(n.split()), raw_name_counts[n]))
            aliases = sorted(variants)
        if gender_resolver:
            gender = gender_resolver(root, variants, is_narrator)
        else:
            # When the narrator's name was discovered but never tagged (the
            # "I said" cluster owns no name strings), `variants` is just the
            # sentinel and carries no gender evidence.  Ask about the name too.
            gender_variants = variants
            if is_narrator and narrator_name:
                gender_variants = variants | {narrator_name}
            gender = _cluster_gender(gender_variants, gender_evidence)
        characters.append({
            "id": root_to_id[root],
            "name": canonical,
            "aliases": aliases,
            "person": "first" if is_narrator else "third",
            "narrator": bool(is_narrator),
            "gender": gender,
            "line_count": 0,  # filled in after attribution, below
        })
    characters.append({"id": "unknown", "name": "Unknown / Minor Characters",
                       "aliases": [], "gender": None, "line_count": 0})
    return characters, root_to_id


def add_minor_characters(characters: List[dict]) -> List[dict]:
    """Insert gendered minor buckets before the final Unknown state.

    Unnamed speakers otherwise share one genderless voice; a man described as
    "he" then sounds like the female narrator.  The buckets carry a gender so
    ``assign_voices`` gives them a matching voice, and they stay *after* the
    named cast so they never take a named character's voice slot.
    """
    if any(c["id"] in ("minor_m", "minor_f") for c in characters):
        return characters
    unknown = characters.pop()
    characters.append({"id": "minor_f", "name": "Unnamed woman (minor)",
                       "aliases": [], "gender": "female", "line_count": 0})
    characters.append({"id": "minor_m", "name": "Unnamed man (minor)",
                       "aliases": [], "gender": "male", "line_count": 0})
    characters.append(unknown)
    return characters


def _cluster_names(raw_name_counts: Counter):
    """Group raw name strings that are probably the same character — "Tom",
    "Tom Sawyer", "Mr. Sawyer" — via union-find over the conservative
    ``_same_person`` test.
    Returns (clusters: {root: {"names": set, "count": int}}, name_to_root: dict).
    """
    names = list(raw_name_counts.keys())
    uf = _UnionFind()
    for n in names:
        uf.find(n)
    # Which honorifics does the book ever use with a given core name? Used to
    # refuse ambiguous bare-surname bridging (see _same_person).
    core_honors: Dict[tuple, set] = defaultdict(set)
    for n in names:
        core, honors = _name_parts(n)
        if core:
            core_honors[core].add(honors)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if _same_person(names[i], names[j], core_honors):
                uf.union(names[i], names[j])

    clusters: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"names": set(), "count": 0})
    name_to_root = {}
    for n in names:
        root = uf.find(n)
        name_to_root[n] = root
        clusters[root]["names"].add(n)
        clusters[root]["count"] += raw_name_counts[n]
    return clusters, name_to_root


def _same_gender_candidate(gender: Optional[str], candidates: List[str],
                           char_by_id: dict) -> Optional[str]:
    """First character in ``candidates`` (ordered, most likely first) of `gender`."""
    if not gender:
        return None
    for cid in candidates:
        if char_by_id.get(cid, {}).get("gender") == gender:
            return cid
    return None


def assign_voices(characters: List[dict]):
    """Pick a voice + pitch for each discovered character, grouped by
    inferred gender, cycling the voice bank and nudging pitch once a pool
    is exhausted so characters stay distinguishable even in a large cast."""
    counters = {"male": 0, "female": 0, "neutral": 0}
    pitch_options = ["+0Hz", "+10Hz", "-10Hz", "+20Hz", "-20Hz"]
    used = set()
    for ch in characters:
        gender = ch.get("gender") or "neutral"
        pool = VOICE_BANK.get(gender) or VOICE_BANK["neutral"]
        i = counters[gender]
        counters[gender] += 1
        voice = pool[i % len(pool)]
        pitched = False
        if voice in used:
            # Prefer a still-unused voice: identical voices (e.g. the neutral
            # pool overlapping the female one) make two characters sound like
            # the same person.  Try this gender's pool first, then every voice
            # the profile offers (an Unknown bucket must not sound like the
            # narrator just because one short list ran out).
            free = [v for v in pool if v not in used]
            if not free:
                everywhere = [v for vs in VOICE_BANK.values() for v in vs]
                free = [v for v in everywhere if v not in used]
            if free:
                voice = free[0]
                pitched = True
        used.add(voice)
        ch["voice"] = voice
        steps = (i // len(pool)) + (1 if pitched else 0)
        ch["pitch"] = pitch_options[steps % len(pitch_options)] if steps else "+0Hz"


def analyze_book(book: Book, max_characters: int = 12) -> dict:
    # --- Pass A: scan every dialogue segment for a tag ---
    chapter_scans = []
    raw_name_counts = Counter()

    midsentence_caps = _collect_midsentence_caps(book)

    for chapter in book.chapters:
        segments = split_dialogue_narration(chapter.text)
        tags = _tag_dialogue_segments(segments, midsentence_caps)
        scan = []
        for seg_idx, (kind, seg_text) in enumerate(segments):
            if kind != "dialogue":
                continue
            raw_name, snippet, method = tags.get(seg_idx, (None, "", "unresolved"))
            if raw_name and raw_name not in PRONOUN_SENTINELS:
                raw_name_counts[raw_name] += 1
            scan.append({"seg_idx": seg_idx, "raw_name": raw_name, "method": method})
        chapter_scans.append((segments, scan))

    # Book-wide pronoun evidence for gender is gathered inside _build_cast.
    clusters, name_to_root = _cluster_names(raw_name_counts)
    tagged_raw_names = {item["raw_name"] for _segs, scan in chapter_scans
                        for item in scan
                        if item["raw_name"] and item["raw_name"] not in PRONOUN_SENTINELS
                        and item["method"] == "tag"}
    characters, root_to_id = _build_cast(
        book, clusters, name_to_root, raw_name_counts, tagged_raw_names,
        midsentence_caps, max_characters)
    add_minor_characters(characters)
    assign_voices(characters)
    char_by_id = {c["id"]: c for c in characters}

    # --- Pass B: attribute each dialogue segment ---
    attribution: Dict[str, List[dict]] = {}
    minor_mentions: Counter = Counter()

    for chapter_num, (segments, scan) in enumerate(chapter_scans, start=1):
        active: List[str] = []  # last up-to-2 distinct speaker ids, for ping-pong guessing
        chapter_attr = []
        for item in scan:
            raw_name = item["raw_name"]
            found_method = item["method"]
            character_id = confidence = method = None

            pronoun_gender = None
            if raw_name in PRONOUN_SENTINELS:
                pronoun_gender = ("male" if raw_name == PRONOUN_TAG_M else "female")
                # Most recent speaker of that gender is the natural referent
                # ("... she says" right after Amanda spoke).
                cid = _same_gender_candidate(pronoun_gender, list(reversed(active)),
                                             char_by_id)
                if cid:
                    character_id, confidence, method = cid, "medium", "pronoun"

            elif raw_name:
                root = name_to_root.get(raw_name)
                cid = root_to_id.get(root)
                if cid:
                    character_id = cid
                    method = found_method  # 'tag' / 'narrator' / 'action-beat'
                    confidence = "high" if found_method in ("tag", "narrator") else "medium"
                else:
                    character_id, confidence, method = "unknown", "low", f"{found_method}-minor"
                    if raw_name != NARRATOR_SENTINEL:
                        minor_mentions[raw_name] += 1

            if character_id is None or character_id in ("minor_m", "minor_f"):
                # '"Amanda Thomas," she says.' — a quote that is just a full
                # name identifies the speaker even when the tag is a pronoun.
                cid = self_introduced_name(segments[item["seg_idx"]][1],
                                           name_to_root, root_to_id)
                if cid:
                    character_id, confidence, method = cid, "medium", "self-intro"

            if character_id is None:
                if len(active) == 2:
                    character_id, confidence, method = active[0], "medium", "alternation"
                else:
                    character_id, confidence, method = "unknown", "low", "unresolved"

            if character_id != "unknown":
                if not active or active[-1] != character_id:
                    active.append(character_id)
                    if len(active) > 2:
                        active.pop(0)
            else:
                # Give a nameless speaker a voice of the right gender rather
                # than lumping every minor into one generic voice.
                gender = pronoun_gender or minor_gender(segments, item["seg_idx"])
                if gender:
                    character_id = "minor_m" if gender == "male" else "minor_f"
                    method = f"{method}-{gender}"

            chapter_attr.append({
                "seg_idx": item["seg_idx"],
                "character_id": character_id,
                "confidence": confidence,
                "method": method,
            })
        attribution[str(chapter_num)] = chapter_attr

    # Recompute line counts from the actual final attribution (more meaningful
    # than the raw tag-mention counts used only for the keep/cut ranking).
    final_counts = Counter()
    for chapter_attr in attribution.values():
        for a in chapter_attr:
            final_counts[a["character_id"]] += 1
    for ch in characters:
        ch["line_count"] = final_counts.get(ch["id"], 0)

    return {
        "version": HEURISTIC_VERSION,
        "book_title": book.title,
        "max_characters": max_characters,
        "characters": characters,
        "attribution": attribution,
        "minor_mentions": dict(minor_mentions.most_common(30)),
    }


def format_report(cast: dict) -> str:
    total_lines = sum(len(v) for v in cast["attribution"].values())
    named = [c for c in cast["characters"]
             if c["id"] not in ("unknown", "minor_m", "minor_f")]
    lines = [f"'{cast['book_title']}' — {len(named)} named character(s) found, {total_lines} dialogue line(s) total"]
    for ch in sorted(cast["characters"], key=lambda c: -c["line_count"]):
        pct = (ch["line_count"] / total_lines * 100) if total_lines else 0.0
        gender = ch["gender"] or "?"
        lines.append(
            f"  [{ch['id']:>7}] {ch['name']:<28} {ch['line_count']:>4} lines ({pct:4.1f}%)  "
            f"gender={gender:<7} voice={ch['voice']} pitch={ch['pitch']}"
        )

    conf_counts = Counter()
    for chapter in cast["attribution"].values():
        for a in chapter:
            conf_counts[a["confidence"]] += 1
    lines.append("")
    lines.append("Attribution confidence: " + ", ".join(f"{k}={v}" for k, v in conf_counts.most_common()))

    if cast.get("minor_mentions"):
        top_minor = list(cast["minor_mentions"].items())[:8]
        lines.append(
            "Named but folded into Unknown (raise --max-characters to keep them): "
            + ", ".join(f"{n}({c})" for n, c in top_minor)
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Persistence — one cast.json per input book, independent of synthesis settings
# --------------------------------------------------------------------------

def cast_path_for(input_path: Path) -> Path:
    return input_path.parent / f".{input_path.stem}.cast.json"


def _input_signature(input_path: Path) -> Dict[str, Any]:
    stat = input_path.stat()
    return {"path": str(input_path.resolve()), "size": stat.st_size, "mtime": stat.st_mtime}


class CastStore:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {}

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict:
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        return self.data

    def save(self):
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def build(self, book: Book, input_path: Path, max_characters: int):
        self.data = analyze_book(book, max_characters=max_characters)
        self.data["input"] = _input_signature(input_path)
        self.save()

    def matches(self, input_path: Path, max_characters: int) -> bool:
        return (
            bool(self.data)
            and self.data.get("version") == HEURISTIC_VERSION
            and self.data.get("input") == _input_signature(input_path)
            and self.data.get("max_characters") == max_characters
        )
