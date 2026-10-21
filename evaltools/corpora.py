"""Loaders for the public evaluation corpora.

PDNC (Project Dialogism Novel Corpus, LREC 2022)
    ``eval_data/pdnc/data/<Folder>/novel_text.txt``          raw novel text
    ``eval_data/pdnc/data/<Folder>/quotation_info.csv``      per-quote gold
    ``eval_data/pdnc/data/<Folder>/character_info.csv``      cast + aliases + gender
    ``eval_data/pdnc/PDNC-Novel-Index.csv``                  narrator person/gender

LitBank (CC-BY-4.0)
    ``eval_data/litbank/quotations/tsv/<id>_<title>_brat.ann``  QUOTE / ATTRIB rows
    ``eval_data/litbank/quotations/tsv/<id>_<title>_brat.txt``  tokenised, one sentence/line

Both are normalised to a common ``Doc`` shape: one or more chapters of plain
text plus gold quotes carrying character offsets into that text.
"""

import ast
import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from btlib.models import Book, Chapter


@dataclass
class GoldQuote:
    text: str
    start: int            # char offset into the quoting chapter's text
    end: int
    speaker: str          # gold main name / coref name
    addressees: List[str] = field(default_factory=list)
    quote_type: str = ""  # PDNC: Explicit/Implicit; LitBank: ""


@dataclass
class Doc:
    doc_id: str
    title: str
    chapters: List[Tuple[str, str]]          # (title, text)
    quotes: List[Tuple[int, GoldQuote]]      # (chapter_index, quote)
    # gold main name -> gender ('M'/'F'/'A'/'U'); empty when unknown
    genders: Dict[str, str] = field(default_factory=dict)
    # gold main name -> set of lower-cased aliases (includes the main name)
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    narrator_gender: Optional[str] = None
    narrative_person: Optional[float] = None
    source: str = ""

    def to_book(self) -> Book:
        return Book(title=self.title,
                    chapters=[Chapter(title=t, text=x) for t, x in self.chapters])

    @property
    def chapter_starts(self) -> List[int]:
        starts, off = [], 0
        for _, text in self.chapters:
            starts.append(off)
            off += len(text)
        return starts


def _norm(name: str) -> str:
    name = name.lower().replace("\u2019", "'")
    name = re.sub(r"[.,]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# --------------------------------------------------------------------------
# PDNC
# --------------------------------------------------------------------------

_PDNC_CHAPTER_RE = re.compile(
    r"^[ \t]*(?:CHAPTER|Chapter|PART|Part|BOOK|Book)\s+[IVXLCDM\d]+\b.*$",
    re.MULTILINE,
)


def _pd_bool_spans_to_chars(raw_bytes: bytes, spans) -> List[Tuple[int, int]]:
    out = []
    for sp in spans:
        try:
            s, e = int(sp[0]), int(sp[1])
        except (TypeError, ValueError, IndexError):
            continue
        if s < 0 or e > len(raw_bytes) or e <= s:
            continue
        out.append((len(raw_bytes[:s].decode("utf-8", "ignore")),
                    len(raw_bytes[:e].decode("utf-8", "ignore"))))
    return out


def _parse_alias_cell(cell: str) -> List[str]:
    cell = (cell or "").strip()
    if not cell:
        return []
    try:
        val = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return [cell]
    if isinstance(val, (set, list, tuple)):
        return [str(v) for v in val]
    return [str(val)]


def _pd_chapters(text: str) -> List[Tuple[str, int, int]]:
    """Split on chapter/part headings, preserving exact character slices."""
    marks = [m.start() for m in _PDNC_CHAPTER_RE.finditer(text)]
    if not marks:
        return [("Full Text", 0, len(text))]
    marks.append(len(text))
    chapters = []
    for i in range(len(marks) - 1):
        start, end = marks[i], marks[i + 1]
        title = text[start:start + 80].split("\n", 1)[0].strip()
        if end - start > 0:
            chapters.append((title or f"Section {i + 1}", start, end))
    # Attach any preamble before the first heading to the first chapter.
    if marks[0] > 0:
        chapters.insert(0, ("Front Matter", 0, marks[0]))
    return chapters


def load_pdnc(root: Path, folder: Optional[str] = None) -> List[Doc]:
    root = Path(root)
    if root.name == "data" or (root / "data").is_dir():
        data_dir = root if root.name == "data" else root / "data"
    else:
        data_dir = root / "data"
    index_csv = root / "PDNC-Novel-Index.csv"
    if not index_csv.exists():
        index_csv = data_dir.parent / "PDNC-Novel-Index.csv"

    meta = {}
    if index_csv.exists():
        for row in csv.DictReader(index_csv.open(encoding="utf-8")):
            meta[row.get("Folder Name", "").strip()] = row

    docs: List[Doc] = []
    folders = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if folder:
        folders = [p for p in folders if p.name == folder]

    for f in folders:
        text_path = f / "novel_text.txt"
        quote_path = f / "quotation_info.csv"
        char_path = f / "character_info.csv"
        if not (text_path.exists() and quote_path.exists()):
            continue

        raw = text_path.read_bytes()
        text = raw.decode("utf-8", "ignore")

        genders, aliases = {}, {}
        if char_path.exists():
            for row in csv.DictReader(char_path.open(encoding="utf-8")):
                main = (row.get("Main Name") or "").strip()
                if not main:
                    continue
                genders[_norm(main)] = (row.get("Gender") or "U").strip()
                names = [main] + _parse_alias_cell(row.get("Aliases", ""))
                aliases[_norm(main)] = sorted({_norm(n) for n in names if n})

        chaps = _pd_chapters(text)
        starts = [s for _, s, _ in chaps]

        def chap_of(off: int) -> int:
            idx = 0
            for i, s in enumerate(starts):
                if off >= s:
                    idx = i
                else:
                    break
            return idx

        quotes: List[Tuple[int, GoldQuote]] = []
        for row in csv.DictReader(quote_path.open(encoding="utf-8")):
            speaker = (row.get("speaker") or "").strip()
            qtype = (row.get("quoteType") or "").strip()
            addrs = [a.strip() for a in re.split(r"[;,]", row.get("addressees") or "") if a.strip()]
            try:
                spans = ast.literal_eval(row.get("quoteByteSpans") or "[]")
            except (ValueError, SyntaxError):
                spans = []
            for (cs, ce) in _pd_bool_spans_to_chars(raw, spans):
                ci = chap_of(cs)
                cstart = chaps[ci][1]
                quotes.append((ci, GoldQuote(
                    text=text[cs:ce],
                    start=cs - cstart,
                    end=ce - cstart,
                    speaker=speaker,
                    addressees=addrs,
                    quote_type=qtype,
                )))

        m = meta.get(f.name, {})
        ng = (m.get("Narrator Gender (M/F/A/U)") or "").strip() or None
        try:
            person = float(m.get("Narrative Person")) if m.get("Narrative Person") else None
        except ValueError:
            person = None

        docs.append(Doc(
            doc_id=f.name,
            title=(m.get("Novel Title") or f.name).strip(),
            chapters=[(t, text[s:e]) for t, s, e in chaps],
            quotes=quotes,
            genders=genders,
            aliases=aliases,
            narrator_gender=ng,
            narrative_person=person,
            source="pdnc",
        ))
    return docs


# --------------------------------------------------------------------------
# LitBank
# --------------------------------------------------------------------------

_NO_SPACE_BEFORE = set(list(".,!?;:%)]}") + ["\u2019", "'", "\u201d", "\u2014", "\u2013"])
_NO_SPACE_AFTER = set(list("([{") + ["\u201c", "\u2018", "$"])


def _detokenize(sentences: List[List[str]]):
    """Join tokenised sentences back into prose and return (text, positions),
    where ``positions[(sent, tok)] = (start, end)``."""
    out: List[str] = []
    positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    pos = 0
    prev_tok: Optional[str] = None

    for si, toks in enumerate(sentences):
        for ti, tok in enumerate(toks):
            if not out:
                sep = ""
            elif ti == 0:
                sep = "\n"
            elif tok in _NO_SPACE_BEFORE or (prev_tok is not None and prev_tok in _NO_SPACE_AFTER):
                sep = ""
            else:
                sep = " "
            if sep:
                out.append(sep)
                pos += len(sep)
            positions[(si, ti)] = (pos, pos + len(tok))
            out.append(tok)
            pos += len(tok)
            prev_tok = tok
    return "".join(out), positions


def load_litbank(root: Path, ids: Optional[List[str]] = None) -> List[Doc]:
    root = Path(root)
    tsv = root / "quotations" / "tsv"
    if not tsv.is_dir():
        tsv = root / "tsv"
    docs: List[Doc] = []
    for ann in sorted(tsv.glob("*.ann")):
        base = ann.with_suffix("")
        txt_path = base.with_suffix(".txt")
        if not txt_path.exists():
            continue
        doc_id = base.name
        if ids and not any(i in doc_id for i in ids):
            continue

        raw_lines = txt_path.read_text(encoding="utf-8").splitlines()
        sentences = [ln.split() for ln in raw_lines if ln.strip()]
        text, positions = _detokenize(sentences)

        quotes_raw = {}
        attrib = {}
        for line in ann.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if parts[0] == "QUOTE" and len(parts) >= 7:
                qid = parts[1]
                try:
                    s_s, s_t, e_s, e_t = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                except ValueError:
                    continue
                quotes_raw[qid] = (s_s, s_t, e_s, e_t, parts[6])
            elif parts[0] == "ATTRIB" and len(parts) >= 3:
                attrib[parts[1]] = parts[2]

        quotes: List[Tuple[int, GoldQuote]] = []
        for qid, (s_s, s_t, e_s, e_t, qtext) in quotes_raw.items():
            a = positions.get((s_s, s_t))
            b = positions.get((e_s, e_t))
            if not a or not b:
                continue
            speaker = attrib.get(qid, "")
            speaker = speaker.rsplit("-", 1)[0].replace("_", " ").strip() if speaker else ""
            quotes.append((0, GoldQuote(
                text=text[a[0]:b[1]],
                start=a[0],
                end=b[1],
                speaker=speaker,
                quote_type="LitBank",
            )))

        title = re.sub(r"^\d+_", "", doc_id).replace("_brat", "").replace("_", " ").title()
        docs.append(Doc(
            doc_id=doc_id,
            title=title,
            chapters=[(title, text)],
            quotes=quotes,
            source="litbank",
        ))
    return docs
