"""Text extraction from EPUB/PDF, cleaning, and chunking into TTS-sized pieces."""

import re
import sys
from pathlib import Path
from typing import List

from .models import Book, Chapter, merge_small_chapters


# --------------------------------------------------------------------------
# EPUB
# --------------------------------------------------------------------------

def extract_epub(path: Path) -> Book:
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        sys.exit(
            "Missing dependencies for EPUB parsing.\n"
            "Install with: pip install ebooklib beautifulsoup4"
        )

    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else path.stem

    out = Book(title=title)
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")

        for tag in soup(["script", "style", "sup", "nav"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = clean_text(text)
        if not text.strip():
            continue

        chapter_title = None
        heading = soup.find(["h1", "h2", "h3"])
        if heading and heading.get_text(strip=True):
            chapter_title = heading.get_text(strip=True)
        if not chapter_title:
            chapter_title = f"Chapter {len(out.chapters) + 1}"

        out.chapters.append(Chapter(title=chapter_title, text=text))

    if not out.chapters:
        sys.exit("Could not find any readable text inside this EPUB.")
    out.chapters = merge_small_chapters(out.chapters)
    return out


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def extract_pdf(path: Path) -> Book:
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("Missing dependency for PDF parsing.\nInstall with: pip install pypdf")

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            sys.exit("This PDF is password-protected. Decrypt it first.")

    full_text = []
    for page in reader.pages:
        full_text.append(page.extract_text() or "")
    raw = "\n".join(full_text)

    if len(raw.strip()) < 200:
        sys.exit(
            "Almost no text could be extracted from this PDF.\n"
            "It's likely a scanned/image-only PDF, which needs OCR first "
            "(this script doesn't do OCR)."
        )

    text = clean_text(raw)

    chapters = split_pdf_into_chapters(text)
    chapters = merge_small_chapters(chapters)

    title = path.stem
    meta = reader.metadata
    if meta and meta.title:
        title = meta.title

    return Book(title=title, chapters=chapters)


HEADING_RE = re.compile(
    r"^(chapter\s+\d+|chapter\s+[ivxlcdm]+|part\s+\d+|\d+\.\s+[A-Z].{0,60})\s*$",
    re.IGNORECASE,
)


def split_pdf_into_chapters(text: str) -> List[Chapter]:
    lines = text.split("\n")
    chapters: List[Chapter] = []
    current_title = "Chapter 1"
    current_lines: List[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            chapters.append(Chapter(title=current_title, text=body))

    for line in lines:
        if HEADING_RE.match(line.strip()) and len(current_lines) > 20:
            flush()
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if not chapters:
        chapters = [Chapter(title="Full Text", text=text)]
    return chapters


# --------------------------------------------------------------------------
# Cleaning & chunking
# --------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    return text.strip()


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        sentences = SENTENCE_SPLIT_RE.split(para)
        for sentence in sentences:
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 > max_chars:
                if current:
                    chunks.append(current.strip())
                if len(sentence) > max_chars:
                    for i in range(0, len(sentence), max_chars):
                        chunks.append(sentence[i:i + max_chars])
                    current = ""
                else:
                    current = sentence
            else:
                current = f"{current} {sentence}".strip()
        current += "\n"

    if current.strip():
        chunks.append(current.strip())
    return chunks


# --------------------------------------------------------------------------
# Dialogue vs. narration
# --------------------------------------------------------------------------

# Matches a straight-quoted "..." span or a curly-quoted “...” span.
# Deliberately double-quotes only: single/curly-single quotes are left alone
# because they collide too often with apostrophes and contractions to
# reliably tell "it's dialogue" from "it's just an apostrophe".
DIALOGUE_RE = re.compile(r'("[^"]*"|“[^”]*”)', re.DOTALL)


def split_dialogue_narration(text: str) -> List[tuple]:
    """Split chapter text into ('narration' | 'dialogue', text) segments,
    in original order, based on double-quoted spans.

    Known limitation: a speech that runs across multiple paragraphs using
    the classic novel convention (opening quote on each new paragraph, no
    closing quote until the speech actually ends) is not specially handled —
    each paragraph is matched independently, which is usually fine since
    each one still opens and closes its own quote in practice for most
    modern typesetting, but a true unclosed multi-paragraph quote will just
    read as narration until its paragraph's own quotes resolve.
    """
    segments = []
    pos = 0
    for m in DIALOGUE_RE.finditer(text):
        if m.start() > pos:
            narration = text[pos:m.start()]
            if narration.strip():
                segments.append(("narration", narration))
        segments.append(("dialogue", m.group(0)))
        pos = m.end()
    if pos < len(text):
        tail = text[pos:]
        if tail.strip():
            segments.append(("narration", tail))
    if not segments:
        segments = [("narration", text)]
    return segments


def chunk_chapter(text: str, max_chars: int = 3000, distinguish_dialogue: bool = True) -> List[tuple]:
    """Split a chapter into (kind, chunk_text, seg_idx) pieces ready for
    synthesis, where kind is 'narration' or 'dialogue'. Each chunk stays
    within one kind, so it can be handed to the TTS engine with that kind's
    voice settings without mixing narration and dialogue in one audio call.

    seg_idx is the index of the ('narration'|'dialogue', text) segment (as
    produced by split_dialogue_narration) that this chunk came from — stable
    regardless of how long the segment is or how many chunks it got split
    into. A separate speaker-attribution pass keys its character-id guesses
    by this same seg_idx, so a multi-chunk quote still resolves to one voice.
    """
    if not distinguish_dialogue:
        return [("narration", c, 0) for c in chunk_text(text, max_chars)]

    result = []
    for seg_idx, (kind, seg_text) in enumerate(split_dialogue_narration(text)):
        for chunk in chunk_text(seg_text, max_chars=max_chars):
            result.append((kind, chunk, seg_idx))
    return result

