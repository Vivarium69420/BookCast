from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chapter:
    title: str
    text: str


@dataclass
class Book:
    title: str
    chapters: List[Chapter] = field(default_factory=list)


def merge_small_chapters(chapters: List[Chapter], min_words: int = 150) -> List[Chapter]:
    if not chapters:
        return chapters

    merged: List[Chapter] = []
    pending_title: Optional[str] = None
    pending_text = ""

    for chap in chapters:
        if pending_text:
            pending_text = pending_text.rstrip() + "\n\n" + chap.text
        else:
            pending_title = chap.title
            pending_text = chap.text

        accumulated_words = len(pending_text.split())
        if accumulated_words >= min_words or chap is chapters[-1]:
            merged.append(Chapter(title=pending_title, text=pending_text))
            pending_title = None
            pending_text = ""

    return merged
