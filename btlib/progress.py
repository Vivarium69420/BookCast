"""Persistent progress tracking so a long book conversion can be stopped
(by a lost connection, a crash, or Ctrl+C) and resumed later without
redoing work that's already done.

Everything lives under a work directory next to the output:

    <workdir>/
        progress.json     <- book metadata + per-chapter/per-chunk status
        chunks/ch_0001/chunk_0000.mp3, ...   <- synthesized audio (persistent, not tempfile)
        chapters/ch_0001_combined.mp3, ...   <- one combined file per chapter

progress.json is rewritten after every chunk and every chapter, so at any
point the work directory reflects exactly what has been completed.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Book, Chapter

PROGRESS_VERSION = 1


def default_workdir(output: str, per_chapter: bool) -> Path:
    out_path = Path(output)
    if per_chapter:
        return out_path / ".book_to_audio_progress"
    return out_path.parent / f".{out_path.stem}.book_to_audio_progress"


def _input_signature(input_path: Path) -> Dict[str, Any]:
    stat = input_path.stat()
    return {"path": str(input_path.resolve()), "size": stat.st_size, "mtime": stat.st_mtime}


def _params_signature(args) -> Dict[str, Any]:
    # Anything that changes what audio gets produced must match to safely resume.
    return {
        "engine": args.engine,
        "voice": args.voice,
        "rate": args.rate,
        "gap_ms": args.gap_ms,
        "chapter_pause_ms": args.chapter_pause_ms,
        "min_chapter_words": args.min_chapter_words,
        "per_chapter": args.per_chapter,
        "output": str(Path(args.output).resolve()),
        "no_dialogue_voice": getattr(args, "no_dialogue_voice", False),
        "dialogue_voice": getattr(args, "dialogue_voice", None),
        "dialogue_rate": getattr(args, "dialogue_rate", None),
        "dialogue_pitch": getattr(args, "dialogue_pitch", None),
        "dialogue_pause_ms": getattr(args, "dialogue_pause_ms", 0),
        "multi_voice": getattr(args, "multi_voice", False),
        "max_characters": getattr(args, "max_characters", None),
        "cast_file": getattr(args, "cast_file", None),
    }


class ProgressStore:
    """Reads/writes progress.json and tracks per-chunk / per-chapter status."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.chunks_dir = workdir / "chunks"
        self.chapters_dir = workdir / "chapters"
        self.path = workdir / "progress.json"
        self.data: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def save(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        tmp.replace(self.path)  # atomic-ish: never leaves a half-written progress.json

    def init_new(self, book: Book, input_path: Path, args):
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.chapters_dir.mkdir(parents=True, exist_ok=True)

        chapters = []
        for idx, chap in enumerate(book.chapters, start=1):
            chapters.append({
                "index": idx,
                "title": chap.title,
                "text": chap.text,
                "num_chunks": None,           # filled in once chunked
                "chunk_done": [],             # list[bool], one per chunk
                "chapter_combined_done": False,
                "chapter_output_done": False, # for --per-chapter: final file copied out
            })

        self.data = {
            "version": PROGRESS_VERSION,
            "input": _input_signature(input_path),
            "params": _params_signature(args),
            "book_title": book.title,
            "chapters": chapters,
            "final_done": False,
        }
        self.save()

    def matches(self, input_path: Path, args) -> bool:
        """True if a saved progress file was produced by the same input file
        and the same TTS settings, so it's safe to resume from.

        The output path is ignored: it only decides where the final file is
        written, not how cached chunks sound, so renaming the output (for
        example to match the EPUB filename) must not discard finished audio.
        """
        if not self.data:
            return False
        saved = dict(self.data.get("params") or {})
        current = _params_signature(args)
        saved.pop("output", None)
        current.pop("output", None)
        return (
            self.data.get("input") == _input_signature(input_path)
            and saved == current
        )

    # -- book reconstruction ------------------------------------------

    def book(self) -> Book:
        chapters = [Chapter(title=c["title"], text=c["text"]) for c in self.data["chapters"]]
        return Book(title=self.data["book_title"], chapters=chapters)

    # -- chunk-level state ---------------------------------------------

    def chunk_path(self, chapter_idx: int, chunk_idx: int, suffix: str = ".mp3") -> Path:
        d = self.chunks_dir / f"ch_{chapter_idx:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"chunk_{chunk_idx:04d}{suffix}"

    def set_num_chunks(self, chapter_idx: int, n: int):
        c = self.data["chapters"][chapter_idx - 1]
        if c["num_chunks"] != n:
            c["num_chunks"] = n
            c["chunk_done"] = [False] * n
            self.save()

    def is_chunk_done(self, chapter_idx: int, chunk_idx: int) -> bool:
        c = self.data["chapters"][chapter_idx - 1]
        return bool(c["chunk_done"][chunk_idx])

    def mark_chunk_done(self, chapter_idx: int, chunk_idx: int):
        c = self.data["chapters"][chapter_idx - 1]
        c["chunk_done"][chunk_idx] = True
        self.save()

    # -- chapter-level state ---------------------------------------------

    def chapter_combined_path(self, chapter_idx: int) -> Path:
        return self.chapters_dir / f"ch_{chapter_idx:04d}_combined.mp3"

    def is_chapter_combined_done(self, chapter_idx: int) -> bool:
        return bool(self.data["chapters"][chapter_idx - 1]["chapter_combined_done"])

    def mark_chapter_combined_done(self, chapter_idx: int):
        self.data["chapters"][chapter_idx - 1]["chapter_combined_done"] = True
        self.save()

    def is_chapter_output_done(self, chapter_idx: int) -> bool:
        return bool(self.data["chapters"][chapter_idx - 1]["chapter_output_done"])

    def mark_chapter_output_done(self, chapter_idx: int):
        self.data["chapters"][chapter_idx - 1]["chapter_output_done"] = True
        self.save()

    # -- final state ---------------------------------------------

    def is_final_done(self) -> bool:
        return bool(self.data.get("final_done"))

    def mark_final_done(self):
        self.data["final_done"] = True
        self.save()

    # -- reporting ---------------------------------------------

    def summary(self) -> str:
        lines = [f"'{self.data.get('book_title')}' — resuming from saved progress:"]
        for c in self.data["chapters"]:
            done = sum(1 for x in c["chunk_done"]) if c["chunk_done"] else 0
            total = c["num_chunks"] if c["num_chunks"] is not None else "?"
            done_n = sum(1 for x in c["chunk_done"] if x)
            status = "combined" if c["chapter_combined_done"] else f"{done_n}/{total} chunks"
            lines.append(f"  [{c['index']}] {c['title'][:60]!r}: {status}")
        return "\n".join(lines)
