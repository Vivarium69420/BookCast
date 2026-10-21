"""Background CLI runner: one render at a time, streamed log, progress parsing."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

CHAPTER_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.*)$")
CHUNK_RE = re.compile(r"chunk\s+(\d+)/(\d+)")
CAST_SAVED_RE = re.compile(r"(?:Cast analysis saved to|Using cached cast analysis:)\s*(.+?)(?:\s+—.*)?$")
DONE_RE = re.compile(r"Done\s*->\s*(.+)$")
ERROR_RE = re.compile(r"(Traceback \(most recent call last\)|^Error|^error:|Killed by|Missing dependencies)")

STALL_SECONDS = 300


@dataclass
class Job:
    kind: str                       # "analyze" | "render"
    argv: List[str]
    cwd: Path
    log_path: Path
    proc: Optional[subprocess.Popen] = None
    lines: Deque[str] = field(default_factory=lambda: deque(maxlen=3000))
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None
    returncode: Optional[int] = None
    last_output: float = field(default_factory=time.time)

    chapter_index: int = 0
    chapter_total: int = 0
    chapter_title: str = ""
    chunk_done: int = 0
    chunk_total: int = 0
    output: Optional[str] = None
    cast_path: Optional[str] = None
    error: Optional[str] = None
    consumed: bool = False          # UI has already reacted to completion
    cancelled: bool = False

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def done(self) -> bool:
        return self.ended is not None

    @property
    def stalled(self) -> bool:
        return self.running and (time.time() - self.last_output) > STALL_SECONDS

    def percent(self) -> float:
        if self.kind == "analyze":
            return 100.0 if self.done else 0.0
        if not self.chapter_total:
            return 0.0
        frac = (self.chunk_done / self.chunk_total) if self.chunk_total else 0.0
        pct = ((self.chapter_index - 1) + frac) / self.chapter_total * 100
        return max(0.0, min(100.0, pct))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "running": self.running,
            "done": self.done,
            "returncode": self.returncode,
            "percent": self.percent(),
            "chapter_index": self.chapter_index,
            "chapter_total": self.chapter_total,
            "chapter_title": self.chapter_title,
            "chunk_done": self.chunk_done,
            "chunk_total": self.chunk_total,
            "output": self.output,
            "cast_path": self.cast_path,
            "error": self.error,
            "stalled": self.stalled,
            "cancelled": self.cancelled,
            "elapsed": (self.ended or time.time()) - self.started,
            "log": "\n".join(self.lines),
        }


class JobManager:
    def __init__(self) -> None:
        self.current: Optional[Job] = None
        self._lock = threading.Lock()

    def start(self, kind: str, args: List[str], cwd: Path, log_path: Path) -> Job:
        with self._lock:
            if self.current and self.current.running:
                raise RuntimeError("A job is already running")
            argv = [PYTHON, str(ROOT / "book_to_audio.py")] + [str(a) for a in args]
            job = Job(kind=kind, argv=argv, cwd=Path(cwd), log_path=Path(log_path))
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            job.proc = subprocess.Popen(
                argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
                start_new_session=True,
            )
            self.current = job
            threading.Thread(target=self._reader, args=(job,), daemon=True).start()
            return job

    def _reader(self, job: Job) -> None:
        assert job.proc is not None and job.proc.stdout is not None
        try:
            with open(job.log_path, "a", encoding="utf-8") as log:
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(job.argv)}\n")
                for line in job.proc.stdout:
                    line = line.rstrip("\n")
                    job.lines.append(line)
                    job.last_output = time.time()
                    log.write(line + "\n")
                    log.flush()
                    self._parse(job, line)
        except Exception as exc:            # pragma: no cover - defensive
            job.error = f"reader crashed: {exc}"
        finally:
            rc = job.proc.wait()
            job.returncode = rc
            job.ended = time.time()
            if rc != 0 and not job.error:
                job.error = f"exit code {rc}"

    def _parse(self, job: Job, line: str) -> None:
        m = CHAPTER_RE.match(line)
        if m:
            job.chapter_index = int(m.group(1))
            job.chapter_total = int(m.group(2))
            job.chapter_title = m.group(3).strip()
            job.chunk_done = 0
            job.chunk_total = 0
            return
        m = CHUNK_RE.search(line)
        if m:
            job.chunk_done = int(m.group(1))
            job.chunk_total = int(m.group(2))
            return
        m = CAST_SAVED_RE.search(line)
        if m:
            job.cast_path = m.group(1).strip()
            return
        m = DONE_RE.search(line)
        if m:
            job.output = m.group(1).strip()
            return
        if ERROR_RE.search(line):
            job.error = line.strip()

    def cancel(self, grace: float = 5.0) -> bool:
        job = self.current
        if not job or not job.running:
            return False
        job.cancelled = True
        assert job.proc is not None
        try:
            os.killpg(job.proc.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            try:
                job.proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                return False

        def _later() -> None:
            time.sleep(grace)
            if job.running:
                try:
                    os.killpg(job.proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

        threading.Thread(target=_later, daemon=True).start()
        return True

    def snapshot(self) -> Optional[Dict[str, Any]]:
        job = self.current
        return job.snapshot() if job else None
