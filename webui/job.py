"""Background CLI runner: a bounded render queue with streamed logs.

Jobs are captured up front (book, cast, workdir, argv), so the UI can load a
different book while renders wait or run.  A workspace is only ever rendered
by one process at a time, enforced in-process and with a PID lock on disk.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
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
MAX_SLOTS = 3


@dataclass
class JobSpec:
    """Everything a render needs, frozen at the moment it is queued."""
    book: Path
    title: str
    cast_file: Path
    workdir: Path
    out_path: Path
    args: List[str]                 # CLI tokens, without python/script
    chapters: str = ""
    per_chapter: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)

    def short(self) -> str:
        return self.id[:6]


@dataclass
class Job:
    spec: JobSpec
    log_path: Path
    cwd: Path = ROOT
    argv: List[str] = field(default_factory=list)
    proc: Optional[subprocess.Popen] = None
    lines: Deque[str] = field(default_factory=lambda: deque(maxlen=4000))
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
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
    cancelled: bool = False

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.ended is None

    @property
    def queued(self) -> bool:
        return self.proc is None and self.ended is None and not self.cancelled

    @property
    def done(self) -> bool:
        return self.ended is not None

    @property
    def stalled(self) -> bool:
        return self.running and (time.time() - self.last_output) > STALL_SECONDS

    @property
    def state(self) -> str:
        if self.cancelled and self.done:
            return "cancelled"
        if self.done:
            return "done" if (self.returncode == 0 and not self.error) else "error"
        if self.proc is None:
            return "queued"
        if self.running:
            return "stalled" if self.stalled else "running"
        rc = self.proc.poll()
        if rc is None:
            return "running"
        return "done" if rc == 0 and not self.error else "error"

    def percent(self) -> float:
        if self.done and self.returncode == 0 and not self.error:
            return 100.0
        if not self.chapter_total:
            return 0.0
        frac = (self.chunk_done / self.chunk_total) if self.chunk_total else 0.0
        pct = ((self.chapter_index - 1) + frac) / self.chapter_total * 100
        return max(0.0, min(100.0, pct))

    def elapsed(self) -> float:
        start = self.started or self.created
        return (self.ended or time.time()) - start

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "short": self.spec.short(),
            "book": self.spec.title,
            "chapters": self.spec.chapters or "all",
            "state": self.state,
            "running": self.running,
            "queued": self.queued,
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
            "elapsed": self.elapsed(),
            "queued_for": (self.started or time.time()) - self.created,
            "per_chapter": self.spec.per_chapter,
            "log": "\n".join(self.lines),
        }


class JobQueue:
    """Runs up to ``slots`` jobs at once; the rest wait in FIFO order."""

    def __init__(self, slots: int = 2) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._slots = max(1, min(MAX_SLOTS, int(slots)))

    # -- configuration -------------------------------------------------
    @property
    def slots(self) -> int:
        return self._slots

    def set_slots(self, slots: int) -> int:
        with self._lock:
            self._slots = max(1, min(MAX_SLOTS, int(slots)))
            self._start_pending_locked()
            return self._slots

    # -- inspection ----------------------------------------------------
    @property
    def jobs(self) -> List[Job]:
        with self._lock:
            return [self._jobs[i] for i in self._order if i in self._jobs]

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)
        return job.snapshot() if job else None

    def snapshots(self) -> List[Dict[str, Any]]:
        return [job.snapshot() for job in self.jobs]

    def running_for(self, workdir: Path) -> Optional[Job]:
        target = Path(workdir).resolve()
        for job in self.jobs:
            if job.running and Path(job.spec.workdir).resolve() == target:
                return job
        return None

    def has_workspace(self, workdir: Path) -> Optional[Job]:
        """A queued or running job already owns this workspace."""
        target = Path(workdir).resolve()
        for job in self.jobs:
            if job.state in ("queued", "running", "stalled") and \
                    Path(job.spec.workdir).resolve() == target:
                return job
        return None

    def position(self, job_id: str) -> int:
        """1-based position among queued jobs, or 0 if not waiting."""
        queued = [j.id for j in self.jobs if j.state == "queued"]
        return queued.index(job_id) + 1 if job_id in queued else 0

    # -- lifecycle -----------------------------------------------------
    def enqueue(self, spec: JobSpec, cwd: Path = ROOT,
                log_path: Optional[Path] = None) -> Job:
        argv = [PYTHON, str(ROOT / "book_to_audio.py")] + [str(a) for a in spec.args]
        path = Path(log_path) if log_path else \
            Path(spec.workdir).parent / "logs" / f"{spec.id}.log"
        job = Job(spec=spec, argv=argv, cwd=Path(cwd), log_path=path)
        with self._lock:
            busy = self.has_workspace(spec.workdir)
            if busy:
                raise RuntimeError(f"this book is already {busy.state} as job {busy.spec.short()}")
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._start_pending_locked()
            return job

    def _start_pending_locked(self) -> None:
        while True:
            running = sum(1 for j in self.jobs if j.running)
            if running >= self._slots:
                return
            nxt = next((j for j in self.jobs if j.queued), None)
            if nxt is None:
                return
            self._launch(nxt)

    def _launch(self, job: Job) -> None:
        from webui import store

        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        ok, why = store.acquire_job_lock(job.spec.workdir, job.id)
        if not ok:
            job.error = f"workspace is already rendering ({why})"
            job.returncode = -1
            job.ended = time.time()
            return

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            job.proc = subprocess.Popen(
                job.argv, cwd=str(job.cwd), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, env=env, start_new_session=True,
            )
        except Exception as exc:
            store.release_job_lock(job.spec.workdir, job.id)
            job.error = f"could not start: {exc}"
            job.returncode = -1
            job.ended = time.time()
            return

        job.started = time.time()
        job.last_output = job.started
        store.update_job_lock(job.spec.workdir, job.id, job.proc.pid)
        threading.Thread(target=self._reader, args=(job,), daemon=True).start()

    def _reader(self, job: Job) -> None:
        assert job.proc is not None and job.proc.stdout is not None
        try:
            with open(job.log_path, "a", encoding="utf-8") as log:
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"{' '.join(job.argv)}\n")
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
            from webui import store
            store.release_job_lock(job.spec.workdir, job.id)
            with self._lock:
                self._start_pending_locked()

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

    def cancel(self, job_id: Optional[str] = None, grace: float = 5.0) -> bool:
        job = self.get(job_id) if job_id else self.current
        if job is None:
            return False
        with self._lock:
            if job.queued:
                job.cancelled = True
                job.ended = time.time()
                job.returncode = -1
                self._start_pending_locked()
                return True
            if not job.running:
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

    @property
    def current(self) -> Optional[Job]:
        running = [j for j in self.jobs if j.running]
        return running[-1] if running else None

    def cancel_all(self) -> int:
        n = 0
        for job in self.jobs:
            if job.state in ("queued", "running", "stalled"):
                n += 1 if self.cancel(job.id) else 0
        return n

    def clear_finished(self) -> int:
        with self._lock:
            keep = [i for i in self._order if self._jobs[i].state in
                    ("queued", "running", "stalled")]
            removed = len(self._order) - len(keep)
            self._order = keep
            self._jobs = {i: self._jobs[i] for i in keep}
            return removed

    def pump(self) -> None:
        with self._lock:
            self._start_pending_locked()
