"""Gradio UI for book_to_audio.

One page, one open-book rail, one shared activity panel:
- each open book keeps its own cast, applied settings, and chapter range;
- renders continue when you switch books;
- the Activity column is the only place for the queue, logs, audio, and
  Continue/Cancel.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import gradio as gr
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btlib.extract import split_dialogue_narration          # noqa: E402
from btlib.pipeline import extract_book                     # noqa: E402

from webui import cast as castlib                           # noqa: E402
from webui import params as pspec                           # noqa: E402
from webui import preview, store                            # noqa: E402
from webui.job import JobQueue, JobSpec                     # noqa: E402
from webui.theme import (                                   # noqa: E402
    ACTIVITY_HTML,
    FOOTER_HTML,
    HEAD,
    HEADER_HTML,
    RENDER_NOTE_HTML,
    THEME,
)

MANAGER = JobQueue(slots=store.queue_slots())
RUNTIME = store.runtime_settings()
SHARING = store.sharing_settings()
SERVER_PORT = 7860
SERVER_HOST = "127.0.0.1"
SELECTED_JOB: Optional[str] = None
JOB_VIEW: List[str] = []
LAST_AUDIO: Optional[tuple] = None
_ANALYZE_SYNC_LOCK = threading.Lock()

SESSIONS: Dict[str, Dict[str, Any]] = {}
OPEN_ORDER: List[str] = []
ACTIVE_SLUG: Optional[str] = None

PARAM_COMPONENTS: Dict[str, Any] = {}
PARAM_ORDER: List[str] = []

# These do not change the audio already produced, so they can change between a
# stopped job and its continuation.
RUNTIME_KEYS = {"concurrency", "on_chunk_failure", "keep_workdir"}
ACTION_KEYS = {"fresh"}
# Only these are hidden from the per-book form: concurrency and failure policy
# live in Activity, and "fresh" is the Render section's Start over checkbox.
HIDDEN_KEYS = {"concurrency", "on_chunk_failure", "fresh"}
CONTENT_KEYS = [p.key for p in pspec.PARAMS
                if p.key not in RUNTIME_KEYS and p.key not in ACTION_KEYS]

CAST_SAVED_RE = re.compile(
    r"(?:Cast saved to|Cast analysis saved to|Using cached cast analysis:)\s*(.+?)(?:\s+—.*)?$")

SUPPORTED_SUFFIXES = {".epub", ".pdf"}
SESSION_FILE = ROOT / "runs" / "session.json"


# --------------------------------------------------------------------------
# session helpers
# --------------------------------------------------------------------------

def _blank_session() -> Dict[str, Any]:
    return {
        "slug": "",
        "book": None,
        "path": None,
        "ws": None,
        "cast": None,
        "texts": {},
        "dirty": set(),
        "params": pspec.default_values(),
        "extra": "",
        "chapter_range": "all",
        "needs_rerender": False,
        "last_render_params": None,
        "last_job": None,
        "pending_params": None,
        "pending_extra": "",
        "pending_cast": None,
        "analyze_job": None,
        "last_analyzed_job": None,
        "analyze_prev_chapters": set(),
    }


S: Dict[str, Any] = _blank_session()


def _activate(slug: Optional[str]) -> None:
    global S, ACTIVE_SLUG
    if slug and slug in SESSIONS:
        S = SESSIONS[slug]
        S["pending_cast"] = None
        ACTIVE_SLUG = slug
        job = _book_job(S)
        if job is not None and job.spec.kind == "analyze" and not job.done:
            S["analyze_job"] = job.id
        _poll_analyze_job(S)
    else:
        S = _blank_session()
        ACTIVE_SLUG = None


def _book_job(session: Optional[Dict[str, Any]] = None):
    session = session or S
    ws = session.get("ws")
    if not ws:
        return None
    target = (ws / "work").resolve()
    for job in reversed(MANAGER.jobs):
        if Path(job.spec.workdir).resolve() == target:
            return job
    return None


def _settings_differ(session: Optional[Dict[str, Any]] = None) -> bool:
    session = session or S
    last = session.get("last_render_params")
    if not last:
        return False
    current = _content_fingerprint(_effective_values(session), session.get("extra", ""))
    return current != last


def _save_session_meta(**extra) -> None:
    if not S.get("ws"):
        return
    store.save_meta(
        S["ws"],
        params=S.get("params") or pspec.default_values(),
        extra=S.get("extra", ""),
        chapter_range=S.get("chapter_range", "all"),
        needs_rerender=bool(S.get("needs_rerender")),
        last_render_params=S.get("last_render_params"),
        **extra,
    )


def _load_session(path: Path, book=None) -> Dict[str, Any]:
    book = book or _extract(path)
    ws = store.workspace_for(path)
    slug = ws.name
    if slug in SESSIONS:
        session = SESSIONS[slug]
        session["book"] = book
        session["path"] = path
        session["ws"] = ws
        session["texts"] = {i: ch.text for i, ch in enumerate(book.chapters, start=1)}
        return session

    meta = store.load_meta(ws)
    params = pspec.default_values()
    if isinstance(meta.get("params"), dict):
        params.update(meta["params"])
    elif meta.get("last_argv"):
        params.update(pspec.parse_argv(meta["last_argv"]))
    params["fresh"] = False
    params["narrator_gender"] = params.get("narrator_gender") or ""
    if params.get("voice_profile") is None:
        params["voice_profile"] = ""

    session = _blank_session()
    session.update({
        "slug": slug,
        "book": book,
        "path": path,
        "ws": ws,
        "cast": castlib.load(ws),
        "texts": {i: ch.text for i, ch in enumerate(book.chapters, start=1)},
        "params": params,
        "extra": meta.get("extra", ""),
        "chapter_range": meta.get("chapter_range") or meta.get("chapters") or "all",
        "last_render_params": meta.get("last_render_params"),
        "last_job": meta.get("last_job"),
    })
    session["needs_rerender"] = _settings_differ(session)
    job = _book_job(session)
    if job is not None and job.spec.kind == "analyze" and not job.done:
        session["analyze_job"] = job.id
    SESSIONS[slug] = session
    if slug not in OPEN_ORDER:
        OPEN_ORDER.append(slug)
    return session


# --------------------------------------------------------------------------
# value helpers
# --------------------------------------------------------------------------

def _ui_value(key: str, value: Any) -> Any:
    if key == "narrator_gender":
        return "auto" if value in (None, "", "auto") else value
    if key == "voice_profile" and value is None:
        return ""
    return value


def _stored_value(key: str, value: Any) -> Any:
    if key == "narrator_gender" and value in (None, "auto"):
        return ""
    if key == "voice_profile" and value is None:
        return ""
    return value


def _values_from_widgets(seq) -> Dict[str, Any]:
    """Values posted by the visible widgets, plus global runtime settings."""
    vals = pspec.default_values()
    for key, value in zip(PARAM_ORDER, seq):
        vals[key] = _stored_value(key, value)
    vals["concurrency"] = RUNTIME["concurrency"]
    vals["on_chunk_failure"] = RUNTIME["on_chunk_failure"]
    vals["fresh"] = False
    return vals


def _effective_values(session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    session = session or S
    vals = dict(session.get("params") or pspec.default_values())
    vals["narrator_gender"] = vals.get("narrator_gender") or ""
    if vals.get("voice_profile") is None:
        vals["voice_profile"] = ""
    vals["concurrency"] = RUNTIME["concurrency"]
    vals["on_chunk_failure"] = RUNTIME["on_chunk_failure"]
    vals["fresh"] = False
    return vals


def _content_fingerprint(values: Dict[str, Any], extra: str = "") -> Dict[str, Any]:
    return {
        "params": {key: values.get(key) for key in CONTENT_KEYS},
        "extra": extra,
    }


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "(default)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _param_widget_values(session: Optional[Dict[str, Any]] = None) -> List[Any]:
    session = session or S
    params = session.get("params") or pspec.default_values()
    return [_ui_value(key, params.get(key, pspec.BY_KEY[key].default))
            for key in PARAM_ORDER]


# --------------------------------------------------------------------------
# extraction / chapter helpers
# --------------------------------------------------------------------------

def _extract(path: Path):
    args = argparse.Namespace(input_text=None, epub_encoding=None, min_chapter_words=150)
    try:
        return extract_book(path, args)
    except SystemExit as exc:
        # btlib reports expected failures (bad type, corrupt EPUB, missing PDF
        # deps, ...) with sys.exit().  In a server thread SystemExit would tear
        # down the ASGI lifespan and freeze every request, so turn it into a
        # normal error the callback can display.
        raise RuntimeError(str(exc) or "Could not read this file") from None


def _book_path_error(path: Path) -> str:
    if not path.is_file():
        return f"Not a file: `{path}`"
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        suffix = path.suffix or "(no extension)"
        return f"Unsupported file type `{suffix}` — use .epub or .pdf."
    return ""


def _chapter_rows(book) -> List[List[Any]]:
    rows = []
    for i, ch in enumerate(book.chapters, start=1):
        quotes = sum(1 for kind, _ in split_dialogue_narration(ch.text) if kind == "dialogue")
        rows.append([i, ch.title[:90], len(ch.text.split()), quotes])
    return rows


def _chapter_labels(book) -> List[str]:
    return [f"{i}. {ch.title[:70]}" for i, ch in enumerate(book.chapters, start=1)]


def _label_index(label: str) -> Optional[int]:
    try:
        return int(str(label).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _compress_indices(indices: List[int]) -> str:
    idx = sorted(set(indices))
    if not idx:
        return ""
    parts, start, prev = [], idx[0], idx[0]
    for n in idx[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        start = prev = n
    parts.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(parts)


def _parse_chapter_range(text: str, total: int) -> tuple:
    """Return (cli_spec, indices, error).

    ``all``/blank means every chapter.  Otherwise accept printer-style ranges:
    ``1-3, 6-8`` (also en/em dashes and semicolons).
    """
    raw = (text or "").strip()
    if total <= 0:
        return "", [], "Load a book first."
    if not raw or raw.lower() in ("all", "*"):
        return "", list(range(1, total + 1)), ""

    normalized = raw.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    tokens = [t for t in re.split(r"[,;\s]+", normalized) if t]
    chosen: List[int] = []
    for token in tokens:
        if "-" in token:
            a, _, b = token.partition("-")
            if not a.isdigit() or not b.isdigit():
                return "", [], f"`{token}` is not a valid range."
            lo, hi = int(a), int(b)
            if lo > hi:
                return "", [], f"`{token}` is reversed; write {hi}-{lo}."
            if lo < 1 or hi > total:
                return "", [], f"`{token}` is outside 1-{total}."
            chosen.extend(range(lo, hi + 1))
        elif token.isdigit():
            n = int(token)
            if n < 1 or n > total:
                return "", [], f"Chapter {n} is outside 1-{total}."
            chosen.append(n)
        else:
            return "", [], f"`{token}` is not a chapter number or range."

    idx = sorted(set(chosen))
    if not idx:
        return "", [], "No chapters selected."
    spec = "" if len(idx) == total else _compress_indices(idx)
    return spec, idx, ""


def _range_summary(session: Optional[Dict[str, Any]] = None) -> str:
    session = session or S
    book = session.get("book")
    if not book:
        return "Open a book first."
    text = session.get("chapter_range") or "all"
    spec, idx, error = _parse_chapter_range(text, len(book.chapters))
    if error:
        return f"⚠ {error}"
    if len(idx) == len(book.chapters):
        return f"All **{len(book.chapters)}** chapters."
    return f"**{len(idx)} of {len(book.chapters)}** chapters: `{spec}`"


def _book_info_md(session: Optional[Dict[str, Any]] = None) -> str:
    session = session or S
    book = session.get("book")
    path = session.get("path")
    if not book or not path:
        return "No book open. Use **Open a book** below."
    return (
        f"### {book.title}\n"
        f"`{Path(path).name}` · {len(book.chapters)} chapters · "
        f"{sum(len(c.text.split()) for c in book.chapters):,} words  \n"
        f"Workspace: `{session.get('ws')}`"
    )


def _quote_labels(book, cast) -> List[str]:
    if not book or not cast:
        return []
    labels = []
    for key in castlib.chapter_keys(cast):
        try:
            title = book.chapters[int(key) - 1].title
        except (TypeError, ValueError, IndexError):
            title = f"Chapter {key}"
        labels.append(f"{key}. {title[:60]}")
    return labels


def _cast_note(session: Optional[Dict[str, Any]] = None) -> str:
    session = session or S
    job_id = session.get("analyze_job")
    job = MANAGER.get(job_id) if job_id else None
    if job is not None and not job.done:
        return (f"**Analysis {job.state}** as job `{job.spec.short()}` — "
                f"watch Activity. The table shows the previous cast until it finishes.")
    cast = session.get("cast")
    if cast:
        return (f"Loaded saved cast ({len(cast.get('characters', []))} characters) · "
                f"{castlib.source_label(cast)}. The **Narrator (I)** row is editable; "
                "its voice is reserved and cannot be reused by a character.")
    return "No cast yet — press **Analyze cast**."


# --------------------------------------------------------------------------
# analysis / cast helpers
# --------------------------------------------------------------------------

def _cast_chapters(data: Optional[Dict[str, Any]]) -> set:
    return {int(k) for k in (data or {}).get("attribution") or {}
            if str(k).isdigit()}


def _poll_analyze_job(session: Optional[Dict[str, Any]] = None) -> tuple:
    """Import a finished background analysis into the session.

    Returns ``(cast_changed, status_message)``.  The cast is read from disk only
    after the job exits, so the UI never sees a partially written JSON file.
    """
    session = session or S
    job_id = session.get("analyze_job")
    if not job_id:
        return False, ""
    job = MANAGER.get(job_id)
    if job is None:
        session["analyze_job"] = None
        return False, ""
    if not job.done:
        return False, ""

    with _ANALYZE_SYNC_LOCK:
        # Re-check under the lock: the timer and book switching can race.
        if session.get("analyze_job") != job_id:
            return False, ""
        session["analyze_job"] = None
        if job.cancelled:
            session["last_analyzed_job"] = job.id
            return False, (f"**Analysis cancelled** · job `{job.spec.short()}`. "
                           f"Use **Continue selected job** to re-run it.")
        if job.returncode != 0:
            session["last_analyzed_job"] = job.id
            return False, (f"**Analysis failed** · job `{job.spec.short()}`: "
                           f"`{job.error or f'exit code {job.returncode}'}`. "
                           f"See the Activity log.")
        data = castlib.load(session["ws"])
        if not data:
            session["last_analyzed_job"] = job.id
            return False, ("**Analysis finished, but no cast file was found.** "
                           "See the Activity log.")
        ok, problems, _summary = castlib.validate_import(
            data, session.get("book"), session.get("cast"))
        if not ok:
            # Never leave an invalid cast on disk for a later render to pick up.
            target = store.cast_path(session["ws"])
            bak = target.with_suffix(".json.bak")
            try:
                if bak.exists():
                    target.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                pass
            session["last_analyzed_job"] = job.id
            detail = "; ".join(problems[:3])
            return False, (f"**Analysis produced an invalid cast:** `{detail}`. "
                           f"See the Activity log.")

        old = session.get("analyze_prev_chapters") or set()
        new = _cast_chapters(data)
        redone = store.invalidate_chapters(session["ws"], old | new)
        session["cast"] = data
        session["analyze_prev_chapters"] = set()
        session["last_analyzed_job"] = job.id
        session["dirty"].clear()
        msg = (f"**Analysis complete** · {castlib.source_label(data)}. "
               f"Review the cast below.")
        if redone:
            msg += f" {len(redone)} chapter(s) will re-render."
        return True, msg


def _run_sync(argv: List[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "book_to_audio.py")] + argv,
        cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
    )


def _analyze_now(values: Dict[str, Any], extra: str = "") -> tuple:
    if not S["path"] or not S["ws"]:
        return False, "Load a book first."
    busy = _busy_reason()
    if busy:
        return False, f"Cannot analyze: {busy}."
    argv = [str(S["path"]), "--analyze-only"] + pspec.build_argv(values, extra=extra)
    try:
        proc = _run_sync(argv)
    except subprocess.TimeoutExpired:
        return False, "Analysis timed out."
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, "Analysis failed:\n" + out[-2000:]
    cast_file = None
    for line in out.splitlines():
        match = CAST_SAVED_RE.search(line)
        if match:
            cast_file = Path(match.group(1).strip())
    if not cast_file or not cast_file.exists():
        cached = sorted(
            Path(S["path"]).parent.glob(f".{Path(S['path']).stem}.cast*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if cached:
            cast_file = cached[0]
    if not cast_file or not cast_file.exists():
        return False, "Analysis finished but no cast file was found:\n" + out[-1500:]
    data = json.loads(cast_file.read_text(encoding="utf-8"))
    target = store.cast_path(S["ws"])
    old = _cast_chapters(S.get("cast"))
    if target.exists():
        target.with_suffix(".json.bak").write_text(
            target.read_text(encoding="utf-8"), encoding="utf-8")
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    redone = store.invalidate_chapters(S["ws"], old | _cast_chapters(data))
    S["cast"] = data
    S["dirty"].clear()
    S["analyze_job"] = None
    S["analyze_prev_chapters"] = set()
    note = f" {len(redone)} chapter(s) will re-render." if redone else ""
    return True, f"Cast saved to {target}{note}"


def _queue_analysis(values: Dict[str, Any], extra: str = "") -> tuple:
    """Start a long-running (light/heavy) cast analysis in the background."""
    global SELECTED_JOB
    if not S["path"] or not S["ws"] or not S["book"]:
        return False, "Load a book first."
    busy = _busy_reason()
    if busy:
        return False, f"Cannot analyze: {busy}."

    target = store.cast_path(S["ws"])
    if target.exists():
        try:
            target.with_suffix(".json.bak").write_text(
                target.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass
    S["analyze_prev_chapters"] = _cast_chapters(S.get("cast"))
    S["analyze_job"] = None
    S["last_analyzed_job"] = None

    workdir = S["ws"] / "work"
    argv = [str(S["path"]), "--analyze-only"]
    argv += pspec.build_argv(values, extra=extra)
    argv += ["--cast-out", str(target), "--workdir", str(workdir)]
    job_spec = JobSpec(
        book=S["path"], title=f"Analyze · {S['book'].title}",
        cast_file=target, workdir=workdir, out_path=target,
        args=argv, chapters="", kind="analyze",
    )
    try:
        job = MANAGER.enqueue(job_spec, cwd=ROOT,
                              log_path=store.job_log_path(S["ws"], job_spec.id))
    except Exception as exc:
        return False, f"Could not queue analysis: {exc}"

    store.save_meta(
        S["ws"], last_job=job.id, params=S.get("params"),
        extra=S.get("extra", ""), chapter_range=S.get("chapter_range", "all"),
    )
    S["analyze_job"] = job.id
    S["last_job"] = job.id
    SELECTED_JOB = job.id
    position = MANAGER.position(job.id)
    where = f"queued at position {position}" if position else "started"
    return True, (f"Analysis {where} as job `{job.spec.short()}`. "
                  f"Track it in **Activity**; the cast updates when it finishes.")


def _reload_cast(data: Dict[str, Any]):
    S["cast"] = data
    rows = castlib.characters_rows(data)
    speakers = castlib.character_names(data, include_buckets=True)
    keys = castlib.chapter_keys(data)
    labels = []
    for key in keys:
        try:
            i = int(key)
            title = (S["book"].chapters[i - 1].title
                     if S["book"] and i - 1 < len(S["book"].chapters) else f"Chapter {i}")
        except (TypeError, ValueError, IndexError):
            title = f"Chapter {key}"
        labels.append(f"{key}. {title[:60]}")
    return (
        rows,
        gr.update(choices=speakers, value=speakers[0] if speakers else None),
        gr.update(choices=labels, value=labels[0] if labels else None),
    )


def _chapters_for(cid: str) -> List[int]:
    data = S.get("cast")
    if not data:
        return []
    out = []
    for key, entries in (data.get("attribution") or {}).items():
        if any(e.get("character_id") == cid for e in entries):
            try:
                out.append(int(key))
            except (TypeError, ValueError):
                pass
    return out


def _busy_reason() -> str:
    """Why this book's workspace cannot be edited/rendered right now."""
    ws = S.get("ws")
    if not ws:
        return ""
    job = MANAGER.running_for(ws / "work")
    if job is None:
        # A queued analysis will overwrite cast.json when it starts, so block
        # edits even before it gets a slot.  Queued renders only read the cast.
        target = (ws / "work").resolve()
        job = next((j for j in MANAGER.jobs
                    if j.state == "queued" and j.spec.kind == "analyze"
                    and Path(j.spec.workdir).resolve() == target), None)
    if job:
        if job.spec.kind == "analyze":
            action = "queued to analyze" if job.state == "queued" else "analyzing"
        else:
            action = "rendering"
        return f"job `{job.spec.short()}` is {action} this book"
    pid = store.external_render_pid(ws / "work")
    if pid:
        return f"another process (PID {pid}) is working on this book"
    return ""


def _invalidate(chapters) -> str:
    if _busy_reason():
        return ""
    chapters = set(chapters) & S.get("dirty", set())
    if not chapters:
        return ""
    done = store.invalidate_chapters(S["ws"], chapters)
    S["dirty"] -= set(done)
    return f" (re-render: {', '.join(map(str, done))})" if done else ""


def _last_book_path() -> str:
    """Path of the most recently rendered book, for the open panel."""
    runs = ROOT / "runs"
    best = (0.0, "")
    if runs.is_dir():
        for meta_path in runs.glob("*/meta.json"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                argv = data.get("last_argv") or []
                updated = float(data.get("updated") or 0)
                if not argv:
                    continue
                path = Path(argv[0])
                if path.is_file() and updated >= best[0]:
                    best = (updated, str(path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    return best[1]


# --------------------------------------------------------------------------
# render queue helpers
# --------------------------------------------------------------------------

def _jobs_rows() -> List[List[Any]]:
    global JOB_VIEW
    jobs = MANAGER.jobs
    JOB_VIEW = [j.id for j in jobs]
    rows = []
    for job in jobs:
        snap = job.snapshot()
        out = Path(job.spec.out_path).name + ("/" if job.spec.per_chapter else "")
        rows.append([
            job.spec.short(), job.spec.title[:46], job.spec.chapters or "all",
            snap["state"], round(snap["percent"]), f"{snap['elapsed']:.0f}s", out,
        ])
    return rows


def _selected_job():
    if SELECTED_JOB:
        job = MANAGER.get(SELECTED_JOB)
        if job is not None:
            return job
    running = [j for j in MANAGER.jobs if j.state in ("running", "stalled")]
    return running[0] if running else (MANAGER.jobs[-1] if MANAGER.jobs else None)


def _resolve_job(value: str):
    raw = str(value or "").strip()
    jobs = MANAGER.jobs
    if not raw:
        return _selected_job()
    for job in jobs:
        if job.id == raw:
            return job
    matches = [j for j in jobs if j.id.startswith(raw)]
    if len(matches) == 1:
        return matches[0]
    try:
        i = int(raw) - 1
        if 0 <= i < len(jobs):
            return jobs[i]
    except ValueError:
        pass
    return None


def _queue_summary() -> str:
    jobs = MANAGER.jobs
    running = sum(1 for j in jobs if j.state in ("running", "stalled"))
    queued = sum(1 for j in jobs if j.state == "queued")
    return f"{running}/{MANAGER.slots} running · {queued} queued"


def _job_status(job) -> str:
    snap = job.snapshot()
    state = snap["state"]
    analyze = job.spec.kind == "analyze"
    scope = "text-only analysis" if analyze else f"chapters {job.spec.chapters or 'all'}"
    lines = [f"### {job.spec.title}",
             f"Job `{job.spec.short()}` · {scope}"]
    if state == "queued":
        pos = MANAGER.position(job.id)
        lines.append(f"**Queued** · position {pos} · waiting {snap['queued_for']:.0f}s")
    elif state == "running":
        lines.append(f"**Running** · {snap['elapsed']:.0f}s")
    elif state == "stalled":
        lines.append(f"**Stalled** · no output for 5+ minutes · {snap['elapsed']:.0f}s")
    elif state == "cancelled":
        lines.append(f"**Cancelled** · {snap['elapsed']:.0f}s — "
                     + ("click **Continue selected job** to re-run the analysis."
                        if analyze else
                        "finished chunks are kept. Click **Continue selected job** to resume."))
    elif state == "done":
        lines.append(f"**Finished** (exit {snap['returncode']}) · {snap['elapsed']:.0f}s")
    else:
        lines.append(f"**Failed** (exit {snap['returncode']}) · {snap['elapsed']:.0f}s — "
                     + ("click **Continue selected job** to re-run the analysis."
                        if analyze else
                        "finished chunks are kept. Click **Continue selected job** to resume."))

    if snap["chapter_total"]:
        unit = "Step" if analyze else "Chapter"
        lines.append(f"{unit} {snap['chapter_index']}/{snap['chapter_total']}: "
                     f"{snap['chapter_title']}")
        if not analyze:
            lines.append(f"Chunk {snap['chunk_done']}/{snap['chunk_total']}")
    if state == "stalled":
        if analyze:
            lines.append("⚠ **No output for 15+ minutes** — the model request may be "
                         "stuck. Cancel this job, then click **Continue selected job** "
                         "to re-run the analysis.")
        else:
            lines.append("⚠ **No output for 5+ minutes** — the TTS request may be stuck. "
                         "Cancel this job, then click **Continue selected job**; finished "
                         "chunks are kept.")
    if snap["error"] and not snap.get("cancelled"):
        lines.append(f"❌ `{snap['error']}`")
    outs = store.job_outputs(job.spec) if job.done else []
    if outs:
        lines.append(f"Output: `{outs[0]}`" +
                     (f" (+{len(outs) - 1} more)" if len(outs) > 1 else ""))
    elif analyze and job.done and job.returncode == 0:
        lines.append(f"Cast: `{job.spec.out_path}`")
    lines.append(f"Queue: {_queue_summary()}")
    return "\n\n".join(lines)


def _job_view(job, force_audio: bool = False):
    global LAST_AUDIO
    status = _job_status(job)
    pct = job.percent()
    log = job.log_text()
    outs = [str(p) for p in store.job_outputs(job.spec)] if job.done else []
    audio = gr.skip()
    download = gr.skip()
    if outs:
        key = (job.id, tuple(outs))
        if force_audio or key != LAST_AUDIO:
            LAST_AUDIO = key
            audio = outs[0]
            download = outs if len(outs) > 1 else outs[0]
    elif force_audio:
        LAST_AUDIO = None
        audio = gr.update(value=None)
        download = gr.update(value=None)
    return status, pct, log, audio, download


def _book_run_md(session: Optional[Dict[str, Any]] = None) -> str:
    session = session or S
    book = session.get("book")
    if not book:
        return "No book open."
    job = _book_job(session)
    if job:
        snap = job.snapshot()
        state = snap["state"]
        analyze = job.spec.kind == "analyze"
        head = (f"**{state.title()}** · "
                f"{'analysis' if analyze else 'job'} `{job.spec.short()}`")
        if state == "queued":
            head += f" · position {MANAGER.position(job.id)}"
        if snap["chapter_total"]:
            unit = "step" if analyze else "chapter"
            head += (f" · {unit} {snap['chapter_index']}/{snap['chapter_total']}"
                     f" · {snap['percent']:.0f}%")
        if snap["chunk_total"]:
            head += f" · chunks {snap['chunk_done']}/{snap['chunk_total']}"
        if state in ("cancelled", "error"):
            head += " — use **Continue selected job** in Activity."
        if state == "done":
            if analyze:
                head += " · cast ready to review in **Cast & quotes**"
            else:
                outs = store.job_outputs(job.spec)
                if outs:
                    head += f" · `{outs[0]}`"
        return head
    if session.get("ws"):
        pid = store.external_render_pid(session["ws"] / "work")
        if pid:
            return (f"⚠ An external render (PID {pid}) is writing this book's "
                    f"workspace. Wait for it to finish before rendering here.")
    if session.get("needs_rerender"):
        return ("Ready, but applied settings differ from the last render. "
                "The next render will start over after confirmation.")
    return "Ready. Saved chunks are reused automatically."


def _render_warning(session: Optional[Dict[str, Any]] = None) -> str:
    session = session or S
    if not session.get("needs_rerender"):
        return ""
    return ("⚠ **Applied content settings differ from the last render.** "
            "Starting now will discard cached chunks and re-render from scratch. "
            "Tick the confirmation box to continue, or revert the settings.")


def _radio_choices() -> List[tuple]:
    choices = []
    for slug in OPEN_ORDER:
        session = SESSIONS.get(slug)
        if not session or not session.get("book"):
            continue
        title = session["book"].title
        choices.append((f"{title}  ·  {slug}", slug))
    return choices


def _radio_update():
    return gr.update(choices=_radio_choices(), value=ACTIVE_SLUG)


def _persist_session() -> None:
    """Remember open books and the active one across UI restarts."""
    data = {
        "open": [str(Path(SESSIONS[s]["path"]).resolve()) for s in OPEN_ORDER
                 if SESSIONS.get(s, {}).get("path")],
        "active": ACTIVE_SLUG,
    }
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _restore_session(limit: int = 5) -> None:
    """Reopen the books that were open in the last UI session."""
    if not SESSION_FILE.exists():
        return
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for raw in (data.get("open") or [])[:limit]:
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            _load_session(path)
        except Exception:
            continue
    active = data.get("active")
    if active in SESSIONS:
        _activate(active)
    elif OPEN_ORDER:
        _activate(OPEN_ORDER[0])


def _workspace_values(status: Optional[str] = None,
                      session: Optional[Dict[str, Any]] = None) -> List[Any]:
    """Values for WORKSPACE_OUTPUTS, in the same order as the components."""
    session = session or S
    book = session.get("book")
    if not book:
        return [
            "No book open. Use **Open a book** below.",
            [],
            "all",
            "Open a book first.",
            status or "",
            [],
            "Analyze a book first.",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            [],
            "",
            "",
            "No book open.",
            "",
            gr.update(visible=False),
            "No book open.",
            "",
            gr.update(value=False, visible=False),
            False,
            "",
            gr.update(visible=False),
            *_param_widget_values(session),
            gr.update(open=True),
        ]

    cast = session.get("cast")
    cast_rows = castlib.characters_rows(cast) if cast else []
    speakers = castlib.character_names(cast, include_buckets=True) if cast else []
    qlabels = _quote_labels(book, cast)
    return [
        _book_info_md(session),
        _chapter_rows(book),
        session.get("chapter_range") or "all",
        _range_summary(session),
        status or "Book ready.",
        cast_rows,
        _cast_note(session),
        gr.update(choices=qlabels, value=qlabels[0] if qlabels else None),
        gr.update(choices=speakers, value=speakers[0] if speakers else None),
        [],
        "",
        session.get("extra", ""),
        "Settings loaded." if not session.get("needs_rerender")
        else "Applied settings differ from the last render.",
        "",
        gr.update(visible=False),
        _book_run_md(session),
        _render_warning(session),
        gr.update(value=False,
                  visible=bool(session.get("needs_rerender"))),
        False,
        "",
        gr.update(visible=False),
        *_param_widget_values(session),
        gr.update(open=False),
    ]


def _idle_activity() -> List[Any]:
    return ["", "Idle — queue empty.", 0.0, "",
            gr.update(value=None), gr.update(value=None)]


def _activity_values(session: Optional[Dict[str, Any]] = None) -> List[Any]:
    global SELECTED_JOB
    session = session or S
    # Prefer this book's latest job, but keep the shared panel on the last
    # selected job when this book has none.  That way closing a tab while its
    # render continues does not hide the running job.
    job = _book_job(session) or _selected_job()
    if job is None:
        return _idle_activity()
    SELECTED_JOB = job.id
    status, pct, log, audio, download = _job_view(job, force_audio=True)
    return [job.id, status, pct, log, audio, download]


def _book_view(status: Optional[str] = None,
               session: Optional[Dict[str, Any]] = None) -> List[Any]:
    _persist_session()
    return [_radio_update()] + _workspace_values(status, session) + _activity_values(session)


# --------------------------------------------------------------------------
# book callbacks
# --------------------------------------------------------------------------

def on_open_upload(file):
    if not file:
        return _book_view("Choose an EPUB or PDF first.")
    src = Path(file)
    if src.suffix.lower() not in SUPPORTED_SUFFIXES:
        suffix = src.suffix or "(no extension)"
        return _book_view(f"**Unsupported file type `{suffix}` — use .epub or .pdf.**")
    try:
        book = _extract(src)          # parse before copying, so bad files never land in books/
        path = store.ingest_upload(file, src.name)
        session = _load_session(path, book=book)
        _activate(session["slug"])
        return _book_view(f"Opened **{book.title}**.", session)
    except Exception as exc:
        return _book_view(f"**Upload failed:** `{exc}`")


def on_open_path(path_text: str):
    raw = (path_text or "").strip()
    if not raw:
        return _book_view("**Enter a path to an .epub or .pdf file.**")
    path = Path(raw).expanduser()
    if not path.exists():
        return _book_view(f"**Not found:** `{path}`")
    problem = _book_path_error(path)
    if problem:
        return _book_view(f"**{problem}**")
    try:
        session = _load_session(path)
        _activate(session["slug"])
        return _book_view(f"Opened **{session['book'].title}**.", session)
    except Exception as exc:
        return _book_view(f"**Failed to open:** `{exc}`")


def on_switch_book(slug: str):
    if not slug:
        return _book_view()
    _activate(slug)
    return _book_view()


def on_close_book():
    global ACTIVE_SLUG
    if not ACTIVE_SLUG:
        return _book_view("No book is open.")
    slug = ACTIVE_SLUG
    session = SESSIONS.get(slug)
    running = _book_job(session) if session else None
    if slug in OPEN_ORDER:
        OPEN_ORDER.remove(slug)
    SESSIONS.pop(slug, None)
    if OPEN_ORDER:
        _activate(OPEN_ORDER[0])
    else:
        _activate(None)
    msg = "Closed the tab."
    if running and running.running:
        action = "analysis" if running.spec.kind == "analyze" else "render"
        msg += f" The {action} continues in Activity."
        # Keep the running job selected so it stays visible.
        global SELECTED_JOB
        SELECTED_JOB = running.id
    return _book_view(f"**{msg}**")


def on_analyze():
    if not S.get("book"):
        return [], _cast_note(), gr.update(), gr.update(), "Load a book first.", "No book open."
    busy = _busy_reason()
    if busy:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                f"**Cannot analyze: {busy}.**", _book_run_md())
    values = _effective_values(S)
    if values.get("attribution_mode") in ("light", "heavy"):
        ok, msg = _queue_analysis(values, S.get("extra", ""))
        if not ok:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    f"**{msg}**", _book_run_md())
        return (gr.update(), _cast_note(), gr.update(), gr.update(),
                f"**{msg}**", _book_run_md())

    ok, msg = _analyze_now(values, S.get("extra", ""))
    if not ok or not S["cast"]:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                f"**{msg}**", _book_run_md())
    rows, speakers, qs = _reload_cast(S["cast"])
    return rows, _cast_note(), qs, speakers, f"**{msg}**", _book_run_md()


def on_range_change(text: str):
    if not S.get("book"):
        return "Open a book first."
    S["chapter_range"] = (text or "").strip() or "all"
    _save_session_meta()
    return _range_summary()


def on_select_all_range():
    if not S.get("book"):
        return "all", "Open a book first."
    S["chapter_range"] = "all"
    _save_session_meta()
    return "all", _range_summary()


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

def on_apply_params(*args):
    if not S.get("book"):
        return "Open a book first.", "", gr.update(visible=False)
    *widgets, extra = args
    extra = extra or ""
    draft = _values_from_widgets(widgets)
    applied = _effective_values(S)

    changed = []
    for key in PARAM_ORDER:
        old = applied.get(key)
        new = draft.get(key)
        if old != new:
            changed.append((key, old, new))
    extra_changed = extra != S.get("extra", "")

    if not changed and not extra_changed:
        return "No changes.", "", gr.update(visible=False)

    busy = _busy_reason()
    if busy:
        return (f"**Cannot apply settings while {busy}.** Stop the job first; "
                f"runtime settings in Activity can still change."), "", gr.update(visible=False)

    content_changed = extra_changed or any(key not in RUNTIME_KEYS for key, _, _ in changed)
    if not content_changed:
        S["params"] = draft
        S["extra"] = extra
        _save_session_meta()
        return ("Applied. These are runtime-only changes and will be used by the "
                "next render."), "", gr.update(visible=False)

    S["pending_params"] = draft
    S["pending_extra"] = extra
    lines = ["**Apply these changes?**"]
    for key, old, new in changed:
        label = pspec.BY_KEY[key].label
        suffix = "next run only" if key in RUNTIME_KEYS else "content change — re-renders"
        lines.append(f"- **{label}**: {_format_value(old)} → {_format_value(new)}  ·  {suffix}")
    if extra_changed:
        lines.append(f"- **Extra CLI args**: `{S.get('extra', '')}` → `{extra}`  ·  content change")
    lines.append("\nContent changes do not touch the current render. The **next** "
                 "render will start over after you confirm it.")
    return ("Review the changes below and confirm.", "\n".join(lines),
            gr.update(visible=True))


def on_confirm_params():
    if S.get("pending_params") is None:
        return ("Nothing to confirm.", "", gr.update(visible=False),
                _book_run_md(), _render_warning(), gr.update(visible=False, value=False))
    S["params"] = S["pending_params"]
    S["extra"] = S.get("pending_extra", "")
    S["pending_params"] = None
    S["pending_extra"] = ""
    S["needs_rerender"] = _settings_differ()
    _save_session_meta()
    if S["needs_rerender"]:
        status = ("Applied. The next render will start over because content settings "
                  "differ from the last render.")
    else:
        status = "Applied."
    return (status, "", gr.update(visible=False), _book_run_md(), _render_warning(),
            gr.update(value=False, visible=bool(S["needs_rerender"])))


def on_revert_params():
    S["pending_params"] = None
    S["pending_extra"] = ""
    S["needs_rerender"] = _settings_differ()
    _save_session_meta()
    return [
        *_param_widget_values(S),
        S.get("extra", ""),
        "Reverted to the applied settings.",
        "",
        gr.update(visible=False),
        _book_run_md(),
        _render_warning(),
        gr.update(value=False, visible=bool(S.get("needs_rerender"))),
    ]


def on_preview_command():
    if not S.get("path") or not S.get("book"):
        return "Open a book first."
    values = _effective_values(S)
    argv = [str(S["path"])] + pspec.build_argv(values, extra=S.get("extra", ""))
    if S.get("ws"):
        per_chapter = bool(values.get("per_chapter"))
        argv += [
            "--cast-file", str(store.cast_path(S["ws"])),
            "--workdir", str(S["ws"] / "work"),
            "-o", str(store.out_path(S["ws"], per_chapter, S["path"])),
        ]
    spec, _, error = _parse_chapter_range(S.get("chapter_range", "all"),
                                          len(S["book"].chapters))
    if error:
        return f"**Chapter selection error:** {error}"
    if spec:
        argv += ["--chapters", spec]
    return "python book_to_audio.py " + shlex.join(str(a) for a in argv)


def on_load_preset(name: str):
    vals = store.load_preset(name) if name else {}
    out = []
    for key in PARAM_ORDER:
        p = pspec.BY_KEY[key]
        value = vals.get(key, p.default)
        if p.kind == "choice" and key != "narrator_gender" and value not in p.choices:
            value = p.default
        out.append(_ui_value(key, value))
    return out


def on_save_preset(name: str, *args):
    if not name.strip():
        return "**Preset name is empty.**", gr.update()
    vals = _values_from_widgets(args)
    vals["narrator_gender"] = "auto" if not vals.get("narrator_gender") else vals["narrator_gender"]
    path = store.save_preset(name, vals)
    names = store.preset_names()
    return f"Saved `{path}`.", gr.update(choices=names, value=path.stem)


# --------------------------------------------------------------------------
# cast / quotes
# --------------------------------------------------------------------------

def on_cast_select(evt: gr.SelectData):
    if not S["cast"]:
        return (gr.update(),) * 7
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        row = int(row)
    except (TypeError, ValueError):
        return (gr.update(),) * 7
    ordered = castlib.ordered_characters(S["cast"])
    if not (0 <= row < len(ordered)):
        return (gr.update(),) * 7
    c = ordered[row]
    return (row + 1, c["id"], c.get("name") or "", c.get("gender") or "",
            c.get("voice") or "", c.get("pitch") or "+0Hz", gr.update())


def on_row_number(row):
    if not S["cast"] or not row:
        return (gr.update(),) * 6
    ordered = castlib.ordered_characters(S["cast"])
    i = int(row) - 1
    if not (0 <= i < len(ordered)):
        return (gr.update(),) * 6
    c = ordered[i]
    return (c["id"], c.get("name") or "", c.get("gender") or "",
            c.get("voice") or "", c.get("pitch") or "+0Hz", gr.update())


def on_gender_change(cid: str, gender: str, profile: str = ""):
    if not S["cast"] or not cid:
        return gr.update()
    if gender not in ("male", "female"):
        return gr.update()
    voice = castlib.auto_voice(S["cast"], gender, exclude_cid=cid, profile=profile or None)
    if voice:
        return gr.update(value=voice)
    return gr.update()


def on_apply(cid: str, name: str, gender: str, voice: str, pitch: str):
    if not S["cast"] or not cid:
        return gr.update(), "No character selected."
    busy = _busy_reason()
    if busy:
        return gr.update(), f"**Cannot edit: {busy}.**"
    data = S["cast"]
    old = castlib.find(data, cid) or {}
    gender = gender or None
    if gender != (old.get("gender") or None) and not (voice or "").strip():
        voice = castlib.auto_voice(data, gender, exclude_cid=cid) or (old.get("voice") or "")
    unchanged = (
        (name or "").strip() == (old.get("name") or "")
        and (gender or None) == (old.get("gender") or None)
        and (voice or "").strip() == (old.get("voice") or "")
        and (pitch or "").strip() == (old.get("pitch") or "+0Hz")
    )
    if unchanged:
        return gr.update(), "No changes."
    ok, msg = castlib.apply(data, cid, name=name, gender=gender or "", voice=voice, pitch=pitch)
    if not ok:
        return gr.update(), f"**{msg}**"
    castlib.save(S["ws"], data)
    chapters = _chapters_for(cid)
    S["dirty"] |= set(chapters)
    note = _invalidate(chapters)
    rows, speakers, qs = _reload_cast(data)
    return rows, f"{msg}: `{cid}`{note}"


def on_merge(src_label: str, dst_label: str):
    if not S["cast"]:
        return gr.update(), "No cast."
    busy = _busy_reason()
    if busy:
        return gr.update(), f"**Cannot edit: {busy}.**"
    src = (src_label or "").split("\t")[0]
    dst = (dst_label or "").split("\t")[0]
    data = S["cast"]
    if not src or not dst:
        return gr.update(), "Pick both characters."
    ok, msg = castlib.merge(data, src, dst)
    if not ok:
        return gr.update(), f"**{msg}**"
    castlib.save(S["ws"], data)
    chapters = _chapters_for(dst)
    S["dirty"] |= set(chapters)
    note = _invalidate(chapters)
    rows, speakers, qs = _reload_cast(data)
    return rows, f"{msg}{note}"


def on_delete(cid: str):
    if not S["cast"] or not cid:
        return gr.update(), "No character selected."
    busy = _busy_reason()
    if busy:
        return gr.update(), f"**Cannot edit: {busy}.**"
    chapters = _chapters_for(cid)
    ok, msg = castlib.delete(S["cast"], cid)
    if not ok:
        return gr.update(), f"**{msg}**"
    castlib.save(S["ws"], S["cast"])
    S["dirty"] |= set(chapters)
    note = _invalidate(chapters)
    rows, speakers, qs = _reload_cast(S["cast"])
    return rows, f"{msg}{note}"


def on_audition(cid: str, voice: str, pitch: str, rate: str, engine: str):
    if not S["cast"] or not cid:
        return gr.update(), "No character selected."
    text = castlib.character_sample(S["cast"], S["texts"], cid) or "This is a voice preview."
    try:
        p = preview.synth(text, voice, S["ws"] / "preview", pitch=pitch or "+0Hz",
                          rate=rate or "+0%", engine=engine or "edge")
        return str(p), f"Preview: *{text[:120]}*"
    except Exception as exc:
        return gr.update(), f"**Preview failed:** `{exc}`"


def on_save_cast():
    if not S["cast"]:
        return gr.update(), "No cast."
    busy = _busy_reason()
    if busy:
        return gr.update(), f"**Cannot edit: {busy}.**"
    castlib.save(S["ws"], S["cast"])
    note = _invalidate(S["dirty"])
    rows, speakers, qs = _reload_cast(S["cast"])
    return rows, "Cast saved." + note


def on_revert():
    if not S["ws"]:
        return gr.update(), "No workspace."
    busy = _busy_reason()
    if busy:
        return gr.update(), f"**Cannot edit: {busy}.**"
    bak = store.cast_path(S["ws"]).with_suffix(".json.bak")
    if not bak.exists():
        return gr.update(), "No backup yet."
    data = json.loads(bak.read_text(encoding="utf-8"))
    castlib.save(S["ws"], data)
    chapters = []
    for key in castlib.chapter_keys(data):
        try:
            chapters.append(int(key))
        except (TypeError, ValueError):
            pass
    note = ""
    if store.invalidate_chapters(S["ws"], chapters):
        note = f" (re-render: {', '.join(map(str, chapters))})"
    rows, speakers, qs = _reload_cast(data)
    S["dirty"].clear()
    return rows, "Reverted to the previous cast." + note


def on_cast_import_preview(file):
    """Validate an LLM/Colab cast before it replaces the workspace cast."""
    if not S.get("ws"):
        S["pending_cast"] = None
        return "Load a book first.", gr.update(visible=False)
    if not file:
        S["pending_cast"] = None
        return "Choose a cast JSON file first.", gr.update(visible=False)
    path = Path(file[0] if isinstance(file, (list, tuple)) else file)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        S["pending_cast"] = None
        return f"**Could not read `{path}`:** {exc}", gr.update(visible=False)
    ok, _problems, summary = castlib.validate_import(data, S.get("book"), S.get("cast"))
    S["pending_cast"] = data if ok else None
    return summary, gr.update(visible=ok)


def on_cast_import_apply():
    data = S.get("pending_cast")
    if not data:
        return (gr.update(), _cast_note(), gr.update(), gr.update(),
                "Nothing validated to import.", gr.update(visible=False))
    if not S.get("ws"):
        return (gr.update(), _cast_note(), gr.update(), gr.update(),
                "Load a book first.", gr.update(visible=False))
    busy = _busy_reason()
    if busy:
        return (gr.update(), _cast_note(), gr.update(), gr.update(),
                f"**Cannot replace the cast: {busy}.**", gr.update(visible=False))
    castlib.save(S["ws"], data)
    S["pending_cast"] = None
    S["dirty"].clear()
    # The CLI's resume signature hashes the cast *path*, not its contents.
    # Drop any completed audio that used the old cast so a later render cannot
    # silently keep the old speakers.
    old_att = (S.get("cast") or {}).get("attribution") or {}
    chapters = {int(k) for k in list(data.get("attribution") or {}) + list(old_att)
                if str(k).isdigit()}
    redone = store.invalidate_chapters(S["ws"], chapters)
    rows, speakers, qs = _reload_cast(data)
    note = f" Re-render queued for {len(redone)} chapter(s)." if redone else ""
    return (rows, _cast_note(), qs, speakers,
            f"**Imported cast applied** · {castlib.source_label(data)}. The previous "
            f"cast is in `cast.json.bak`.{note}", gr.update(visible=False))


def on_q_chapter(label: str, only: bool = True):
    if not S["cast"] or not label or not S["book"]:
        return []
    key = str(_label_index(label) or "")
    if not key:
        return []
    text = S["texts"].get(int(key), "")
    return castlib.quote_rows(S["cast"], key, text, bool(only))


def on_q_filter(label: str, only: bool):
    if not S["cast"] or not label:
        return []
    key = str(_label_index(label) or "")
    if not key:
        return []
    text = S["texts"].get(int(key), "")
    return castlib.quote_rows(S["cast"], key, text, only)


def on_q_select(evt: gr.SelectData, chapter_label: str = "", only: bool = True):
    if not S["cast"]:
        return gr.update(), gr.update()
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        key = str(_label_index(chapter_label) or "")
        text = S["texts"].get(int(key), "")
        rows = castlib.quote_rows(S["cast"], key, text, bool(only))
        seg = rows[int(row)][0]
    except (TypeError, ValueError, IndexError, KeyError):
        return gr.update(), gr.update()
    return seg, gr.update()


def on_assign(seg, speaker_label: str, chapter_label: str = "", only: bool = True):
    if not S["cast"]:
        return gr.update(), "No cast."
    busy = _busy_reason()
    if busy:
        return gr.update(), f"**Cannot edit: {busy}.**"
    key = str(_label_index(chapter_label) or "")
    if not key:
        return gr.update(), "Pick a chapter."
    cid = (speaker_label or "").split("\t")[0]
    if not cid:
        return gr.update(), "Pick a speaker."
    try:
        seg = int(seg)
    except (TypeError, ValueError):
        return gr.update(), "Pick a segment."
    ok, msg = castlib.assign(S["cast"], key, seg, cid)
    if not ok:
        return gr.update(), f"**{msg}**"
    castlib.save(S["ws"], S["cast"])
    chapter = int(key)
    S["dirty"].add(chapter)
    note = _invalidate({chapter})
    rows, speakers, qs = _reload_cast(S["cast"])
    return on_q_chapter(chapter_label, only), f"{msg}{note}"


# --------------------------------------------------------------------------
# render callbacks
# --------------------------------------------------------------------------

def _set_flag(args: List[str], flag: str, value: Any) -> List[str]:
    out, i = [], 0
    while i < len(args):
        token = args[i]
        if token == flag:
            i += 2 if i + 1 < len(args) and not args[i + 1].startswith("-") else 1
            continue
        if token.startswith(flag + "="):
            i += 1
            continue
        out.append(token)
        i += 1
    if value is not None:
        out += [flag, str(value)]
    return out


def on_render(chapter_range: str, start_over: bool, confirm_restart: bool):
    global SELECTED_JOB
    _poll_analyze_job()
    rows = _jobs_rows()
    book_md = _book_run_md()
    if not S.get("path") or not S.get("ws") or not S.get("book"):
        return ("Load a book first.", rows, 0.0, "", book_md,
                "Open a book first.", "No book open.", "", gr.update(visible=False),
                gr.update(value=False))

    text = (chapter_range or "").strip() or "all"
    S["chapter_range"] = text
    _save_session_meta()
    spec, _indices, error = _parse_chapter_range(text, len(S["book"].chapters))
    if error:
        return (f"**Chapter selection error:** {error}", rows, 0.0, "", book_md,
                _range_summary(), "Fix the chapter range.", "", gr.update(visible=False),
                gr.update(value=False))

    needs_restart = bool(S.get("needs_rerender")) and not start_over
    if needs_restart and not confirm_restart:
        return ("**Settings changed.** Review the warning below and tick the "
                "confirmation box, or enable **Start over**.", rows, 0.0, "",
                book_md, _range_summary(), "Settings need confirmation.",
                _render_warning(), gr.update(value=False, visible=True),
                gr.update(value=False))

    workdir = S["ws"] / "work"
    busy = MANAGER.has_workspace(workdir)
    if busy:
        return (f"**Already {busy.state}** as job `{busy.spec.short()}`. "
                f"Cancel it first, or wait for it to finish.", rows, 0.0, "",
                book_md, _range_summary(), "Wait for the current job.",
                _render_warning(), gr.update(visible=needs_restart),
                gr.update(value=False))
    pid = store.external_render_pid(workdir)
    if pid:
        return (f"**PID {pid} is already rendering this book.** Wait for it to "
                f"finish (or stop that process) before queuing another.", rows, 0.0, "",
                book_md, _range_summary(), "Wait for the external render.",
                _render_warning(), gr.update(visible=needs_restart),
                gr.update(value=False))

    values = _effective_values(S)
    if start_over or needs_restart:
        values["fresh"] = True

    if not store.cast_path(S["ws"]).exists():
        if values.get("attribution_mode") in ("light", "heavy"):
            return ("**No cast yet.** Click **Analyze cast** first; light/heavy "
                    "analysis runs in Activity and can take longer than the old "
                    "synchronous timeout. Render again when the cast is ready.",
                    rows, 0.0, "", book_md, _range_summary(), "Analyze first.",
                    _render_warning(), gr.update(visible=needs_restart),
                    gr.update(value=False))
        ok, msg = _analyze_now(values, S.get("extra", ""))
        if not ok:
            return (f"**{msg}**", rows, 0.0, msg, book_md,
                    _range_summary(), "Analysis failed.", _render_warning(),
                    gr.update(visible=needs_restart), gr.update(value=False))
    if not store.cast_path(S["ws"]).exists():
        return ("No cast file — run Analyze first.", rows, 0.0, "", book_md,
                _range_summary(), "No cast.", "", gr.update(visible=False),
                gr.update(value=False))

    per_chapter = bool(values.get("per_chapter"))
    out = store.out_path(S["ws"], per_chapter, S["path"])
    if per_chapter:
        out.mkdir(parents=True, exist_ok=True)
    argv = [str(S["path"])] + pspec.build_argv(values, extra=S.get("extra", ""))
    argv += ["--cast-file", str(store.cast_path(S["ws"])),
             "--workdir", str(workdir), "-o", str(out)]
    if spec:
        argv += ["--chapters", spec]

    job_spec = JobSpec(
        book=S["path"], title=S["book"].title,
        cast_file=store.cast_path(S["ws"]), workdir=workdir,
        out_path=out, args=argv, chapters=spec, per_chapter=per_chapter,
    )
    fingerprint = _content_fingerprint(values, S.get("extra", ""))
    store.save_meta(
        S["ws"], last_argv=argv, chapters=spec, chapter_range=text,
        per_chapter=per_chapter, last_job=job_spec.id, output=str(out),
        input=str(S["path"]), params=S["params"], extra=S.get("extra", ""),
        last_render_params=fingerprint, needs_rerender=False,
    )
    try:
        job = MANAGER.enqueue(job_spec, cwd=ROOT,
                              log_path=store.job_log_path(S["ws"], job_spec.id))
    except Exception as exc:
        return (f"**Could not queue:** `{exc}`", rows, 0.0, "", book_md,
                _range_summary(), "Queue failed.", _render_warning(),
                gr.update(visible=needs_restart), gr.update(value=False))
    S["needs_rerender"] = False
    S["last_render_params"] = fingerprint
    S["last_job"] = job.id
    SELECTED_JOB = job.id
    rows = _jobs_rows()
    if job.state == "queued":
        head = f"**Queued** `{job.spec.short()}` (position {MANAGER.position(job.id)})."
    elif job.state in ("running", "stalled"):
        head = f"**Started** `{job.spec.short()}`."
    else:
        head = f"**Job `{job.spec.short()}` failed:** `{job.error}`"
    return (
        f"{head} Chapters: {spec or 'all'} · output `{out}`",
        rows, 0.0, "", _book_run_md(), _range_summary(), "Rendering.",
        "", gr.update(value=False, visible=False), gr.update(value=False),
    )


def on_continue_job(selector: str):
    global SELECTED_JOB
    rows = _jobs_rows()
    book_md = _book_run_md()
    job = _resolve_job(selector)
    if job is None:
        return ("Select a **cancelled** or **failed** job in the table first.",
                rows, 0.0, "", book_md)
    if job.state in ("queued", "running", "stalled"):
        return (f"Job `{job.spec.short()}` is already {job.state}; wait for it "
                f"or cancel it first."), rows, 0.0, "", book_md

    spec = job.spec
    analyze = spec.kind == "analyze"
    if not spec.workdir.exists():
        return (f"No saved work folder for `{spec.short()}`. Use "
                f"**Render / continue current book** to start from scratch."), \
            rows, 0.0, "", book_md
    if not analyze and not spec.cast_file.exists():
        return (f"Cast file for `{spec.short()}` is missing: `{spec.cast_file}`."), \
            rows, 0.0, "", book_md
    if not spec.book.exists():
        return (f"Book for `{spec.short()}` is missing: `{spec.book}`."), \
            rows, 0.0, "", book_md

    busy = MANAGER.has_workspace(spec.workdir)
    if busy:
        return (f"This book is already {busy.state} as job `{busy.spec.short()}`."), \
            rows, 0.0, "", book_md
    pid = store.external_render_pid(spec.workdir)
    if pid:
        return (f"PID {pid} is already rendering this book. Wait for it to "
                f"finish before continuing."), rows, 0.0, "", book_md

    # Continue keeps the job's original content settings and chapters, but
    # applies the current global runtime values and the EPUB-named output.
    args = [a for a in spec.args if a != "--fresh" and not a.startswith("--fresh=")]
    args = _set_flag(args, "--concurrency", RUNTIME["concurrency"])
    args = _set_flag(args, "--on-chunk-failure", RUNTIME["on_chunk_failure"])
    out_path = spec.out_path
    if not analyze:
        out_path = store.out_path(spec.workdir.parent, spec.per_chapter, spec.book)
        args = _set_flag(args, "-o", None)
        args = _set_flag(args, "--output", None)
        args += ["-o", str(out_path)]
    new_spec = JobSpec(
        book=spec.book, title=spec.title, cast_file=spec.cast_file,
        workdir=spec.workdir, out_path=out_path, args=args,
        chapters=spec.chapters, per_chapter=spec.per_chapter,
        kind=spec.kind,
    )
    if spec.per_chapter:
        Path(out_path).mkdir(parents=True, exist_ok=True)
    ws = spec.workdir.parent
    meta_update = {"last_job": new_spec.id, "input": str(spec.book)}
    if not analyze:
        meta_update["output"] = str(out_path)
        meta_update["per_chapter"] = spec.per_chapter
    store.save_meta(ws, **meta_update)
    try:
        new_job = MANAGER.enqueue(new_spec, cwd=ROOT,
                                  log_path=store.job_log_path(ws, new_spec.id))
    except Exception as exc:
        return f"**Could not continue:** `{exc}`", rows, 0.0, "", book_md
    SELECTED_JOB = new_job.id
    owner = next((session for session in SESSIONS.values()
                  if session.get("ws")
                  and Path(session["ws"]).resolve() == Path(ws).resolve()), None)
    if owner is not None:
        owner["last_job"] = new_job.id
        if analyze:
            owner["analyze_job"] = new_job.id
            owner["analyze_prev_chapters"] = _cast_chapters(owner.get("cast"))
    rows = _jobs_rows()
    state = "Queued" if new_job.state == "queued" else "Started"
    if analyze:
        return (f"**{state}** `{new_spec.short()}` — re-running analysis "
                f"`{spec.short()}` · cast `{out_path}`"), \
            rows, 0.0, "", _book_run_md()
    return (f"**{state}** `{new_spec.short()}` — continuing `{spec.short()}` "
            f"from saved chunks · output `{out_path}`"), \
        rows, 0.0, "", _book_run_md()


def on_cancel_job(selector: str):
    global SELECTED_JOB
    job = _resolve_job(selector)
    if job is None:
        return "No job selected.", _jobs_rows(), _book_run_md()
    SELECTED_JOB = job.id
    was_running = job.running
    ok = MANAGER.cancel(job.id)
    if not ok:
        return (f"Job `{job.spec.short()}` is not queued or running.",
                _jobs_rows(), _book_run_md())
    if was_running:
        if job.spec.kind == "analyze":
            return (f"Cancelling `{job.spec.short()}`… The previous cast is kept. "
                    f"Click **Continue selected job** to re-run the analysis."), \
                _jobs_rows(), _book_run_md()
        return (f"Cancelling `{job.spec.short()}`… Saved chunks are kept — "
                f"click **Continue selected job** to resume."), \
            _jobs_rows(), _book_run_md()
    return f"Removed `{job.spec.short()}` from the queue.", _jobs_rows(), _book_run_md()


def on_cancel_all():
    n = MANAGER.cancel_all()
    return ((f"Cancelling {n} job(s)." if n else "No queued or running jobs."),
            _jobs_rows(), _book_run_md())


def on_delete_job(selector: str):
    """Delete one finished job from Activity; rendered files are kept."""
    global SELECTED_JOB
    job = _resolve_job(selector)
    rows = _jobs_rows()
    if job is None:
        return "Select a finished job in the table first.", rows, _book_run_md()
    if job.state in ("queued", "running", "stalled"):
        return (f"Job `{job.spec.short()}` is {job.state}. Cancel it first, "
                f"then delete it."), rows, _book_run_md()
    for session in SESSIONS.values():
        if session.get("analyze_job") == job.id:
            return ("Wait for this analysis result to load before deleting "
                    "the job."), rows, _book_run_md()
    if not MANAGER.delete(job.id):
        return f"Could not delete job `{job.spec.short()}`.", rows, _book_run_md()
    if SELECTED_JOB == job.id:
        SELECTED_JOB = None
    for session in SESSIONS.values():
        if session.get("last_job") == job.id:
            session["last_job"] = None
        if session.get("analyze_job") == job.id:
            session["analyze_job"] = None
    return (f"Deleted job `{job.spec.short()}` from Activity. "
            f"Its log was removed; rendered audio and cast files were kept."), \
        _jobs_rows(), _book_run_md()


def on_clear_finished():
    global SELECTED_JOB
    n = MANAGER.clear_finished()
    if SELECTED_JOB and MANAGER.get(SELECTED_JOB) is None:
        SELECTED_JOB = None
    return ((f"Cleared {n} finished job(s)." if n else "No finished jobs to clear."),
            _jobs_rows(), _book_run_md())


def on_slots(slots):
    n = MANAGER.set_slots(int(slots or 1))
    store.save_queue_slots(n)
    return (f"Parallel renders: **{n}**. Keep parallel × TTS concurrency ≤ ~12 "
            f"to avoid Edge rate limiting."), _jobs_rows(), _book_run_md()


def on_runtime_change(concurrency, on_chunk_failure):
    global RUNTIME
    try:
        conc = max(1, min(16, int(concurrency or 6)))
    except (TypeError, ValueError):
        conc = 6
    failure = on_chunk_failure if on_chunk_failure in ("stop", "skip") else "stop"
    RUNTIME = store.save_runtime_settings(conc, failure)
    return (f"Runtime saved: **{conc}** TTS requests, failure policy **{failure}**. "
            f"This applies to new jobs and **Continue**; running jobs keep their "
            f"captured settings.")


# --------------------------------------------------------------------------
# sharing: rclone/Google Drive + phone download link + QR
# --------------------------------------------------------------------------

def _local_ips() -> List[str]:
    """Reachable physical IPv4 addresses, LAN first, then Tailscale."""
    ips: List[str] = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"],
                             capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            m = re.match(r"\d+:\s+(\S+)\s+inet (\d+\.\d+\.\d+\.\d+)", line)
            if not m:
                continue
            iface, ip = m.group(1), m.group(2)
            if iface == "lo" or iface.startswith(("docker", "br-", "veth")):
                continue
            ips.append(ip)
    except (OSError, subprocess.SubprocessError):
        ips = []
    if not ips:
        try:
            out = subprocess.run(["hostname", "-I"], capture_output=True,
                                 text=True, timeout=3).stdout
            ips = out.split()
        except (OSError, subprocess.SubprocessError):
            ips = []
    ips = [ip for ip in ips if ip and not ip.startswith("127.")]
    if not ips:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips = [s.getsockname()[0]]
        except OSError:
            ips = []
        finally:
            s.close()

    def rank(ip: str) -> int:
        if ip.startswith(("192.168.", "10.", "172.")):  # private LAN
            return 0
        if ip.startswith("100."):                       # Tailscale CGNAT
            return 1
        return 2
    return sorted(dict.fromkeys(ips), key=rank)


def _autodetect_base() -> str:
    ips = _local_ips()
    return f"http://{ips[0] if ips else '127.0.0.1'}:{SERVER_PORT}"


def _make_qr(url: str, slug: str) -> Optional[str]:
    qdir = store.RUNS / "share"
    try:
        qdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    out_png = qdir / f"qr_{slug}.png"
    try:
        subprocess.run(["qrencode", "-o", str(out_png), "-s", "6", "-m", "2", url],
                       check=True, capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return str(out_png)


def _share_target(selector: str):
    """The job (if any) and workspace owning the output to share."""
    job = _resolve_job(selector)
    if job is not None:
        return job, Path(job.spec.workdir).parent
    ws = S.get("ws")
    return None, Path(ws) if ws else None


def on_detect_base():
    return _autodetect_base()


def on_phone_link(selector, base_url, request: gr.Request = None):
    job, ws = _share_target(selector)
    if not ws:
        return "", None, "Open a book or select a finished job first.", gr.skip()
    file = store.output_for_sharing(ws, job.spec.out_path if job else None)
    if not file:
        return "", None, ("No finished MP3 yet for this book. Render it first, "
                          "or click a finished job's row."), gr.skip()
    base = (base_url or "").strip().rstrip("/")
    if not base:
        port = SERVER_PORT
        if request is not None:
            host = request.headers.get("host") or ""
            if ":" in host:
                port = host.rsplit(":", 1)[1]
        ips = _local_ips()
        base = f"http://{ips[0] if ips else '127.0.0.1'}:{port}"
    url = f"{base}/share/{ws.name}"
    qr = _make_qr(url, ws.name)
    store.save_sharing_settings(base_url=base)
    size_mb = file.stat().st_size / 1048576
    lines = [f"Scan the QR, or open the link on your phone. "
             f"File: `{store.share_name(ws, file)}` ({size_mb:.1f} MB)."]
    if job is not None and job.state in ("queued", "running", "stalled"):
        lines.append("⚠ This job is still running; the link serves the current file.")
    port = urlparse(base).port or SERVER_PORT
    alternates = [f"http://{ip}:{port}" for ip in _local_ips()
                  if f"http://{ip}:{port}" != base]
    if alternates:
        lines.append("Other addresses to try: " + ", ".join(f"`{b}`" for b in alternates))
    if qr is None:
        lines.append("_Could not build a QR (`qrencode` missing); use the URL above._")
    if SERVER_HOST in ("127.0.0.1", "localhost", "::1"):
        lines.append("⚠ The UI is bound to loopback, so your phone cannot reach it. "
                     "Restart with `./run_ui.sh --host 0.0.0.0` (or bind to your "
                     "Tailscale address).")
    return url, qr, "\n\n".join(lines), base


def on_drive_upload(selector, remote, path):
    job, ws = _share_target(selector)
    file = store.output_for_sharing(ws, job.spec.out_path if job else None) if ws else None
    if not file:
        yield "Nothing to upload yet — render a book, or click a finished job's row."
        return
    remote = (remote or "").strip()
    folder = (path or "").strip().strip("/")
    if not remote:
        yield ("Set an **rclone remote** first (for example `gdrive`). Create it once "
               "with `rclone config` → Google Drive, then check it with "
               "`rclone lsd gdrive:`.")
        return
    name = store.share_name(ws, file)
    target = f"{remote}:{folder}/{name}" if folder else f"{remote}:{name}"
    store.save_sharing_settings(drive_remote=remote, drive_path=folder)
    size_mb = file.stat().st_size / 1048576
    note = ""
    if job is not None and job.state in ("queued", "running", "stalled"):
        note = " (job still running — uploading the current file)"
    yield (f"⏳ Uploading `{name}` ({size_mb:.1f} MB) to `{target}`{note}… "
           f"Keep this tab open until it finishes.")
    cmd = ["rclone", "copyto", str(file), target, "--stats", "0", "--stats-one-line"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        yield "❌ `rclone` is not installed. Install it, then run `rclone config`."
        return
    except subprocess.TimeoutExpired:
        yield "❌ Upload timed out after 60 minutes."
        return
    if proc.returncode == 0:
        yield f"✅ Uploaded to `{target}` — open Google Drive on any device to fetch it."
    else:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        yield f"❌ rclone failed (exit {proc.returncode}): `{tail[-1] if tail else 'unknown'}`"


def on_job_select(evt: gr.SelectData):
    global SELECTED_JOB
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    jid = ""
    row_value = getattr(evt, "row_value", None)
    if row_value is not None:
        try:
            jid = str(row_value[0])
        except (TypeError, IndexError, KeyError):
            jid = ""
    if not jid:
        try:
            jid = JOB_VIEW[int(row)]
        except (TypeError, ValueError, IndexError):
            jid = ""
    job = MANAGER.get(jid) if jid else None
    if job is None:
        return gr.skip(), "Select a job row.", gr.skip(), gr.skip(), gr.skip(), gr.skip()
    SELECTED_JOB = job.id
    status, pct, log, audio, download = _job_view(job, force_audio=True)
    return job.id, status, pct, log, audio, download


def on_job_pick(selector: str):
    global SELECTED_JOB
    job = _resolve_job(selector)
    if job is None:
        return (gr.skip(), "No job matches that id or number.",
                gr.skip(), gr.skip(), gr.skip(), gr.skip())
    SELECTED_JOB = job.id
    status, pct, log, audio, download = _job_view(job, force_audio=True)
    return job.id, status, pct, log, audio, download


def on_tick():
    global SELECTED_JOB
    MANAGER.pump()
    cast_changed, cast_msg = _poll_analyze_job(S)
    rows = _jobs_rows()
    book_md = _book_run_md()
    if cast_changed:
        cast_rows, cast_speakers, cast_qs = _reload_cast(S["cast"])
        cast_note = _cast_note()
    else:
        cast_rows = cast_speakers = cast_qs = gr.skip()
        # Keep "Analysis queued/running…" current, and replace it when the job
        # failed or was cancelled (no cast change in either case).
        cast_note = _cast_note() if (cast_msg or S.get("analyze_job")) else gr.skip()
    book_status = cast_msg if cast_msg else gr.skip()
    cast_outputs = (book_status, cast_rows, cast_note, cast_qs, cast_speakers)
    job = _selected_job()
    if job is None:
        msg = "Idle — queue empty."
        if S.get("ws"):
            pid = store.external_render_pid(S["ws"] / "work")
            if pid:
                msg += (f"\n\n⚠ An external render (PID {pid}) is writing this book's "
                        f"workspace. It was started outside this queue; wait for it to "
                        f"finish before rendering this book here.")
        return (rows, msg, 0.0, "", gr.skip(), gr.skip(), book_md, *cast_outputs)
    SELECTED_JOB = job.id
    status, pct, log, audio, download = _job_view(job)
    return (rows, status, pct, log, audio, download, book_md, *cast_outputs)


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

# Reopen the books from the previous UI session before building the page.
_restore_session()


def build_param_widget(p: pspec.Param):
    if p.kind == "bool":
        return gr.Checkbox(label=p.label, value=bool(p.default), info=p.help)
    if p.kind == "choice":
        choices = p.choices
        value = p.default
        if p.key == "narrator_gender":
            choices = ["auto", "male", "female"]
            value = "auto"
        return gr.Dropdown(label=p.label, choices=choices, value=value, info=p.help)
    if p.kind == "int":
        extra = {}
        if p.minimum is not None:
            extra["minimum"] = p.minimum
        if p.maximum is not None:
            extra["maximum"] = p.maximum
        return gr.Number(label=p.label, value=p.default, precision=0, info=p.help, **extra)
    if p.kind == "float":
        return gr.Number(label=p.label, value=p.default, precision=2, info=p.help,
                         minimum=p.minimum, maximum=p.maximum, step=p.step)
    if p.key in ("voice", "dialogue_voice"):
        return gr.Dropdown(label=p.label, choices=store.all_voices(), value="",
                           allow_custom_value=True, info=p.help)
    if p.key == "voice_profile":
        return gr.Dropdown(label=p.label, choices=sorted(store.voice_profiles()),
                           value=None, allow_custom_value=True, info=p.help)
    return gr.Textbox(label=p.label, value="" if p.default is None else str(p.default),
                      info=p.help)


with gr.Blocks(title="BookCast Studio", theme=THEME, head=HEAD) as demo:
    gr.HTML(HEADER_HTML, elem_classes=["bc-plain"])

    with gr.Row(elem_classes=["bc-shell"]):
        # ------------------------------------------------------------ workspace
        with gr.Column(scale=3, min_width=720, elem_classes=["bc-workspace"]):
            with gr.Row(elem_classes=["bc-library"]):
                # A Dropdown with allow_custom_value keeps a newly opened book
                # selectable even though Gradio validates Radio inputs against
                # the choices the server component was built with.
                book_radio = gr.Dropdown(label="Open books",
                                         choices=_radio_choices(),
                                         value=ACTIVE_SLUG,
                                         interactive=True, elem_classes=["book-rail"],
                                         allow_custom_value=True, scale=4)
                close_book_btn = gr.Button("Close book", scale=1)

            with gr.Accordion("Open a book", open=True,
                              elem_classes=["bc-card", "bc-open-card"]) as open_book_accordion:
                open_upload = gr.File(label="Upload EPUB / PDF", type="filepath",
                                      file_types=[".epub", ".pdf"], height=190)
                open_path = gr.Textbox(
                    label="…or a path on disk",
                    value=_last_book_path(),
                    placeholder="/home/you/books/Book.epub",
                    info="The last rendered book is pre-filled.",
                )
                open_path_btn = gr.Button("Open path", variant="primary")

            book_info = gr.Markdown(
                "No book open yet. Open an EPUB or PDF above to build a cast, "
                "review the chapters, and render.",
                elem_classes=["bc-book-card"],
            )

            with gr.Accordion("Book & chapters", open=True,
                              elem_classes=["bc-card"]):
                chapter_table = gr.Dataframe(
                    headers=["#", "Chapter", "Words", "Dialogue lines"],
                    datatype=["number", "str", "number", "number"],
                    interactive=False, wrap=True, max_height=280,
                    elem_classes=["bc-table"])
                with gr.Row():
                    chapter_range = gr.Textbox(
                        label="Chapters to render", value="all", scale=3,
                        placeholder="all, or 1-3, 6-8",
                        info="Printer-style ranges. Blank or 'all' = every chapter.",
                    )
                    range_all_btn = gr.Button("All chapters", scale=1)
                chapter_summary = gr.Markdown("Open a book first.")
                with gr.Row():
                    analyze_btn = gr.Button("Analyze cast (no TTS)", variant="secondary",
                                            scale=2)
                with gr.Accordion("High-accuracy analysis (optional)", open=False,
                                  elem_classes=["bc-subcard"]):
                    gr.Markdown(
                        "_High accuracy: start Ollama in the Colab notebook with the "
                        "cloudflared tunnel, paste its `/api/generate` URL into **LLM "
                        "endpoint**, choose `heavy`, then press **Analyze cast**. The "
                        "analysis runs in **Activity** (no 15-minute timeout) and the cast "
                        "updates here when it finishes. You can also import a Colab cast "
                        "JSON in **Cast & quotes**._")
                book_status = gr.Markdown("")

            with gr.Accordion("Parameters", open=False,
                              elem_classes=["bc-card"]):
                gr.Markdown("_Runtime settings — TTS concurrency, failure policy, and "
                            "parallel slots — are in the Activity panel. Apply changes "
                            "here before rendering._")
                for group in pspec.GROUPS:
                    visible = [p for p in pspec.PARAMS
                               if p.group == group and p.key not in HIDDEN_KEYS]
                    if not visible:
                        continue
                    with gr.Accordion(group, open=(group in ("Voice", "Attribution")),
                                      elem_classes=["bc-subcard"]):
                        for p in visible:
                            PARAM_COMPONENTS[p.key] = build_param_widget(p)
                            PARAM_ORDER.append(p.key)

                extra_args = gr.Textbox(
                    label="Extra CLI args (appended verbatim)",
                    placeholder="--unknown-below 0.35",
                )
                with gr.Row():
                    apply_params_btn = gr.Button("Apply changes", variant="primary")
                    revert_params_btn = gr.Button("Revert")
                params_status = gr.Markdown("No book open.")
                params_review = gr.Markdown("")
                confirm_params_btn = gr.Button("Confirm changes", variant="primary",
                                               visible=False)

                with gr.Accordion("Presets", open=False,
                                  elem_classes=["bc-subcard"]):
                    with gr.Row():
                        preset_dd = gr.Dropdown(label="Preset", choices=store.preset_names(),
                                                value=None, allow_custom_value=True)
                        load_preset_btn = gr.Button("Load")
                        preset_name = gr.Textbox(label="Save as",
                                                 placeholder="verity-chapter-preview")
                        save_preset_btn = gr.Button("Save preset")
                    preset_status = gr.Markdown("")

                with gr.Accordion("Command preview", open=False,
                                  elem_classes=["bc-subcard"]):
                    preview_btn = gr.Button("Show exact command")
                    cmd_preview = gr.Code(label="Command", interactive=False)

            with gr.Accordion("Cast & quotes", open=False,
                              elem_classes=["bc-card"]):
                cast_note = gr.Markdown("Analyze a book first.")
                with gr.Row():
                    cast_import = gr.File(label="Import cast JSON (Colab)", file_types=[".json"],
                                          type="filepath", scale=3)
                    cast_import_preview_btn = gr.Button("Validate import", scale=1)
                cast_import_preview = gr.Markdown("")
                cast_import_apply_btn = gr.Button("Apply imported cast", variant="primary",
                                                  visible=False)
                with gr.Row():
                    with gr.Column(scale=3):
                        cast_table = gr.Dataframe(
                            headers=["id", "name", "lines", "gender", "voice", "pitch"],
                            datatype=["str", "str", "number", "str", "str", "str"],
                            interactive=False, wrap=True, max_height=300,
                            elem_classes=["bc-table"])
                        row_number = gr.Number(label="Row # (1-based)", precision=0, value=1)
                    with gr.Column(scale=2):
                        cid_box = gr.Textbox(label="Character id", interactive=False)
                        name_box = gr.Textbox(label="Name")
                        gender_dd = gr.Dropdown(
                            label="Gender", choices=["", "male", "female", "neutral"],
                            allow_custom_value=True)
                        voice_dd = gr.Dropdown(label="Voice", choices=store.all_voices(),
                                               allow_custom_value=True)
                        pitch_box = gr.Textbox(label="Pitch", value="+0Hz")
                        with gr.Row():
                            apply_btn = gr.Button("Apply", variant="primary")
                            audition_btn = gr.Button("Audition")
                        with gr.Row():
                            merge_dst = gr.Dropdown(label="Merge into", choices=[],
                                                    allow_custom_value=True)
                            merge_btn = gr.Button("Merge")
                            delete_btn = gr.Button("Delete")
                        audition_audio = gr.Audio(label="Preview", type="filepath",
                                                 interactive=False)
                        cast_status = gr.Markdown("")
                        with gr.Row():
                            save_cast_btn = gr.Button("Save cast")
                            revert_btn = gr.Button("Revert to backup")

                with gr.Accordion("Quotes", open=True, elem_classes=["bc-subcard"]):
                    with gr.Row():
                        q_chapter = gr.Dropdown(label="Chapter", choices=[],
                                                allow_custom_value=True)
                        q_only = gr.Checkbox(
                            label="Only uncertain (low confidence / Unknown)", value=True)
                    q_table = gr.Dataframe(
                        headers=["seg", "speaker", "id", "method", "confidence", "text"],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        interactive=False, wrap=True, max_height=340,
                        elem_classes=["bc-table"])
                    with gr.Row():
                        q_seg = gr.Number(label="Segment #", precision=0, value=0)
                        q_speaker = gr.Dropdown(label="Assign to", choices=[],
                                                allow_custom_value=True)
                        q_assign = gr.Button("Assign", variant="primary")
                    q_status = gr.Markdown("")

            with gr.Accordion("Render this book", open=True,
                              elem_classes=["bc-card", "bc-render-card"]):
                gr.HTML(RENDER_NOTE_HTML, elem_classes=["bc-plain"])
                book_run_md = gr.Markdown("No book open.")
                with gr.Row():
                    render_btn = gr.Button("Render / continue current book",
                                           variant="primary", scale=3,
                                           elem_classes=["bc-render-btn"])
                    start_over = gr.Checkbox(
                        label="Start over", value=False, scale=1,
                        info="Ignore saved chunks for this run only.")
                render_warning = gr.Markdown("")
                confirm_restart = gr.Checkbox(
                    label="I understand this will discard saved chunks and re-render",
                    value=False, visible=False)

        # ------------------------------------------------------------ activity
        with gr.Column(scale=2, min_width=420, elem_classes=["bc-activity"]):
            gr.HTML(ACTIVITY_HTML, elem_classes=["bc-plain"])
            with gr.Accordion("Runtime", open=True, elem_classes=["bc-card"]):
                slots_dd = gr.Dropdown(label="Parallel renders", choices=[1, 2, 3],
                                       value=store.queue_slots(),
                                       info="How many books may render at once.")
                with gr.Row():
                    runtime_conc = gr.Number(
                        label="TTS concurrency", value=RUNTIME["concurrency"],
                        precision=0, minimum=1, maximum=16,
                        info="Requests per job; applies to the next job or Continue.")
                    runtime_fail = gr.Dropdown(
                        label="On chunk failure", choices=["stop", "skip"],
                        value=RUNTIME["on_chunk_failure"])
                runtime_apply_btn = gr.Button("Apply runtime settings")
                runtime_status = gr.Markdown(
                    "_Keep parallel renders × TTS concurrency around 12 or less._")
                gr.Markdown("_Running jobs keep their captured runtime settings. "
                            "Changes apply to new jobs and **Continue**._")

            jobs_table = gr.Dataframe(
                headers=["id", "book", "range", "state", "%", "time", "out"],
                datatype=["str", "str", "str", "str", "number", "str", "str"],
                interactive=False, wrap=True, max_height=240,
                elem_classes=["bc-table"])
            with gr.Row():
                job_pick = gr.Textbox(
                    label="Selected job (id or #)", scale=3,
                    placeholder="Click a job row, or type its id",
                    info="Continue, Cancel, and Delete act on this job.")
                select_btn = gr.Button("Show job", scale=1)
            with gr.Row():
                continue_btn = gr.Button("Continue selected job", scale=1)
                cancel_btn = gr.Button("Cancel selected job", variant="stop", scale=1)
            with gr.Row():
                delete_job_btn = gr.Button("Delete selected job", scale=1)
                cancel_all_btn = gr.Button("Cancel all jobs", scale=1)
                clear_btn = gr.Button("Clear finished jobs", scale=1)

            run_status = gr.Markdown("Idle — queue empty.")
            progress = gr.Slider(0, 100, value=0, label="Progress", interactive=False)
            log_box = gr.Textbox(label="Log", lines=14, max_lines=14, autoscroll=True,
                                 elem_classes=["bc-log"])
            with gr.Row():
                audio_out = gr.Audio(label="Output", type="filepath", interactive=False)
                download_out = gr.File(label="Download")

            with gr.Accordion("Share output", open=False, elem_classes=["bc-card"]):
                gr.Markdown("### Google Drive\n_Uses your local `rclone` config._")
                with gr.Row():
                    drive_remote = gr.Textbox(
                        label="rclone remote", value=SHARING["drive_remote"],
                        info="Create once with `rclone config` (Google Drive).")
                    drive_path = gr.Textbox(
                        label="Folder on Drive", value=SHARING["drive_path"])
                drive_btn = gr.Button("Upload selected output to Drive")
                drive_status = gr.Markdown(
                    "_Check a remote with `rclone lsd <remote>:`._")

                gr.Markdown("### Phone\n_Scan to download directly._")
                with gr.Row():
                    share_base = gr.Textbox(
                        label="Base URL for your phone", value=SHARING["base_url"],
                        placeholder="http://192.168.1.10:7860", scale=4)
                    detect_btn = gr.Button("Detect", scale=1)
                phone_btn = gr.Button("Create phone link + QR")
                share_url = gr.Textbox(label="Phone URL", interactive=False)
                qr_image = gr.Image(label="QR", type="filepath", height=220,
                                    interactive=False, show_download_button=False)
                phone_status = gr.Markdown(
                    "_The phone must share your Wi-Fi or be on your Tailscale network._")

    gr.HTML(FOOTER_HTML, elem_classes=["bc-plain"])

    # ------------------------------------------------------------- wiring
    PARAM_INPUTS = [PARAM_COMPONENTS[k] for k in PARAM_ORDER]
    WORKSPACE_OUTPUTS = [
        book_info, chapter_table, chapter_range, chapter_summary, book_status,
        cast_table, cast_note, q_chapter, q_speaker, q_table, q_status,
        extra_args, params_status, params_review, confirm_params_btn,
        book_run_md, render_warning, confirm_restart, start_over,
        cast_import_preview, cast_import_apply_btn,
        *PARAM_INPUTS, open_book_accordion,
    ]
    ACTIVITY_OUTPUTS = [job_pick, run_status, progress, log_box, audio_out, download_out]
    BOOK_OUTPUTS = [book_radio] + WORKSPACE_OUTPUTS + ACTIVITY_OUTPUTS

    open_upload.change(on_open_upload, inputs=[open_upload], outputs=BOOK_OUTPUTS)
    open_path_btn.click(on_open_path, inputs=[open_path], outputs=BOOK_OUTPUTS)
    book_radio.input(on_switch_book, inputs=[book_radio], outputs=BOOK_OUTPUTS)
    close_book_btn.click(on_close_book, outputs=BOOK_OUTPUTS)

    range_all_btn.click(on_select_all_range, outputs=[chapter_range, chapter_summary])
    chapter_range.change(on_range_change, inputs=[chapter_range],
                         outputs=[chapter_summary])
    analyze_btn.click(on_analyze,
                      outputs=[cast_table, cast_note, q_chapter, q_speaker,
                               book_status, book_run_md])
    cast_import_preview_btn.click(on_cast_import_preview, inputs=[cast_import],
                                  outputs=[cast_import_preview, cast_import_apply_btn])
    cast_import_apply_btn.click(
        on_cast_import_apply,
        outputs=[cast_table, cast_note, q_chapter, q_speaker, cast_status,
                 cast_import_apply_btn],
    )

    apply_params_btn.click(on_apply_params, inputs=PARAM_INPUTS + [extra_args],
                           outputs=[params_status, params_review, confirm_params_btn])
    confirm_params_btn.click(on_confirm_params,
                             outputs=[params_status, params_review, confirm_params_btn,
                                      book_run_md, render_warning, confirm_restart])
    revert_params_btn.click(
        on_revert_params,
        outputs=[*PARAM_INPUTS, extra_args, params_status, params_review,
                 confirm_params_btn, book_run_md, render_warning, confirm_restart],
    )
    load_preset_btn.click(on_load_preset, inputs=[preset_dd], outputs=PARAM_INPUTS)
    save_preset_btn.click(on_save_preset, inputs=[preset_name] + PARAM_INPUTS,
                          outputs=[preset_status, preset_dd])
    preview_btn.click(on_preview_command, outputs=[cmd_preview])

    cast_table.select(on_cast_select, outputs=[row_number, cid_box, name_box, gender_dd,
                                               voice_dd, pitch_box, audition_audio])
    row_number.change(on_row_number, inputs=[row_number],
                      outputs=[cid_box, name_box, gender_dd, voice_dd,
                               pitch_box, audition_audio])
    gender_dd.input(on_gender_change,
                    inputs=[cid_box, gender_dd, PARAM_COMPONENTS["voice_profile"]],
                    outputs=[voice_dd])
    apply_btn.click(on_apply, inputs=[cid_box, name_box, gender_dd, voice_dd, pitch_box],
                    outputs=[cast_table, cast_status])
    audition_btn.click(on_audition,
                       inputs=[cid_box, voice_dd, pitch_box, PARAM_COMPONENTS["rate"],
                               PARAM_COMPONENTS["engine"]],
                       outputs=[audition_audio, cast_status])
    merge_btn.click(on_merge, inputs=[cid_box, merge_dst],
                    outputs=[cast_table, cast_status])
    delete_btn.click(on_delete, inputs=[cid_box], outputs=[cast_table, cast_status])
    save_cast_btn.click(on_save_cast, outputs=[cast_table, cast_status])
    revert_btn.click(on_revert, outputs=[cast_table, cast_status])

    q_chapter.change(on_q_chapter, inputs=[q_chapter, q_only], outputs=[q_table])
    q_only.change(on_q_filter, inputs=[q_chapter, q_only], outputs=[q_table])
    q_table.select(on_q_select, inputs=[q_chapter, q_only], outputs=[q_seg, q_speaker])
    q_assign.click(on_assign, inputs=[q_seg, q_speaker, q_chapter, q_only],
                   outputs=[q_table, q_status])

    render_btn.click(
        on_render,
        inputs=[chapter_range, start_over, confirm_restart],
        outputs=[run_status, jobs_table, progress, log_box, book_run_md,
                 chapter_summary, params_status, render_warning,
                 confirm_restart, start_over],
    )
    continue_btn.click(on_continue_job, inputs=[job_pick],
                       outputs=[run_status, jobs_table, progress, log_box, book_run_md])
    cancel_btn.click(on_cancel_job, inputs=[job_pick],
                     outputs=[run_status, jobs_table, book_run_md])
    delete_job_btn.click(on_delete_job, inputs=[job_pick],
                         outputs=[run_status, jobs_table, book_run_md])
    cancel_all_btn.click(on_cancel_all,
                         outputs=[run_status, jobs_table, book_run_md])
    clear_btn.click(on_clear_finished,
                    outputs=[run_status, jobs_table, book_run_md])
    slots_dd.change(on_slots, inputs=[slots_dd],
                    outputs=[run_status, jobs_table, book_run_md])
    runtime_apply_btn.click(on_runtime_change,
                            inputs=[runtime_conc, runtime_fail],
                            outputs=[runtime_status])

    detect_btn.click(on_detect_base, outputs=[share_base])
    phone_btn.click(on_phone_link, inputs=[job_pick, share_base],
                    outputs=[share_url, qr_image, phone_status, share_base])
    drive_btn.click(on_drive_upload, inputs=[job_pick, drive_remote, drive_path],
                    outputs=[drive_status])

    jobs_table.select(on_job_select,
                      outputs=[job_pick, run_status, progress, log_box,
                               audio_out, download_out])
    select_btn.click(on_job_pick, inputs=[job_pick],
                     outputs=[job_pick, run_status, progress, log_box,
                              audio_out, download_out])
    job_pick.submit(on_job_pick, inputs=[job_pick],
                    outputs=[job_pick, run_status, progress, log_box,
                             audio_out, download_out])

    timer = gr.Timer(1.0, active=True)
    timer.tick(on_tick, outputs=[jobs_table, run_status, progress, log_box,
                                 audio_out, download_out, book_run_md, book_status,
                                 cast_table, cast_note, q_chapter, q_speaker])

    def _refresh_merge(rows):
        if not S["cast"]:
            return gr.update()
        return gr.update(choices=castlib.character_names(S["cast"], include_buckets=False))
    cast_table.change(_refresh_merge, inputs=[cast_table], outputs=[merge_dst])

    # Populate the page from the restored session (and re-assert the active
    # book after a browser refresh).
    demo.load(_book_view, outputs=BOOK_OUTPUTS)


_SHARE_SECURITY = HTTPBasic(auto_error=False)


def _share_authorized(credentials: Optional[HTTPBasicCredentials]) -> bool:
    """Apply the optional BOOKCAST_USER/PASSWORD auth to /share too."""
    expected = _launch_auth()
    if not expected:
        return True
    if credentials is None:
        return False
    user, password = expected
    return (
        secrets.compare_digest(credentials.username, user)
        and secrets.compare_digest(credentials.password, password)
    )


def _share_route(slug: str,
                 credentials: Optional[HTTPBasicCredentials] = Depends(_SHARE_SECURITY)):
    """Serve a book's finished MP3 (or per-chapter zip) to a phone/browser."""
    if not _share_authorized(credentials):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": 'Basic realm="BookCast"'},
        )
    runs = store.RUNS.resolve()
    try:
        ws = (store.RUNS / slug).resolve()
    except OSError:
        return JSONResponse({"error": "bad path"}, status_code=400)
    if ws != runs and runs not in ws.parents:
        return JSONResponse({"error": "bad path"}, status_code=400)
    if not ws.is_dir():
        return JSONResponse({"error": "no such book"}, status_code=404)
    f = store.output_for_sharing(ws)
    if not f or not f.exists():
        return JSONResponse({"error": "no output yet"}, status_code=404)
    media = "audio/mpeg" if f.suffix == ".mp3" else "application/zip"
    return FileResponse(str(f), media_type=media, filename=store.share_name(ws, f))


_SHARE_ROUTE_ADDED = False


def _register_share_route(app) -> None:
    """Attach /share/{slug} to the app Gradio builds during launch()."""
    global _SHARE_ROUTE_ADDED
    if _SHARE_ROUTE_ADDED:
        return
    app.add_api_route("/share/{slug}", _share_route, methods=["GET", "HEAD"],
                      name="share_output")
    _SHARE_ROUTE_ADDED = True


def _launch_auth() -> Optional[tuple[str, str]]:
    """Optional app-level basic auth for personal LAN/Termux hosting."""
    user = (os.environ.get("BOOKCAST_USER") or "").strip()
    password = os.environ.get("BOOKCAST_PASSWORD") or ""
    if user and password:
        return (user, password)
    return None


def main() -> None:
    import argparse as _argparse
    global SERVER_PORT, SERVER_HOST
    ap = _argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Public Gradio link (Colab)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    SERVER_PORT = a.port
    SERVER_HOST = a.host
    # Launch without blocking so we can add the phone-download route to the
    # FastAPI app Gradio only creates inside launch().
    auth = _launch_auth()
    demo.launch(server_name=a.host, server_port=a.port, share=a.share,
                auth=auth, inbrowser=not a.no_browser, show_error=True,
                max_file_size="2gb", prevent_thread_lock=True)
    _register_share_route(demo.app)
    demo.block_thread()


if __name__ == "__main__":
    main()
