"""Gradio UI for book_to_audio.

Run:  .venv/bin/python -m webui.app        (or ./run_ui.sh)
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btlib.extract import split_dialogue_narration          # noqa: E402
from btlib.pipeline import extract_book                     # noqa: E402

from webui import cast as castlib                           # noqa: E402
from webui import params as pspec                           # noqa: E402
from webui import preview, store                            # noqa: E402
from webui.job import JobQueue, JobSpec                     # noqa: E402

MANAGER = JobQueue(slots=store.queue_slots())
SELECTED_JOB: Optional[str] = None
JOB_VIEW: List[str] = []
LAST_AUDIO: Optional[tuple] = None
S: Dict[str, Any] = {
    "book": None,
    "path": None,
    "ws": None,
    "cast": None,
    "texts": {},
    "dirty": set(),
}

CAST_SAVED_RE = re.compile(
    r"(?:Cast saved to|Cast analysis saved to|Using cached cast analysis:)\s*(.+?)(?:\s+—.*)?$")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

SUPPORTED_SUFFIXES = {".epub", ".pdf"}


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
    """Return a human-readable problem with ``path``, or '' when it is usable."""
    if not path.is_file():
        return f"Not a file: `{path}`"
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        suffix = path.suffix or "(no extension)"
        return f"Unsupported file type `{suffix}` — use .epub or .pdf."
    return ""



def _chapter_rows(book) -> List[List[Any]]:
    rows = []
    for i, ch in enumerate(book.chapters, start=1):
        quotes = sum(1 for k, _ in split_dialogue_narration(ch.text) if k == "dialogue")
        rows.append([i, ch.title[:90], len(ch.text.split()), quotes])
    return rows


def _chapter_labels(book) -> List[str]:
    return [f"{i}. {ch.title[:70]}" for i, ch in enumerate(book.chapters, start=1)]


def _label_index(label: str) -> Optional[int]:
    try:
        return int(str(label).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _chapter_spec(labels: List[str], total: int) -> str:
    idx = sorted({i for i in (_label_index(l) for l in labels or []) if i})
    if not idx or len(idx) == total:
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


def _values_from(seq) -> Dict[str, Any]:
    """Values posted by the browser, in PARAM_ORDER."""
    vals = {k: v for k, v in zip(PARAM_ORDER, seq)}
    if vals.get("narrator_gender") in (None, "", "auto"):
        vals["narrator_gender"] = ""
    return vals


def _base_argv(values: Dict[str, Any], extra: str = "") -> List[str]:
    return pspec.build_argv(values, extra=extra)


def _run_sync(argv: List[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "book_to_audio.py")] + argv,
        cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
    )


def _analyze_now(values: Dict[str, Any], extra: str = "") -> tuple:
    """Run --analyze-only synchronously and copy the cast into the workspace."""
    if not S["path"] or not S["ws"]:
        return False, "Load a book first."
    busy = _busy_reason()
    if busy:
        return False, f"Cannot analyze: {busy}."
    argv = [str(S["path"]), "--analyze-only"] + _base_argv(values, extra)
    try:
        proc = _run_sync(argv)
    except subprocess.TimeoutExpired:
        return False, "Analysis timed out."
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, "Analysis failed:\n" + out[-2000:]
    cast_file = None
    for line in out.splitlines():
        m = CAST_SAVED_RE.search(line)
        if m:
            cast_file = Path(m.group(1).strip())
    if not cast_file or not cast_file.exists():
        # fall back to the newest cast cache for this book
        cached = sorted(Path(S["path"]).parent.glob(f".{Path(S['path']).stem}.cast*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if cached:
            cast_file = cached[0]
    if not cast_file or not cast_file.exists():
        return False, "Analysis finished but no cast file was found:\n" + out[-1500:]
    data = json.loads(cast_file.read_text(encoding="utf-8"))
    target = store.cast_path(S["ws"])
    if target.exists():
        target.with_suffix(".json.bak").write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    S["cast"] = data
    S["dirty"].clear()
    return True, f"Cast saved to {target}"


def _reload_cast(data: Dict[str, Any]):
    S["cast"] = data
    rows = castlib.characters_rows(data)
    speakers = castlib.character_names(data, include_buckets=True)
    keys = castlib.chapter_keys(data)
    labels = []
    for k in keys:
        try:
            i = int(k)
            title = S["book"].chapters[i - 1].title if S["book"] and i - 1 < len(S["book"].chapters) else f"Chapter {i}"
        except (TypeError, ValueError, IndexError):
            title = f"Chapter {k}"
        labels.append(f"{k}. {title[:60]}")
    return rows, gr.update(choices=speakers, value=speakers[0] if speakers else None), \
        gr.update(choices=labels, value=labels[0] if labels else None)


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
    if job:
        return f"job `{job.spec.short()}` is rendering this book"
    pid = store.external_render_pid(ws / "work")
    if pid:
        return f"another process (PID {pid}) is rendering this book"
    return ""


def _last_book_path() -> str:
    """Path of the most recently rendered book, for the Book tab's path box."""
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


def _current_book_md() -> str:
    """Show exactly which book the Run tab will render.

    ``Add to queue`` was ambiguous because the book is loaded on another tab.
    Keeping this card visible makes the primary action self-explanatory.
    """
    book = S.get("book")
    path = S.get("path")
    if not book or not path:
        return ("### No book loaded\n"
                "Open the **Book** tab and load a book first. "
                "The buttons below render that loaded book.")
    if S.get("cast"):
        n = len((S["cast"].get("characters") or {}))
        cast_state = f"{n} character(s) ready"
    elif S.get("ws") and store.cast_path(S["ws"]).exists():
        cast_state = "cast file ready"
    else:
        cast_state = "no cast yet (Analyze on the Book tab)"
    ws = S.get("ws")
    workspace = f"`{ws.name}`" if ws else "—"
    return (
        f"### {book.title}\n"
        f"`{Path(path).name}` · {len(book.chapters)} chapters · "
        f"workspace {workspace}  \n"
        f"Cast: {cast_state}. **Render / continue** starts this book or resumes "
        f"its saved chunks. To discard those chunks, enable **Ignore saved "
        f"progress** in Parameters → Output."
    )


def _invalidate(chapters) -> str:
    if _busy_reason():
        return ""
    chapters = set(chapters) & S.get("dirty", set())
    if not chapters:
        return ""
    done = store.invalidate_chapters(S["ws"], chapters)
    S["dirty"] -= set(done)
    return f" (re-render: {', '.join(map(str, done))})" if done else ""


# --------------------------------------------------------------------------
# render queue helpers
# --------------------------------------------------------------------------

def _jobs_rows() -> List[List[Any]]:
    global JOB_VIEW
    jobs = MANAGER.jobs
    JOB_VIEW = [j.id for j in jobs]
    rows = []
    for j in jobs:
        snap = j.snapshot()
        out = Path(j.spec.out_path).name + ("/" if j.spec.per_chapter else "")
        rows.append([
            j.spec.short(), j.spec.title[:46], j.spec.chapters or "all",
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
    lines = [f"### {job.spec.title}", f"Job `{job.spec.short()}` · chapters {job.spec.chapters or 'all'}"]
    if state == "queued":
        pos = MANAGER.position(job.id)
        lines.append(f"**Queued** · position {pos} · waiting {snap['queued_for']:.0f}s")
    elif state == "running":
        lines.append(f"**Running** · {snap['elapsed']:.0f}s")
    elif state == "stalled":
        lines.append(f"**Stalled** · no output for 5+ minutes · {snap['elapsed']:.0f}s")
    elif state == "cancelled":
        lines.append(f"**Cancelled** · {snap['elapsed']:.0f}s — finished chunks are kept. "
                     f"Click **Continue selected job** to resume.")
    elif state == "done":
        lines.append(f"**Finished** (exit {snap['returncode']}) · {snap['elapsed']:.0f}s")
    else:
        lines.append(f"**Failed** (exit {snap['returncode']}) · {snap['elapsed']:.0f}s — "
                     f"finished chunks are kept. Click **Continue selected job** to resume.")

    if snap["chapter_total"]:
        lines.append(f"Chapter {snap['chapter_index']}/{snap['chapter_total']}: "
                     f"{snap['chapter_title']}")
        lines.append(f"Chunk {snap['chunk_done']}/{snap['chunk_total']}")
    if state == "stalled":
        lines.append("⚠ **No output for 5+ minutes** — the TTS request may be stuck. "
                     "Cancel this job, then click **Continue selected job**; finished "
                     "chunks are kept.")
    if snap["error"] and not snap.get("cancelled"):
        lines.append(f"❌ `{snap['error']}`")
    outs = store.job_outputs(job.spec) if job.done else []
    if outs:
        lines.append(f"Output: `{outs[0]}`" + (f" (+{len(outs) - 1} more)" if len(outs) > 1 else ""))
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


# --------------------------------------------------------------------------
# book tab
# --------------------------------------------------------------------------

def on_upload(file):
    if not file:
        return (gr.update(),) * 8
    src = Path(file)
    if src.suffix.lower() not in SUPPORTED_SUFFIXES:
        suffix = src.suffix or "(no extension)"
        return (gr.update(),) * 7 + (
            f"**Unsupported file type `{suffix}` — upload an .epub or .pdf.**",)
    try:
        book = _extract(src)          # parse before copying, so bad files never land in books/
        path = store.ingest_upload(file, src.name)
        return _open_book(path, book=book)
    except Exception as exc:
        return (gr.update(),) * 7 + (f"**Upload failed:** `{exc}`",)


def on_path_load(path_text: str):
    raw = (path_text or "").strip()
    if not raw:
        return (gr.update(),) * 7 + (
            "**Enter a path to an .epub or .pdf file (or use Upload above).**",)
    path = Path(raw).expanduser()
    if not path.exists():
        return (gr.update(),) * 7 + (f"**Not found:** `{path}`",)
    problem = _book_path_error(path)
    if problem:
        return (gr.update(),) * 7 + (f"**{problem}**",)
    try:
        return _open_book(path)
    except Exception as exc:
        return (gr.update(),) * 7 + (f"**Failed to open:** `{exc}`",)


def _open_book(path: Path, book=None):
    book = book or _extract(path)
    S.update({
        "book": book, "path": path, "ws": store.workspace_for(path),
        "cast": None, "audio": None,
        "texts": {i: ch.text for i, ch in enumerate(book.chapters, start=1)},
        "dirty": set(),
    })
    labels = _chapter_labels(book)
    chapter_rows = _chapter_rows(book)
    info = (f"### {book.title}\n"
            f"{len(book.chapters)} chapters · {sum(len(c.text.split()) for c in book.chapters):,} words  \n"
            f"Source: `{path}`  \nWorkspace: `{S['ws']}`")

    data = castlib.load(S["ws"])
    if data:
        S["cast"] = data
        cast_rows = castlib.characters_rows(data)
        speakers = castlib.character_names(data, include_buckets=True)
        qkeys = castlib.chapter_keys(data)
        qlabels = []
        for k in qkeys:
            try:
                title = book.chapters[int(k) - 1].title
            except (TypeError, ValueError, IndexError):
                title = f"Chapter {k}"
            qlabels.append(f"{k}. {title[:60]}")
        cast_note = f"Loaded saved cast ({len(data.get('characters', []))} characters)."
    else:
        cast_rows, speakers, qlabels = [], [], []
        cast_note = "No cast yet — press **Analyze cast**."

    return (
        info,
        chapter_rows,
        gr.update(choices=labels, value=labels),
        gr.update(choices=qlabels, value=qlabels[0] if qlabels else None),
        cast_rows,
        gr.update(choices=speakers, value=speakers[0] if speakers else None),
        cast_note,
        "Book loaded.",
    )


def on_select_all():
    if not S["book"]:
        return gr.update()
    labels = _chapter_labels(S["book"])
    return gr.update(value=labels)


def on_select_none():
    return gr.update(value=[])


def on_analyze(*args):
    values = _values_from(args[:-1])
    extra = args[-1] if args else ""
    if not S["path"]:
        return gr.update(), gr.update(), gr.update(), "Load a book first."
    busy = _busy_reason()
    if busy:
        return gr.update(), gr.update(), gr.update(), f"**Cannot analyze: {busy}.**"
    ok, msg = _analyze_now(values, extra)
    if not ok or not S["cast"]:
        return gr.update(), gr.update(), gr.update(), f"**{msg}**"
    rows, spk, qs = _reload_cast(S["cast"])
    return rows, spk, qs, f"**{msg}**"


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

def on_preview_command(*args):
    values = _values_from(args[:-2])
    extra = args[-2] if len(args) >= 2 else ""
    chapters = args[-1] if args else []
    if S["path"]:
        argv = [str(S["path"])] + _base_argv(values, extra)
        if S["ws"]:
            argv += ["--cast-file", str(store.cast_path(S["ws"])),
                     "--workdir", str(S["ws"] / "work"),
                     "-o", str(store.out_path(S["ws"], bool(values.get("per_chapter"))))]
        spec = _chapter_spec(chapters, len(S["book"].chapters) if S["book"] else 0)
        if spec:
            argv += ["--chapters", spec]
    else:
        argv = ["<book.epub>"] + _base_argv(values, extra)
    return "python book_to_audio.py " + shlex.join(str(a) for a in argv)


def _preset_value(name: str) -> Dict[str, Any]:
    return store.load_preset(name) if name else {}


def on_load_preset(name: str):
    vals = _preset_value(name)
    out = []
    for key in PARAM_ORDER:
        p = pspec.BY_KEY[key]
        v = vals.get(key, p.default)
        if key == "narrator_gender":
            v = "auto" if v in (None, "", "auto") else v
        elif p.kind == "choice" and v not in p.choices:
            v = p.default
        out.append(v)
    return out


def on_save_preset(name: str, *args):
    if not name.strip():
        return "**Preset name is empty.**", gr.update()
    vals = _values_from(args)
    if not vals.get("narrator_gender"):
        vals["narrator_gender"] = "auto"
    p = store.save_preset(name, vals)
    names = store.preset_names()
    return f"Saved `{p}`.", gr.update(choices=names, value=p.stem)


# --------------------------------------------------------------------------
# cast tab
# --------------------------------------------------------------------------

def on_cast_select(evt: gr.SelectData):
    if not S["cast"]:
        return (gr.update(),) * 7
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        row = int(row)
    except (TypeError, ValueError):
        return (gr.update(),) * 7
    ordered = sorted(castlib.characters(S["cast"]), key=lambda x: -(x.get("line_count") or 0))
    if not (0 <= row < len(ordered)):
        return (gr.update(),) * 7
    c = ordered[row]
    return (row + 1, c["id"], c.get("name") or "", c.get("gender") or "",
            c.get("voice") or "", c.get("pitch") or "+0Hz", gr.update())


def on_row_number(row):
    if not S["cast"] or not row:
        return (gr.update(),) * 6
    ordered = sorted(castlib.characters(S["cast"]), key=lambda x: -(x.get("line_count") or 0))
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
    rows, spk, qs = _reload_cast(data)
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
    rows, spk, qs = _reload_cast(data)
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
    rows, spk, qs = _reload_cast(S["cast"])
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
    rows, spk, qs = _reload_cast(S["cast"])
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
    for k in castlib.chapter_keys(data):
        try:
            chapters.append(int(k))
        except (TypeError, ValueError):
            pass
    note = ""
    if store.invalidate_chapters(S["ws"], chapters):
        note = f" (re-render: {', '.join(map(str, chapters))})"
    rows, spk, qs = _reload_cast(data)
    S["dirty"].clear()
    return rows, "Reverted to the previous cast." + note


# --------------------------------------------------------------------------
# quotes tab
# --------------------------------------------------------------------------

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
    rows, spk, qs = _reload_cast(S["cast"])
    return on_q_chapter(chapter_label, only), f"{msg}{note}"


# --------------------------------------------------------------------------
# run tab
# --------------------------------------------------------------------------

def on_enqueue(*args):
    global SELECTED_JOB
    values = _values_from(args[:-2])
    extra = args[-2] if len(args) >= 2 else ""
    chapters = args[-1] if args else []
    rows = _jobs_rows()
    book_md = _current_book_md()
    if not S["path"] or not S["ws"]:
        return "Load a book first.", rows, 0.0, "", book_md

    workdir = S["ws"] / "work"
    busy = MANAGER.has_workspace(workdir)
    if busy:
        return (f"**Already {busy.state}** as job `{busy.spec.short()}`. "
                f"Cancel it first, or wait for it to finish."), rows, 0.0, "", book_md
    pid = store.external_render_pid(workdir)
    if pid:
        return (f"**PID {pid} is already rendering this book.** "
                f"Wait for it to finish (or stop that process) before queuing another."), rows, 0.0, "", book_md

    if not store.cast_path(S["ws"]).exists():
        ok, msg = _analyze_now(values, extra)
        if not ok:
            return f"**{msg}**", rows, 0.0, msg, book_md
    if not store.cast_path(S["ws"]).exists():
        return "No cast file — run Analyze first.", rows, 0.0, "", book_md

    per_chapter = bool(values.get("per_chapter"))
    out = store.out_path(S["ws"], per_chapter)
    if per_chapter:
        out.mkdir(parents=True, exist_ok=True)
    argv = [str(S["path"])] + _base_argv(values, extra)
    argv += ["--cast-file", str(store.cast_path(S["ws"])),
             "--workdir", str(workdir), "-o", str(out)]
    spec = _chapter_spec(chapters, len(S["book"].chapters) if S["book"] else 0)
    if spec:
        argv += ["--chapters", spec]

    job_spec = JobSpec(
        book=S["path"], title=S["book"].title,
        cast_file=store.cast_path(S["ws"]), workdir=workdir,
        out_path=out, args=argv, chapters=spec, per_chapter=per_chapter,
    )
    store.save_meta(S["ws"], last_argv=argv, chapters=spec,
                    per_chapter=per_chapter, last_job=job_spec.id)
    try:
        job = MANAGER.enqueue(job_spec, cwd=ROOT,
                              log_path=store.job_log_path(S["ws"], job_spec.id))
    except Exception as exc:
        return f"**Could not queue:** `{exc}`", rows, 0.0, "", book_md
    SELECTED_JOB = job.id
    rows = _jobs_rows()
    if job.state == "queued":
        pos = MANAGER.position(job.id)
        head = f"**Queued** `{job.spec.short()}` (position {pos})."
    elif job.state in ("running", "stalled"):
        head = f"**Started** `{job.spec.short()}`."
    else:
        head = f"**Job `{job.spec.short()}` failed:** `{job.error}`"
    return (f"{head} Chapters: {spec or 'all'} · output `{out}`\n"
            f"```\npython book_to_audio.py {shlex.join(argv)}\n```"), rows, 0.0, "", book_md


def on_continue_job(selector: str):
    """Re-queue a cancelled/failed job using its original captured settings.

    Renders are resumable, so the right recovery action after Cancel is to
    continue the same job rather than make the user reconstruct it from the
    Book and Parameters tabs.
    """
    global SELECTED_JOB
    rows = _jobs_rows()
    book_md = _current_book_md()
    job = _resolve_job(selector)
    if job is None:
        return ("Select a **cancelled** or **failed** job in the table first.",
                rows, 0.0, "", book_md)
    if job.state in ("queued", "running", "stalled"):
        return (f"Job `{job.spec.short()}` is already {job.state}; wait for it "
                f"or cancel it first."), rows, 0.0, "", book_md

    spec = job.spec
    if not spec.workdir.exists():
        return (f"No saved work folder for `{spec.short()}`. Use "
                f"**Render / continue** to start this book from scratch."), \
            rows, 0.0, "", book_md
    if not spec.cast_file.exists():
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

    # A continue must resume, never replay a captured --fresh flag.
    args = [a for a in spec.args
            if a != "--fresh" and not a.startswith("--fresh=")]
    new_spec = JobSpec(
        book=spec.book, title=spec.title, cast_file=spec.cast_file,
        workdir=spec.workdir, out_path=spec.out_path, args=args,
        chapters=spec.chapters, per_chapter=spec.per_chapter,
    )
    if spec.per_chapter:
        Path(spec.out_path).mkdir(parents=True, exist_ok=True)
    ws = spec.workdir.parent
    store.save_meta(ws, last_argv=args, chapters=spec.chapters,
                    per_chapter=spec.per_chapter, last_job=new_spec.id)
    try:
        new_job = MANAGER.enqueue(new_spec, cwd=ROOT,
                                  log_path=store.job_log_path(ws, new_spec.id))
    except Exception as exc:
        return f"**Could not continue:** `{exc}`", rows, 0.0, "", book_md
    SELECTED_JOB = new_job.id
    rows = _jobs_rows()
    state = "Queued" if new_job.state == "queued" else "Started"
    return (f"**{state}** `{new_spec.short()}` — continuing `{spec.short()}` "
            f"from saved chunks · output `{spec.out_path}`"), \
        rows, 0.0, "", book_md


def on_cancel_job(selector: str):
    global SELECTED_JOB
    job = _resolve_job(selector)
    if job is None:
        return "No job selected.", _jobs_rows()
    SELECTED_JOB = job.id
    was_running = job.running
    ok = MANAGER.cancel(job.id)
    if not ok:
        return f"Job `{job.spec.short()}` is not queued or running.", _jobs_rows()
    if was_running:
        return (f"Cancelling `{job.spec.short()}`… Saved chunks are kept — "
                f"click **Continue selected job** to resume."), _jobs_rows()
    return f"Removed `{job.spec.short()}` from the queue.", _jobs_rows()


def on_cancel_all():
    n = MANAGER.cancel_all()
    return (f"Cancelling {n} job(s)." if n else "No queued or running jobs."), _jobs_rows()


def on_clear_finished():
    n = MANAGER.clear_finished()
    return (f"Cleared {n} finished job(s)." if n else "No finished jobs to clear."), _jobs_rows()


def on_slots(slots):
    n = MANAGER.set_slots(int(slots or 1))
    store.save_queue_slots(n)
    return (f"Parallel renders: **{n}**. Keep parallel × TTS concurrency ≤ ~12 "
            f"to avoid Edge rate limiting."), _jobs_rows()


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
        return gr.skip(), "No job matches that id or number.", gr.skip(), gr.skip(), gr.skip(), gr.skip()
    SELECTED_JOB = job.id
    status, pct, log, audio, download = _job_view(job, force_audio=True)
    return job.id, status, pct, log, audio, download


def on_tick():
    global SELECTED_JOB
    MANAGER.pump()
    rows = _jobs_rows()
    book_md = _current_book_md()
    job = _selected_job()
    if job is None:
        msg = "Idle — queue empty."
        if S.get("ws"):
            pid = store.external_render_pid(S["ws"] / "work")
            if pid:
                msg += (f"\n\n⚠ An external render (PID {pid}) is writing this book's "
                        f"workspace. It was started outside this queue; wait for it to "
                        f"finish before rendering this book here.")
        return rows, msg, 0.0, "", gr.skip(), gr.skip(), book_md
    SELECTED_JOB = job.id
    status, pct, log, audio, download = _job_view(job)
    return rows, status, pct, log, audio, download, book_md


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

PARAM_COMPONENTS: Dict[str, Any] = {}
PARAM_ORDER: List[str] = []


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
    return gr.Textbox(label=p.label, value="" if p.default is None else str(p.default), info=p.help)


with gr.Blocks(title="Book to Audio") as demo:
    gr.Markdown("# Book to Audio\nEPUB/PDF → multi-voice audiobook. "
                "The UI drives the existing CLI; nothing is re-implemented here.")

    with gr.Tabs():
        # ------------------------------------------------------------- Book
        with gr.Tab("Book"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.File(label="Upload EPUB / PDF", type="filepath",
                                     file_types=[".epub", ".pdf"])
                    path_box = gr.Textbox(label="…or a path on disk",
                                          value=_last_book_path(),
                                          placeholder="/home/you/books/Book.epub",
                                          info="The last rendered book is pre-filled; click Load path to reopen it.")
                    load_path = gr.Button("Load path")
                    analyze_btn = gr.Button("Analyze cast (no TTS)", variant="secondary")
                with gr.Column(scale=2):
                    book_info = gr.Markdown("No book loaded.")
                    chapter_table = gr.Dataframe(
                        headers=["#", "Chapter", "Words", "Dialogue lines"],
                        datatype=["number", "str", "number", "number"],
                        interactive=False, wrap=True, max_height=260)
                    with gr.Row():
                        pick_all = gr.Button("Select all")
                        pick_none = gr.Button("Select none")
                    chapter_pick = gr.Dropdown(
                        label="Chapters to render", choices=[], value=[], multiselect=True,
                        allow_custom_value=True,
                        info="Empty = all chapters. Update after loading a book.")
                    book_status = gr.Markdown("")

        # ------------------------------------------------------- Parameters
        with gr.Tab("Parameters"):
            for group in pspec.GROUPS:
                with gr.Accordion(group, open=(group in ("Voice", "Attribution"))):
                    for p in [x for x in pspec.PARAMS if x.group == group]:
                        PARAM_COMPONENTS[p.key] = build_param_widget(p)
                        PARAM_ORDER.append(p.key)


            with gr.Accordion("Presets", open=False):
                with gr.Row():
                    preset_dd = gr.Dropdown(label="Preset", choices=store.preset_names(), value=None,
                                            allow_custom_value=True)
                    load_preset_btn = gr.Button("Load")
                    preset_name = gr.Textbox(label="Save as", placeholder="verity-chapter-preview")
                    save_preset_btn = gr.Button("Save preset")
                preset_status = gr.Markdown("")

            with gr.Accordion("Advanced", open=False):
                extra_args = gr.Textbox(label="Extra CLI args (appended verbatim)",
                                        placeholder="--unknown-below 0.35")
                preview_btn = gr.Button("Show exact command")
                cmd_preview = gr.Code(label="Command", interactive=False)

        # ------------------------------------------------------------- Cast
        with gr.Tab("Cast"):
            cast_note = gr.Markdown("Analyze a book first.")
            with gr.Row():
                with gr.Column(scale=3):
                    cast_table = gr.Dataframe(
                        headers=["id", "name", "lines", "gender", "voice", "pitch"],
                        datatype=["str", "str", "number", "str", "str", "str"],
                        interactive=False, wrap=True, max_height=300)
                    row_number = gr.Number(label="Row # (1-based)", precision=0, value=1)
                with gr.Column(scale=2):
                    cid_box = gr.Textbox(label="Character id", interactive=False)
                    name_box = gr.Textbox(label="Name")
                    gender_dd = gr.Dropdown(label="Gender", choices=["", "male", "female", "neutral"],
                                            allow_custom_value=True)
                    voice_dd = gr.Dropdown(label="Voice", choices=store.all_voices(),
                                           allow_custom_value=True)
                    pitch_box = gr.Textbox(label="Pitch", value="+0Hz")
                    with gr.Row():
                        apply_btn = gr.Button("Apply", variant="primary")
                        audition_btn = gr.Button("Audition")
                    with gr.Row():
                        merge_dst = gr.Dropdown(label="Merge into", choices=[], allow_custom_value=True)
                        merge_btn = gr.Button("Merge")
                        delete_btn = gr.Button("Delete")
                    audition_audio = gr.Audio(label="Preview", type="filepath", interactive=False)
                    cast_status = gr.Markdown("")
                    with gr.Row():
                        save_cast_btn = gr.Button("Save cast")
                        revert_btn = gr.Button("Revert to backup")

            with gr.Accordion("Quotes", open=True):
                with gr.Row():
                    q_chapter = gr.Dropdown(label="Chapter", choices=[], allow_custom_value=True)
                    q_only = gr.Checkbox(label="Only uncertain (low confidence / Unknown)",
                                         value=True)
                q_table = gr.Dataframe(
                    headers=["seg", "speaker", "id", "method", "confidence", "text"],
                    datatype=["number", "str", "str", "str", "str", "str"],
                    interactive=False, wrap=True, max_height=340)
                with gr.Row():
                    q_seg = gr.Number(label="Segment #", precision=0, value=0)
                    q_speaker = gr.Dropdown(label="Assign to", choices=[], allow_custom_value=True)
                    q_assign = gr.Button("Assign", variant="primary")
                q_status = gr.Markdown("")

        # -------------------------------------------------------------- Run
        with gr.Tab("Run"):
            run_book = gr.Markdown(_current_book_md())
            with gr.Row():
                render_btn = gr.Button("Render / continue current book",
                                       variant="primary", scale=3)
                continue_btn = gr.Button("Continue selected job", scale=1)
                cancel_btn = gr.Button("Cancel selected job", variant="stop", scale=1)
            gr.Markdown("_**Render / continue** acts on the book shown above. It starts "
                        "now, or waits if all parallel slots are busy. Saved chunks are "
                        "reused automatically. **Continue selected job** re-queues a "
                        "cancelled or failed job from the table below._")
            with gr.Row():
                slots_dd = gr.Dropdown(label="Parallel renders", choices=[1, 2, 3],
                                       value=store.queue_slots(), scale=1,
                                       info="Each job uses Parameters → TTS concurrency.")
                job_pick = gr.Textbox(label="Selected job (id or #)", scale=2,
                                      placeholder="Click a job row below, or type its id",
                                      info="Continue and Cancel act on this job.")
                select_btn = gr.Button("Show job", scale=1)
                cancel_all_btn = gr.Button("Cancel all jobs", scale=1)
                clear_btn = gr.Button("Clear finished jobs", scale=1)
            jobs_table = gr.Dataframe(
                headers=["id", "book", "chapters", "state", "%", "elapsed", "output"],
                datatype=["str", "str", "str", "str", "number", "str", "str"],
                interactive=False, wrap=True, max_height=220)
            gr.Markdown("_Keep **parallel renders × TTS concurrency** at about 12 or less — "
                        "Edge TTS rate-limits aggressive request counts. The queue holds a job "
                        "back if starting it would exceed 12 in-flight requests._")
            run_status = gr.Markdown("Idle — queue empty.")
            progress = gr.Slider(0, 100, value=0, label="Progress", interactive=False)
            log_box = gr.Textbox(label="Log", lines=14, max_lines=14, autoscroll=True)
            with gr.Row():
                audio_out = gr.Audio(label="Output", type="filepath", interactive=False)
                download_out = gr.File(label="Download")

    # ------------------------------------------------------------- wiring
    PARAM_INPUTS = [PARAM_COMPONENTS[k] for k in PARAM_ORDER]
    OUTPUTS_BOOK = [book_info, chapter_table, chapter_pick, q_chapter,
                    cast_table, q_speaker, cast_note, book_status]
    upload.change(on_upload, inputs=[upload], outputs=OUTPUTS_BOOK)
    load_path.click(on_path_load, inputs=[path_box], outputs=OUTPUTS_BOOK)
    pick_all.click(on_select_all, outputs=[chapter_pick])
    pick_none.click(on_select_none, outputs=[chapter_pick])
    analyze_btn.click(on_analyze, inputs=PARAM_INPUTS + [extra_args],
                      outputs=[cast_table, q_speaker, q_chapter, book_status])

    preview_btn.click(on_preview_command, inputs=PARAM_INPUTS + [extra_args, chapter_pick],
                      outputs=[cmd_preview])
    load_preset_btn.click(on_load_preset, inputs=[preset_dd],
                          outputs=PARAM_INPUTS)
    save_preset_btn.click(on_save_preset, inputs=[preset_name] + PARAM_INPUTS,
                          outputs=[preset_status, preset_dd])

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
    merge_btn.click(on_merge, inputs=[cid_box, merge_dst], outputs=[cast_table, cast_status])
    delete_btn.click(on_delete, inputs=[cid_box], outputs=[cast_table, cast_status])
    save_cast_btn.click(on_save_cast, outputs=[cast_table, cast_status])
    revert_btn.click(on_revert, outputs=[cast_table, cast_status])

    q_chapter.change(on_q_chapter, inputs=[q_chapter, q_only], outputs=[q_table])
    q_only.change(on_q_filter, inputs=[q_chapter, q_only], outputs=[q_table])
    q_table.select(on_q_select, inputs=[q_chapter, q_only], outputs=[q_seg, q_speaker])
    q_assign.click(on_assign, inputs=[q_seg, q_speaker, q_chapter, q_only],
                   outputs=[q_table, q_status])

    render_btn.click(on_enqueue, inputs=PARAM_INPUTS + [extra_args, chapter_pick],
                     outputs=[run_status, jobs_table, progress, log_box, run_book])
    continue_btn.click(on_continue_job, inputs=[job_pick],
                       outputs=[run_status, jobs_table, progress, log_box, run_book])
    cancel_btn.click(on_cancel_job, inputs=[job_pick],
                     outputs=[run_status, jobs_table])
    cancel_all_btn.click(on_cancel_all, outputs=[run_status, jobs_table])
    clear_btn.click(on_clear_finished, outputs=[run_status, jobs_table])
    slots_dd.change(on_slots, inputs=[slots_dd], outputs=[run_status, jobs_table])
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
                                 audio_out, download_out, run_book])

    # keep the "merge into" dropdown in sync with the cast table
    def _refresh_merge(rows):
        if not S["cast"]:
            return gr.update()
        return gr.update(choices=castlib.character_names(S["cast"], include_buckets=False))
    cast_table.change(_refresh_merge, inputs=[cast_table], outputs=[merge_dst])


def main() -> None:
    import argparse as _argparse
    ap = _argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Public Gradio link (Colab)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    demo.launch(server_name=a.host, server_port=a.port, share=a.share,
                inbrowser=not a.no_browser, show_error=True, max_file_size="2gb")


if __name__ == "__main__":
    main()
