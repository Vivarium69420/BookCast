"""Per-book workspace and upload handling for the web UI.

Layout::

    runs/<slug>/
        cast.json     editable cast (always passed to the CLI as --cast-file)
        work/         explicit --workdir (chunks + progress.json)
        out.mp3       combined output (or out/ for --per-chapter)
        log.txt       last run log
        meta.json     last parameters / timestamps
        preview/      audition clips
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"
RUNS = ROOT / "runs"
PRESETS = ROOT / "webui" / "presets"


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return (s or "book")[:60]


def sha256(path: Path, limit: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_upload(tmp_path: str, original_name: Optional[str] = None) -> Path:
    """Copy an uploaded file into books/ (never overwrite; reuse identical)."""
    BOOKS.mkdir(parents=True, exist_ok=True)
    src = Path(tmp_path)
    name = original_name or src.name
    target = BOOKS / name
    if target.exists():
        if sha256(target) == sha256(src):
            return target
        stem, suffix = target.stem, target.suffix
        n = 2
        while (BOOKS / f"{stem} ({n}){suffix}").exists():
            n += 1
        target = BOOKS / f"{stem} ({n}){suffix}"
    shutil.copy2(src, target)
    return target


def workspace_for(book: Path) -> Path:
    ws = RUNS / slugify(book.stem)
    ensure(ws)
    return ws


def ensure(ws: Path) -> Path:
    (ws / "work").mkdir(parents=True, exist_ok=True)
    (ws / "preview").mkdir(parents=True, exist_ok=True)
    return ws


def cast_path(ws: Path) -> Path:
    return ws / "cast.json"


def out_path(ws: Path, per_chapter: bool) -> Path:
    return ws / "out" if per_chapter else ws / "out.mp3"


def load_meta(ws: Path) -> Dict[str, Any]:
    p = ws / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_meta(ws: Path, **kw) -> None:
    meta = load_meta(ws)
    meta.update(kw)
    meta["updated"] = time.time()
    (ws / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def list_outputs(ws: Path) -> List[Path]:
    files: List[Path] = []
    single = ws / "out.mp3"
    if single.exists() and single.stat().st_size > 0:
        files.append(single)
    outdir = ws / "out"
    if outdir.is_dir():
        files.extend(sorted(p for p in outdir.glob("*.mp3") if p.stat().st_size > 0))
    # Older/manual runs may have written a title-named file (e.g. "Book.mp3")
    # instead of out.mp3; keep those visible too.
    named = sorted((p for p in ws.glob("*.mp3")
                    if p.name != "out.mp3" and p.stat().st_size > 0),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    files.extend(p for p in named if p not in files)
    return files


def invalidate_chapters(ws: Path, chapters: Iterable[int]) -> List[int]:
    """Forget cached audio for the given 1-based chapter indices.

    The CLI's progress signature does not hash the cast *content*, only the
    cast file path, so edited voices would otherwise resume with stale audio.
    """
    idxs = sorted({int(c) for c in chapters})
    prog = ws / "work" / "progress.json"
    if not idxs or not prog.exists():
        return []
    try:
        data = json.loads(prog.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    done = []
    for ch in data.get("chapters", []):
        i = ch.get("index")
        if i not in idxs:
            continue
        ch["num_chunks"] = None
        ch["chunk_done"] = []
        ch["chapter_combined_done"] = False
        ch["chapter_output_done"] = False
        done.append(i)
        shutil.rmtree(ws / "work" / "chunks" / f"ch_{i:04d}", ignore_errors=True)
        combined = ws / "work" / "chapters" / f"ch_{i:04d}_combined.mp3"
        combined.unlink(missing_ok=True)
        outdir = ws / "out"
        if outdir.is_dir():
            for p in outdir.glob(f"*{i:02d}*.mp3"):
                p.unlink(missing_ok=True)
    if done:
        data["final_done"] = False
        tmp = prog.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(prog)
        # the combined file is now stale
        (ws / "out.mp3").unlink(missing_ok=True)
    return done


# --------------------------------------------------------------------------
# Render queue: settings, per-job logs, workspace locks
# --------------------------------------------------------------------------

QUEUE_SETTINGS = RUNS / "queue.json"


def queue_slots(default: int = 2) -> int:
    if QUEUE_SETTINGS.exists():
        try:
            n = int(json.loads(QUEUE_SETTINGS.read_text(encoding="utf-8")).get("slots", default))
            return max(1, min(3, n))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return default


def save_queue_slots(slots: int) -> int:
    n = max(1, min(3, int(slots)))
    RUNS.mkdir(parents=True, exist_ok=True)
    QUEUE_SETTINGS.write_text(json.dumps({"slots": n}, indent=2), encoding="utf-8")
    return n


def job_log_path(ws: Path, job_id: str) -> Path:
    return ws / "logs" / f"{job_id}.log"


def job_lock_path(workdir: Path) -> Path:
    return Path(workdir) / ".ui-job.json"


def read_job_lock(workdir: Path) -> Dict[str, Any]:
    p = job_lock_path(workdir)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
        return state != "Z"
    except OSError:
        return True


def _cmdline(pid: int) -> List[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]


def external_render_pid(workdir: Path) -> Optional[int]:
    """PID of a CLI process already rendering this workdir, if any.

    This also catches a render orphaned by a UI restart, which has no lock
    file because it was started by an older version of the UI.
    """
    target = str(Path(workdir).resolve())
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        args = _cmdline(pid)
        if not any(a.endswith("book_to_audio.py") for a in args):
            continue
        for i, arg in enumerate(args):
            value = None
            if arg == "--workdir" and i + 1 < len(args):
                value = args[i + 1]
            elif arg.startswith("--workdir="):
                value = arg.split("=", 1)[1]
            if value and str(Path(value).resolve()) == target:
                return pid
    return None


def acquire_job_lock(workdir: Path, job_id: str) -> tuple:
    """Claim a workspace. Returns (ok, reason). Stale locks are reclaimed."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pid = external_render_pid(workdir)
    if pid:
        return False, f"PID {pid} is already rendering this workspace"
    info = read_job_lock(workdir)
    if info and info.get("job_id") != job_id and _pid_alive(info.get("pid")):
        return False, f"job {info.get('job_id')} is still running as PID {info.get('pid')}"
    lock = job_lock_path(workdir)
    tmp = lock.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "job_id": job_id, "pid": None, "started": time.time(),
    }, indent=2), encoding="utf-8")
    tmp.replace(lock)
    return True, ""


def update_job_lock(workdir: Path, job_id: str, pid: int) -> None:
    info = read_job_lock(workdir)
    if info.get("job_id") != job_id:
        return
    info["pid"] = int(pid)
    lock = job_lock_path(workdir)
    tmp = lock.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    tmp.replace(lock)


def release_job_lock(workdir: Path, job_id: str) -> None:
    info = read_job_lock(workdir)
    if info.get("job_id") == job_id:
        job_lock_path(workdir).unlink(missing_ok=True)


def job_outputs(spec: Any) -> List[Path]:
    # Analyze-only jobs write a cast JSON, not playable audio.
    if getattr(spec, "kind", "render") == "analyze":
        return []
    out = Path(spec.out_path)
    try:
        if getattr(spec, "per_chapter", False):
            if out.is_dir():
                return sorted(p for p in out.glob("*.mp3") if p.stat().st_size > 0)
            return []
        return [out] if out.exists() and out.stat().st_size > 0 else []
    except OSError:
        return []


# --------------------------------------------------------------------------
# Voice bank / presets
# --------------------------------------------------------------------------

def voice_profiles() -> Dict[str, Dict[str, List[str]]]:
    p = ROOT / "voice_priority.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("profiles") or {}


def all_voices(profile: Optional[str] = None) -> List[str]:
    profiles = voice_profiles()
    chosen = profiles.get(profile) if profile else None
    if chosen is None:
        default = None
        p = ROOT / "voice_priority.json"
        if p.exists():
            try:
                default = json.loads(p.read_text(encoding="utf-8")).get("default_profile")
            except (OSError, json.JSONDecodeError):
                default = None
        chosen = profiles.get(default) or next(iter(profiles.values()), {})
    seen: List[str] = []
    for pool in ("female", "male", "neutral"):
        for v in chosen.get(pool, []):
            if v not in seen:
                seen.append(v)
    return seen


def voices_for_gender(gender: Optional[str], profile: Optional[str] = None) -> List[str]:
    profiles = voice_profiles()
    if not profiles:
        return all_voices(profile)
    chosen = profiles.get(profile) if profile else None
    if chosen is None:
        p = ROOT / "voice_priority.json"
        default = None
        if p.exists():
            try:
                default = json.loads(p.read_text(encoding="utf-8")).get("default_profile")
            except (OSError, json.JSONDecodeError):
                default = None
        chosen = profiles.get(default) or next(iter(profiles.values()), {})
    key = {"male": "male", "female": "female"}.get(gender or "", "neutral")
    return list(chosen.get(key) or chosen.get("neutral") or [])


def preset_names() -> List[str]:
    if not PRESETS.exists():
        return []
    return sorted(p.stem for p in PRESETS.glob("*.json"))


def load_preset(name: str) -> Dict[str, Any]:
    p = PRESETS / f"{name}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_preset(name: str, values: Dict[str, Any]) -> Path:
    PRESETS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9 _-]+", "", name).strip() or "preset"
    p = PRESETS / f"{safe}.json"
    p.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


RUNTIME_SETTINGS = RUNS / "runtime.json"


def runtime_settings(default_concurrency: int = 6,
                     default_failure: str = "stop") -> Dict[str, Any]:
    """Global runtime knobs: they do not change the audio/progress signature."""
    data: Dict[str, Any] = {
        "concurrency": max(1, min(16, int(default_concurrency))),
        "on_chunk_failure": default_failure if default_failure in ("stop", "skip") else "stop",
    }
    if RUNTIME_SETTINGS.exists():
        try:
            loaded = json.loads(RUNTIME_SETTINGS.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data["concurrency"] = max(1, min(16, int(loaded.get("concurrency", data["concurrency"]))))
                failure = loaded.get("on_chunk_failure", data["on_chunk_failure"])
                if failure in ("stop", "skip"):
                    data["on_chunk_failure"] = failure
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return data


def save_runtime_settings(concurrency: int, on_chunk_failure: str) -> Dict[str, Any]:
    data = {
        "concurrency": max(1, min(16, int(concurrency))),
        "on_chunk_failure": on_chunk_failure if on_chunk_failure in ("stop", "skip") else "stop",
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    RUNTIME_SETTINGS.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# --------------------------------------------------------------------------
# Sharing: rclone/Google Drive + phone download link + QR
# --------------------------------------------------------------------------

SHARING_SETTINGS = RUNS / "sharing.json"

SHARING_DEFAULTS: Dict[str, Any] = {
    "drive_remote": "gdrive",     # an `rclone config` remote name
    "drive_path": "BookToAudio",  # folder inside that remote
    "base_url": "",               # e.g. http://100.64.0.1:7860 (Tailscale/LAN)
}


def sharing_settings() -> Dict[str, Any]:
    data = dict(SHARING_DEFAULTS)
    if SHARING_SETTINGS.exists():
        try:
            loaded = json.loads(SHARING_SETTINGS.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in SHARING_DEFAULTS:
                    value = loaded.get(key)
                    if isinstance(value, str):
                        data[key] = value
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return data


def save_sharing_settings(**kw: Any) -> Dict[str, Any]:
    data = sharing_settings()
    for key, value in kw.items():
        if key in SHARING_DEFAULTS and value is not None:
            data[key] = str(value)
    RUNS.mkdir(parents=True, exist_ok=True)
    SHARING_SETTINGS.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def output_for_sharing(ws: Path) -> Optional[Path]:
    """The single file to send to Drive or the phone.

    Prefers the combined ``out.mp3``; for ``--per-chapter`` books it caches a
    STORED (no recompress) zip of ``out/*.mp3`` next to the workspace.
    """
    ws = Path(ws)
    single = ws / "out.mp3"
    if single.exists() and single.stat().st_size > 0:
        return single
    outdir = ws / "out"
    if outdir.is_dir():
        files = sorted(p for p in outdir.glob("*.mp3") if p.stat().st_size > 0)
        if files:
            zip_path = ws / "out.zip"
            newest = max(p.stat().st_mtime for p in files)
            if not zip_path.exists() or zip_path.stat().st_mtime < newest:
                tmp = zip_path.with_suffix(".zip.tmp")
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
                    for p in files:
                        z.write(p, arcname=p.name)
                tmp.replace(zip_path)
            return zip_path
    # Fall back to a title-named MP3 written by an older/manual run.
    named = [p for p in ws.glob("*.mp3") if p.stat().st_size > 0]
    if named:
        return max(named, key=lambda p: p.stat().st_size)
    return None


_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def share_name(ws: Path, f: Optional[Path] = None) -> str:
    """A human-readable download name (book title) for the shared file."""
    ws = Path(ws)
    if f is None:
        f = output_for_sharing(ws)
    if f is None:
        return f"{ws.name}.mp3"
    f = Path(f)
    if f.suffix == ".zip":
        return f.name
    if f.name != "out.mp3":
        return f.name
    title = ""
    prog = ws / "work" / "progress.json"
    if prog.exists():
        try:
            title = json.loads(prog.read_text(encoding="utf-8")).get("book_title") or ""
        except (OSError, ValueError, json.JSONDecodeError):
            title = ""
    clean = re.sub(r"\s+", " ", _UNSAFE_NAME.sub(" ", title)).strip()
    return f"{clean or ws.name}{f.suffix or '.mp3'}"
