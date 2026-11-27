#!/usr/bin/env python3
"""Watch finished BookCast renders and copy them to Google Drive with rclone.

The watcher is deliberately separate from the renderer: it can be started
before a render, survive a render restart, and upload only after the progress
file says the final MP3 is complete.  It also works for an MP3 that was
produced by another instance and copied into a workspace.

Usage:
    python auto_upload_to_drive.py              # keep watching
    python auto_upload_to_drive.py --once       # upload anything ready, then exit
    python auto_upload_to_drive.py --force      # ignore the uploaded-state cache

Configuration lives in ``auto_upload.json`` next to this script.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "auto_upload.json"
STATE_PATH = ROOT / "runs" / ".auto_upload_state.json"
LOCK_PATH = ROOT / "runs" / ".auto_upload.lock"
LOG_PATH = ROOT / "runs" / "auto_upload.log"


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def acquire_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another auto-upload watcher is already running ({LOCK_PATH}).",
              file=sys.stderr)
        raise SystemExit(1)
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def remote_size(remote: str) -> Optional[int]:
    """Size of the remote file, or None when it does not exist yet."""
    proc = run(["rclone", "lsjson", remote, "--stat", "--no-mimetype",
                "--no-modtime"], timeout=120)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or "{}")
        return int(data.get("Size", 0))
    except (ValueError, TypeError):
        return None


def probe_duration(path: Path) -> Optional[float]:
    proc = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], timeout=120)
    if proc.returncode != 0:
        return None
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return None


def progress_data(workspace: Path) -> Optional[Dict[str, Any]]:
    path = workspace / "work" / "progress.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def output_for_job(workspace: Path, progress: Optional[Dict[str, Any]]) -> Optional[Path]:
    """Prefer the renderer's recorded output path, then the newest MP3."""
    if progress:
        raw = (progress.get("params") or {}).get("output")
        if raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate

    candidates = []
    for path in workspace.glob("*.mp3"):
        try:
            if path.is_file() and path.stat().st_size > 0:
                candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def is_stable(path: Path, seconds: int) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size <= 0:
        return False
    return (time.time() - stat.st_mtime) >= seconds


def workspaces_for_job(job: Dict[str, Any]) -> list[Path]:
    if job.get("workspace"):
        return [ROOT / str(job["workspace"])]
    pattern = str(job.get("workspace_glob") or "")
    if not pattern:
        return []
    found = [p for p in ROOT.glob(pattern) if p.is_dir()]
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def ready_output(job: Dict[str, Any], stable_seconds: int) -> tuple[Optional[Path], str]:
    workspaces = workspaces_for_job(job)
    if not workspaces:
        return None, f"no workspace matched {job.get('workspace_glob') or job.get('workspace')}"

    reasons = []
    for workspace in workspaces:
        progress = progress_data(workspace)
        output = output_for_job(workspace, progress)

        if progress is not None:
            if not progress.get("final_done"):
                reasons.append(f"{workspace.name}: render still in progress")
                continue
            if output is None:
                reasons.append(f"{workspace.name}: marked done but no MP3 found")
                continue
            if output.stat().st_size <= 0:
                reasons.append(f"{workspace.name}: output MP3 is empty")
                continue
            return output, f"render marked final in {workspace.name}"

        # No progress file: this is usually an MP3 produced by another
        # instance and copied into the workspace.  Require a quiet file so we
        # do not upload a partially written copy.
        if output is None:
            reasons.append(f"{workspace.name}: waiting for an MP3")
            continue
        if not is_stable(output, stable_seconds):
            reasons.append(f"{workspace.name}: waiting for {output.name} to stop changing")
            continue
        return output, f"stable MP3 found in {workspace.name}"

    return None, "; ".join(reasons) or "waiting for a workspace"


def remote_path(config: Dict[str, Any], job: Dict[str, Any]) -> str:
    remote = str(config.get("remote") or "").rstrip("/")
    name = str(job.get("remote_name") or Path(str(job.get("workspace", ""))).name)
    return f"{remote}/{name}"


def already_uploaded(state: Dict[str, Any], key: str, path: Path) -> bool:
    record = state.get(key) or {}
    try:
        stat = path.stat()
    except OSError:
        return False
    return (
        record.get("size") == stat.st_size
        and record.get("mtime_ns") == stat.st_mtime_ns
    )


def upload(path: Path, remote: str, state_key: str, state: Dict[str, Any]) -> bool:
    stat = path.stat()
    logging.info("Uploading %s -> %s (%.1f MiB)",
                 path, remote, stat.st_size / (1024 * 1024))
    cmd = [
        "rclone", "copyto", str(path), remote,
        "--checksum",
        "--retries", "5",
        "--low-level-retries", "10",
        "--timeout", "10m",
        "--stats", "30s",
        "--stats-one-line",
    ]
    try:
        proc = run(cmd)
    except subprocess.TimeoutExpired:
        logging.error("Upload timed out: %s", remote)
        return False

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        logging.error("rclone failed for %s:\n%s", remote, "\n".join(tail))
        return False

    size = remote_size(remote)
    if size is not None and size != stat.st_size:
        logging.error("Remote size mismatch for %s: local=%s remote=%s",
                      remote, stat.st_size, size)
        return False

    state[state_key] = {
        "local": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "uploaded_at": time.time(),
        "remote": remote,
    }
    save_state(state)
    logging.info("Uploaded %s (remote size %s)", remote, size if size is not None else "unknown")
    return True


def process_once(config: Dict[str, Any], state: Dict[str, Any], force: bool) -> int:
    uploaded = 0
    stable_seconds = int(config.get("stable_seconds", 45))
    for job in config.get("jobs") or []:
        name = job.get("name") or job.get("workspace") or "book"
        path, reason = ready_output(job, stable_seconds)
        if path is None:
            logging.info("%s: %s", name, reason)
            continue

        remote = remote_path(config, job)
        if not force and already_uploaded(state, remote, path):
            logging.info("%s: already uploaded, skipping", name)
            continue

        duration = probe_duration(path)
        if duration is None or duration <= 0:
            logging.warning("%s: ffprobe could not read %s; will retry", name, path)
            continue

        logging.info("%s: ready (%.1f min)", name, duration / 60)
        if upload(path, remote, remote, state):
            uploaded += 1
    return uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON config (default: auto_upload.json)")
    parser.add_argument("--once", action="store_true",
                        help="Check once, upload anything ready, then exit")
    parser.add_argument("--force", action="store_true",
                        help="Upload even if this exact file version was already sent")
    args = parser.parse_args()

    setup_logging()
    if shutil.which("rclone") is None:
        logging.error("rclone is not installed or not on PATH")
        return 2
    if not args.config.exists():
        logging.error("Config not found: %s", args.config)
        return 2

    config = load_json(args.config, {})
    if not config.get("remote"):
        logging.error("Config must set a remote, for example maihaidanga6:Audiobooks")
        return 2
    if not config.get("jobs"):
        logging.error("Config must contain at least one job")
        return 2

    lock = acquire_lock()
    state = load_json(STATE_PATH, {})
    logging.info("Watching %d book(s) -> %s", len(config["jobs"]), config["remote"])

    try:
        if args.once:
            count = process_once(config, state, args.force)
            logging.info("One-shot check finished: %d upload(s)", count)
            return 0

        interval = max(5, int(config.get("poll_seconds", 20)))
        while True:
            try:
                process_once(config, state, args.force)
            except Exception:
                logging.exception("Watcher cycle failed; will retry")
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
