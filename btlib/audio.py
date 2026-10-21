"""Audio validation, silence trimming, and ffmpeg-based concatenation.

Functions here raise RuntimeError on failure (instead of sys.exit) so the
pipeline layer can catch them, save progress, and let the user resume later
instead of losing an entire run to one bad chunk or a missing dependency.
"""

import subprocess
import sys
from pathlib import Path
from typing import List


def is_valid_audio(path: Path) -> bool:
    """Tests if an audio file can be fully decoded without errors.

    This spawns ffmpeg and decodes the whole file (~60 ms per chunk), so
    resume checks use ``audio_file_ok`` instead: progress.json only marks a
    chunk done after it has passed this test.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and not result.stderr.strip()


def audio_file_ok(path: Path, min_bytes: int = 1024) -> bool:
    """Cheap resume check: the chunk was validated before being marked done."""
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def trim_silence_edges(path: Path, threshold_db: int = -40):
    tmp_path = path.with_suffix(path.suffix + ".trim.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-af",
        (
            f"silenceremove=start_periods=1:start_duration=0:start_threshold={threshold_db}dB:detection=peak,"
            "areverse,"
            f"silenceremove=start_periods=1:start_duration=0:start_threshold={threshold_db}dB:detection=peak,"
            "areverse"
        ),
        "-q:a", "2",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
        tmp_path.replace(path)
    elif tmp_path.exists():
        tmp_path.unlink()


def make_silence(path: Path, duration_ms: int = 250):
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(duration_ms / 1000),
            "-q:a", "9",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def concat_audio(chunk_paths: List[Path], out_path: Path, gap_ms=350):
    """gap_ms can be a single int (uniform silence between every chunk) or a
    list of ints of length len(chunk_paths)-1 giving the silence before each
    chunk individually — used to put a slightly longer beat wherever the
    narrator hands off to dialogue or back."""
    if not chunk_paths:
        raise ValueError("No audio chunks to concatenate")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    workdir = chunk_paths[0].parent
    if isinstance(gap_ms, int):
        gaps = [gap_ms] * (len(chunk_paths) - 1)
    else:
        gaps = list(gap_ms)
        if len(gaps) != len(chunk_paths) - 1:
            raise ValueError("gap_ms list must have len(chunk_paths) - 1 entries")

    silence_paths = {}
    for g in set(gaps):
        sp = workdir / f"_silence_{g}.mp3"
        if not sp.exists():
            make_silence(sp, g)
        silence_paths[g] = sp

    interleaved: List[Path] = []
    for i, p in enumerate(chunk_paths):
        interleaved.append(p)
        if i != len(chunk_paths) - 1:
            interleaved.append(silence_paths[gaps[i]])

    list_file = workdir / f"_concat_list_{out_path.stem}.txt"
    with open(list_file, "w") as f:
        for p in interleaved:
            f.write(f"file '{p.resolve()}'\n")

    fmt = out_path.suffix.lstrip(".") or "mp3"
    codec = "libmp3lame" if fmt == "mp3" else "copy"

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:a", codec,
    ]
    if codec == "libmp3lame":
        cmd += ["-q:a", "2"]
    cmd.append(str(out_path))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if codec == "libmp3lame" and ("Invalid argument" in result.stderr or "padding" in result.stderr):
            print("\n  [Warning] MP3 encoder padding error detected. Retrying via safe WAV fallback...", file=sys.stderr)
            temp_wav = workdir / f"_fallback_{out_path.stem}.wav"

            wav_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c:a", "pcm_s16le", str(temp_wav)
            ]
            wav_result = subprocess.run(wav_cmd, capture_output=True, text=True)

            if wav_result.returncode == 0:
                mp3_cmd = [
                    "ffmpeg", "-y", "-i", str(temp_wav),
                    "-c:a", "libmp3lame", "-q:a", "2", str(out_path)
                ]
                mp3_result = subprocess.run(mp3_cmd, capture_output=True, text=True)
                if mp3_result.returncode == 0:
                    return
                raise RuntimeError(f"ffmpeg failed during fallback MP3 encode:\n{mp3_result.stderr[-2000:]}")
            raise RuntimeError(f"ffmpeg failed during fallback WAV concat:\n{wav_result.stderr[-2000:]}")
        raise RuntimeError(f"ffmpeg failed to combine audio:\n{result.stderr[-2000:]}")
