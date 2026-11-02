import asyncio
import sys
from pathlib import Path
from typing import List, Optional


EDGE_REQUEST_TIMEOUT = 240.0


async def _edge_tts_synthesize(text: str, voice: str, out_path: Path, rate: str, pitch: str = "+0Hz"):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    # A stalled websocket can otherwise hang a worker forever.  The retry
    # wrapper in pipeline.py treats a timeout like any other failed attempt.
    await asyncio.wait_for(communicate.save(str(out_path)), timeout=EDGE_REQUEST_TIMEOUT)


def synth_chunk_edge(text: str, voice: str, out_path: Path, rate: str = "+0%", pitch: str = "+0Hz"):
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        sys.exit("Missing dependency for --engine edge.\nInstall with: pip install edge-tts")
    asyncio.run(_edge_tts_synthesize(text, voice, out_path, rate, pitch))


def synth_chunk_pyttsx3(text: str, voice: Optional[str], out_path: Path, rate_wpm: int = 175):
    try:
        import pyttsx3
    except ImportError:
        sys.exit("Missing dependency for --engine pyttsx3.\nInstall with: pip install pyttsx3")

    engine = pyttsx3.init()
    engine.setProperty("rate", rate_wpm)
    if voice:
        for v in engine.getProperty("voices"):
            if voice.lower() in (v.id.lower() + v.name.lower()):
                engine.setProperty("voice", v.id)
                break
    wav_path = out_path.with_suffix(".wav")
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()
    return wav_path


VOICE_BANK = {
    "male": [
        "en-US-GuyNeural",
        "en-US-EricNeural",
        "en-US-ChristopherNeural",
        "en-GB-RyanNeural",
        "en-GB-ThomasNeural",
        "en-AU-WilliamNeural",
        "en-IE-ConnorNeural",
        "en-US-RogerNeural",
        "en-US-SteffanNeural",
    ],
    "female": [
        "en-US-AriaNeural",
        "en-US-JennyNeural",
        "en-US-EmmaMultilingualNeural",
        "en-GB-SoniaNeural",
        "en-GB-LibbyNeural",
        "en-AU-NatashaNeural",
        "en-IE-EmilyNeural",
    ],
    "neutral": [
        "en-US-AndrewMultilingualNeural",
        "en-US-EmmaMultilingualNeural",
        "en-GB-RyanNeural",
    ],
}

DEFAULT_PRIORITY_FILE = Path(__file__).resolve().parent.parent / "voice_priority.json"
_GENDERS = ("female", "male", "neutral")
_BUILTIN_BANK = {g: list(v) for g, v in VOICE_BANK.items()}
ACTIVE_VOICE_SOURCE = "built-in list"


def load_voice_priority(path: Optional[str] = None, profile: Optional[str] = None) -> tuple:
    import json

    if path is None:
        if not DEFAULT_PRIORITY_FILE.exists():
            return {g: list(v) for g, v in _BUILTIN_BANK.items()}, "built-in list"
        path = str(DEFAULT_PRIORITY_FILE)
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"--voice-priority file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{p} isn't valid JSON: {e}")

    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f'{p}: expected a non-empty "profiles" object')
    name = profile or data.get("default_profile") or next(iter(profiles))
    if name not in profiles:
        raise ValueError(f"{p}: no profile named {name!r}. Available: {', '.join(profiles)}")

    skip = set(data.get("skip") or [])
    bank = {}
    for gender in _GENDERS:
        raw = profiles[name].get(gender)
        seen, ordered = set(), []
        for v in raw or []:
            if v not in seen and v not in skip:
                seen.add(v)
                ordered.append(v)
        bank[gender] = ordered or list(_BUILTIN_BANK[gender])
    return bank, f"{p.name} -> profile '{name}'"


def configure_voice_bank(path: Optional[str] = None, profile: Optional[str] = None) -> str:
    global ACTIVE_VOICE_SOURCE
    bank, desc = load_voice_priority(path, profile)
    VOICE_BANK.clear()
    VOICE_BANK.update(bank)
    ACTIVE_VOICE_SOURCE = desc
    return desc


def voice_bank_signature() -> str:
    import hashlib
    import json
    return hashlib.sha1(json.dumps(VOICE_BANK, sort_keys=True).encode()).hexdigest()[:12]


def sample_voices(per_gender: int = 3) -> List[str]:
    out: List[str] = []
    for gender in ("female", "male"):
        for v in VOICE_BANK.get(gender, [])[:per_gender]:
            if v not in out:
                out.append(v)
    return out


def get_available_edge_voices() -> Optional[set]:
    try:
        import edge_tts
        voices = asyncio.run(edge_tts.list_voices())
        return {v["ShortName"] for v in voices}
    except Exception:
        return None


SAMPLE_TEXT = "This is a short sample of this voice read at a natural pace."


def run_sample(voices: List[str], out_dir: Path, rate: str, engine: str = "edge"):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Synthesizing {len(voices)} sample(s) into {out_dir}/ ...", file=sys.stderr)
    for voice in voices:
        out_path = out_dir / f"sample_{voice}.mp3"
        try:
            if engine == "kokoro":
                synth_chunk_kokoro(SAMPLE_TEXT, voice, out_path, rate)
            else:
                synth_chunk_edge(SAMPLE_TEXT, voice, out_path, rate)
            print(f"  -> {out_path}", file=sys.stderr)
        except Exception as e:
            print(f"  ! failed for {voice}: {e}", file=sys.stderr)


async def list_edge_voices(lang_filter: Optional[str] = None):
    import edge_tts
    voices = await edge_tts.list_voices()
    for v in sorted(voices, key=lambda x: x["ShortName"]):
        if lang_filter and lang_filter.lower() not in v["Locale"].lower():
            continue
        print(f'{v["ShortName"]:<28} {v["Gender"]:<8} {v["Locale"]}')


# ==============================================================================
# Kokoro: local, offline neural TTS (via kokoro-onnx) - no Microsoft service needed
# ==============================================================================
KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODEL_URL = f"{KOKORO_RELEASE}/kokoro-v1.0.onnx"
KOKORO_INT8_URL = f"{KOKORO_RELEASE}/kokoro-v1.0.int8.onnx"  # ~88 MB, faster on CPU, some quality loss
KOKORO_VOICES_URL = f"{KOKORO_RELEASE}/voices-v1.0.bin"
KOKORO_CACHE_DIR = Path.home() / ".cache" / "book_to_audio"
DEFAULT_KOKORO_VOICE = "af_heart"  # default narrator; kept out of the character pool below

KOKORO_VOICE_BANK = {
    "female": [
        "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova", "af_aoede", "af_kore",
        "af_jessica", "af_river", "af_alloy", "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    ],
    "male": [
        "am_michael", "am_fenrir", "am_puck", "am_adam", "am_eric", "am_liam", "am_onyx",
        "am_echo", "bm_george", "bm_fable", "bm_lewis", "bm_daniel",
    ],
    "neutral": ["am_michael", "af_bella", "am_fenrir", "bf_emma", "bm_george", "af_nicole"],
}
KOKORO_ALL_VOICES = sorted(set(sum(KOKORO_VOICE_BANK.values(), [])) | {DEFAULT_KOKORO_VOICE})

_kokoro_state = {"model_path": None, "voices_path": None, "engine": None, "available": None,
                 "warned": set(), "device": "auto", "fast": False, "threads": 0, "gpu": False, "ctx": None}
KOKORO_STATS = {"audio": 0.0, "wall": 0.0}  # seconds of audio produced / seconds spent (for the speed readout)


def configure_kokoro(model_path: Optional[str] = None, voices_path: Optional[str] = None, device: str = "auto",
                     fast: bool = False, threads: int = 0):
    _kokoro_state["model_path"] = Path(model_path).expanduser() if model_path else None
    _kokoro_state["voices_path"] = Path(voices_path).expanduser() if voices_path else None
    _kokoro_state["device"] = device
    _kokoro_state["fast"] = fast
    _kokoro_state["threads"] = threads


def use_kokoro_voice_bank():
    VOICE_BANK.clear()
    VOICE_BANK.update({g: list(v) for g, v in KOKORO_VOICE_BANK.items()})


def list_kokoro_voices():
    for v in KOKORO_ALL_VOICES:
        accent = "British" if v[:2] in ("bf", "bm") else "American"
        gender = "female" if v[1] == "f" else "male"
        note = "  (default narrator)" if v == DEFAULT_KOKORO_VOICE else ""
        print(f"{v:<14} {gender:<8} {accent}{note}")


def _download(url: str, dest: Path):
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading {dest.name} (one-time) ...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, str(tmp))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"Could not download {url}: {e}\nDownload it manually and pass --kokoro-model / --kokoro-voices.")
    tmp.replace(dest)


WARMUP_TEXT = ("This is a short warm up passage, long enough to exercise the same convolution shapes "
               "that a normal sentence of an audiobook will need, so that problems show up now and not mid-book.")


class _quiet_stderr:
    """Silence C-library noise written straight to stderr (cuDNN prints pages of warnings when it can't start)."""

    def __enter__(self):
        import os
        try:
            sys.stderr.flush()
            self._saved = os.dup(2)
            self._null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._null, 2)
        except Exception:
            self._saved = None
        return self

    def __exit__(self, *exc):
        import os
        if self._saved is not None:
            os.dup2(self._saved, 2)
            os.close(self._saved)
            os.close(self._null)
        return False


def _warmup_voice(engine) -> str:
    try:
        avail = set(engine.get_voices())
    except Exception:
        return DEFAULT_KOKORO_VOICE
    return DEFAULT_KOKORO_VOICE if DEFAULT_KOKORO_VOICE in avail else sorted(avail)[0]


def _make_cpu_engine(Kokoro, ort, model, voices):
    import os
    engine = None
    if ort is not None:
        threads = _kokoro_state["threads"] or (os.cpu_count() or 1)
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads  # ORT's default can leave hyper-threads idle
            session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
            engine = Kokoro.from_session(session, str(voices))
            print(f"Kokoro running on: CPUExecutionProvider ({threads} threads)", file=sys.stderr)
        except Exception as e:
            engine = None
            print(f"  ! custom CPU session failed ({type(e).__name__}: {e}); using defaults.", file=sys.stderr)
    if engine is None:
        engine = Kokoro(str(model), str(voices))
        print("Kokoro running on: CPUExecutionProvider", file=sys.stderr)
    return engine


def _try_cuda_engine(Kokoro, ort, model, voices, cuda_options: dict, label: str):
    """Start a CUDA session and prove it works with a warm-up synthesis. Returns an engine or None."""
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        provider = ("CUDAExecutionProvider", cuda_options) if cuda_options else "CUDAExecutionProvider"
        session = ort.InferenceSession(str(model), sess_options=so, providers=[provider, "CPUExecutionProvider"])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            print(f"  ! GPU attempt ({label}): CUDA provider did not load.", file=sys.stderr)
            return None
        engine = Kokoro.from_session(session, str(voices))
        with _quiet_stderr():
            engine.create(WARMUP_TEXT, voice=_warmup_voice(engine), speed=1.0, lang="en-us")
        print(f"Kokoro running on: CUDAExecutionProvider ({label})", file=sys.stderr)
        return engine
    except Exception as e:
        print(f"  ! GPU attempt ({label}) failed: {type(e).__name__}: {str(e).splitlines()[0][:160]}", file=sys.stderr)
        return None


def _switch_to_cpu():
    ctx = _kokoro_state["ctx"]
    engine = _make_cpu_engine(*ctx)
    _kokoro_state["engine"] = engine
    _kokoro_state["gpu"] = False
    try:
        _kokoro_state["available"] = set(engine.get_voices())
    except Exception:
        pass
    return engine


def _get_kokoro():
    if _kokoro_state["engine"] is not None:
        return _kokoro_state["engine"]
    try:
        import numpy  # noqa: F401
        from kokoro_onnx import Kokoro
    except ImportError:
        sys.exit("Missing dependency for --engine kokoro.\nInstall with: pip install kokoro-onnx")

    fast = _kokoro_state["fast"]
    model = _kokoro_state["model_path"] or (KOKORO_CACHE_DIR / ("kokoro-v1.0.int8.onnx" if fast else "kokoro-v1.0.onnx"))
    voices = _kokoro_state["voices_path"] or (KOKORO_CACHE_DIR / "voices-v1.0.bin")
    if not model.exists() or model.stat().st_size == 0:
        _download(KOKORO_INT8_URL if fast else KOKORO_MODEL_URL, model)
    if not voices.exists() or voices.stat().st_size == 0:
        _download(KOKORO_VOICES_URL, voices)

    print(f"Loading Kokoro model ({'fast int8' if fast else 'full quality'}) ...", file=sys.stderr)
    try:
        import onnxruntime as ort
        gpu_ok = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        ort, gpu_ok = None, False

    device = _kokoro_state["device"]
    if device == "cuda" and not gpu_ok:
        print("  ! --kokoro-device cuda requested but onnxruntime-gpu isn't available; using CPU.", file=sys.stderr)

    engine = None
    if device != "cpu" and gpu_ok:
        # Newer onnxruntime-gpu + cuDNN 9 can fail on some GPUs (e.g. T4); try a safer conv setting before giving up.
        for label, opts in (("default", {}), ("safe conv algorithm", {"cudnn_conv_algo_search": "DEFAULT"})):
            engine = _try_cuda_engine(Kokoro, ort, model, voices, opts, label)
            if engine is not None:
                _kokoro_state["gpu"] = True
                break
        if engine is None:
            print("  ! GPU isn't usable here; falling back to CPU." +
                  ("" if fast else " For more CPU speed, add --kokoro-fast."), file=sys.stderr)
    if engine is None:
        engine = _make_cpu_engine(Kokoro, ort, model, voices)

    _kokoro_state["ctx"] = (Kokoro, ort, model, voices)
    try:
        _kokoro_state["available"] = set(engine.get_voices())
    except Exception:
        _kokoro_state["available"] = None
    _kokoro_state["engine"] = engine
    return engine


def _resolve_kokoro_voice(voice: Optional[str]) -> str:
    voice = voice or DEFAULT_KOKORO_VOICE
    avail = _kokoro_state["available"]
    if avail is None or voice in avail:
        return voice
    if voice not in _kokoro_state["warned"]:
        _kokoro_state["warned"].add(voice)
        print(f"  ! Kokoro voice {voice!r} isn't in the voices file; using a fallback.", file=sys.stderr)
    for cand in (DEFAULT_KOKORO_VOICE, *KOKORO_ALL_VOICES):
        if cand in avail:
            return cand
    return sorted(avail)[0]


def rate_to_speed(rate: Optional[str]) -> float:
    """Convert an Edge-style rate like '+10%' / '-15%' to Kokoro's speed multiplier (1.0 = normal)."""
    try:
        pct = float(str(rate or "+0%").strip().rstrip("%"))
    except ValueError:
        return 1.0
    return max(0.5, min(2.0, 1.0 + pct / 100.0))


def _split_for_kokoro(text: str, max_chars: int = 350) -> List[str]:
    import re
    text = " ".join(text.split())

    def hard_split(s: str):
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            yield s[:cut]
            s = s[cut:].lstrip()
        if s:
            yield s

    pieces, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for part in hard_split(sentence):
            if cur and len(cur) + len(part) + 1 > max_chars:
                pieces.append(cur)
                cur = part
            else:
                cur = f"{cur} {part}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def synth_chunk_kokoro(text: str, voice: Optional[str], out_path: Path, rate: str = "+0%", pitch: Optional[str] = None):
    """Synthesize `text` with Kokoro and write an MP3 to out_path. (`pitch` is ignored; use different voices.)"""
    import subprocess
    import time
    import wave

    import numpy as np

    t_start = time.time()
    engine = _get_kokoro()
    voice = _resolve_kokoro_voice(voice)
    lang = "en-gb" if voice[:2] in ("bf", "bm") else "en-us"
    speed = rate_to_speed(rate)

    parts, sample_rate = [], 24000
    for piece in _split_for_kokoro(text):
        try:
            samples, sample_rate = engine.create(piece, voice=voice, speed=speed, lang=lang)
        except Exception as e:
            if not _kokoro_state["gpu"]:
                raise
            print(f"  ! GPU synthesis failed ({type(e).__name__}: {str(e).splitlines()[0][:120]}); "
                  "switching to CPU for the rest of the run.", file=sys.stderr)
            engine = _switch_to_cpu()
            samples, sample_rate = engine.create(piece, voice=voice, speed=speed, lang=lang)
        parts.append(np.asarray(samples, dtype="float32").reshape(-1))
    if not parts:
        raise ValueError("nothing to synthesize")

    pcm = (np.clip(np.concatenate(parts), -1.0, 1.0) * 32767).astype("<i2")
    wav_path = out_path.with_name(out_path.stem + ".kokoro.wav")
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-ar", "24000", "-ac", "1", "-codec:a", "libmp3lame", "-q:a", "2", str(out_path)],
        capture_output=True, text=True,
    )
    wav_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to encode Kokoro audio: {result.stderr[-300:]}")
    KOKORO_STATS["audio"] += len(pcm) / float(sample_rate)
    KOKORO_STATS["wall"] += time.time() - t_start
