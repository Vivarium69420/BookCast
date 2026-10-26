import concurrent.futures
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, List

from .audio import (audio_file_ok, concat_audio, is_valid_audio, make_silence,
                    trim_silence_edges)
from .cast import format_report
from .cast_dispatcher import generate_cast, resolve_cast_file
from .extract import chunk_chapter, extract_epub, extract_pdf
from .models import Book, merge_small_chapters
from .progress import ProgressStore, default_workdir
from .tts import (DEFAULT_KOKORO_VOICE, VOICE_BANK, get_available_edge_voices,
                  synth_chunk_edge, synth_chunk_kokoro, synth_chunk_pyttsx3, KOKORO_STATS)

MAX_ATTEMPTS_PER_CHUNK = 8
BACKOFF_SCHEDULE = [3, 10, 30, 60, 60, 120, 120, 120]  # seconds between attempts
DEFAULT_EDGE_VOICE = "en-US-AndrewMultilingualNeural"
NO_AUDIO_BEFORE_FALLBACK = 3

EDGE_403_HELP = (
    "Microsoft's speech service rejected the request (HTTP 403): {err}\n"
    "  1) Update the library:  pip install -U edge-tts   (old versions get blocked)\n"
    "  2) If it still fails, your network/IP is being blocked (common on cloud hosts such as Google Colab).\n"
    "     Run it on your own computer or another network. Progress is saved, so re-running resumes."
)


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "_", text)[:60] or "chapter"


def parse_chapter_spec(spec: str, total: int) -> set:
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            result.update(range(a, b + 1))
        else:
            result.add(int(part))
    return result


class GracefulStop(Exception):
    pass


_WORD_CHAR_RE = re.compile(r"\w")


def is_speakable(text: str) -> bool:
    return bool(_WORD_CHAR_RE.search(text))


def voice_profile_for(kind: str, args) -> dict:
    if kind != "dialogue" or getattr(args, "no_dialogue_voice", False):
        return {"voice": args.voice, "rate": args.rate, "pitch": "+0Hz"}

    voice = args.dialogue_voice or args.voice
    rate = args.dialogue_rate or args.rate
    pitch = args.dialogue_pitch if args.dialogue_pitch is not None else ("+0Hz" if args.dialogue_voice else "+18Hz")
    return {"voice": voice, "rate": rate, "pitch": pitch}


def first_person_narration_voice(args, char_by_id: dict):
    if getattr(args, "voice", None):
        return None
    narrator = next((c for c in char_by_id.values() if c.get("name") == "Narrator (I)"), None)
    return (narrator or {}).get("voice") or None


def resolve_unavailable_voices(args, char_by_id: dict):
    available = get_available_edge_voices()
    if available is None:
        return

    for flag, name in (("--voice", args.voice), ("--dialogue-voice", getattr(args, "dialogue_voice", None))):
        if name and name not in available:
            sys.exit(f"{flag} {name!r} isn't served by Edge TTS. Run --list-voices.")

    used = {c.get("voice") for c in char_by_id.values() if c.get("voice") in available}
    for ch in char_by_id.values():
        voice = ch.get("voice")
        if not voice or voice in available:
            continue
        gender = ch.get("gender") or "neutral"
        candidates = [v for v in VOICE_BANK.get(gender, []) + VOICE_BANK["neutral"] if v in available]
        replacement = next((v for v in candidates if v not in used), None) or args.voice or DEFAULT_EDGE_VOICE
        ch["voice"] = replacement
        used.add(replacement)


def synthesize_chunk_with_retries(chunk_str: str, engine: str, profile: dict, out_path: Path, on_permanent_failure: str = "stop"):
    last_err = None
    no_audio_streak = 0
    forbidden = 0
    for attempt in range(MAX_ATTEMPTS_PER_CHUNK):
        is_no_audio = False
        try:
            if engine == "edge":
                voice = profile["voice"] or DEFAULT_EDGE_VOICE
                synth_chunk_edge(chunk_str, voice, out_path, profile["rate"], profile["pitch"])
                result_path = out_path
            elif engine == "kokoro":
                synth_chunk_kokoro(chunk_str, profile["voice"] or DEFAULT_KOKORO_VOICE, out_path, profile["rate"], profile.get("pitch"))
                result_path = out_path
            elif engine == "pyttsx3":
                result_path = synth_chunk_pyttsx3(chunk_str, profile["voice"], out_path)
            else:
                raise ValueError(f"Unknown engine: {engine}")

            trim_silence_edges(result_path)
            if is_valid_audio(result_path):
                return result_path
            last_err = "corrupted/empty audio"
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = str(e)
            is_no_audio = type(e).__name__ == "NoAudioReceived"
            if engine == "edge" and "403" in last_err:
                forbidden += 1
                if forbidden >= 2:
                    raise GracefulStop(EDGE_403_HELP.format(err=last_err[:160]))

        no_audio_streak = no_audio_streak + 1 if is_no_audio else 0
        if engine == "edge" and no_audio_streak >= NO_AUDIO_BEFORE_FALLBACK:
            fallback = profile.get("fallback_voice") or DEFAULT_EDGE_VOICE
            profile = {**profile, "voice": fallback, "pitch": "+0Hz"}
            no_audio_streak = 0
            continue

        if engine == "kokoro":
            break

        wait = BACKOFF_SCHEDULE[min(attempt, len(BACKOFF_SCHEDULE) - 1)]
        time.sleep(wait)

    if on_permanent_failure == "skip":
        make_silence(out_path, duration_ms=300)
        return out_path
    raise GracefulStop(f"Failed chunk ({engine}): {last_err}")


def run_pending_chunks(pending: list, engine: str, max_workers: int, on_result: Callable, on_permanent_failure: str = "stop"):
    if max_workers <= 1 or engine != "edge" or len(pending) <= 1:
        for c_idx, chunk_str, profile, out_chunk in pending:
            res = synthesize_chunk_with_retries(chunk_str, engine, profile, out_chunk, on_permanent_failure)
            on_result(c_idx, res)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(synthesize_chunk_with_retries, c_str, engine, prof, out_p, on_permanent_failure): idx
            for idx, c_str, prof, out_p in pending
        }
        for fut in concurrent.futures.as_completed(futs):
            c_idx = futs[fut]
            on_result(c_idx, fut.result())


def extract_book(in_path: Path, args) -> Book:
    suffix = in_path.suffix.lower()
    from . import cast as _cast
    _cast.PARAGRAPH_AWARE = suffix != ".pdf"
    if suffix == ".epub":
        book = extract_epub(in_path)
    elif suffix == ".pdf":
        book = extract_pdf(in_path)
    else:
        sys.exit(f"Unsupported file type: {suffix}")
    if args.min_chapter_words != 150:
        book.chapters = merge_small_chapters(book.chapters, min_words=args.min_chapter_words)
    return book


def load_or_resume_progress(in_path: Path, args) -> ProgressStore:
    workdir = Path(args.workdir) if args.workdir else default_workdir(args.output, args.per_chapter)
    store = ProgressStore(workdir)
    if store.exists():
        store.load()
        if args.fresh:
            shutil.rmtree(workdir)
            store = ProgressStore(workdir)
        elif store.matches(in_path, args):
            print(f"Found saved progress in {workdir} - resuming.", file=sys.stderr)
            return store
    book = extract_book(in_path, args)
    store.init_new(book, in_path, args)
    return store


def list_chapters(args):
    in_path = Path(args.input)
    book = extract_book(in_path, args)
    print(f"'{book.title}' - {len(book.chapters)} chapter(s):", file=sys.stderr)
    for i, ch in enumerate(book.chapters, start=1):
        print(f"  [{i:>3}] {ch.title} (~{len(ch.text.split())} words)", file=sys.stderr)


def analyze_only(args):
    in_path = Path(args.input)
    book = extract_book(in_path, args)
    cast_data = generate_cast(in_path, book, args)
    target_path = resolve_cast_file(in_path, args)
    print(format_report(cast_data), file=sys.stderr)
    print(f"\nCast saved to {target_path}", file=sys.stderr)


def run(args):
    in_path = Path(args.input)
    store = load_or_resume_progress(in_path, args)
    book = store.book()
    out_path = Path(args.output)

    selected_chapters = parse_chapter_spec(args.chapters, len(book.chapters)) if args.chapters else None

    char_by_id = {}
    attribution_by_chapter = {}
    narration_voice = None
    if args.multi_voice:
        cast_data = generate_cast(in_path, book, args)
        char_by_id = {c["id"]: c for c in cast_data["characters"]}
        for ch_num, entries in cast_data["attribution"].items():
            attribution_by_chapter[int(ch_num)] = {e["seg_idx"]: e for e in entries}
        if args.engine == "edge":
            resolve_unavailable_voices(args, char_by_id)
        narration_voice = first_person_narration_voice(args, char_by_id)
        print(format_report(cast_data), file=sys.stderr)

    for idx, chapter in enumerate(book.chapters, start=1):
        if selected_chapters is not None and idx not in selected_chapters:
            continue  # not requested this run — leave its state untouched
        ch_out = out_path / f"{idx:02d}_{slugify(chapter.title)}.mp3"
        if store.is_chapter_combined_done(idx):
            if not args.per_chapter:
                continue  # this chapter is entirely finished already
            if store.is_chapter_output_done(idx) and ch_out.exists():
                continue
            comb = store.chapter_combined_path(idx)
            if comb.exists():
                out_path.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(comb, ch_out)
                store.mark_chapter_output_done(idx)
                continue

        print(f"[{idx}/{len(book.chapters)}] {chapter.title}", file=sys.stderr)
        segments = chunk_chapter(chapter.text, max_chars=3000, distinguish_dialogue=not args.no_dialogue_voice)
        store.set_num_chunks(idx, len(segments))

        chunk_suffix = ".wav" if args.engine == "pyttsx3" else ".mp3"
        chunk_kinds: List[str] = [k for k, _, _ in segments]
        chunk_paths = {}
        pending = []

        for c_idx, (kind, chunk_str, seg_idx) in enumerate(segments):
            out_chunk = store.chunk_path(idx, c_idx, suffix=chunk_suffix)
            if store.is_chunk_done(idx, c_idx) and audio_file_ok(out_chunk):
                chunk_paths[c_idx] = out_chunk
                continue

            if not is_speakable(chunk_str):
                make_silence(out_chunk, duration_ms=250)
                store.mark_chunk_done(idx, c_idx)
                chunk_paths[c_idx] = out_chunk
                continue

            if args.multi_voice and kind == "dialogue":
                entry = attribution_by_chapter.get(idx, {}).get(seg_idx)
                cid = entry["character_id"] if entry else "unknown"
                char = char_by_id.get(cid) or char_by_id.get("unknown") or {}
                profile = {
                    "voice": char.get("voice") or args.voice,
                    "rate": args.dialogue_rate or args.rate,
                    "pitch": char.get("pitch", "+0Hz"),
                    "fallback_voice": args.voice or narration_voice or DEFAULT_EDGE_VOICE,
                }
            else:
                profile = voice_profile_for(kind, args)
                if narration_voice and kind == "narration":
                    profile["voice"] = narration_voice

            pending.append((c_idx, chunk_str, profile, out_chunk))

        n_pending = len(pending)
        done_count = [0]
        t_chapter = time.time()

        def _on_done(c_idx, path):
            store.mark_chunk_done(idx, c_idx)
            chunk_paths[c_idx] = path
            done_count[0] += 1
            if args.engine == "kokoro":
                speed = ""
                if KOKORO_STATS["wall"] > 0:
                    speed = f", {KOKORO_STATS['audio'] / KOKORO_STATS['wall']:.2f}x realtime"
                print(f"  chunk {done_count[0]}/{n_pending} ({time.time() - t_chapter:.0f}s{speed})", file=sys.stderr)
            elif done_count[0] % 10 == 0:
                print(f"  chunk {done_count[0]}/{n_pending}", file=sys.stderr)

        run_pending_chunks(pending, args.engine, args.concurrency if args.engine == "edge" else 1,
                           _on_done, on_permanent_failure=args.on_chunk_failure)

        ordered_paths = [chunk_paths[i] for i in range(len(segments))]
        if not store.is_chapter_combined_done(idx):
            comb = store.chapter_combined_path(idx)
            gaps = [args.gap_ms + (args.dialogue_pause_ms if chunk_kinds[i] != chunk_kinds[i + 1] else 0)
                    for i in range(len(ordered_paths) - 1)]
            concat_audio(ordered_paths, comb, gap_ms=gaps)
            store.mark_chapter_combined_done(idx)
        else:
            comb = store.chapter_combined_path(idx)

        if args.per_chapter:
            out_path.mkdir(parents=True, exist_ok=True)
            # The directory may have changed while the chunks stayed valid
            # (for example when switching to an EPUB-named output folder).
            if not store.is_chapter_output_done(idx) or not ch_out.exists():
                shutil.copyfile(comb, ch_out)
                store.mark_chapter_output_done(idx)

    if not args.per_chapter and (not store.is_final_done()
                                 or not out_path.exists()
                                 or out_path.stat().st_size == 0):
        # With --chapters, only the selected chapters were synthesized; the
        # others have no combined file and must not be referenced.
        all_comb = [store.chapter_combined_path(c["index"])
                    for c in store.data["chapters"]
                    if selected_chapters is None or c["index"] in selected_chapters]
        all_comb = [p for p in all_comb if p.exists()]
        if not all_comb:
            print("No chapters were produced; nothing to concatenate.", file=sys.stderr)
            return 1
        concat_audio(all_comb, out_path, gap_ms=args.chapter_pause_ms)
        store.mark_final_done()
        print(f"Done -> {out_path}", file=sys.stderr)
