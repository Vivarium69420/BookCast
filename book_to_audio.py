#!/usr/bin/env python3
"""
book_to_audio.py - Turn an .epub or .pdf into an audiobook (MP3).
Resumable progress with multi-engine dialogue attribution (Heuristic, Light HF Model, Heavy LLM).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from btlib import pipeline, tts
from btlib.pipeline import GracefulStop


def main():
    parser = argparse.ArgumentParser(description="Convert an EPUB or PDF into an audiobook.")
    parser.add_argument("input", nargs="?", help="Path to the .epub or .pdf file")
    parser.add_argument("-o", "--output", help="Output .mp3 file, or a directory when using --per-chapter")
    parser.add_argument("--engine", choices=["edge", "kokoro", "pyttsx3"], default="edge",
                        help="TTS engine: 'edge' (online, natural), 'kokoro' (offline neural, no Microsoft "
                             "service needed) or 'pyttsx3' (offline, robotic)")
    parser.add_argument("--kokoro-model", metavar="FILE", default=None,
                        help="Path to kokoro-v1.0.onnx (downloaded here automatically if missing).")
    parser.add_argument("--kokoro-fast", action="store_true",
                        help="Use the smaller int8 Kokoro model: faster on CPU, with some quality loss.")
    parser.add_argument("--kokoro-threads", type=int, default=0, metavar="N",
                        help="CPU threads for Kokoro (default: all cores).")
    parser.add_argument("--kokoro-device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Kokoro device: 'auto' uses the GPU when onnxruntime-gpu is installed (default).")
    parser.add_argument("--kokoro-voices", metavar="FILE", default=None,
                        help="Path to voices-v1.0.bin (downloaded here automatically if missing).")
    parser.add_argument("--voice", help="Voice name for narration. Use --list-voices to view options.")
    parser.add_argument("--rate", default="+0%", help="Speaking rate adjustment for edge-tts, e.g. +15%% or -10%%")
    parser.add_argument("--per-chapter", action="store_true", help="Write one MP3 per chapter instead of one file")
    parser.add_argument("--gap-ms", type=int, default=350,
                        help="Silence (ms) between sentence-chunks within a chapter (default: 350)")
    parser.add_argument("--chapter-pause-ms", type=int, default=900,
                        help="Silence (ms) between chapters in the combined book (default: 900)")
    parser.add_argument("--min-chapter-words", type=int, default=150,
                        help="Merge chapters smaller than this many words into the next one (default: 150)")
    parser.add_argument("--dialogue-voice", help="Different voice for dialogue when not using --multi-voice.")
    parser.add_argument("--dialogue-rate", help="Speaking rate for dialogue chunks (edge only).")
    parser.add_argument("--dialogue-pitch", help="Pitch shift for dialogue chunks (edge only).")
    parser.add_argument("--dialogue-pause-ms", type=int, default=200,
                        help="Extra silence (ms) between narration and dialogue handoffs (default: 200)")
    parser.add_argument("--no-dialogue-voice", action="store_true",
                        help="Disable narration/dialogue distinction entirely (read in one voice).")
    parser.add_argument("--multi-voice", action="store_true",
                        help="Detect characters and give each major speaker their own voice.")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Run character-detection and attribution pass, save cast file, and exit.")
    parser.add_argument("--cast-file", help="Use an existing cast.json instead of generating one.")
    parser.add_argument("--whole-book-context", "--whole-book", dest="whole_book_context", action="store_true",
                        help="Global HMM attribution heuristic across the entire book.")
    parser.add_argument("--scene-context", "--scene", dest="scene_context", action="store_true",
                        help="Scene-structured attribution: segment chapters into scenes, track who is "
                             "present, and alternate only among a scene's participants (faster and more "
                             "accurate than --whole-book on the public-domain eval corpora).")
    parser.add_argument("--unknown-below", type=float, default=None, metavar="P",
                        help="Posterior threshold under which a speaker is left as Unknown (default 0.30).")
    parser.add_argument("--narrator-gender", choices=["male", "female"], default=None,
                        help="Gender of a first-person narrator ('I').")
    parser.add_argument("--voice-priority", metavar="FILE", default=None,
                        help="Voice priority JSON list. Default: voice_priority.json")
    parser.add_argument("--voice-profile", metavar="NAME", default=None,
                        help="Voice profile name inside priority file (novel, news, conversation).")
    parser.add_argument("--max-characters", type=int, default=12,
                        help="Maximum characters to give unique voices; rest grouped as Unknown (default: 12)")
    parser.add_argument("--chapters", help="Only synthesize specified chapters (e.g., '1,3,5-8').")
    parser.add_argument("--list-chapters", action="store_true",
                        help="Print chapter numbers/titles and exit.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel edge-tts request concurrency (default: 4).")
    parser.add_argument("--on-chunk-failure", choices=["stop", "skip"], default="stop",
                        help="Action when a chunk fails retries: 'stop' or 'skip'.")
    parser.add_argument("--list-voices", action="store_true", help="List available edge-tts voices and exit.")
    parser.add_argument("--lang", help="Filter --list-voices by language, e.g. en-US")
    parser.add_argument("--sample", nargs="*", metavar="VOICE",
                        help="Synthesize sample sentence in voice(s) and exit.")
    parser.add_argument("--workdir", help="Persistent progress directory.")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved progress and start over.")
    parser.add_argument("--keep-workdir", action="store_true", help="Retain progress directory after run.")

    # Model Attribution Options
    ai_group = parser.add_argument_group("AI Attribution Options")
    ai_group.add_argument("--light-model", action="store_true",
                          help="Use local Hugging Face transformer (helliun/quote-attribution) for attribution.")
    ai_group.add_argument("--heavy-model", action="store_true",
                          help="Use local LLM (Ollama/OpenAI compatible) for attribution.")
    ai_group.add_argument("--llm-model", default="llama3:8b",
                          help="Model tag for --heavy-model (default: 'llama3:8b').")
    ai_group.add_argument("--llm-endpoint", default="http://localhost:11434/api/generate",
                          help="API URL for --heavy-model (default: 'http://localhost:11434/api/generate').")

    args = parser.parse_args()

    try:
        tts.configure_voice_bank(args.voice_priority, args.voice_profile)
    except ValueError as e:
        parser.error(str(e))

    if args.engine == "kokoro":
        tts.configure_kokoro(args.kokoro_model, args.kokoro_voices, args.kokoro_device,
                             fast=args.kokoro_fast, threads=args.kokoro_threads)
        tts.use_kokoro_voice_bank()
        for flag, name in (("--voice", args.voice), ("--dialogue-voice", args.dialogue_voice)):
            if name and name not in tts.KOKORO_ALL_VOICES:
                parser.error(f"{flag} {name!r} isn't a Kokoro voice. Run: --engine kokoro --list-voices")

    if args.list_voices:
        if args.engine == "kokoro":
            tts.list_kokoro_voices()
        else:
            asyncio.run(tts.list_edge_voices(args.lang))
        return

    if args.sample is not None:
        voices = args.sample if args.sample else tts.sample_voices()
        out_dir = Path(args.output) if args.output else Path("voice_samples")
        tts.run_sample(voices, out_dir, args.rate, args.engine)
        return

    if args.list_chapters:
        if not args.input:
            parser.error("input is required for --list-chapters")
        pipeline.list_chapters(args)
        return

    if args.analyze_only:
        if not args.input:
            parser.error("input is required for --analyze-only")
        pipeline.analyze_only(args)
        return

    if not args.input or not args.output:
        parser.error("input and -o/--output are required (unless using --list-voices, --sample, or --analyze-only)")

    try:
        pipeline.run(args)
    except GracefulStop as e:
        sys.exit(f"\nStopped: {e}")


if __name__ == "__main__":
    main()
