# book_to_audio

Audiobook creator from EPUB/PDF with multi-voice character attribution and crash-resilient resuming.

## Requirements

- Python 3.9+
- `ffmpeg` on your PATH
- `pip install -r requirements.txt`

## Project layout

```
book_to_audio/
├── book_to_audio.py        # CLI entry point
├── voice_priority.json     # Voice preference profiles (novel / news / conversation)
├── requirements.txt
├── requirements-ui.txt     # Gradio (web UI)
├── run_ui.sh               # Launch the web UI
├── webui/
│   ├── app.py              # Gradio layout + callbacks
│   ├── params.py           # CLI flag <-> widget mapping
│   ├── store.py            # Per-book workspace, uploads, presets
│   ├── cast.py             # Cast/quote editing helpers
│   ├── job.py              # Background CLI runner + progress parser
│   └── preview.py          # Audition clips
└── btlib/
    ├── models.py           # Book / Chapter dataclasses, chapter merging
    ├── extract.py          # EPUB + PDF extraction, chunking, dialogue splitting
    ├── audio.py            # ffmpeg helpers: validate, trim, silence, concat
    ├── tts.py              # Edge-TTS / pyttsx3 + voice bank loading
    ├── progress.py         # Resumable progress store
    ├── cast.py             # Regex heuristic attribution + CastStore cache
    ├── cast_whole.py       # Whole-book HMM attribution
    ├── cast_light.py       # Local Hugging Face attribution (--light-model)
    ├── cast_heavy.py       # Local LLM / Ollama attribution (--heavy-model)
    ├── cast_dispatcher.py  # Picks the attribution engine
    └── pipeline.py         # Orchestration: synth, retry, concat
```

## Web UI

A local Gradio front-end drives the same CLI (it does not re-implement the pipeline):

```bash
pip install -r requirements-ui.txt
./run_ui.sh                 # http://127.0.0.1:7860
./run_ui.sh --share         # public link (Colab)
```

In Colab:

```python
!pip install -q -r requirements-ui.txt
!python -m webui.app --share
```

Workflow:

1. **Book** – upload an EPUB/PDF (or paste a path), then **Analyze cast** (no audio).
2. **Parameters** – every CLI flag, grouped; presets can be saved and reloaded.
3. **Cast** – edit names/genders/voices/pitch, merge or delete characters, audition a line.
   Changing a gender offers the first unused voice from that gender's pool; any voice can be overridden.
   **Quotes** below lists each dialogue line (optionally only uncertain ones) and lets you reassign it.
4. **Run** – pick chapters, **Render / continue current book**, watch the jobs table/log/progress,
   download the MP3. The Run tab shows which book it will render; saved chunks are reused automatically.

### Render queue

The Run tab runs a bounded queue instead of a single job:

- **Parallel renders** (1–3, default 2) controls how many jobs run at once. Waiting jobs start
  automatically; you can load and edit another book while renders run.
- Every job keeps its own log (`runs/<slug>/logs/<job-id>.log`), progress, and output.
- Select a row (or type its id / queue number) to show that job's status, log, audio, and download.
- **Render / continue current book** applies to the book named at the top of the Run tab. It starts
  now or waits for a slot, and resumes from saved chunks automatically.
- **Continue selected job** re-queues a cancelled or failed job with its original settings, resuming
  from its saved chunks. **Cancel selected job** stops one job; **Cancel all jobs** stops the queue;
  **Clear finished jobs** tidies the table.
- A book can only have one queued/running job. The UI also writes a PID lock in
  `runs/<slug>/work/.ui-job.json`, so two UI instances (or a UI restarted while a CLI render is
  still running) cannot write to the same workspace.
- Cast/quote edits are blocked while that book is rendering. Queued jobs read the cast when they
  start, so editing before then is fine.

**Concurrency:** each job uses Parameters → **TTS concurrency** (default 6). Edge TTS is
network-bound; 6–8 is stable on a typical home connection, 10–12 is often faster but can trigger
HTTP 403 rate limits. Keep `parallel renders × TTS concurrency` around 12 or less; the queue also
holds a job back if starting it would exceed 12 in-flight requests. If a job fails with 403s,
lower both and re-run — completed chunks are kept.

Each book gets a workspace under `runs/<book-slug>/`:

```
cast.json     editable cast (always passed to the CLI as --cast-file)
work/         chunks + progress.json (--workdir); .ui-job.json lock while rendering
out.mp3       combined output, or out/ with --per-chapter
logs/         one log per queued job (<job-id>.log)
meta.json     last parameters / last job
preview/      audition clips
```

Edits that affect a chapter delete that chapter's cached chunks, so the next render redoes only what changed.
Renders are resumable: cancel any time (finished chunks are kept).

**Throughput notes:** the pipeline is network-bound, not CPU-bound. Measured on a 45-chapter
book: ~1.0 chunks/s at concurrency 6 (about 11× realtime, ~60 min for ~10.9 h of audio).
Raising concurrency is the main lever; per-chunk ffmpeg trimming and validation are only a few
percent. Resume checks now use a size check instead of fully decoding every completed chunk,
which makes restarting a long book much faster.

## Basic usage

```bash
python book_to_audio.py book.epub -o book.mp3
python book_to_audio.py book.epub -o out_dir --per-chapter
python book_to_audio.py book.epub --list-chapters
python book_to_audio.py --list-voices --lang en-US
python book_to_audio.py --sample
```

Interrupted runs resume automatically. Use `--fresh` to start over.

## 3 modes for character attribution

1. **Heuristic (default, fast, no extra dependencies)**
   ```bash
   python book_to_audio.py book.epub -o book.mp3 --multi-voice
   # Smarter global pass across the whole book:
   python book_to_audio.py book.epub -o book.mp3 --multi-voice --whole-book-context
   ```

2. **Lightweight local transformer** (`pip install transformers torch`)
   ```bash
   python book_to_audio.py book.epub -o book.mp3 --multi-voice --light-model
   ```

3. **Heavy-hitter local LLM** (Ollama or compatible endpoint)
   ```bash
   ollama pull llama3:8b
   python book_to_audio.py book.epub -o book.mp3 --multi-voice --heavy-model
   python book_to_audio.py book.epub -o book.mp3 --multi-voice --heavy-model \
       --llm-model qwen2.5:14b --llm-endpoint http://localhost:11434/api/generate
   ```

Preview the cast without synthesizing:

```bash
python book_to_audio.py book.epub --analyze-only --multi-voice
```

## Voice priority

Edit `voice_priority.json` or pass `--voice-priority FILE` / `--voice-profile NAME`
(`novel`, `news`, `conversation`).

## Troubleshooting

- **`403 Invalid response status` from Edge-TTS** - run `pip install -U edge-tts`. If it still happens, your
  network/IP is blocked by Microsoft (common on cloud hosts like Google Colab); run on your own machine.
  The program stops after two 403s; progress is saved and re-running resumes.
- **Lots of Unknown speakers** - try `--whole-book-context`, and set `--narrator-gender` for first-person books.

## Offline voices with Kokoro (no Edge-TTS needed)

```bash
pip install kokoro-onnx
python book_to_audio.py book.epub -o book.mp3 --engine kokoro --multi-voice --whole-book-context
python book_to_audio.py --engine kokoro --list-voices
python book_to_audio.py --engine kokoro --sample
```

The model (~340 MB) downloads once to `~/.cache/book_to_audio/` (or use `--kokoro-model` / `--kokoro-voices`).
Voices are names like `af_heart` (narrator default), `am_michael`, `bf_emma`. Kokoro has no pitch control, so
characters are told apart by different voices. `--rate +10%` speeds speech up.

**Speed:** Kokoro prints `x realtime` while it works (2.0x = two seconds of audio per second of compute). On a CPU it can
be slow for a whole novel. With a GPU and `pip install onnxruntime-gpu` it is much faster (`--kokoro-device auto`).
Convert in batches with `--chapters 1-10`, `--chapters 11-20`, ...; finished chapters are skipped on resume.

**Fastest settings:** use a GPU (`--kokoro-device auto` with `onnxruntime-gpu`); on CPU add `--kokoro-fast` (int8 model,
~88 MB, faster but a little lower quality: listen to `--chapters 1` first). `--rate +10%` also speeds speech up slightly.

## How speakers are attributed (all multi-voice modes)

- Tags and actions only count when they are in the **same paragraph** as the quote (a tag after a paragraph break
  belongs to the next speaker). PDFs keep the older behaviour because their line breaks are visual wraps.
- A turn-taking model uses: same paragraph = same speaker, new paragraph = the other person, long narration = new scene;
  he/she clues; "the woman said" descriptors; first-person beats; people being addressed by name; and who is mentioned in the scene.
- Pronoun-only tags (`"..." she says.`) resolve to the most recent speaker of that gender, so an explicit `she`/`he` cue
  cannot be given to a character of the other gender. A quote that is just a full name (`"Amanda Thomas," she says.`)
  identifies the speaker.
- Unnamed people get **gender-matched "minor" voices** instead of one generic Unknown voice.
- `--light-model` / `--heavy-model` supply extra names to this model (cached after the first run).
- Check the result without making audio: `--analyze-only --multi-voice` (add `--whole-book-context`, `--light-model` or `--heavy-model`).
