# book_to_audio

Audiobook creator from EPUB/PDF with multi-voice character attribution and crash-resilient resuming.

## BookCast website

The product landing page is the static site in `site/` (brand: **BookCast**).
Preview it with `python -m http.server 8765 -d site`, or see
[`site/README.md`](site/README.md) for GitHub Pages deployment.

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
    ├── cast_llm.py         # Scene + LLM cast reconciliation / weak-line adjudication
    ├── cast_heavy.py       # Legacy local-LLM wrapper (kept for old imports)
    ├── llm.py              # Ollama / OpenAI-compatible client + response cache
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

Workflow (one page per open book, plus a shared Activity panel):

1. **Open a book** – upload an EPUB/PDF or paste a path. Every opened book is added to the
   **Open books** rail; switching between books never stops their renders.
2. **Book & chapters** – review the chapter table and type a printer-style range such as
   `1-3, 6-8` (blank or `all` renders every chapter). **Analyze cast** runs without TTS;
   `light`/`heavy` analysis runs as a background job in **Activity** so long Colab analyses
   neither block the page nor hit the old synchronous timeout.
3. **Parameters** – every CLI flag except the global runtime knobs, grouped. Edit, then
   **Apply changes**. Content changes show a review first and need **Confirm changes**; the next
   render then warns and asks for confirmation because it will discard saved chunks. **Revert**
   restores the applied settings. Presets and an exact-command preview live in the same panel.
4. **Cast & quotes** – edit names/genders/voices/pitch, merge or delete characters, audition a
   line. **Import cast JSON** accepts a cast produced by the Colab `analyze_only_heavy` mode;
   it validates the chapter/segment references and shows a diff before replacing the workspace
   cast. Changing a gender offers the first unused voice from that gender's pool; any voice can
   be overridden, but a named character cannot reuse another named character's voice (the editor
   rejects the duplicate so the narrator and a major character cannot collapse into one voice).
   **Quotes** lists each dialogue line (optionally only uncertain ones) and lets you reassign it.
5. **Render this book** – **Render / continue current book** starts the active book now (or waits
   for a slot) and resumes from saved chunks automatically. The output file is named after the input
   EPUB/PDF filename (for example `My Book.epub` → `My Book.mp3`). **Start over** ignores saved
   chunks for that run only.
6. **Activity** – shared by every open book: **Parallel renders**, global **TTS concurrency** and
   **On chunk failure**, the jobs table, status/log/progress, audio/download, and
   **Continue selected job** / **Cancel selected job** / **Delete selected job** / **Cancel all
   jobs** / **Clear finished jobs**. Deleting a job clears its Activity row and log; rendered audio
   and cast files stay on disk.

Open books are remembered in `runs/session.json`: refreshing the page or restarting the UI reopens
the same books and the active one (up to five).

### Render queue

The Activity panel runs a bounded queue instead of a single job:

- **Parallel renders** (1–3) controls how many books render at once. Waiting jobs start
  automatically; you can open and edit other books while renders run.
- Every job keeps its own log (`runs/<slug>/logs/<job-id>.log`), progress, and output.
  Light/heavy cast analyses use the same queue as renders and write `cast.json`
  instead of audio.
- Select a row (or type its id / queue number) to show that job's status, log, audio, and download.
- **Continue selected job** re-queues a cancelled or failed job with its original content settings;
  renders resume from saved chunks, while analyses re-run. Both use the **current** global
  runtime settings.
- A book can only have one queued/running job. The UI also writes a PID lock in
  `runs/<slug>/work/.ui-job.json`, so two UI instances (or a UI restarted while a CLI render is
  still running) cannot write to the same workspace.
- Cast/quote edits are blocked while that book is rendering or analyzing. Queued jobs read the
  cast when they start, so editing before then is fine.

**Runtime settings vs content settings:** **TTS concurrency** and **On chunk failure** are global
and safe to change without invalidating saved chunks; they are in Activity and apply to new and
continued jobs (a running job keeps the settings it started with). Parameters such as **Rate**,
**Max characters**, or **One MP3 per chapter** change the audio and therefore force the next
render to start from scratch; the UI requires confirmation for those.

**Concurrency:** the default is 6 TTS requests per job. Edge TTS is network-bound; 6–8 is stable on
a typical home connection, 10–12 is often faster but can trigger HTTP 403 rate limits. Keep
`parallel renders × TTS concurrency` around 12 or less; the queue also holds a job back if starting
it would exceed 12 in-flight requests. If a job fails with 403s, lower both and re-run — completed
chunks are kept.

Each book gets a workspace under `runs/<book-slug>/`:

```
cast.json           editable cast (always passed to the CLI as --cast-file)
work/               chunks + progress.json (--workdir); .ui-job.json lock while rendering
<EPUB filename>.mp3 combined output, or <EPUB filename>/ with --per-chapter
logs/               one log per queued job (<job-id>.log)
meta.json           last parameters / last job
preview/            audition clips
```

Edits that affect a chapter delete that chapter's cached chunks, so the next render redoes only what changed.
Renders are resumable: cancel any time (finished chunks are kept).

Global UI state lives next to the workspaces: `runs/runtime.json` (TTS concurrency / failure policy),
`runs/queue.json` (parallel slots), `runs/session.json` (open books), and `runs/sharing.json`
(Drive remote / phone base URL).

### Sharing the output

The **Share output** accordion in Activity acts on the selected job, or on the active book when no
job is selected:

- **Google Drive** runs `rclone copyto` to `<remote>:<folder>/<output filename>` and remembers the
  remote and folder in `runs/sharing.json`. Add a Google Drive remote once with `rclone config` (the
  field defaults to `gdrive`), then check it with `rclone lsd gdrive:`. Large files can take a while;
  the status line streams while the upload runs, so keep the tab open.
- **Phone** builds `http://<host>:<port>/share/<slug>` and a QR code. The route serves the most recent
  output, named after the input EPUB/PDF (for example `The Secret Garden.epub` →
  `The Secret Garden.mp3`), or `<EPUB filename>.zip` for `--per-chapter` books. **Detect** fills in
  the best local address; edit **Base URL** to your Tailscale `100.x` address to reach the phone
  away from home.

The phone must be able to reach the Gradio server, so start it on the LAN/Tailscale interface:

```bash
./run_ui.sh --host 0.0.0.0
```

The UI has no authentication, so use this on a trusted home network or over Tailscale. If you leave
`--host` at its loopback default, the phone link warns that the server is not reachable.

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

## Attribution engines

1. **Scene (UI default; recommended offline engine)**
   ```bash
   python book_to_audio.py book.epub -o book.mp3 --multi-voice --scene-context
   ```
   The bare CLI still defaults to the legacy regex heuristic; the web UI defaults
   to scene because it is the best measured offline engine on PDNC. It uses
   paragraph/scene turn-taking, action beats, vocatives, and presence cues.

2. **Heuristic (legacy, fastest)**
   ```bash
   python book_to_audio.py book.epub -o book.mp3 --multi-voice
   ```

3. **Whole-book HMM**
   ```bash
   python book_to_audio.py book.epub -o book.mp3 --multi-voice --whole-book-context
   ```

4. **Lightweight local transformer** (`pip install transformers torch`)
   ```bash
   python book_to_audio.py book.epub -o book.mp3 --multi-voice --light-model
   ```

5. **Scene + LLM adjudication** (Colab GPU or a local Ollama)
   ```bash
   ollama pull qwen3:14b
   python book_to_audio.py book.epub -o book.mp3 --multi-voice --heavy-model \
       --llm-model qwen3:14b --llm-strategy weak
   ```
   This keeps the deterministic scene engine as the base. The model only:
   - reconciles the cast (split identities, junk names, gender, first-person narrator), and
   - re-asks the uncertain fallback lines (`scene-single`, `pronoun`, `action-beat`, ...)
     with the full scene and a **closed** cast list.

   Explicit `"X said"` tags are never overridden. Every response is cached in
   `runs/llm_cache/` (or `--llm-cache-dir`), so a rerun/Colab reconnect resumes.
   `--llm-strategy cast` does the cast pass only; `--llm-strategy all` is an ablation.
   The OpenAI-compatible `/v1/chat/completions` shape is also supported; point
   `--llm-endpoint` at it and set `LLM_API_KEY` / `OPENAI_API_KEY` if needed.

### Colab: generate a cast, render locally

The free T4 route is the recommended one for this CPU-only machine. There are two
workflows; both keep rendering local.

**A. Live remote Ollama (no cast import)**

1. Put the EPUB in `MyDrive/book_to_audio/books/` and open `book_to_audio_colab.ipynb`.
2. Set `MODE = "analyze_only_heavy"`, `LLM_MODEL = "qwen3:14b"`, and
   `EXPOSE_OLLAMA = True`, then run steps 1–4b. Step **4b** prints a free
   `https://….trycloudflare.com/api/generate` URL. **Skip step 5** in this workflow.
3. In the local UI, set **Attribution engine → heavy**, paste that URL into
   **LLM endpoint**, and use the same model tag. Press **Analyze cast**.
   The analysis runs in **Activity** against the Colab GPU; the UI stays responsive
   and there is no 15-minute subprocess timeout. When it finishes, review the cast
   and render locally.
4. Keep the Colab runtime and tunnel alive while the analysis runs. The response
   cache defaults to the local `runs/llm_cache/`, so a dropped tunnel does not
   throw away completed prompts. The tunnel URL is public and Ollama has no login;
   stop it when done (`pkill -x cloudflared`).

**B. Download the cast (offline after analysis)**

1. Set `MODE = "analyze_only_heavy"` and run steps 1–5 with the tunnel disabled.
   This is text-only (no TTS) and writes `MyDrive/book_to_audio/casts/<book>.cast.json`.
2. Download the small cast JSON, use **Cast & quotes → Import cast JSON** in the local UI,
   then **Validate import → Apply imported cast**.
3. Render locally. The pipeline already receives the imported cast as `--cast-file`.

Apply/importing a new cast invalidates previously rendered audio for the affected
chapters; the next render regenerates them with the new voices.

The notebook also has an optional PDNC evaluation cell (six novels, text-only) so the
LLM change is measured before it becomes trusted. The LLM engine is **not yet
validated with a real model in this repo**; the scene engine remains the UI
default and the only benchmark-proven path until that cell clears the gate.

Preview the cast without synthesizing:

```bash
python book_to_audio.py book.epub --analyze-only --multi-voice
# or, on a machine with Ollama:
python book_to_audio.py book.epub --analyze-only --multi-voice --heavy-model \
    --llm-model qwen3:14b --cast-out /tmp/book.cast.json
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
  identifies the speaker, as does an embedded introduction (`"Hi, I'm Willem."`).
- Unnamed people get **gender-matched "minor" voices** instead of one generic Unknown voice; this includes a character
  whose first lines are only `he says` / `she says` before the book names them.
- The scene engine ignores a name that is merely mentioned several sentences away, is the object of a preposition
  (`hit on Melanie`), or is a one-word quoted fragment (`Louise.`). That keeps a two-person conversation from
  picking up false third participants.
- `--light-model` / `--heavy-model` supply extra names to this model (cached after the first run).
- Check the result without making audio: `--analyze-only --multi-voice` (add `--whole-book-context`, `--light-model` or `--heavy-model`).
