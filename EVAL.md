# Accuracy evaluation for speaker attribution

This is a self-contained harness for measuring how well the pipeline decides
**who says what**. It is deliberately separate from the runtime code: nothing
under `evaltools/` is imported by `book_to_audio.py` or `btlib/`.

## Corpora

Everything lives under `eval_data/` (git-ignored; re-download with the commands
at the bottom of this file).

| Corpus  | What it gives us | Role |
|---------|------------------|------|
| **PDNC** — Project Dialogism Novel Corpus (LREC 2022) | 28 full novels, every quote with `speaker`, `addressees`, `quoteType`, `referringExpression`; plus `character_info.csv` (aliases + gender) and narrator person/gender | **primary judge** (full-length books, character metadata) |
| **LitBank** (CC-BY-4.0) | ~2,000-word samples of 100 works; per-quote `QUOTE`/`ATTRIB` records with coref speaker ids | **secondary** (out-of-domain style check) |

Caveats worth remembering:

* PDNC and LitBank both draw on public-domain prose that large language models
  were trained on, so any *LLM-based* engine may look better here than it
  really is. Treat PDNC as held-out only in the sense that we tune against it.
* 14 of LitBank's 100 quotation `.ann` files are empty upstream; the loader
  skips documents with no gold quotes.

## What is measured

For each engine, over every book in the selection:

1. **Quote detection** — precision / recall / F1 of the pipeline's
   double-quoted dialogue spans against gold quote spans (matched by IoU ≥ 0.5).
2. **Attribution given detection** — of the quotes that *were* detected, what
   fraction got the right character. This isolates attribution from detection.
3. **Attribution end-to-end** — right character out of *all* gold quotes
   (so a missed quote also counts against you).
4. **Back-and-forth accuracy** — accuracy restricted to `A B A` runs, the exact
   case the current turn-taking heuristic is weak at.
5. **Unknown / unattributed rate** — how often the pipeline punts.
6. **Gender accuracy** — computed per *predicted cluster* (does the character's
   gender match the gold character it maps to), so attribution mistakes don't
   count against it. PDNC only.
7. **Cluster merges / character splits** — clustering health: how many
   predicted clusters contain more than one gold speaker (merges) and how many
   gold speakers are spread across more than one predicted cluster (splits).
8. **Character precision / recall** — did we find the *cast* at all? Precision
   is the fraction of named cast members that map to a real gold character
   (catches spurious characters — place names, adjectives, chapter-heading
   phrases); recall is the fraction of gold speakers with ≥ 5 quotes that at
   least one named cast member covers (catches missed characters).

> **`detect F1` is not character detection.** It only measures whether the
> double-quoted *spans* were found. Cast quality is items 7–8.

Predicted character names are matched to gold main names through the alias
tables in PDNC, with a title-stripping / token-subset fallback. Note that a
gold quote's `speaker` may be an *alias* of the character's main name (PDNC
lists Dorian Gray's main name as "Prince Charming"); the harness resolves the
speaker to the main name before comparing (see `_alias_to_main` in
`evaltools/runner.py`).

Character precision/recall are only meaningful when the cast is not severely
capped: `--max-characters 12` gives ~12 slots against ~20 major characters per
book on PDNC, so recall is bounded above by roughly `slots / majors`.

## Usage

```bash
# sanity check that offset segmentation == the runtime splitter
python run_eval.py --selftest

# one engine on a few books, with a per-book breakdown
python run_eval.py --corpus pdnc --engine heuristic --books AHandfulOfDust DaisyMiller
python run_eval.py --corpus pdnc --engine whole     --books AHandfulOfDust DaisyMiller

# engine vs engine, compact table, saved to JSON
python run_baseline.py --corpus pdnc --limits 8 --engines heuristic whole
python run_baseline.py --corpus litbank --limits 20 --engines heuristic

# machine-readable single run
python run_eval.py --corpus pdnc --engine whole --limit 10 --save-json eval_results/whole.json
```

Engines: `heuristic` (regex), `scene` (scene-structured turn-taking, `btlib/cast_scene.py`),
`whole` (turn-taking HMM), `light` (HF QA model), `heavy` (scene + LLM cast reconciliation
and weak-line adjudication in `btlib/cast_llm.py`; needs Ollama / an OpenAI-compatible
server and reports `SKIPPED` if absent).

The `scene` engine is also available at runtime:

```bash
python book_to_audio.py --analyze-only --multi-voice --scene-context book.epub
```

## Pre-fix baselines (historical)

These are the numbers produced before the scorer fix described below.  Keep
them only for comparison with older notes; use the corrected table for current
work.

`python run_baseline.py --corpus pdnc --engines heuristic scene whole` (all 28 novels)
and the same on LitBank:

| corpus | engine | detect F1 | attr(given) | attr(e2e) | back-and-forth | unknown | gender | charP | charR | wall |
|--------|--------|-----------|-------------|-----------|----------------|---------|--------|-------|-------|------|
| PDNC (28) | heuristic | 0.956 | 0.295 | 0.287 | 0.320 | 0.188 | 0.931 | 0.874 | 0.476 | 7.4 s |
| PDNC (28) | **scene** | 0.956 | **0.456** | **0.444** | **0.428** | **0.074** | 0.931 | 0.874 | 0.476 | 7.6 s |
| PDNC (28) | whole | 0.956 | 0.387 | 0.377 | 0.405 | 0.177 | 0.973 | 0.863 | 0.476 | 36.2 s |
| PDNC (28, cap 40) | heuristic | 0.956 | 0.323 | 0.315 | 0.343 | 0.129 | 0.938 | 0.838 | 0.728 | 7.3 s |
| PDNC (28, cap 40) | **scene** | 0.956 | **0.493** | **0.480** | **0.446** | **0.046** | 0.938 | 0.838 | 0.728 | 7.6 s |
| LitBank | heuristic | 0.877 | 0.211 | 0.182 | 0.230 | 0.480 | — | — | — | 0.4 s |
| LitBank | **scene** | 0.877 | **0.331** | **0.286** | **0.313** | **0.295** | — | — | — | 0.4 s |
| LitBank | whole | 0.877 | 0.258 | 0.223 | 0.288 | 0.463 | — | — | — | 0.5 s |

Character precision/recall are properties of the shared cast layer, so they are
identical across engines. At the operational cap of 12 characters per book,
**87 % of named cast members are real characters**, and **48 % of the major gold
characters are present** under some name — but the cap itself bounds recall
(~20 major characters per book vs. 12 slots), so the cap-free row (40 slots) is
the truer measure of detection: precision 0.838, recall 0.728.

### Corrected scorer (Oct 2026)

The table above predates the scoring fix in `evaltools/runner.py` (both sides of
the alias comparison are now canonicalised, and a token-subset fallback covers
short forms missing from PDNC's alias tables). Re-running the same engines gives:

| PDNC (28) | attr(given) | attr(e2e) | back-and-forth | unknown |
|-----------|-------------|-----------|----------------|---------|
| heuristic | 0.360 | 0.350 | 0.402 | 0.188 |
| **scene** | **0.566** | **0.551** | **0.554** | **0.093** |
| whole | 0.471 | 0.459 | 0.509 | 0.177 |

Saved as `eval_results/corrected_{heuristic,scene,whole}.json`. Use these
numbers for any "did the LLM help?" comparison; the old strict table is kept
only as a historical record. The corrected scorer is now the default.

### Scene + LLM engine

`--engine heavy` now runs the scene engine and then uses a local Ollama (or
OpenAI-compatible) model for two narrow jobs:

1. **Cast reconciliation** — merge split identities, drop non-people, fix
   gender/narrator. One request per book.
2. **Weak-line adjudication** — only the fallback buckets are sent back to the
   model, with the full scene and a closed cast list. Explicit tags are not
   overridden.

Run it on Colab (free T4) with the notebook's `analyze_only_heavy` mode or the
optional PDNC cell. To drive the **local UI** from Colab without downloading a
cast, set `EXPOSE_OLLAMA = True`, run notebook step **4b**, and paste its
`https://….trycloudflare.com/api/generate` URL into the UI's **LLM endpoint**.
`Analyze cast` then runs as a background job in **Activity**. On a local machine
with Ollama:

```bash
python run_eval.py --corpus pdnc --engine heavy --books ARoomWithAView Emma \
    --llm-model qwen3:14b --llm-strategy weak --llm-cache-dir runs/llm_cache
```

`--llm-max-calls N` caps uncached requests for a smoke test. During a render a
failed model call falls back to the scene result, so a listener never loses an
audiobook because Ollama went away. `--analyze-only` and this evaluation harness
are stricter: they raise instead of silently labelling scene numbers as "LLM".

**Status:** the engine and harness are implemented, but no real model run has
been recorded in this repo yet (the development machine has no GPU/Ollama). Do
not treat the LLM path as an accuracy win until the optional Colab PDNC cell
shows it clears the scene baseline. On the notebook's six-book subset, the
current scene baseline is **0.571 attr(given) / 0.547 e2e, gender 0.938,
charP 0.944**; the gate is **+3 points attribution, gender ≥ 0.95, no charP
regression**.

The precision work (v8) rejects three classes of rubbish before they earn a
voice — bare honorifics (`Mr` with no surname), non-persons (places, demonyms,
nationalities, calendar words, spelled-out numbers, `Milsom Street`,
`Misselthwaite Manor`, `New Yorkers`), and names mentioned fewer than four
times that no real speech tag ever vouches for — and names the first-person
narrator (see below). On PDNC precision went 0.548 → 0.810 and attribution
0.398 → 0.414; on LitBank scene attribution 0.304 → 0.313.

## Pronoun tags and paragraph boundaries (v16)

Three tag-extraction fixes, each measured on PDNC before keeping it:

- **Pronoun-only tags.** `"..." she says.` / `"...", said he.` were invisible,
  so the line fell through to alternation and could land on a speaker of the
  wrong gender. They are now their own pass (`method: "pronoun"`), resolved
  against the most recent character of that gender in the scene. The pass runs
  before the action-beat pass (a pronoun tag is stronger evidence than a name
  merely being nearby) but does **not** consume the narration, so it cannot
  starve the pass that discovers characters. Resolving to `minor_m`/`minor_f`
  only happens if the scene has no character of that gender.
- **Paragraph boundaries.** A narration that *starts* a paragraph is the next
  speaker's cue, not a tag for the quote before the break. `"... ?"` followed
  by a blank line and `Barron answers this question.` used to tag the preceding
  question; it now tags the answer that follows.
- **Lead-in interjections.** `Goddammit Barron answers this question.` used to
  be captured whole and rejected as a name. The interjection is stripped
  *before* validation, so `Barron` becomes a character.
- **Self-introductions.** A 2+-word quote that is exactly a cast name
  (`"Amanda Thomas," she says.`) identifies the speaker.

On PDNC this moved scene attribution from 0.384 to **0.456** (e2e 0.374 to
0.444), and cap-40 from 0.414 to **0.493**, with character precision and recall
also up (0.851/0.469 to 0.874/0.476). LitBank moved 0.313 to 0.331. The whole-book
HMM keeps the old tag stream (it has no notion of the pronoun sentinels), so its
row is unchanged apart from the shared cast improvements.

Rejected experiments, kept here so they are not retried blindly: restricting
tags to the sentence touching the quote (helps precision on distant names, but
costs ~0.02 attribution because legitimate beats get dropped), and running the
pronoun pass with narration consumption (starves character discovery; charP
0.851 to 0.818).

## First appearances and duplicate voices (v17)

Willem's first scene in *Just One Day* exposed two separate failure modes:
the scene engine gave his first `he says` lines to Melanie/Allyson, and the
saved cast gave Allyson and Melanie the same Edge voice.  The fixes are:

- **Embedded self-introductions.** `self_introduced_name` now accepts
  `Hi, I'm Willem.` / `My name is Willem.` in addition to a quote that is
  exactly a cast name.  A self-introduction also joins the scene's participant
  list, so the alternation knows the new speaker before narration tags him.
- **Strict action beats for attribution.**  Cast discovery still uses the
  liberal bare-name pass (a mention in the narrator's thoughts helps us learn
  the cast).  Attribution uses a strict variant that only looks at the sentence
  touching the quote, rejects questions, prepositional objects (`hit on
  Melanie`), and one-word fragments (`Louise.`).  This removes the false
  participants that made a two-hander look like a three-hander.
- **Pronoun fallback.**  A `he says` / `she says` tag with no named character
  of that gender in the scene now goes to `minor_m` / `minor_f`, not to a
  named character of the opposite gender.  Adjacent narration can continue a
  recently-heard minor through the same turn (`he opens his arms`), which is
  how Willem is voiced before the book names him.
- **Narrator in one-person scenes.**  A first-person narrator joins a scene
  that names only one other character, so a reply such as `No. It's already
  all arranged.` alternates back to the narrator instead of staying on the
  named character.
- **Unique named voices.**  The cast editor rejects assigning a voice already
  used by another named character.  The narrator and a major character can no
  longer silently collapse into the same voice; use a different voice or pitch
  instead.

Measured on all 28 PDNC novels with the corrected scorer:

| PDNC (28) | attr(given) | attr(e2e) | back-and-forth | unknown | gender | charP | charR |
|-----------|-------------|-----------|----------------|---------|--------|-------|-------|
| scene before v17 | 0.551 | 0.537 | 0.532 | 0.074 | 0.931 | 0.874 | 0.476 |
| **scene after v17** | **0.566** | **0.551** | **0.554** | **0.093** | **0.934** | **0.868** | **0.476** |
| heuristic after v17 | 0.360 | 0.350 | 0.402 | 0.188 | 0.931 | 0.874 | 0.476 |

The heuristic keeps its old pronoun-reuse behavior (the unique-narration rule is
opt-in for the scene engine), so its attribution is unchanged.  Character
precision gives up 0.006 because the stricter attribution surfaces a few more
gender-matched minor speakers; recall and gender accuracy do not regress.  The
scene engine runs the tag pass twice (broad discovery + strict attribution),
which adds about 0.1 s per book.

## Paragraph-scoped tags and missing speech verbs (v18)

The scene engine used to read the last 160 characters of the preceding
narration and the first 160 of the following narration, even when those windows
crossed a blank line.  That made a tag in one paragraph attribute a quote in the
next one: `Willem says there's another station.` in the chapter intro could
claim the first line of dialogue, and `Surprise. And relief.` /blank line/
`And then Willem says...` could claim the line *before* the blank line.  The
scene engine now uses only the paragraph adjacent to the quote: the first
paragraph after it and the last paragraph before it.  A tag in an earlier or
later paragraph no longer bleeds across.

Three related fixes ship with it:

- **Missing speech verbs.**  `tell/tells`, `explain(s)`, `respond(s)`,
  `acknowledge(s)`, `confide(s)`, `prompt(s)`, `press(es)`, `question(s)` and
  `remind(s)` are now tags.  `Willem tells me.` used to fall through to
  alternation; it now names Willem.
- **Contracted narrator/pronoun tags.**  `I'd asked`, `I'd said`, `he'd said`
  and `she'd replied` are recognised.  This matters in flashbacks, where the
  narration often uses past perfect instead of simple past.
- **Verb-before-name needs no subject.**  `I tell Willem.` is the narrator
  speaking *to* Willem, not Willem speaking.  The verb-before-name form is only
  a tag when the verb has no subject of its own (`said Mary`, `then asked
  Mary`).
- **Object mentions are not action beats.**  Possessive and prepositional
  mentions (`with Melanie's family`, `looked at Melanie`) no longer make
  Melanie the speaker of the nearest quote.  Ordinary subject mentions still
  work, because PDNC relies on them.
- **Self-introductions win over Unknown.**  `"Actually, my name is Allyson."`
  now overrides a nearby object mention that had already put the line in the
  Unknown bucket.

The paragraph-scoped context is scene-engine only; the legacy heuristic keeps
the old paragraph-wide window so its score does not move.

Measured on all 28 PDNC novels with the corrected scorer:

| PDNC (28) | attr(given) | attr(e2e) | back-and-forth | unknown | gender | charP | charR |
|-----------|-------------|-----------|----------------|---------|--------|-------|-------|
| scene after v17 | 0.566 | 0.551 | 0.554 | 0.093 | 0.934 | 0.868 | 0.476 |
| **scene after v18** | **0.572** | **0.557** | **0.559** | **0.087** | **0.934** | **0.870** | **0.476** |
| heuristic after v17 | 0.360 | 0.350 | 0.402 | 0.188 | 0.931 | 0.874 | 0.476 |
| heuristic after v18 | 0.359 | 0.349 | 0.399 | 0.187 | 0.931 | 0.871 | 0.478 |

The scene engine gains 0.006 on both attribution measures and 0.005 on
back-and-forth while lowering Unknown.  The heuristic is flat within noise
(-0.001), which is the intended trade: the new context rule is opt-in for the
scene engine.

## Unnamed speakers (v10)

Unresolved lines used to land in one genderless `Unknown` bucket, so an unnamed
speaker described as "he" could be read in the female narrator's voice. They now
get a gender from the narration hugging the quote — a descriptor ("the man
said", "said the older woman") first, then subject pronouns, with the text after
the quote trusted first — and are routed to `Unnamed man (minor)` /
`Unnamed woman (minor)`. Those buckets carry a gender, so they get a matching
voice; only lines with no gender evidence at all stay `Unknown`.

On *Verity*, 23 of the 31 previously `Unknown` lines were male and moved to
`minor_m` (the remaining 4 are genuinely genderless). In the opening scene every
line Lowen does not speak is now male — before, they were all voiced as Lowen.

Minors are appended after the named cast, so they never take a named
character's voice, and `assign_voices` now prefers a voice that is not already
in use — the neutral pool overlaps the male/female pools, which made `Unknown`
sound exactly like the narrator. Scoring is unaffected: the harness counts
`minor_m`/`minor_f` as unknown, so the PDNC/LitBank tables above are unchanged.

## First-person narrators

In a first-person novel the narrator's own lines are tagged `I said` and land in
a `Narrator (I)` pseudo-character. Because other characters *address* the
narrator by name while the narrator says "I", the narrator's name is the one
most widely used as a **vocative** in dialogue and (almost) never as a
speech-tag subject. v8 uses that to name the narrator — recovering `Jake`
(*The Sun Also Rises*), `Watson` (*The Sign of the Four*), `Hastings` (*The
Mysterious Affair at Styles*) and `Lowen` (*Verity*) — and merges any stray
cluster holding the same name into one character. Each character now carries
`person` (`first`/`third`) and `narrator` flags, and inherits the narrator's
gender.

Guards, all of which a false positive has to clear:

| guard | why |
|-------|-----|
| the `I said` cluster recurs ≥ 8 times | a lone "I said" is an aside, not a narrative voice |
| the name is a vocative ≥ 5 times | enough evidence to be the narrator's own name |
| the vocatives span ≥ 3 chapters | a minor character is only addressed locally; the narrator is addressed throughout (*Prascovia*, *The Gambler*: 1 chapter → rejected) |
| used mid-sentence as a proper noun | rejects interjections (`Indeed`, `Yeah`) |
| not title-led | rejects `Madame`, `General`, `Grandmamma` |
| addressed ≥ 2× as often as it tags speech | rejects characters who merely say a lot (`Jeremy` 18 vs 178 tags) |

The `scene` engine beats the whole-book HMM on both corpora at roughly 6× the
speed, and punts far less often. The `whole` engine still resolves gender better
(it casts a wider net of gender evidence).

## Re-downloading the corpora

```bash
cd eval_data
git clone --depth 1 https://github.com/Priya22/project-dialogism-novel-corpus.git pdnc
git clone --depth 1 https://github.com/dbamman/litbank.git litbank
```

## How the alignment works

`evaltools/align.segment_spans` reproduces `btlib.extract.split_dialogue_narration`
byte-for-byte but also returns each segment's character offsets. `--selftest`
asserts the two agree, so the `seg_idx` keys the attribution engine emits can be
mapped straight back onto gold character spans. `evaltools` then greedily
matches gold quotes to detected dialogue spans by IoU and scores them.
