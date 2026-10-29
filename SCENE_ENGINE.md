# Scene-structured attribution engine (`btlib/cast_scene.py`)

A third attribution engine alongside the regex `heuristic` (`btlib/cast.py`) and
the whole-book HMM (`btlib/cast_whole.py`). Select it with `--scene-context`
(or `--scene`) on the CLI, or `--engine scene` in the evaluation harness.

## Why

Untagged dialogue is the hard case, and both older engines resolve it by
turn-taking over the whole chapter. On the public-domain corpora, the scene
engine is both **more accurate** and **much faster** than the HMM:

| corpus | engine | attr(given) | back-and-forth | unknown | wall |
|--------|--------|-------------|----------------|---------|------|
| PDNC (28 novels) | heuristic | 0.247 | 0.263 | 0.222 | 4.8 s |
| PDNC (28 novels) | **scene** | **0.370** | 0.354 | **0.130** | 5.0 s |
| PDNC (28 novels) | whole | 0.352 | 0.358 | 0.232 | 28.1 s |
| LitBank | **scene** | **0.304** | **0.298** | **0.292** | 0.2 s |
| LitBank | whole | 0.260 | 0.288 | 0.450 | 0.4 s |

These are the original pre-scorer-fix numbers.  Current corrected PDNC numbers
and the v17/v18 attribution changes are in `EVAL.md`.

## How it works

1. **Scene segmentation.** A chapter is cut into scenes at long narration gaps
   (`SCENE_GAP`, 500 chars) and typographic scene breaks. Presence is carried
   across a scene boundary when the new scene names nobody.

2. **Presence.** Character ids mentioned in each scene's narration (including
   vocatives) via a book-wide token → character map.

3. **Participants.** Characters actually *tagged as speaking* in the scene,
   ordered by how often, plus anyone who introduces themselves ("Hi, I'm
   Willem"). When a scene has more than two, alternatives are drawn from the
   top two, because the overwhelming majority of scenes are two-handers.  A
   first-person narrator joins a scene that names only one other character, so
   replies alternate back to the narrator instead of staying on that one name.

4. **Paragraph groups.** Consecutive quotes are grouped when the narration
   between them has no blank line. Measured on PDNC, this is a very strong
   signal:

   | gap between consecutive quotes | P(same speaker) | n |
   |--------------------------------|-----------------|---|
   | narration, no blank line (same paragraph) | **96.8%** | 5,730 |
   | narration with a blank line (new paragraph) | 23.1% | 6,732 |
   | directly adjacent quotes (no narration) | **0.7%** | 9,655 |

   So a same-paragraph group is resolved **once** — anchored on any real tag in
   the group — rather than letting one mistake cascade down the paragraph, and
   back-to-back quotes are treated as a turn change (not a continuation).

5. **Constrained turn-taking.** New-paragraph untagged lines alternate among the
   participants (or, failing that, the scene's presence), with vocatives marking
   the addressee so the speaker is someone else.

6. **First appearances.** A pronoun tag with no named character of that gender
   in the scene falls back to `minor_m` / `minor_f` rather than a named
   character of the opposite gender.  If the adjacent narration continues to
   say "he"/"she", a recently-heard minor keeps the turn.  This is how an
   unnamed character ("I think she's right," he says) gets a male voice before
   the book reveals his name.

7. **Paragraph-scoped tags (v18).** The scene engine only lets the paragraph
   adjacent to a quote tag it: the first paragraph after the quote and the last
   paragraph before it.  A name in an earlier or later paragraph no longer
   bleeds across a blank line.  This is scene-engine only; the legacy heuristic
   keeps the old paragraph-wide window so its PDNC score stays flat.

   The same pass also recognises contracted tags (`I'd asked`, `he'd said`),
   more speech verbs (`tell/tells`, `explain(s)`, `respond(s)`, ...), and
   rejects verb-before-name matches that already have a subject (`I tell
   Willem` is the narrator speaking, not Willem).  Possessive and prepositional
   action beats (`with Melanie's family`, `looked at Melanie`) are no longer
   treated as the speaker.

## Shared improvements this work also landed

The tag layer (`btlib/cast.py`) is used by every engine, and two fixes there
lifted all of them:

* snippet windows are snapped to word boundaries (they used to truncate names,
  e.g. `Marilla Cu`);
* names containing a speech verb are rejected, which stops chapter headings
  like *"CHAPTER VII. Anne Says Her Prayers"* from becoming a character.

Attribution on PDNC (28 novels) for the **whole** engine rose 0.352 from 0.31
before these fixes.
