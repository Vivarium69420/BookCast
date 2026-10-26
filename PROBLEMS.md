# Survey of failure modes (Oct 2026)

Broad sweep over **PDNC** (28 novels, 47,528 matched quotes), **LitBank**
(86 excerpts, 1,522 quotes) and the four EPUBs on disk, run with the scene
engine unless stated.  The point of this pass was to find problem classes the
existing metrics do not show — not to re-rank the engines.

Raw data: `eval_results/survey_{pdnc,litbank}_{scene,whole,heuristic}_cap12.json`
and the per-quote files `eval_results/survey_*_rows_*.jsonl`.

---

## 0. Headline table (PDNC, cap 12, strict scoring)

| engine | exact | bucket (Unknown/minor) | bucket but speaker *is* in cast | wrong-gender lines | named gender = None |
|---|---|---|---|---|---|
| heuristic | 0.295 | 15.9% | 4,529 (9.5%) | 1,647 | — |
| **scene** | **0.456** | **4.1%** | **637 (1.3%)** | **1,918 (4.0%)** | 1,358 (2.9%) |
| whole | 0.387 | 14.0% | 3,993 (8.4%) | 657 | 7,196 (15.1%) |

Scene has the best attribution *and* the lowest unknown rate, but the worst
wrong-gender rate.  Whole's "better gender" is partly bought by giving 15% of
quotes a gender-less character (neutral voice), so its total gender problem is
16.5% of quotes versus scene's 6.9%.

---

## 1. First-person narrators are left without a gender

| book | narrator | quotes on the narrator |
|---|---|---|
| The Sign of the Four | Watson | 35.3% of the book |
| The Gambler | Narrator (I) | 35.4% |
| The Mysterious Affair at Styles | Hastings | 31.3% |

The narrator entry is found and marked `narrator: true`, but `gender` is
`None`, so a third of the dialogue gets the neutral voice.  In *The Gambler*
the cast entry is literally called `Narrator (I)` even though the text names
him Alexis Ivanovitch 266+ times.

`--narrator-gender` exists but defaults to unset, and nothing infers it from
the name once the name is known.  This is the largest *audible* gender failure
by book.

## 2. Mis-gendered characters — 4.0% of all quotes

Top offenders (lines spoken with the wrong-gender voice):

| lines | character | cast | gold | book |
|---|---|---|---|---|
| 617 | Brett | male | F | The Sun Also Rises |
| 218 | Nancy | male | F | Oliver Twist |
| 196 | Madame Olenska | male | F | The Age of Innocence |
| 129 | Darcy | female | M | Pride and Prejudice |
| 115 | Mrs. John Dashwood | male | F | Sense and Sensibility |
| 80 | Willoughby | female | M | Sense and Sensibility |
| 67 | Janey Archer | male | F | The Age of Innocence |
| 66 | Bingley | female | M | Pride and Prejudice |
| 66 | Babs | male | F | A Handful of Dust |
| 64 | Kanga | male | F | Winnie-the-Pooh |
| 62 | Sibyl Vane | male | F | The Picture of Dorian Gray |
| 46 | Thorpe | female | M | Northanger Abbey |
| 45 | Wickham | female | M | Pride and Prejudice |

Causes, in order of size:

* **Pronoun co-occurrence leaks the other participant's gender.**  A narration
  sentence that names two people credits its pronouns to both.  "Nancy" almost
  always appears in a sentence with Bill Sikes → male votes.
* **The first-name lexicon is applied to surnames.**  `name_gender("darcy")`
  returns `female`, so Mr. Darcy is voiced by a woman (129 lines).  The lexicon
  should not fire when the cluster's dominant form is `Mr. X` / `Mrs. X`.
* **Honorific gaps.** `Madame`, `Mademoiselle`, `Lord`, `King`, `Queen`,
  `Prince`, `Princess` (and the abbreviations `Mlle`, `Mme`) are missing from
  `HONORIFICS`, so `Madame Olenska` and `Mlle` are resolved by pronouns.
  Adding them fixes **178 lines** net, including all 196 of Madame Olenska.
* **Unisex names with no lexicon entry** (Brett, Janey, Sibyl, Babs) fall
  through to pronoun evidence and inherit the narrator's gender field.

## 3. Everything with a name tag is fine; the fallback is where it breaks

Scene engine, PDNC, by attribution method (strict / after repairing the
alias resolver, see §6):

| method | quotes | strict | fixed |
|---|---|---|---|
| `tag` (name in the tag) | 7,695 | 0.807 | **0.849** |
| `scene-same-par` | 8,473 | 0.572 | 0.596 |
| `scene-alternate` | 5,718 | 0.439 | 0.470 |
| `pronoun` (`he said`/`she said`) | 6,823 | 0.427 | 0.464 |
| `action-beat` | 3,421 | 0.388 | 0.409 |
| `scene-core-alt` | 6,977 | 0.311 | 0.335 |
| `scene-presence-alt` | 1,642 | 0.296 | 0.328 |
| `scene-recent` | 2,502 | 0.237 | 0.266 |
| `scene-single` | 1,195 | 0.229 | 0.249 |
| `scene-roundrobin` | 499 | 0.204 | 0.216 |

Two of the largest buckets are close to random:

* `scene-single` (1,195 quotes, 0.23) — a scene with one known candidate:
  every quote goes to that candidate even when the real speaker is somebody
  who just walked in (*Just One Day* ch. 8: Willem's first lines are given to
  Allyson and Melanie).
* `scene-recent` (2,502, 0.24) — alternation between the two most recent
  speakers collapses in any scene with three or more participants
  (*Little Women* ch. 1: four sisters, 5/13 correct on `scene-core-alt`).

`pronoun` at 0.46 is also worth attention: 3,900 pronoun-tagged quotes are
wrong, usually because the nearest `he/she` belongs to a different character.

## 4. Addressee mentions are used as speaker evidence

1,182 errors (4.6% of all errors) have the **predicted speaker's name inside
the quote itself**, versus 0.8% of correct predictions — a 6× enrichment.
These are vocatives: the speaker is addressing the person we picked.

```
"That's true, my son."                      → Beaver, should be Mrs Beaver
"Thank you, Mrs Beaver."                    → Beaver, should be Joyce
"Oh, Mr Beaver, I am so sorry to trouble you" → Beaver, should be Mrs Tipping
"Darling, you're being heroic with Beaver." → Beaver, should be Tony Last
```

*A Handful of Dust* is the worst book in the corpus (0.51 lenient) almost
entirely because of this.  The scene engine already has `scene-addressee`
methods, but the tag/mention passes do not discount names in vocative position
(`, Name` or `Name,` at a clause boundary).

## 5. The Unknown bucket hides real misroutes

* scene: 637 quotes (1.3%) routed to `minor_*`/`unknown` although a matching
  character **is** in the cast.
* whole: 3,993 (8.4%); heuristic: 4,529 (9.5%).
* 1,291 quotes (2.7%) go to a bucket because the speaker is genuinely absent
  from the cast.

The metric counts these as "unknown", which is honest for accuracy but hides
the audible effect: the line is read by the wrong minor voice with no warning.

## 6. The benchmark itself is understating accuracy

Two independent scoring problems in `evaltools/runner.py`:

1. **Alias resolver compares a mapped prediction to an unmapped gold string.**
   `resolve()` does `alias_to_main.get(pred) == norm(gold)`.  When the gold
   quote uses an alias (`Lucy`, `Dorian Gray`) whose main name differs
   (`Miss Honeychurch`, `Prince Charming`), a *correct* prediction is scored
   wrong.  Fixing this alone: **+0.028** (0.456 → 0.484).
2. **PDNC's alias tables omit common short forms.**  Jake Barnes' aliases are
   `{_narr, Barnes, Mr. Barnes, Jacob, Jake Barnes}` — no `Jake`.  The engine
   says "Jake", gold says "Jake Barnes", nobody can win.  2,127 further quotes
   (4.5%) are lost this way, only recoverable with a token-subset fallback.

Combined, PDNC scene accuracy is **0.529**, not 0.456.  The "worst books" list
is also distorted: *A Room with a View* is 0.269 strict but **0.516** corrected;
*Dorian Gray* 0.195 → 0.411.  The genuinely bad books are different:
*The Age of Innocence* 0.149 (Mrs. Archer absorbing Newland Archer's lines,
323 quotes), *A Handful of Dust* 0.51 (vocatives).

> **Fixed (Oct 2026).** `evaltools/runner.py` now canonicalises both sides and
> uses a token-subset fallback. Re-measured scene on all 28 PDNC novels:
> **attr(given) 0.551, e2e 0.537, back-and-forth 0.532, unknown 0.074**
> (`eval_results/corrected_scene.json`). The old strict table in `EVAL.md` is
> kept only as a historical record.

Any engine comparison or "did this change help?" measurement made before
fixing the scorer is measuring the scorer as much as the engine.

## 7. Split characters (one person, two voices)

20 PDNC documents have a gold speaker covered by more than one cast id:

```
Mr Thomas Marvel   ×3 ids   The Invisible Man
A high piping voice ×3 ids  The Sign of the Four
Daisy Miller       ×2      Daisy Miller
Mary Crawford      ×2      Mansfield Park
Mr. Wilcox         ×2      Howards End
```

On the EPUBs the same bug shows as **"Mistress Mary"** (22 lines) living
alongside **"Mary"** (839 lines) in *The Secret Garden* — the protagonist
changes voice whenever a character addresses her by the nursery-rhyme nickname.

Also in *The Sign of the Four*: `Holmes` / `Sherlock Holmes` and
`Jones` / `Athelney Jones` are separate entries.

## 8. Junk entities with real dialogue lines

The cast is not filtered for "is this a person":

```
Rajah       13 lines, male   The Secret Garden   (a word for an Indian prince)
New York    —                The Age of Innocence (place name)
Luyden Newland —             The Age of Innocence (garbled merge of two names)
```

`Rajah` is the worst kind: 13 quotes are read by a fabricated character with
its own voice.  In Verity the equivalent entries are `Sasha` (4) and `Barron`
(3), both with no gender.

## 9. EPUB spot-check

| book | casts | problems found |
|---|---|---|
| Verity | 12 + 3 buckets | Sasha (4), Barron (3) ungendered; bucket 1.6% |
| Just One Day | 12 + 3 buckets | Jenn (17) ungendered → male voice; bucket **7.0%**; Willem's entrance in ch. 8 read by Allyson/Melanie |
| Secret Garden | 12 + 3 buckets | Mistress Mary duplicated (22); Rajah invented (13); bucket 2.4% |
| Little Women ch. 1 | 4 | 28/38 = 0.737; 10 errors, all untagged fallback |

*Just One Day* has the highest bucket rate of the four, and it is the only one
where a **main character's entrance** is misattributed in the first minutes.
The text gives two explicit cues the engine ignores:

```
"That's a very confused breakfast."        → heard as Allyson, is Willem
That voice. ... I look up, and he's right there.
"In so many ways," he says, sitting down.  → heard as Melanie, is Willem
```

---

## Priority (audible impact × fixability)

1. **Narrator gender** — up to 35% of a first-person book; one inference rule.
2. **Gender resolver** — 4.0% wrong + 2.9% unknown; honorific list is a
   one-line fix (178 lines), surname/lexicon guard is the next.
3. **Scoring repair** — without it every future measurement is 7 points off
   and improvements are invisible.
4. **Scene-entry cue** — a participant introduced in the narration
   (`he's right there`, `X walked in`) should get a candidate slot; fixes
   `scene-single` (1,195 quotes, 0.23) and the Willem case.
5. **Vocative discount** — 1,182 errors, concentrated in dialogue-heavy books.
6. **Split/junk cast cleanup** — 20 split speakers, a handful of junk entities,
   but each one is a whole voice that is wrong for many lines.
