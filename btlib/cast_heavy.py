import json
import re
import sys
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .cast import NARRATOR_SENTINEL
from .cast_whole import analyze_book_whole
from .cast_light import _clean_speaker
from .extract import split_dialogue_narration
from .models import Book

DEFAULT_MODEL = "llama3:8b"
DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
BATCH_SIZE = 20          # dialogue lines per request (a whole chapter overflows Ollama's default context)
NUM_CTX = 8192           # context window requested from Ollama
REQUEST_TIMEOUT = 600


def _query_model(prompt: str, model: str, endpoint: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": NUM_CTX},
        "think": False,  # "thinking" models (Qwen3, DeepSeek-R1) would otherwise spend the answer on reasoning text
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res.get("response", "{}")
    except Exception as e:
        raise RuntimeError(f"Failed to query local LLM at {endpoint}: {e}")


def _mark_paragraphs(text: str) -> str:
    """Keep paragraph breaks visible to the model as a pilcrow."""
    return re.sub(r"(\s*\n\s*)+", " \u00b6 ", text).strip()


def _build_batch_prompt(chapter_title: str, segments: List[tuple], indices: List[int]) -> str:
    lines_to_tag = []
    for idx in indices:
        text = segments[idx][1]
        prev_narr = _mark_paragraphs(segments[idx - 1][1][-300:]) if idx > 0 and segments[idx - 1][0] == "narration" else ""
        next_narr = _mark_paragraphs(segments[idx + 1][1][:300]) if idx + 1 < len(segments) and segments[idx + 1][0] == "narration" else ""
        lines_to_tag.append({
            "seg_idx": idx,
            "dialogue": text.strip()[:300],
            "context_before": prev_narr,
            "context_after": next_narr,
        })

    return f"""You are an expert dialogue annotator for the book chapter "{chapter_title}".
For each numbered utterance, identify who speaks it and that person's gender.
Rules: answer with the character's name as written in the text, never a pronoun like "he" or "she".
The symbol \u00b6 marks a paragraph break. A speech tag or action only describes an utterance when no \u00b6 lies between them;
text after a \u00b6 usually describes the NEXT speaker. If the same paragraph names nobody, answer "Unknown".
If the speaker is the first-person narrator, answer "Narrator (I)". If you cannot tell, answer "Unknown".

Return strictly JSON:
{{
  "results": [
    {{"seg_idx": <int>, "speaker": "<Name>", "gender": "male" | "female" | "neutral"}}
  ]
}}

Utterances:
{json.dumps(lines_to_tag, indent=2)}
"""


def _parse_entries(resp_json: str) -> Dict[int, Tuple[str, str]]:
    try:
        parsed = json.loads(resp_json)
    except json.JSONDecodeError:
        return {}
    raw = parsed.get("results", []) if isinstance(parsed, dict) else []
    out = {}
    for e in raw:
        if not isinstance(e, dict):
            continue
        try:
            seg = int(e.get("seg_idx"))
        except (TypeError, ValueError):
            continue
        out[seg] = (str(e.get("speaker", "Unknown")).strip(), str(e.get("gender", "neutral")).lower().strip())
    return out


def _annotate(book: Book, model: str, endpoint: str) -> Dict[Tuple[int, int], Tuple[str, str]]:
    """Ask the LLM about every dialogue line, in small batches. Returns {(chapter, seg_idx): (speaker, gender)}."""
    answers: Dict[Tuple[int, int], Tuple[str, str]] = {}
    hard_failures = successes = 0

    for ch_idx, chapter in enumerate(book.chapters, start=1):
        segments = split_dialogue_narration(chapter.text)
        d_idx = [i for i, (kind, _t) in enumerate(segments) if kind == "dialogue"]
        if not d_idx:
            continue
        print(f"  [Heavy Hitter] Chapter {ch_idx}/{len(book.chapters)} ({len(d_idx)} lines)...", file=sys.stderr)

        for start in range(0, len(d_idx), BATCH_SIZE):
            batch = d_idx[start:start + BATCH_SIZE]
            prompt = _build_batch_prompt(chapter.title, segments, batch)
            entries: Dict[int, Tuple[str, str]] = {}
            for attempt in range(2):
                try:
                    entries = _parse_entries(_query_model(prompt, model, endpoint))
                    successes += 1
                    break
                except RuntimeError as e:
                    if attempt == 1:
                        hard_failures += 1
                        print(f"  ! batch failed: {e}", file=sys.stderr)
                        if successes == 0 and hard_failures >= 2:
                            sys.exit(f"Can't reach the LLM. Is it running and is the model pulled?\n{e}")
            for seg_idx, ans in entries.items():
                answers[(ch_idx, seg_idx)] = ans
    return answers


def analyze_book_heavy(
    book: Book,
    max_characters: int = 12,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    narrator_gender: Optional[str] = None,
) -> dict:
    """Hybrid attribution: explicit 'Name said' tags (regex) win; the LLM answers the rest; whatever it can't name
    falls back to the regex heuristics and turn-taking logic."""
    answers = _annotate(book, model, endpoint)
    stats = Counter()
    gender_votes = Counter()

    def resolver(ch_num, segments, seg_idx, h_name, h_snip, h_method):
        if h_name and h_method in ("tag", "narrator"):
            stats["regex tag"] += 1
            return h_name, h_snip, h_method

        ans = answers.get((ch_num, seg_idx))
        if ans:
            speaker, gender = ans
            if speaker.lower().startswith("narrator"):
                name = NARRATOR_SENTINEL
            else:
                name = _clean_speaker(speaker)
            if name:
                stats["llm"] += 1
                if name != NARRATOR_SENTINEL and gender in ("male", "female"):
                    gender_votes[(name, gender)] += 1
                prev_narr = segments[seg_idx - 1][1][-120:] if seg_idx > 0 and segments[seg_idx - 1][0] == "narration" else ""
                next_narr = segments[seg_idx + 1][1][:120] if seg_idx + 1 < len(segments) and segments[seg_idx + 1][0] == "narration" else ""
                return name, f"{prev_narr} {next_narr}", "heavy-llm"
        stats["fallback"] += 1
        return h_name, h_snip, h_method

    hints = {}
    for (name, g), n in gender_votes.items():  # the LLM's gender answers feed the model's he/she evidence
        hints.setdefault(name, Counter())[g] += n
    data = analyze_book_whole(book, max_characters=max_characters, narrator_gender=narrator_gender,
                              resolver=resolver, gender_hints=hints)

    data["mode"] = f"heavy-llm ({model})"
    print("  [Heavy Hitter] speaker source: " + ", ".join(f"{k}={v}" for k, v in stats.items()), file=sys.stderr)
    return data
