import re
import sys
from collections import Counter
from typing import List, Optional, Tuple

from . import cast as _cast
from .cast import NARRATOR_SENTINEL, _valid_name
from .cast_whole import analyze_book_whole
from .models import Book

MODEL_ID = "helliun/quote-attribution"
MAX_ANSWER_TOKENS = 8
MAX_QUOTE_CHARS = 300
MIN_SCORE = 0.15
_NON_NAMES = {"unknown", "none", "n/a", "na", "nobody", "someone", "somebody", "unsure", "unclear", "narrator", "speaker"}
_NAME_WORD_RE = re.compile(r"[A-Z][A-Za-z'’-]*\.?")


def _best_span(start_probs: List[float], end_probs: List[float], offsets: List[Tuple[int, int]],
               seq_ids: List[Optional[int]], max_len: int = MAX_ANSWER_TOKENS) -> Tuple[Optional[Tuple[int, int]], float]:
    """Pick the best (char_start, char_end) answer span inside the context (sequence id 1)."""
    ctx = [i for i, s in enumerate(seq_ids) if s == 1]
    best_score, best_span = 0.0, None
    for a, i in enumerate(ctx):
        for j in ctx[a:a + max_len]:
            score = start_probs[i] * end_probs[j]
            if score > best_score:
                best_score = score
                best_span = (offsets[i][0], offsets[j][1])
    return best_span, best_score


class _QA:
    """Minimal extractive-QA wrapper.

    Replaces transformers.pipeline("question-answering"), which was removed in
    transformers v5, so this works on both old and new versions.
    """

    def __init__(self, model_id: str):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError:
            sys.exit("Missing dependencies for lightweight model.\nInstall with: pip install transformers torch")

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading local model ({model_id}) on {'GPU' if self.device == 'cuda' else 'CPU'}...", file=sys.stderr)

        cfg = AutoConfig.from_pretrained(model_id)
        archs = getattr(cfg, "architectures", None) or []
        if archs and not any("QuestionAnswering" in a for a in archs):
            sys.exit(
                f"{model_id} is not an extractive question-answering model (architectures={archs}), "
                "so --light-model can't use it.\nTry --whole-book-context (no model needed) or --heavy-model instead."
            )

        self.tok = AutoTokenizer.from_pretrained(model_id)
        if not getattr(self.tok, "is_fast", False):
            sys.exit("--light-model needs a fast tokenizer (offset mapping) for this model.")
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_id).to(self.device).eval()

    def answer(self, question: str, context: str) -> Tuple[str, float]:
        enc = self.tok(
            question, context,
            truncation="only_second", max_length=512,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        seq_ids = enc.sequence_ids(0)
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
        start = out.start_logits[0].softmax(-1).cpu().tolist()
        end = out.end_logits[0].softmax(-1).cpu().tolist()
        span, score = _best_span(start, end, offsets, seq_ids)
        if span is None:
            return "", 0.0
        return context[span[0]:span[1]].strip(), score


def _clean_speaker(answer: str) -> Optional[str]:
    """Accept only name-like answers. Pronouns, stop words and fragments ('he', 'The', 'His') return None."""
    ans = answer.strip().strip("\"'“”‘’.,;:!?—-").strip()
    if not ans:
        return None
    if ans == "I":
        return NARRATOR_SENTINEL
    if ans.lower() in _NON_NAMES or ans.lower().startswith(("unknown", "unclear", "not ")):
        return None
    words = ans.split()
    if len(words) > 3 or not all(_NAME_WORD_RE.fullmatch(w) for w in words):
        return None
    return ans if _valid_name(ans) else None


def analyze_book_light(book: Book, max_characters: int = 12, narrator_gender: Optional[str] = None) -> dict:
    """Hybrid attribution: explicit 'Name said' tags from the regex pass win; the QA model fills the
    gaps; anything it can't name falls back to the regex heuristics and turn-taking logic."""
    qa = _QA(MODEL_ID)
    stats = Counter()

    def resolver(ch_num, segments, seg_idx, h_name, h_snip, h_method):
        if h_name and h_method in ("tag", "narrator"):
            stats["regex tag"] += 1
            return h_name, h_snip, h_method

        text = segments[seg_idx][1]
        prev_narr = segments[seg_idx - 1][1] if seg_idx > 0 and segments[seg_idx - 1][0] == "narration" else ""
        next_narr = segments[seg_idx + 1][1] if seg_idx + 1 < len(segments) and segments[seg_idx + 1][0] == "narration" else ""
        if _cast.PARAGRAPH_AWARE:  # a tag in another paragraph belongs to someone else
            prev_narr, next_narr = prev_narr.rsplit("\n", 1)[-1], next_narr.split("\n", 1)[0]
        prev_narr, next_narr = prev_narr[-350:], next_narr[:350]
        context_window = f"{prev_narr} {text} {next_narr}".strip()

        name = None
        if prev_narr.strip() or next_narr.strip():
            try:
                answer, score = qa.answer(f"Who said: {text[:MAX_QUOTE_CHARS]}", context_window)
                if score >= MIN_SCORE:
                    name = _clean_speaker(answer)
            except Exception as e:
                print(f"  [light-model] line failed ({type(e).__name__}: {e})", file=sys.stderr)

        done = sum(stats.values()) + 1
        if done % 100 == 0:
            print(f"  [light-model] {done} dialogue lines processed...", file=sys.stderr)

        if name:
            stats["model"] += 1
            return name, f"{prev_narr} {next_narr}", "lightweight-qa"
        stats["fallback"] += 1
        return h_name, h_snip, h_method

    data = analyze_book_whole(book, max_characters=max_characters, narrator_gender=narrator_gender, resolver=resolver)
    data["mode"] = "lightweight-transformer"
    print("  [light-model] speaker source: " + ", ".join(f"{k}={v}" for k, v in stats.items()), file=sys.stderr)
    return data
