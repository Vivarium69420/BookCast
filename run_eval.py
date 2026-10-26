#!/usr/bin/env python3
"""Score the speaker-attribution pipeline against public gold corpora.

Examples
--------
    # whole-book engine over all of PDNC
    python run_eval.py --corpus pdnc --engine whole

    # compare engines on a few books
    python run_eval.py --corpus pdnc --engine heuristic --books AHandfulOfDust Emma
    python run_eval.py --corpus pdnc --engine whole     --books AHandfulOfDust Emma

    # sanity-check that our offset segmentation matches the runtime splitter
    python run_eval.py --selftest
"""

import argparse
import sys
import time
from pathlib import Path

from btlib.llm import LLMError
from evaltools.align import verify_segmentation_matches_extract
from evaltools.corpora import Doc, load_litbank, load_pdnc
from evaltools.runner import evaluate


def _selftest() -> int:
    samples = [
        'He said "Hello there." Then left.',
        'She whispered, \u201cWait for me,\u201d and ran.',
        'No quotes at all here.',
        '"One," he said. "Two."',
        '',
        '"only a quote"',
    ]
    ok = all(verify_segmentation_matches_extract(s) for s in samples)
    print("segmentation self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--corpus", choices=["pdnc", "litbank"], default="pdnc")
    ap.add_argument("--data", default="eval_data", help="root holding pdnc/ and litbank/")
    ap.add_argument("--books", nargs="*", help="only these doc ids / folder names")
    ap.add_argument("--engine", choices=["heuristic", "scene", "whole", "light", "heavy"],
                    default="whole")
    ap.add_argument("--max-characters", type=int, default=12)
    ap.add_argument("--narrator-gender", default=None, choices=[None, "M", "F", "A", "U"])
    ap.add_argument("--unknown-below", type=float, default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-endpoint", default=None)
    ap.add_argument("--llm-strategy", choices=["cast", "weak", "all"], default="weak")
    ap.add_argument("--llm-batch-size", type=int, default=8)
    ap.add_argument("--llm-confidence", type=float, default=0.6)
    ap.add_argument("--llm-cache-dir", default=None)
    ap.add_argument("--llm-max-calls", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N books only")
    ap.add_argument("--per-book", action="store_true")
    ap.add_argument("--save-json", default=None, help="write metrics to this JSON file")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    data_root = Path(args.data)
    if args.corpus == "pdnc":
        docs = load_pdnc(data_root / "pdnc")
        if args.books:
            docs = [d for d in docs if d.doc_id in set(args.books)]
    else:
        docs = load_litbank(data_root / "litbank", ids=args.books)

    if args.limit:
        docs = docs[:args.limit]
    empty = [d.doc_id for d in docs if not d.quotes]
    docs = [d for d in docs if d.quotes]
    if empty:
        print(f"Skipped {len(empty)} doc(s) with no gold quotes: {', '.join(empty[:6])}"
              + (" ..." if len(empty) > 6 else ""), file=sys.stderr)
    if not docs:
        print("No documents loaded — check --data / --books.", file=sys.stderr)
        return 2

    total_quotes = sum(len(d.quotes) for d in docs)
    print(f"Loaded {len(docs)} book(s), {total_quotes} gold quotes "
          f"({args.corpus}). Engine = {args.engine}.", file=sys.stderr)

    t0 = time.time()
    try:
        total, per_doc = evaluate(
            docs, args.engine,
            max_characters=args.max_characters,
            narrator_gender=args.narrator_gender,
            unknown_below=args.unknown_below,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_strategy=args.llm_strategy,
            llm_batch_size=args.llm_batch_size,
            llm_confidence=args.llm_confidence,
            llm_cache_dir=args.llm_cache_dir,
            llm_max_calls=args.llm_max_calls,
        )
    except LLMError as exc:
        print(f"SKIPPED {args.engine}: {exc}", file=sys.stderr)
        return 1
    wall = time.time() - t0

    label = f"{args.corpus} / {args.engine} / {len(docs)} books / {wall:.1f}s"
    print()
    print(total.report(label))

    if args.per_book:
        print()
        for doc_id, m, secs in per_doc:
            print(m.report(f"{doc_id} ({secs:.1f}s)"))
            print()

    if args.save_json:
        import json
        payload = {
            "corpus": args.corpus,
            "engine": args.engine,
            "books": [d.doc_id for d in docs],
            "wall_seconds": wall,
            "metrics": total.as_dict(),
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved metrics to {args.save_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
