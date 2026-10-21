#!/usr/bin/env python3
"""Run several attribution engines over the same books and print a baseline table.

Example:
    python run_baseline.py --corpus pdnc --limits 8 --engines heuristic whole
    python run_baseline.py --corpus litbank --limits 20 --engines heuristic whole

The whole/light/heavy engines load heavier dependencies, so they are only
attempted when requested; a missing dependency is reported, not fatal.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from evaltools.corpora import load_litbank, load_pdnc
from evaltools.runner import evaluate


def _load(corpus: str, data_root: Path, limit, books):
    if corpus == "pdnc":
        docs = load_pdnc(data_root / "pdnc")
        if books:
            docs = [d for d in docs if d.doc_id in set(books)]
    else:
        docs = load_litbank(data_root / "litbank", ids=books)
    docs = [d for d in docs if d.quotes]
    return docs[:limit] if limit else docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["pdnc", "litbank"], default="pdnc")
    ap.add_argument("--data", default="eval_data")
    ap.add_argument("--engines", nargs="*", default=["heuristic", "whole"])
    ap.add_argument("--limits", type=int, default=None, help="cap number of books")
    ap.add_argument("--books", nargs="*", default=None)
    ap.add_argument("--max-characters", type=int, default=12)
    ap.add_argument("--out", default="eval_results/baseline.json")
    args = ap.parse_args()

    data_root = Path(args.data)
    docs = _load(args.corpus, data_root, args.limits, args.books)
    if not docs:
        print("No documents.", file=sys.stderr)
        return 2
    quotes = sum(len(d.quotes) for d in docs)
    print(f"{args.corpus}: {len(docs)} books, {quotes} gold quotes\n", file=sys.stderr)

    rows = []
    payload = {"corpus": args.corpus, "books": [d.doc_id for d in docs], "engines": {}}
    header = (f"{'engine':<12} {'detect F1':>10} {'attr(given)':>12} {'attr(e2e)':>10} "
              f"{'bf':>7} {'unknown':>8} {'gender':>7} {'charP':>7} {'charR':>7} {'wall s':>8}")
    print(header)
    print("-" * len(header))
    for engine in args.engines:
        try:
            t0 = time.time()
            total, _ = evaluate(docs, engine, max_characters=args.max_characters)
            wall = time.time() - t0
        except Exception as exc:  # missing deps / no LLM server etc.
            print(f"{engine:<12} SKIPPED: {type(exc).__name__}: {exc}")
            continue
        d = total.as_dict()
        charp = f"{d['character_precision']:>7.3f}" if d["cast_named"] else f"{'—':>7}"
        charr = f"{d['character_recall']:>7.3f}" if d["gold_major"] else f"{'—':>7}"
        print(f"{engine:<12} "
              f"{d['quote_detection']['f1']:>10.3f} "
              f"{d['attribution_given_detection']:>12.3f} "
              f"{d['attribution_end_to_end']:>10.3f} "
              f"{d['back_and_forth']:>7.3f} "
              f"{d['unknown_rate']:>8.3f} "
              f"{d['gender_accuracy']:>7.3f} "
              f"{charp} "
              f"{charr} "
              f"{wall:>8.1f}")
        payload["engines"][engine] = {"metrics": d, "wall_seconds": wall}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
