"""Metric accumulation and reporting for attribution evaluation."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r)
    return p, r, f1


@dataclass
class Metrics:
    # detection
    det_tp: int = 0
    det_fp: int = 0
    det_fn: int = 0
    # attribution over matched (quote detection given)
    attr_given_correct: int = 0
    attr_given_total: int = 0
    # end-to-end attribution over all gold quotes
    attr_e2e_correct: int = 0
    attr_e2e_total: int = 0
    unknown: int = 0
    # back-and-forth subset (A B A ...)
    bf_correct: int = 0
    bf_total: int = 0
    # gender
    gender_correct: int = 0
    gender_total: int = 0
    # clustering (over matched quotes)
    merge_clusters: int = 0      # predicted clusters spanning >1 gold speaker
    clusters: int = 0
    split_speakers: int = 0      # gold speakers spread over >1 predicted cluster
    speakers: int = 0
    # character detection: did we find the cast?
    cast_named: int = 0          # named (non-Unknown) characters in the cast
    cast_real: int = 0           # ... of which map to a real gold character
    gold_major: int = 0          # gold speakers with >= MAJOR_MIN_QUOTES quotes
    gold_major_covered: int = 0  # ... of which some named cast member covers
    # per-book breakdown
    per_doc: List[tuple] = field(default_factory=list)

    def add(self, other: "Metrics") -> None:
        for f in ("det_tp", "det_fp", "det_fn", "attr_given_correct",
                  "attr_given_total", "attr_e2e_correct", "attr_e2e_total",
                  "unknown", "bf_correct", "bf_total", "gender_correct",
                  "gender_total", "merge_clusters", "clusters", "split_speakers",
                  "speakers", "cast_named", "cast_real", "gold_major",
                  "gold_major_covered"):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def as_dict(self) -> dict:
        p, r, f1 = prf(self.det_tp, self.det_fp, self.det_fn)
        return {
            "quote_detection": {"precision": p, "recall": r, "f1": f1,
                                "tp": self.det_tp, "fp": self.det_fp, "fn": self.det_fn},
            "attribution_given_detection": _safe_div(self.attr_given_correct, self.attr_given_total),
            "attribution_end_to_end": _safe_div(self.attr_e2e_correct, self.attr_e2e_total),
            "attribution_given_correct": self.attr_given_correct,
            "attribution_given_total": self.attr_given_total,
            "attribution_e2e_correct": self.attr_e2e_correct,
            "attribution_e2e_total": self.attr_e2e_total,
            "back_and_forth": _safe_div(self.bf_correct, self.bf_total),
            "back_and_forth_correct": self.bf_correct,
            "back_and_forth_total": self.bf_total,
            "unknown_rate": _safe_div(self.unknown, self.attr_e2e_total),
            "gender_accuracy": _safe_div(self.gender_correct, self.gender_total),
            "gender_correct": self.gender_correct,
            "gender_total": self.gender_total,
            "cluster_merges": self.merge_clusters,
            "clusters": self.clusters,
            "character_splits": self.split_speakers,
            "speakers": self.speakers,
            "cast_named": self.cast_named,
            "cast_real": self.cast_real,
            "character_precision": _safe_div(self.cast_real, self.cast_named),
            "gold_major": self.gold_major,
            "gold_major_covered": self.gold_major_covered,
            "character_recall": _safe_div(self.gold_major_covered, self.gold_major),
        }

    def report(self, label: str = "") -> str:
        p, r, f1 = prf(self.det_tp, self.det_fp, self.det_fn)
        lines = []
        if label:
            lines.append(f"== {label} ==")
        lines.append(f"  quote detection : P={p:.3f} R={r:.3f} F1={f1:.3f} "
                     f"(tp={self.det_tp} fp={self.det_fp} fn={self.det_fn})")
        lines.append(f"  attribution (given detection) : "
                     f"{_safe_div(self.attr_given_correct, self.attr_given_total):.3f} "
                     f"({self.attr_given_correct}/{self.attr_given_total})")
        lines.append(f"  attribution (end-to-end)      : "
                     f"{_safe_div(self.attr_e2e_correct, self.attr_e2e_total):.3f} "
                     f"({self.attr_e2e_correct}/{self.attr_e2e_total})")
        lines.append(f"  back-and-forth accuracy       : "
                     f"{_safe_div(self.bf_correct, self.bf_total):.3f} "
                     f"({self.bf_correct}/{self.bf_total})")
        lines.append(f"  unknown / unattributed rate   : "
                     f"{_safe_div(self.unknown, self.attr_e2e_total):.3f} "
                     f"({self.unknown}/{self.attr_e2e_total})")
        if self.gender_total:
            lines.append(f"  gender accuracy (clusters)    : "
                         f"{_safe_div(self.gender_correct, self.gender_total):.3f} "
                         f"({self.gender_correct}/{self.gender_total})")
        lines.append(f"  cluster merges                : "
                     f"{self.merge_clusters}/{self.clusters} predicted clusters "
                     f"span >1 gold speaker")
        lines.append(f"  character splits              : "
                     f"{self.split_speakers}/{self.speakers} gold speakers "
                     f"span >1 predicted cluster")
        if self.cast_named:
            lines.append(f"  character precision           : "
                         f"{_safe_div(self.cast_real, self.cast_named):.3f} "
                         f"({self.cast_real}/{self.cast_named} named cast members are real characters)")
        if self.gold_major:
            lines.append(f"  character recall (>=5 quotes) : "
                         f"{_safe_div(self.gold_major_covered, self.gold_major):.3f} "
                         f"({self.gold_major_covered}/{self.gold_major} major gold characters found)")
        return "\n".join(lines)
