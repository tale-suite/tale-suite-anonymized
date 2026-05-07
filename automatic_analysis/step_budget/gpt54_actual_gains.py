#!/usr/bin/env python3
"""Measure actual score gains in 100-step buckets for gpt-5.4 400-step runs.

For comparison, also computes the same for any other model with 400-step transcripts
present in transcripts_slim (matched by `_n400_` in the filename).

Best-so-far is padded with the run's final value beyond its actual end, so a run that
terminates at step 50 with score 1.0 contributes 1.0 at every later step. This is
correct for "is more budget useful?" — once a run ends naturally, the best-so-far
score is fixed."""

from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

EPSILON = 1e-9
SLIM_ROOT = Path(__file__).resolve().parents[2] / "transcripts_slim"


def best_so_far_curve(transcript: dict, max_steps: int) -> list[float]:
    """Return a length-`max_steps` best-so-far curve, padded with last value."""
    score_by_step: dict[int, float] = {}
    for turn in transcript.get("turns", []):
        step = turn.get("step")
        if not isinstance(step, int) or step < 1 or step > max_steps:
            continue
        score = turn.get("normalized_score_at_step")
        if isinstance(score, (int, float)):
            score_by_step[step] = float(score)
    out: list[float] = []
    cur = 0.0
    for s in range(1, max_steps + 1):
        cur = max(cur, score_by_step.get(s, cur))
        out.append(cur)
    return out


def bucket_metrics(curve: list[float]) -> dict[str, float]:
    return {
        "best_at_100": curve[99],
        "best_at_200": curve[199],
        "best_at_300": curve[299],
        "best_at_400": curve[399],
        "gain_101_200": curve[199] - curve[99],
        "gain_201_300": curve[299] - curve[199],
        "gain_301_400": curve[399] - curve[299],
        "any_gain_after_100": 1.0 if curve[399] > curve[99] + EPSILON else 0.0,
        "any_gain_201_400": 1.0 if curve[399] > curve[199] + EPSILON else 0.0,
        "first_step_to_final_best": next(
            (i + 1 for i, v in enumerate(curve) if v >= curve[-1] - EPSILON),
            len(curve),
        ),
    }


def avg(rs: list[dict], k: str) -> float:
    return statistics.fmean(r[k] for r in rs)


def line(label: str, rs: list[dict]) -> str:
    if not rs:
        return f"{label:<32} {'0':>4}"
    return (
        f"{label:<32} {len(rs):>4} "
        f"{avg(rs,'best_at_100'):6.3f} {avg(rs,'best_at_200'):6.3f} {avg(rs,'best_at_300'):6.3f} {avg(rs,'best_at_400'):6.3f} "
        f"{avg(rs,'gain_101_200'):9.3f} {avg(rs,'gain_201_300'):9.3f} {avg(rs,'gain_301_400'):9.3f} "
        f"{avg(rs,'any_gain_after_100')*100:5.1f}% {avg(rs,'any_gain_201_400')*100:5.1f}% "
        f"{avg(rs,'first_step_to_final_best'):7.1f}"
    )


def main() -> None:
    rows: dict[str, list[dict]] = defaultdict(list)
    fw_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for path in sorted(glob.glob(str(SLIM_ROOT / "*" / "*_n400_*.json"))):
        d = json.load(open(path))
        model = d.get("model") or d.get("metadata", {}).get("llm") or "?"
        framework = d.get("framework") or d.get("metadata", {}).get("framework") or "?"
        m = bucket_metrics(best_so_far_curve(d, 400))
        m["total_steps"] = d.get("total_steps", 0)
        rows[model].append(m)
        fw_rows[(model, framework)].append(m)

    if not rows:
        print("No 400-step transcripts found.")
        return

    header = (
        f"{'Model':<32} {'N':>4} "
        f"{'@100':>6} {'@200':>6} {'@300':>6} {'@400':>6} "
        f"{'g101_200':>9} {'g201_300':>9} {'g301_400':>9} "
        f"{'%>100':>6} {'%>200':>6} {'BestAt':>7}"
    )

    print("Best-so-far normalized score and bucket gains across 400-step runs.")
    print("Curves are padded with the run's final value, so runs that ended early")
    print("contribute their terminal score at every later step.\n")
    print(header)
    print("-" * len(header))
    for model in sorted(rows):
        print(line(model, rows[model]))

    print("\nPer-framework breakdown:\n")
    print(header)
    print("-" * len(header))
    for (model, fw) in sorted(fw_rows):
        print(line(f"{model} / {fw}", fw_rows[(model, fw)]))

    print("\nWhere does each model's 100->400 gain come from? (% of total gain by framework)\n")
    fw_header = f"{'Model':<22} {'TotalGain':>10}  jericho  twx  scienceworld  textworld  alfworld"
    print(fw_header)
    print("-" * len(fw_header))
    for model in sorted(rows):
        totals = {"jericho": 0.0, "textworld_express": 0.0, "scienceworld": 0.0, "textworld": 0.0, "alfworld": 0.0}
        weighted_total = 0.0
        for (m, fw), rs in fw_rows.items():
            if m != model or fw not in totals:
                continue
            avg_gain = avg(rs, "best_at_400") - avg(rs, "best_at_100")
            share = len(rs) / len(rows[model])
            contribution = avg_gain * share
            totals[fw] = contribution
            weighted_total += contribution
        if abs(weighted_total) < EPSILON:
            print(f"{model:<22} {weighted_total:>10.3f}   (no gains)")
            continue
        pcts = {fw: (v / weighted_total * 100 if weighted_total > 0 else 0) for fw, v in totals.items()}
        print(
            f"{model:<22} {weighted_total:>10.3f}   "
            f"{pcts['jericho']:5.1f}%  {pcts['textworld_express']:5.1f}%  "
            f"{pcts['scienceworld']:5.1f}%        {pcts['textworld']:5.1f}%      {pcts['alfworld']:5.1f}%"
        )


if __name__ == "__main__":
    main()
