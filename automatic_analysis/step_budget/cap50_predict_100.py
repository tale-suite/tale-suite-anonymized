#!/usr/bin/env python3
"""Cap-at-50 analogue of cap_appropriateness.py.

Question: for the rank-10-15 'middle tier' models (per the existing
beyond-100 priority ranking), can a 50-step cap predict which will show
increased performance with a 100-step cap?

We use the same machinery as the 100→400 analysis, just shifted:
  - saturation_at_50 = best_at_50 / best_at_100   per (model, framework)
  - any_gain_after_50  = pct of runs where best_at_100 > best_at_50
  - smart-vs-brute behavior in the late window 51..100

For reference we include the strong-tier anchors (opus-4.6, sonnet-4.6,
gpt-5.4) and the weak-tier anchors (gpt-5.4-mini, gpt-5.4-nano), so we can
see whether mid-tier models cluster with the strong or the weak.

Frameworks other than jericho terminate in ~10-20 steps for these models, so
the 50→100 question is effectively a jericho question. We still include the
other frameworks for completeness.
"""

from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path

EPSILON = 1e-9
SLIM_ROOT = Path(__file__).resolve().parents[2] / "transcripts_slim"

MID_TIER = [
    "claude-3.5-sonnet-latest",
    "gpt-4.1",
    "gpt-5-mini",
    "o1",
    "o4-mini",
    "meta-llama/Llama-3.1-405B-Instruct",
]
STRONG_REFS = ["claude-opus-4.6", "claude-sonnet-4.6", "gpt-5.4"]
WEAK_REFS = ["gpt-5.4-mini", "gpt-5.4-nano"]
ALL_TARGETS = set(MID_TIER + STRONG_REFS + WEAK_REFS)

FW_DIR_TO_NAME = {
    "alfworld": "alfworld",
    "textworld": "textworld",
    "textworldexpress": "textworld_express",
    "scienceworld": "scienceworld",
    "jericho": "jericho",
}


def best_so_far_curve(transcript: dict, max_steps: int) -> list[float]:
    score_by_step: dict[int, float] = {}
    for turn in transcript.get("turns", []):
        step = turn.get("step")
        if not isinstance(step, int) or step < 1 or step > max_steps:
            continue
        s = turn.get("normalized_score_at_step")
        if isinstance(s, (int, float)):
            score_by_step[step] = float(s)
    out, cur = [], 0.0
    for s in range(1, max_steps + 1):
        cur = max(cur, score_by_step.get(s, cur))
        out.append(cur)
    return out


def actions_by_step(transcript: dict) -> dict[int, str]:
    out = {}
    for t in transcript.get("turns", []):
        s = t.get("step")
        if not isinstance(s, int):
            continue
        if t.get("role") == "agent":
            out[s] = (t.get("content") or "").strip().lower()
    return out


def per_run_metrics(d: dict) -> dict | None:
    total_steps = d.get("total_steps", 0)
    if total_steps < 100:
        return None
    curve = best_so_far_curve(d, 100)
    actions = actions_by_step(d)
    early = [actions[s] for s in range(1, 51) if s in actions]
    late = [actions[s] for s in range(51, 101) if s in actions]
    final_best = curve[-1]
    best50 = curve[49]
    saturation = (best50 / final_best) if final_best > EPSILON else 1.0
    return {
        "best_at_50": best50,
        "best_at_100": curve[-1],
        "gain_after_50": curve[-1] - best50,
        "any_gain_after_50": curve[-1] > best50 + EPSILON,
        "saturation_at_50": saturation,
        "n_actions_late": len(late),
        "uniq_action_ratio_late": len(set(late)) / len(late) if late else 0.0,
        "uniq_action_ratio_early": len(set(early)) / len(early) if early else 0.0,
    }


def fmean(xs):
    return statistics.fmean(xs) if xs else 0.0


def section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    by_mf: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(str(SLIM_ROOT / "*" / "*.json"))):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        m = d.get("model")
        if m not in ALL_TARGETS:
            continue
        fw = FW_DIR_TO_NAME.get(Path(path).parent.name, Path(path).parent.name)
        rec = per_run_metrics(d)
        if rec is None:
            continue
        rec["model"] = m
        rec["framework"] = fw
        by_mf[(m, fw)].append(rec)
        by_model[m].append(rec)

    if not by_model:
        print("No 100-step transcripts for target models.")
        return

    print("Loaded 100-step runs (one row per (model, framework)):")
    print(f"  {'tier':<8} {'model':<40} {'framework':<20} {'N':>4}")
    print("  " + "-" * 74)
    for tier_name, tier_models in (("STRONG", STRONG_REFS), ("MID", MID_TIER), ("WEAK", WEAK_REFS)):
        for m in tier_models:
            for (mm, fw), rs in sorted(by_mf.items()):
                if mm != m:
                    continue
                print(f"  {tier_name:<8} {m:<40} {fw:<20} {len(rs):>4d}")

    section("(A) Saturation at step 50 — best_at_50 / best_at_100 (per model, framework)")
    print(f"  {'tier':<8} {'model':<40} {'framework':<20} {'N':>4} "
          f"{'sat@50':>7} {'best@50':>8} {'best@100':>9} {'gain':>7} {'%anyGain>50':>12}")
    print("  " + "-" * 116)
    for tier_name, tier_models in (("STRONG", STRONG_REFS), ("MID", MID_TIER), ("WEAK", WEAK_REFS)):
        for m in tier_models:
            for (mm, fw), rs in sorted(by_mf.items()):
                if mm != m:
                    continue
                sat = fmean([r["saturation_at_50"] for r in rs])
                b50 = fmean([r["best_at_50"] for r in rs])
                b100 = fmean([r["best_at_100"] for r in rs])
                ag = fmean([1.0 if r["any_gain_after_50"] else 0.0 for r in rs])
                print(f"  {tier_name:<8} {m:<40} {fw:<20} {len(rs):>4d} "
                      f"{sat:7.3f} {b50:8.3f} {b100:9.3f} {b100-b50:+7.3f} {ag*100:11.1f}%")

    section("(B) Cross-tier comparison — JERICHO only (where the 50→100 question lives)")
    print("If the 50-step signal is useful, we expect a clean separation between the strong/mid")
    print("tiers (saturation@50 < 1, real late gains) and the weak tier (saturation@50 ≈ 1, no gain).")
    print()
    print(f"  {'tier':<8} {'model':<40} {'N':>4} {'sat@50':>7} {'gain50→100':>11} {'%anyGain':>10} {'best@50':>8} {'best@100':>9}")
    print("  " + "-" * 110)
    rows_for_pred = []
    for tier_name, tier_models in (("STRONG", STRONG_REFS), ("MID", MID_TIER), ("WEAK", WEAK_REFS)):
        for m in tier_models:
            rs = by_mf.get((m, "jericho"), [])
            if not rs:
                continue
            sat = fmean([r["saturation_at_50"] for r in rs])
            gain = fmean([r["gain_after_50"] for r in rs])
            ag = fmean([1.0 if r["any_gain_after_50"] else 0.0 for r in rs])
            b50 = fmean([r["best_at_50"] for r in rs])
            b100 = fmean([r["best_at_100"] for r in rs])
            rows_for_pred.append((tier_name, m, sat, gain, ag, b50, b100))
            print(f"  {tier_name:<8} {m:<40} {len(rs):>4d} "
                  f"{sat:7.3f} {gain:+11.3f} {ag*100:9.1f}% {b50:8.3f} {b100:9.3f}")

    section("(C) Prediction quality: does saturation@50 (or %anyGain>50) order the models correctly?")
    print("If saturation@50 is high (≈1.0) the prediction is 'no benefit from extending to 100'.")
    print("If saturation@50 is low, the model is using the late budget productively.")
    print()
    print("Models ranked by gain50→100 in jericho (descending — these are the 'true' top extenders):")
    rows_for_pred.sort(key=lambda r: -r[3])
    for i, (tier, m, sat, gain, ag, b50, b100) in enumerate(rows_for_pred, 1):
        print(f"  {i:>2}. {tier:<8} {m:<40} sat@50={sat:.3f}  gain50→100={gain:+.3f}  %anyGain={ag*100:5.1f}%")
    print()
    print("Same models ranked by saturation@50 (ascending — predicted top extenders):")
    rows_for_pred.sort(key=lambda r: r[2])
    for i, (tier, m, sat, gain, ag, b50, b100) in enumerate(rows_for_pred, 1):
        print(f"  {i:>2}. {tier:<8} {m:<40} sat@50={sat:.3f}  gain50→100={gain:+.3f}  %anyGain={ag*100:5.1f}%")

    section("(D) Smart vs brute-force in the 51-100 jericho window (mid-tier focus)")
    print("uniq_action_ratio_late = unique-actions / total-actions in steps 51-100 of each run.")
    print("Conditional on any_gain_after_50.")
    print()
    print(f"  {'tier':<8} {'model':<40} {'N_gainers':>10} {'uniqA early':>11} {'uniqA late':>11} {'uniqOdrop':>10}")
    print("  " + "-" * 96)
    for tier_name, tier_models in (("STRONG", STRONG_REFS), ("MID", MID_TIER), ("WEAK", WEAK_REFS)):
        for m in tier_models:
            rs = [r for r in by_mf.get((m, "jericho"), []) if r["any_gain_after_50"] and r["n_actions_late"] > 0]
            if not rs:
                print(f"  {tier_name:<8} {m:<40} {'(none)':>10}")
                continue
            ua_e = fmean([r["uniq_action_ratio_early"] for r in rs])
            ua_l = fmean([r["uniq_action_ratio_late"] for r in rs])
            print(f"  {tier_name:<8} {m:<40} {len(rs):>10d} {ua_e:>11.3f} {ua_l:>11.3f} {ua_e-ua_l:>10.3f}")


if __name__ == "__main__":
    main()
