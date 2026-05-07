#!/usr/bin/env python3
"""Test the claim: '100 steps is a good cross-framework cap because it acts as
an appropriate ceiling for early frameworks while demonstrating a good signal
for which models deserve more in challenging frameworks. In early frameworks,
an excess of steps simply allows the model to exhaustively try all
possibilities until success rather than smartly explore the environment.'

We test three sub-claims using all 400-step transcripts in transcripts_slim/:
  (A) 100 is an appropriate ceiling for early frameworks
      → measure saturation:  best_at_100 / best_at_400  per (model, framework)
  (B) Excess steps just enable brute-force trial-and-error in early frameworks
      → for runs that DO improve past step 100, measure how 'brute-force' the
        late behavior looks: action repetition rate, observation novelty rate
  (C) 100-step performance signals which models deserve more steps in
      challenging frameworks
      → correlate per-model gain in jericho (challenging) vs %-saturated at
        step 100 in early frameworks

Frameworks are tiered as follows (consistent with TALES grouping):
  early       : alfworld, textworld, textworld_express  (synthetic, narrow)
  mid         : scienceworld                            (synthetic, broader)
  challenging : jericho                                 (human-written IF)
"""

from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

EPSILON = 1e-9
SLIM_ROOT = Path(__file__).resolve().parents[2] / "transcripts_slim"

EARLY = {"alfworld", "textworld", "textworld_express"}
MID = {"scienceworld"}
HARD = {"jericho"}

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


def first_step_to_value(curve: list[float], target: float) -> int | None:
    for i, v in enumerate(curve):
        if v >= target - EPSILON:
            return i + 1
    return None


def actions_and_obs_by_step(transcript: dict) -> tuple[dict[int, str], dict[int, str]]:
    """Return {step: agent_action_text} and {step: env_observation_text}."""
    actions: dict[int, str] = {}
    obs: dict[int, str] = {}
    for turn in transcript.get("turns", []):
        step = turn.get("step")
        if not isinstance(step, int):
            continue
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == "agent":
            actions[step] = content.strip().lower()
        elif role == "environment":
            obs[step] = content.strip()
    return actions, obs


def unique_ratio(items: list[str]) -> float:
    if not items:
        return 0.0
    return len(set(items)) / len(items)


def per_run_metrics(d: dict) -> dict:
    curve = best_so_far_curve(d, 400)
    actions, obs = actions_and_obs_by_step(d)

    # Window [1..100], [101..400]
    early_actions = [actions[s] for s in range(1, 101) if s in actions]
    late_actions = [actions[s] for s in range(101, 401) if s in actions]
    early_obs = [obs[s] for s in range(1, 101) if s in obs]
    late_obs = [obs[s] for s in range(101, 401) if s in obs]

    last_step_with_action = max(actions.keys()) if actions else 0

    final_best = curve[-1]
    best100 = curve[99]
    saturation = (best100 / final_best) if final_best > EPSILON else 1.0  # 1.0 means already saturated

    return {
        "best_at_100": best100,
        "best_at_400": curve[-1],
        "gain_after_100": curve[-1] - best100,
        "any_gain_after_100": curve[-1] > best100 + EPSILON,
        "saturation_at_100": saturation,
        "first_step_final_best": first_step_to_value(curve, curve[-1]) or 400,
        # behavior in late window
        "n_actions_early": len(early_actions),
        "n_actions_late": len(late_actions),
        "uniq_action_ratio_early": unique_ratio(early_actions),
        "uniq_action_ratio_late": unique_ratio(late_actions),
        "uniq_obs_ratio_early": unique_ratio(early_obs),
        "uniq_obs_ratio_late": unique_ratio(late_obs),
        "last_step": last_step_with_action,
        "ran_full_400": last_step_with_action >= 400,
    }


def fmean(xs):
    return statistics.fmean(xs) if xs else 0.0


def fmean_key(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return fmean(vals)


def section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(SLIM_ROOT / "*" / "*_n400_*.json"))):
        d = json.load(open(path))
        model = d.get("model") or d.get("metadata", {}).get("llm") or "?"
        fw_dir = Path(path).parent.name
        framework = FW_DIR_TO_NAME.get(fw_dir, fw_dir)
        m = per_run_metrics(d)
        m["model"] = model
        m["framework"] = framework
        m["tier"] = "early" if framework in EARLY else "mid" if framework in MID else "hard"
        rows.append(m)

    if not rows:
        print("No 400-step transcripts found.")
        return

    print(f"Loaded {len(rows)} 400-step runs.\n")

    # --- (A) Saturation at 100 by tier and (model, framework) ---
    section("(A) Saturation at step 100 — best_at_100 / best_at_400")
    print("If close to 1.0, the cap captures essentially all the score the run will ever reach.")
    print("Tier average (across all runs):")
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    for tier in ("early", "mid", "hard"):
        rs = by_tier[tier]
        if not rs:
            continue
        sat = fmean([r["saturation_at_100"] for r in rs])
        any_gain = fmean([1.0 if r["any_gain_after_100"] else 0.0 for r in rs])
        avg_gain = fmean([r["gain_after_100"] for r in rs])
        print(
            f"  {tier:<5}  N={len(rs):4d}  saturation_at_100={sat:.3f}  "
            f"avg_gain_after_100={avg_gain:+.3f}  pct_runs_with_any_gain_after_100={any_gain*100:5.1f}%"
        )

    print("\nPer (model, framework):")
    print(
        f"  {'model':<22} {'framework':<20} {'N':>3}  "
        f"{'sat@100':>8}  {'best@100':>8}  {'best@400':>8}  {'gain':>7}  {'%ran400':>8}  {'%anyGain>100':>12}"
    )
    print("  " + "-" * 102)
    by_mf: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_mf[(r["model"], r["framework"])].append(r)
    for (m, fw) in sorted(by_mf):
        rs = by_mf[(m, fw)]
        sat = fmean([r["saturation_at_100"] for r in rs])
        b100 = fmean([r["best_at_100"] for r in rs])
        b400 = fmean([r["best_at_400"] for r in rs])
        ran = fmean([1.0 if r["ran_full_400"] else 0.0 for r in rs])
        anygain = fmean([1.0 if r["any_gain_after_100"] else 0.0 for r in rs])
        print(
            f"  {m:<22} {fw:<20} {len(rs):3d}  "
            f"{sat:8.3f}  {b100:8.3f}  {b400:8.3f}  {b400-b100:+7.3f}  {ran*100:7.1f}%  {anygain*100:11.1f}%"
        )

    # --- (B) Smart-exploration vs brute-force in late window ---
    section("(B) When agents DO improve past step 100, how does the late window look?")
    print("uniq_action_ratio_late = unique-actions / total-actions in steps 101-400.")
    print("Lower ratio = more action repetition = more 'brute-force trial-and-error'.")
    print("uniq_obs_ratio_late  = unique-observations / total-observations in steps 101-400.")
    print("Lower ratio = revisiting the same states.\n")

    print(
        f"  {'tier':<6} {'N_runs_with_late_gain':>22}  "
        f"{'uniqA early':>12} {'uniqA late':>11}  {'uniqO early':>12} {'uniqO late':>11}  {'avg_gain':>9}"
    )
    print("  " + "-" * 100)
    for tier in ("early", "mid", "hard"):
        late_gainers = [r for r in by_tier[tier] if r["any_gain_after_100"]]
        if not late_gainers:
            print(f"  {tier:<6}  (no runs with late gain)")
            continue
        ua_e = fmean([r["uniq_action_ratio_early"] for r in late_gainers])
        ua_l = fmean([r["uniq_action_ratio_late"] for r in late_gainers])
        uo_e = fmean([r["uniq_obs_ratio_early"] for r in late_gainers])
        uo_l = fmean([r["uniq_obs_ratio_late"] for r in late_gainers])
        gain = fmean([r["gain_after_100"] for r in late_gainers])
        print(
            f"  {tier:<6} {len(late_gainers):22d}  "
            f"{ua_e:12.3f} {ua_l:11.3f}  {uo_e:12.3f} {uo_l:11.3f}  {gain:+9.3f}"
        )

    print("\nSame breakdown for ALL runs (gain or no gain), to show base rates:")
    print(
        f"  {'tier':<6} {'N_runs':>8}  "
        f"{'uniqA early':>12} {'uniqA late':>11}  {'uniqO early':>12} {'uniqO late':>11}"
    )
    print("  " + "-" * 78)
    for tier in ("early", "mid", "hard"):
        rs = [r for r in by_tier[tier] if r["n_actions_late"] > 0]
        if not rs:
            continue
        ua_e = fmean([r["uniq_action_ratio_early"] for r in rs])
        ua_l = fmean([r["uniq_action_ratio_late"] for r in rs])
        uo_e = fmean([r["uniq_obs_ratio_early"] for r in rs])
        uo_l = fmean([r["uniq_obs_ratio_late"] for r in rs])
        print(
            f"  {tier:<6} {len(rs):8d}  "
            f"{ua_e:12.3f} {ua_l:11.3f}  {uo_e:12.3f} {uo_l:11.3f}"
        )

    # --- (B2) Did the cap really only enable brute-force? Were there models that improved a lot? ---
    section("(B2) Per (model, framework, tier) view of late-window behavior, conditional on late gain")
    print(
        f"  {'model':<22} {'framework':<20} {'N_gain':>6}  "
        f"{'uniqA late':>11}  {'uniqO late':>11}  {'avg_gain':>9}"
    )
    print("  " + "-" * 80)
    for (m, fw) in sorted(by_mf):
        rs = [r for r in by_mf[(m, fw)] if r["any_gain_after_100"] and r["n_actions_late"] > 0]
        if not rs:
            continue
        ua_l = fmean([r["uniq_action_ratio_late"] for r in rs])
        uo_l = fmean([r["uniq_obs_ratio_late"] for r in rs])
        gain = fmean([r["gain_after_100"] for r in rs])
        print(
            f"  {m:<22} {fw:<20} {len(rs):6d}  "
            f"{ua_l:11.3f}  {uo_l:11.3f}  {gain:+9.3f}"
        )

    # --- (B3) Among early-tier runs, how many improved past 100 at all? ---
    section("(B3) Late-gain rates in early-tier frameworks (does the cap actually 'cap' anything?)")
    print("If excess steps mostly let agents brute-force successes, we'd expect a non-trivial")
    print("rate of late gains. If the rate is near zero, the early-tier cap claim is fine on its")
    print("own merits — there's nothing to brute-force toward.\n")
    print(f"  {'model':<22} {'framework':<20} {'N':>4} {'%any_gain_after_100':>20}  {'avg_gain_after_100':>20}")
    print("  " + "-" * 90)
    for (m, fw) in sorted(by_mf):
        if fw not in EARLY:
            continue
        rs = by_mf[(m, fw)]
        any_gain = fmean([1.0 if r["any_gain_after_100"] else 0.0 for r in rs])
        avg_gain = fmean([r["gain_after_100"] for r in rs])
        print(f"  {m:<22} {fw:<20} {len(rs):4d} {any_gain*100:19.1f}%  {avg_gain:+20.3f}")

    # --- (C) Does 100-step performance predict who deserves more in jericho? ---
    section("(C) Does 100-step performance predict who deserves more steps in jericho?")
    print("For each model, compare:")
    print("  - avg best_at_100 (across all 400-step runs in any framework)")
    print("  - avg gain_after_100 in jericho (the challenging tier)")
    print("If the claim holds, the strongest @100 models are also the ones that benefit most from")
    print("extra steps in jericho.\n")
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print(f"  {'model':<22}  {'avg_best@100':>13}  {'jericho_gain_after_100':>23}  {'jericho_sat@100':>16}")
    print("  " + "-" * 85)
    summary = []
    for m in sorted(by_model):
        rs = by_model[m]
        b100 = fmean([r["best_at_100"] for r in rs])
        jrs = [r for r in rs if r["framework"] == "jericho"]
        jgain = fmean([r["gain_after_100"] for r in jrs]) if jrs else 0.0
        jsat = fmean([r["saturation_at_100"] for r in jrs]) if jrs else 1.0
        summary.append((m, b100, jgain, jsat))
        print(f"  {m:<22}  {b100:13.3f}  {jgain:+23.3f}  {jsat:16.3f}")

    # Ordered ranking by best@100 — does jericho_gain track it?
    print("\nRanked by avg_best@100 (descending):")
    for m, b100, jgain, jsat in sorted(summary, key=lambda x: -x[1]):
        print(f"  {m:<22}  best@100={b100:.3f}   jericho_gain_after_100={jgain:+.3f}")

    # --- Final tally ---
    section("Verdict tally — what the data says about the user's claim")
    early_sat = fmean([r["saturation_at_100"] for r in rows if r["tier"] == "early"])
    early_late_gain_rate = fmean([1.0 if r["any_gain_after_100"] else 0.0 for r in rows if r["tier"] == "early"])
    hard_sat = fmean([r["saturation_at_100"] for r in rows if r["tier"] == "hard"])
    hard_late_gain_rate = fmean([1.0 if r["any_gain_after_100"] else 0.0 for r in rows if r["tier"] == "hard"])

    print(f"  Sub-claim A  (100 is an appropriate ceiling for early frameworks):")
    print(f"    avg saturation_at_100 in early tier = {early_sat:.3f}")
    print(f"    avg saturation_at_100 in hard  tier = {hard_sat:.3f}")
    print(f"    pct early runs with any gain after step 100 = {early_late_gain_rate*100:.1f}%")
    print(f"    pct hard  runs with any gain after step 100 = {hard_late_gain_rate*100:.1f}%")
    early_late_gainers = [r for r in rows if r["tier"] == "early" and r["any_gain_after_100"]]
    hard_late_gainers = [r for r in rows if r["tier"] == "hard" and r["any_gain_after_100"]]
    print(f"\n  Sub-claim B  (excess steps in early frameworks just enable brute-force):")
    if early_late_gainers:
        ua = fmean([r["uniq_action_ratio_late"] for r in early_late_gainers])
        uo = fmean([r["uniq_obs_ratio_late"] for r in early_late_gainers])
        print(f"    early-tier late-gainers: uniq_action_ratio_late={ua:.3f}, uniq_obs_ratio_late={uo:.3f}")
    else:
        print(f"    (no early-tier runs with late gain — claim is vacuous in early tier)")
    if hard_late_gainers:
        ua = fmean([r["uniq_action_ratio_late"] for r in hard_late_gainers])
        uo = fmean([r["uniq_obs_ratio_late"] for r in hard_late_gainers])
        print(f"    hard-tier  late-gainers: uniq_action_ratio_late={ua:.3f}, uniq_obs_ratio_late={uo:.3f}")
    print()
    print(f"  Sub-claim C  (100-step performance signals who deserves more in jericho): see section (C).")


if __name__ == "__main__":
    main()
