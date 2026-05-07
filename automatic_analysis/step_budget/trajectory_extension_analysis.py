#!/usr/bin/env python3
"""Analyze trajectory signals related to beyond-100-step headroom.

This script focuses on two signals from the first 100 agent steps:
- marginal_gain_curve: score gains in fixed step buckets using best-so-far score
- state_novelty_decay: how much observation novelty survives into the tail

It also reports non-Jericho 100-step performance as a descriptive comparison metric.

It uses the failure-analysis JSONs only to find the matching local transcripts and to
avoid double-counting duplicated seeds/configs. For each (framework, game, model, seed),
it keeps the longest available transcript.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

EPSILON = 1e-9
DEFAULT_SIGNALS = (
    "gain_51_75",
    "gain_76_100",
    "late_gain_share",
    "best_score_recency",
    "late_gain_consistency",
    "plateau_then_recover_rate",
    "novelty_decay",
    "new_tail_state_rate",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Summarize score momentum and state novelty for beyond-100 analysis."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing failure-analysis JSON files.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=repo_root / "transcripts_slim",
        help="Primary transcript directory.",
    )
    parser.add_argument(
        "--fallback-transcripts-dir",
        type=Path,
        default=repo_root / "transcripts",
        help="Fallback transcript directory.",
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=100,
        help="Compute trajectory signals through this many agent steps.",
    )
    parser.add_argument(
        "--model-regex",
        help="Optional regex filter applied to model names.",
    )
    parser.add_argument(
        "--positive-models",
        required=True,
        help="Comma-separated list of positive anchor models.",
    )
    parser.add_argument(
        "--negative-models",
        required=True,
        help="Comma-separated list of negative anchor models.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=20,
        help="Only rank models with at least this many runs reaching max-agent-steps.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of non-anchor models to show in the ranking section.",
    )
    return parser.parse_args()


def iter_failure_files(analysis_dir: Path) -> Iterable[Path]:
    for path in sorted(analysis_dir.rglob("*.json")):
        if path.name.startswith("."):
            continue
        yield path


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"['\"`]", "", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def resolve_transcript_path(
    relpath: Path,
    transcripts_dir: Path,
    fallback_dir: Path,
) -> Path | None:
    transcript_path = transcripts_dir / relpath
    if transcript_path.exists():
        return transcript_path
    fallback_path = fallback_dir / relpath
    if fallback_path.exists():
        return fallback_path
    return None


def transcript_identity(data: dict, path: Path) -> tuple[str, str, str, str, int]:
    metadata = data.get("metadata") or {}
    framework = data.get("framework") or metadata.get("framework") or path.parent.name
    game = data.get("game") or metadata.get("game") or path.stem.split("-")[0]
    model = data.get("model") or metadata.get("llm") or "unknown"
    seed = str(metadata.get("seed") or data.get("transcript_id") or path.stem)
    max_step = 0
    for turn in data.get("turns", []):
        step = turn.get("step")
        if isinstance(step, int):
            max_step = max(max_step, step)
    if not max_step:
        total_steps = data.get("total_steps")
        if isinstance(total_steps, int):
            max_step = total_steps
    return framework, game, model, seed, max_step


def select_longest_runs(
    analysis_dir: Path,
    transcripts_dir: Path,
    fallback_dir: Path,
    model_pattern: re.Pattern[str] | None,
) -> list[dict]:
    selected: dict[tuple[str, str, str, str], dict] = {}
    for failure_path in iter_failure_files(analysis_dir):
        with failure_path.open() as handle:
            failure_data = json.load(handle)
        model = failure_data.get("model", "unknown")
        if model_pattern and not model_pattern.search(model):
            continue
        relpath = failure_path.relative_to(analysis_dir)
        transcript_path = resolve_transcript_path(relpath, transcripts_dir, fallback_dir)
        if transcript_path is None:
            continue
        with transcript_path.open() as handle:
            transcript = json.load(handle)
        framework, game, transcript_model, seed, max_step = transcript_identity(transcript, transcript_path)
        key = (framework, game, transcript_model, seed)
        entry = {
            "framework": framework,
            "game": game,
            "model": transcript_model,
            "seed": seed,
            "max_step": max_step,
            "transcript_path": transcript_path,
            "relpath": str(relpath),
        }
        if key not in selected or max_step > selected[key]["max_step"]:
            selected[key] = entry
    return list(selected.values())


def build_step_maps(data: dict, max_steps: int) -> tuple[dict[int, float], dict[int, str], int]:
    score_by_step: dict[int, float] = {}
    obs_by_step: dict[int, str] = {}
    max_seen_step = 0
    for turn in data.get("turns", []):
        step = turn.get("step")
        if not isinstance(step, int) or step < 1:
            continue
        max_seen_step = max(max_seen_step, step)
        if step > max_steps:
            continue
        score = turn.get("normalized_score_at_step")
        if isinstance(score, (int, float)):
            score_by_step[step] = float(score)
        if turn.get("role") == "environment":
            content = turn.get("content") or ""
            if content:
                obs_by_step[step] = content
    return score_by_step, obs_by_step, max_seen_step


def first_step_reaching_final_best(best_scores: list[float]) -> int:
    final_best = best_scores[-1]
    for step, score in enumerate(best_scores, start=1):
        if score >= final_best - EPSILON:
            return step
    return len(best_scores)


def has_plateau_then_recover(
    best_scores: list[float],
    plateau_len: int = 15,
    min_plateau_start: int = 26,
) -> bool:
    max_steps = len(best_scores)
    for start_step in range(min_plateau_start, max_steps - plateau_len + 1):
        start_idx = start_step - 1
        end_idx = start_idx + plateau_len - 1
        plateau_score = best_scores[start_idx]
        if best_scores[end_idx] > plateau_score + EPSILON:
            continue
        if max(best_scores[end_idx + 1 :], default=plateau_score) > plateau_score + EPSILON:
            return True
    return False


def compute_run_metrics(data: dict, max_steps: int) -> dict | None:
    score_by_step, obs_by_step, available_steps = build_step_maps(data, max_steps)
    if available_steps < max_steps:
        return None

    best_scores: list[float] = []
    current_best = 0.0
    for step in range(1, max_steps + 1):
        current_best = max(current_best, score_by_step.get(step, current_best))
        best_scores.append(current_best)

    score_25 = best_scores[24]
    score_50 = best_scores[49]
    score_75 = best_scores[74]
    score_100 = best_scores[99]
    gain_1_25 = score_25
    gain_26_50 = score_50 - score_25
    gain_51_75 = score_75 - score_50
    gain_76_100 = score_100 - score_75
    total_gain = score_100
    late_gain_share = gain_76_100 / total_gain if total_gain > EPSILON else 0.0
    best_score_recency = float(first_step_reaching_final_best(best_scores))
    late_gain_consistency = 1.0 if gain_76_100 > EPSILON else 0.0
    plateau_then_recover_rate = 1.0 if has_plateau_then_recover(best_scores) else 0.0

    early_obs = [normalize_text(obs_by_step.get(step, "")) for step in range(1, 31)]
    early_obs = [obs for obs in early_obs if obs]
    tail_obs = [normalize_text(obs_by_step.get(step, "")) for step in range(71, 101)]
    tail_obs = [obs for obs in tail_obs if obs]
    prefix_obs = [normalize_text(obs_by_step.get(step, "")) for step in range(1, 71)]
    prefix_seen = {obs for obs in prefix_obs if obs}

    early_unique_rate = len(set(early_obs)) / len(early_obs) if early_obs else None
    tail_unique_rate = len(set(tail_obs)) / len(tail_obs) if tail_obs else None
    novelty_decay = None
    if early_unique_rate not in (None, 0) and tail_unique_rate is not None:
        novelty_decay = tail_unique_rate / early_unique_rate
    new_tail_state_rate = None
    if tail_obs:
        new_tail_state_rate = sum(1 for obs in tail_obs if obs not in prefix_seen) / len(tail_obs)

    return {
        "gain_1_25": gain_1_25,
        "gain_26_50": gain_26_50,
        "gain_51_75": gain_51_75,
        "gain_76_100": gain_76_100,
        "total_gain_1_100": total_gain,
        "late_gain_share": late_gain_share,
        "best_score_recency": best_score_recency,
        "late_gain_consistency": late_gain_consistency,
        "plateau_then_recover_rate": plateau_then_recover_rate,
        "novelty_decay": novelty_decay,
        "new_tail_state_rate": new_tail_state_rate,
    }


def summarize_models(selected_runs: list[dict], max_steps: int) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "runs_total": 0,
            "runs_100": 0,
            "non_jericho_runs_100": 0,
            "gain_1_25": [],
            "gain_26_50": [],
            "gain_51_75": [],
            "gain_76_100": [],
            "total_gain_1_100": [],
            "late_gain_share": [],
            "best_score_recency": [],
            "late_gain_consistency": [],
            "plateau_then_recover_rate": [],
            "novelty_decay": [],
            "new_tail_state_rate": [],
            "non_jericho_score_100": [],
        }
    )

    for entry in selected_runs:
        model = entry["model"]
        grouped[model]["runs_total"] += 1
        with entry["transcript_path"].open() as handle:
            transcript = json.load(handle)
        metrics = compute_run_metrics(transcript, max_steps)
        if metrics is None:
            continue
        grouped[model]["runs_100"] += 1
        if entry["framework"] != "jericho":
            grouped[model]["non_jericho_runs_100"] += 1
            grouped[model]["non_jericho_score_100"].append(metrics["total_gain_1_100"])
        for key, value in metrics.items():
            if value is not None:
                grouped[model][key].append(value)

    summary: dict[str, dict[str, float | int | None]] = {}
    for model, acc in grouped.items():
        summary[model] = {
            "runs_total": int(acc["runs_total"]),
            "runs_100": int(acc["runs_100"]),
            "non_jericho_runs_100": int(acc["non_jericho_runs_100"]),
            "non_jericho_score_100": mean_or_none(acc["non_jericho_score_100"]),
            "gain_1_25": mean_or_none(acc["gain_1_25"]),
            "gain_26_50": mean_or_none(acc["gain_26_50"]),
            "gain_51_75": mean_or_none(acc["gain_51_75"]),
            "gain_76_100": mean_or_none(acc["gain_76_100"]),
            "total_gain_1_100": mean_or_none(acc["total_gain_1_100"]),
            "late_gain_share": mean_or_none(acc["late_gain_share"]),
            "best_score_recency": mean_or_none(acc["best_score_recency"]),
            "late_gain_consistency": mean_or_none(acc["late_gain_consistency"]),
            "plateau_then_recover_rate": mean_or_none(acc["plateau_then_recover_rate"]),
            "novelty_decay": mean_or_none(acc["novelty_decay"]),
            "new_tail_state_rate": mean_or_none(acc["new_tail_state_rate"]),
        }
    return summary


def format_value(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def build_priority_rows(
    summary: dict[str, dict[str, float | int | None]],
    positive_models: list[str],
    negative_models: list[str],
    min_runs: int,
) -> list[tuple[float, float, float, str]]:
    usable_models = [
        model
        for model, metrics in summary.items()
        if int(metrics.get("runs_100") or 0) >= min_runs
        and all(metrics.get(signal) is not None for signal in DEFAULT_SIGNALS)
    ]
    if not usable_models:
        return []

    means = {
        signal: statistics.fmean(float(summary[model][signal]) for model in usable_models)
        for signal in DEFAULT_SIGNALS
    }
    stdevs = {
        signal: statistics.pstdev(float(summary[model][signal]) for model in usable_models) or 1.0
        for signal in DEFAULT_SIGNALS
    }

    def zvec(model: str) -> dict[str, float]:
        return {
            signal: (float(summary[model][signal]) - means[signal]) / stdevs[signal]
            for signal in DEFAULT_SIGNALS
        }

    zscores = {model: zvec(model) for model in usable_models}

    def centroid(models: list[str]) -> dict[str, float]:
        return {
            signal: statistics.fmean(zscores[model][signal] for model in models)
            for signal in DEFAULT_SIGNALS
        }

    positive_models = [model for model in positive_models if model in zscores]
    negative_models = [model for model in negative_models if model in zscores]
    if not positive_models or not negative_models:
        return []

    pos_center = centroid(positive_models)
    neg_center = centroid(negative_models)

    def distance(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        return math.sqrt(sum((vec_a[signal] - vec_b[signal]) ** 2 for signal in DEFAULT_SIGNALS))

    rows: list[tuple[float, float, float, str]] = []
    for model in usable_models:
        pos_dist = distance(zscores[model], pos_center)
        neg_dist = distance(zscores[model], neg_center)
        priority = neg_dist - pos_dist
        rows.append((priority, pos_dist, neg_dist, model))
    rows.sort(reverse=True)
    return rows


def print_report(
    summary: dict[str, dict[str, float | int | None]],
    positive_models: list[str],
    negative_models: list[str],
    min_runs: int,
    top_k: int,
) -> str:
    rows = build_priority_rows(summary, positive_models, negative_models, min_runs)
    if not rows:
        return "No models had enough complete data to build a trajectory ranking.\n"

    ranked_non_anchors = [
        row for row in rows if row[3] not in set(positive_models + negative_models)
    ]

    lines: list[str] = []
    lines.append("Trajectory Signal Summary")
    lines.append("")
    lines.append("Signals: marginal gain curve over 1-25 / 26-50 / 51-75 / 76-100, plus state novelty decay.")
    lines.append("NonJer100 = average best-so-far normalized score at 100 on all non-Jericho frameworks.")
    lines.append("BestStep = average first step at which the run reaches its final best score by step 100.")
    lines.append("LateCons = fraction of 100-step runs with any positive gain in steps 76-100.")
    lines.append("PlatRec = fraction of runs that plateau for at least 15 steps after step 25 and then recover before step 100.")
    lines.append("")
    lines.append(
        f"{'Model':<28} {'Runs100':>7} {'NJRuns':>7} {'NonJer100':>9} {'G1_25':>8} {'G26_50':>8} {'G51_75':>8} {'G76_100':>9} {'LateShare':>10} {'BestStep':>9} {'LateCons':>9} {'PlatRec':>8} {'NovDecay':>9} {'NewTail':>8}"
    )
    lines.append("-" * 168)
    for model, metrics in sorted(summary.items()):
        lines.append(
            f"{model:<28} {format_value(metrics['runs_100']):>7} {format_value(metrics['non_jericho_runs_100']):>7} {format_value(metrics['non_jericho_score_100']):>9} "
            f"{format_value(metrics['gain_1_25']):>8} {format_value(metrics['gain_26_50']):>8} {format_value(metrics['gain_51_75']):>8} {format_value(metrics['gain_76_100']):>9} "
            f"{format_value(metrics['late_gain_share']):>10} {format_value(metrics['best_score_recency']):>9} {format_value(metrics['late_gain_consistency']):>9} {format_value(metrics['plateau_then_recover_rate']):>8} "
            f"{format_value(metrics['novelty_decay']):>9} {format_value(metrics['new_tail_state_rate']):>8}"
        )

    lines.append("")
    lines.append("Beyond-100 Trajectory Ranking")
    lines.append("")
    lines.append("Positive anchors: " + ", ".join(positive_models))
    lines.append("Negative anchors: " + ", ".join(negative_models))
    lines.append(
        "PriorityScore = distance_to_negative_centroid - distance_to_positive_centroid over "
        + ", ".join(DEFAULT_SIGNALS)
    )
    lines.append("Higher means the model looks more like the score-upside anchors.")
    lines.append("")
    lines.append(
        f"{'Rank':<4} {'Model':<28} {'Priority':>8} {'PosDist':>8} {'NegDist':>8} {'NonJer100':>9} {'G51_75':>8} {'G76_100':>9} {'BestStep':>9} {'LateCons':>9} {'PlatRec':>8} {'NovDecay':>9} {'NewTail':>8}"
    )
    lines.append("-" * 156)
    for rank, (priority, pos_dist, neg_dist, model) in enumerate(ranked_non_anchors[:top_k], start=1):
        metrics = summary[model]
        lines.append(
            f"{rank:<4} {model:<28} {priority:8.3f} {pos_dist:8.3f} {neg_dist:8.3f} {format_value(metrics['non_jericho_score_100']):>9} "
            f"{format_value(metrics['gain_51_75']):>8} {format_value(metrics['gain_76_100']):>9} {format_value(metrics['best_score_recency']):>9} {format_value(metrics['late_gain_consistency']):>9} {format_value(metrics['plateau_then_recover_rate']):>8} "
            f"{format_value(metrics['novelty_decay']):>9} {format_value(metrics['new_tail_state_rate']):>8}"
        )

    lines.append("")
    lines.append("Anchor Check")
    lines.append("-" * 70)
    lines.append(f"{'Type':<10} {'Model':<28} {'Priority':>8} {'PosDist':>8} {'NegDist':>8}")
    lines.append("-" * 70)
    for label, models in (("positive", positive_models), ("negative", negative_models)):
        for model in models:
            match = next((row for row in rows if row[3] == model), None)
            if match is None:
                continue
            priority, pos_dist, neg_dist, _ = match
            lines.append(
                f"{label:<10} {model:<28} {priority:8.3f} {pos_dist:8.3f} {neg_dist:8.3f}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    model_pattern = re.compile(args.model_regex) if args.model_regex else None
    positive_models = [item.strip() for item in args.positive_models.split(",") if item.strip()]
    negative_models = [item.strip() for item in args.negative_models.split(",") if item.strip()]

    selected_runs = select_longest_runs(
        analysis_dir=args.analysis_dir,
        transcripts_dir=args.transcripts_dir,
        fallback_dir=args.fallback_transcripts_dir,
        model_pattern=model_pattern,
    )
    summary = summarize_models(selected_runs, args.max_agent_steps)
    report = print_report(
        summary=summary,
        positive_models=positive_models,
        negative_models=negative_models,
        min_runs=args.min_runs,
        top_k=args.top_k,
    )
    print(report, end="")


if __name__ == "__main__":
    main()
