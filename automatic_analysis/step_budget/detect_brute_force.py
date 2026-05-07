#!/usr/bin/env python3
"""Programmatic brute-force detector for agent trajectories.

Brute-force in TALES manifests in three concrete patterns visible in transcripts:

  1. Parser-variant explosion: same conceptual action retried with many phrasings
     (`grill pork chop with stove`, `fry pork chop on stove`, `heat pork chop in oven`,
     ...).  Strings differ, but a primary noun/object is shared.

  2. Failure-response saturation: high fraction of agent actions trigger explicit
     parser/game failures ("Unknown action", "Nothing happens", "You can't ...").

  3. Score-stagnation followed by sudden gain: long consecutive run of zero-progress
     actions immediately followed by a score increment, indicating the gain came
     from exhaustive search rather than incremental reasoning.

This file computes all three signals per run and a composite score in [0, 1].

Run with: python3 detect_brute_force.py [--filter SUBSTRING]  to scan
all transcripts under transcripts_slim/.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EPSILON = 1e-9
SLIM_ROOT = Path(__file__).resolve().parents[2] / "transcripts_slim"

# Heuristic markers for parser/game failure responses.  These cover the major
# frameworks (jericho, scienceworld, twx, alfworld, textworld) without false
# positives on legitimate observations.
FAILURE_PATTERNS = re.compile(
    r"\b(?:"
    r"unknown action|i'?m not sure what you mean|"          # twx
    r"nothing happens|that'?s not a verb|"                  # jericho/alfworld
    r"i don'?t know the word|i don'?t understand|"          # jericho
    r"you can'?t (?:see|do|go|use|put|take|reach)|"
    r"you do not see|you cannot|that doesn'?t seem|"
    r"no such action|invalid action|action not (?:recognized|allowed)|"
    r"ambiguous|which (?:do|one) you mean"
    r")\b",
    re.IGNORECASE,
)

# A small stoplist of game-state words that aren't meaningful "primary objects."
STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "from", "with", "of", "for",
    "and", "or", "into", "onto", "by", "is", "be", "it", "this", "that",
    "go", "look", "examine", "take", "drop", "open", "close", "put", "give",
    "use", "read", "north", "south", "east", "west", "up", "down", "all",
    "my", "your", "i", "you", "me", "us", "we", "them", "they", "here", "there",
    "inventory", "help", "verbs", "score", "wait", "again",
}


def extract_primary_object(action: str) -> str:
    """Return the longest content noun (>2 chars, non-stop) in the action.

    For 'grill chopped pork chop with stove' this returns 'chopped' (len 7) or
    'pork' (len 4) — we use the LAST long content word, which empirically
    tracks the action target better than the first."""
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]+", action.lower())
    candidates = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    if not candidates:
        return ""
    # Take the rightmost long content word as the target object
    return candidates[-1]


def first_token(action: str) -> str:
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]+", action.lower())
    return toks[0] if toks else ""


def collect_pairs(transcript: dict) -> list[dict]:
    """Build a list of (step, action, env_response, score, score_delta) records."""
    turns = transcript.get("turns", [])
    out = []
    last_score = 0.0
    for i, t in enumerate(turns):
        if t.get("role") != "agent":
            continue
        step = t.get("step")
        if not isinstance(step, int):
            continue
        action = (t.get("content") or "").strip()
        # Find the next environment turn
        env_resp = ""
        for j in range(i + 1, min(i + 3, len(turns))):
            if turns[j].get("role") == "environment":
                env_resp = (turns[j].get("content") or "").strip()
                break
        score = t.get("normalized_score_at_step")
        score = float(score) if isinstance(score, (int, float)) else last_score
        out.append({
            "step": step,
            "action": action,
            "env": env_resp,
            "score": score,
            "delta": score - last_score,
        })
        last_score = score
    return out


def detect(records: list[dict], window: int = 30) -> dict:
    """Compute brute-force signals over the full record list and a sliding window."""
    if not records:
        return {
            "n_actions": 0,
            "parser_failure_rate": 0.0,
            "verb_explosion_max": 0.0,
            "wasted_streak_max": 0,
            "late_gain_after_stagnation": False,
            "brute_force_score": 0.0,
        }

    # 1. Parser failure rate
    n_fail = sum(1 for r in records if FAILURE_PATTERNS.search(r["env"] or ""))
    parser_failure_rate = n_fail / len(records)

    # 2. Verb explosion: in any rolling window, what is the max fraction of
    #    actions that target the same primary object with distinct phrasings?
    objects = [extract_primary_object(r["action"]) for r in records]
    actions_lower = [r["action"].lower() for r in records]
    verb_explosion_max = 0.0
    explosion_window = None
    if len(records) >= window:
        for start in range(0, len(records) - window + 1):
            slc_obj = objects[start:start + window]
            slc_act = actions_lower[start:start + window]
            counter = Counter(o for o in slc_obj if o)
            if not counter:
                continue
            top_obj, top_n = counter.most_common(1)[0]
            distinct_phrasings = len({a for a, o in zip(slc_act, slc_obj) if o == top_obj})
            score = (top_n / window) * (distinct_phrasings / max(top_n, 1))
            if score > verb_explosion_max:
                verb_explosion_max = score
                explosion_window = (records[start]["step"], records[start + window - 1]["step"], top_obj, distinct_phrasings, top_n)

    # 3. Wasted streak: longest consecutive sequence of zero-delta actions
    wasted_streak_max = 0
    cur = 0
    for r in records:
        if r["delta"] <= EPSILON:
            cur += 1
            wasted_streak_max = max(wasted_streak_max, cur)
        else:
            cur = 0

    # 4. Late-gain-after-stagnation: was the longest stagnation streak followed
    #    by a positive score delta?
    late_gain_after_stagnation = False
    cur = 0
    streak_start = 0
    for i, r in enumerate(records):
        if r["delta"] <= EPSILON:
            if cur == 0:
                streak_start = i
            cur += 1
        else:
            if cur >= 20:
                late_gain_after_stagnation = True
                break
            cur = 0

    # Composite: weighted average of signals each clipped to [0,1]
    sig_failure = min(parser_failure_rate / 0.5, 1.0)        # 50% failure → max
    sig_explosion = min(verb_explosion_max / 0.5, 1.0)       # 50% window-locked → max
    sig_streak = min(wasted_streak_max / 50, 1.0)            # 50-step streak → max
    sig_late_gain = 1.0 if late_gain_after_stagnation else 0.0
    brute_force_score = 0.30 * sig_failure + 0.35 * sig_explosion + 0.20 * sig_streak + 0.15 * sig_late_gain

    return {
        "n_actions": len(records),
        "parser_failure_rate": parser_failure_rate,
        "verb_explosion_max": verb_explosion_max,
        "explosion_window": explosion_window,
        "wasted_streak_max": wasted_streak_max,
        "late_gain_after_stagnation": late_gain_after_stagnation,
        "brute_force_score": brute_force_score,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="", help="Substring filter on transcript path")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--per-run-detail", action="store_true",
                    help="Print one-line per-run breakdown for top 20 brute-force runs.")
    args = ap.parse_args()

    rows: list[dict] = []
    for path in sorted(glob.glob(str(SLIM_ROOT / "*" / "*.json"))):
        if args.filter and args.filter not in path:
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        recs = collect_pairs(d)
        sig = detect(recs, window=args.window)
        sig["path"] = Path(path).name
        sig["model"] = d.get("model", "?")
        sig["framework"] = Path(path).parent.name
        sig["game"] = d.get("game", "?")
        sig["best_score"] = (recs[-1]["score"] if recs else 0.0)
        rows.append(sig)

    if not rows:
        print("No transcripts matched filter.")
        return

    # Aggregate by (model, framework)
    by_mf = defaultdict(list)
    for r in rows:
        by_mf[(r["model"], r["framework"])].append(r)

    print(f"Analyzed {len(rows)} transcripts.\n")
    print(f"{'model':<32} {'framework':<20} {'N':>4} {'BF_score':>9} {'fail%':>6} {'explode':>8} {'streak':>7} {'%lateGain':>10}")
    print("-" * 100)
    for (m, fw), rs in sorted(by_mf.items()):
        bf = statistics.fmean([r["brute_force_score"] for r in rs])
        pf = statistics.fmean([r["parser_failure_rate"] for r in rs])
        ex = statistics.fmean([r["verb_explosion_max"] for r in rs])
        st = statistics.fmean([r["wasted_streak_max"] for r in rs])
        lg = statistics.fmean([1.0 if r["late_gain_after_stagnation"] else 0.0 for r in rs])
        print(f"{m:<32} {fw:<20} {len(rs):>4} {bf:>9.3f} {pf*100:>5.1f}% {ex:>8.3f} {st:>7.1f} {lg*100:>9.1f}%")

    if args.per_run_detail:
        print("\nTop 20 brute-force runs by composite score:")
        for r in sorted(rows, key=lambda x: -x["brute_force_score"])[:20]:
            ew = r["explosion_window"]
            ew_str = f"steps {ew[0]}-{ew[1]} obj={ew[2]!r} ({ew[3]} phrasings of {ew[4]} hits)" if ew else "-"
            print(f"  {r['model']:<26} {r['game']:<28} BF={r['brute_force_score']:.3f}  "
                  f"fail={r['parser_failure_rate']:.2f} explode={r['verb_explosion_max']:.2f} "
                  f"streak={r['wasted_streak_max']}  | {ew_str}")


if __name__ == "__main__":
    main()
