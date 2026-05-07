#!/usr/bin/env python3
"""
Detect Rule Learning (Inductive Reasoning) failures in game transcripts.

Uses the master error message lists to identify when an agent repeats
the exact same (state, action, next_state) tuple — indicating it failed
to learn from identical prior negative feedback.

Also detects cyclical action loops in error regions (period 1-10).

Usage:
    python3 scripts/detect_rule_learning.py                          # all transcripts
    python3 scripts/detect_rule_learning.py --game alfworld          # one game family
    python3 scripts/detect_rule_learning.py --file transcripts_slim/alfworld/foo.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = BASE / "transcripts_slim"
MASTER_DIR = BASE / "error_messages" / "master"
OUTPUT_DIR = BASE / "analysis_outputs" / "rule_learning"


# ─── Error list loading ──────────────────────────────────────────────────

def load_error_set(game: str, task: str, master_dir: Path = MASTER_DIR) -> set:
    """Load the set of error message strings for a given game/task."""
    master_file = master_dir / game / f"{task}.txt"
    if not master_file.exists():
        return set()

    errors = set()
    with open(master_file) as f:
        for line in f:
            line = line.rstrip("\n")
            # Skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            # Format: "  count | message"
            if " | " not in line:
                continue
            _, msg = line.split(" | ", 1)
            # Unescape the stored format
            msg = msg.replace("\\n", "\n").replace("\\t", "\t")
            errors.add(msg)
    return errors


def get_task_from_filename(filename: str) -> str:
    """Extract task prefix from transcript filename.
    e.g. 'TWCookingLevel1-Qwen-Qwen2.5-7B-Instruct_zero-shot_s202411061.json'
    -> 'TWCookingLevel1'
    """
    return filename.split("-")[0]


# ─── Tuple-repeat detection ──────────────────────────────────────────────

def detect_tuple_repeats(turns: list, error_set: set) -> tuple:
    """
    Detect repeated errored actions where, between the first and second
    attempts, the agent did not successfully advance the world.

    An agent turn at index i is flagged as a tuple_repeat of an earlier
    errored turn j iff:
      - both turns produced an environment response in error_set,
      - the action strings are identical,
      - every agent turn in (j, i) also produced an error response.

    Operationally: maintain a map last_errored_at[action] = turn_index.
    On any productive (non-error) agent turn, clear the map — a successful
    action means the agent gained new information and any subsequent
    repeat of a previously errored action is no longer evidence of
    ignored feedback. On an errored agent turn, emit a flag if the
    action is already in the map; then update the map.

    Returns:
        failure_events: list of dicts
        agent_turn_indices: list of all agent turn indices
        error_turn_indices: list of agent turn indices that produced errors
    """
    failure_events = []
    # action -> most recent agent turn index where this action errored,
    # cleared on any productive turn.
    last_errored_at = {}

    agent_turn_indices = []
    error_turn_indices = []

    i = 0
    prev_env = ""  # most recent env text — diagnostic only, not a match key
    while i < len(turns):
        turn = turns[i]
        if turn["role"] == "environment":
            prev_env = turn["content"].strip()
            i += 1
            continue
        if turn["role"] == "agent":
            agent_action = turn["content"].strip()
            agent_turn_idx = i

            next_env = ""
            if i + 1 < len(turns) and turns[i + 1]["role"] == "environment":
                next_env = turns[i + 1]["content"].strip()

            agent_turn_indices.append(agent_turn_idx)

            if next_env in error_set:
                error_turn_indices.append(agent_turn_idx)
                if agent_action in last_errored_at:
                    failure_events.append({
                        "event_type": "tuple_repeat",
                        "turn_index": agent_turn_idx,
                        "first_seen_turn": last_errored_at[agent_action],
                        "state": prev_env,
                        "action": agent_action,
                        "error_response": next_env,
                    })
                last_errored_at[agent_action] = agent_turn_idx
            else:
                # Productive turn: world state may have changed, so prior
                # errored attempts are no longer counterfactually equivalent.
                last_errored_at.clear()

            i += 1
            continue
        i += 1

    return failure_events, agent_turn_indices, error_turn_indices


# ─── Cycle detection ─────────────────────────────────────────────────────

def detect_cycles(turns: list, agent_turn_indices: list, max_period: int = 10) -> list:
    """
    Detect behavioral cycles in (action, response) space.

    A cycle of period p is a contiguous sequence of agent turns whose
    (action, response) pairs repeat verbatim at least twice:
        (a_{i+jp+l}, r_{i+jp+l}) == (a_{i+l}, r_{i+l})
        for all l in [0, p) and successive j >= 1, with k >= 2 complete
        repetitions.

    The error-message catalogue is not consulted: any productive turn
    whose (action, response) pair does not match the established pattern
    naturally ends the cycle, since it introduces a non-matching pair.
    This catches non-error loops such as oscillating navigation between
    rooms ("go north -> kitchen, go south -> bedroom, go north ->
    kitchen, go south -> bedroom, ...").

    Period 1 (single-action repeats) is intentionally excluded — that
    case is the responsibility of pass one (detect_tuple_repeats), which
    flags it only in the strong-signal error context with no productive
    turn between attempts.

    The first p turns of a detected cycle are treated as pattern
    discovery; the remaining (k-1)*p turns are recorded as Rule
    Learning failures. Periods are scanned from shortest to longest,
    and positions already attributed to a detected cycle are masked
    from later periods so each phenomenon is reported once at its
    shortest period.

    Returns list of cycle dicts.
    """
    n = len(agent_turn_indices)
    if n < 4:  # need at least 2 * min_period (period >= 2)
        return []

    # Build (action, response) pairs aligned with agent_turn_indices.
    pairs: list[tuple[str, str]] = []
    for idx in agent_turn_indices:
        action = turns[idx]["content"].strip()
        response = ""
        if idx + 1 < len(turns) and turns[idx + 1]["role"] == "environment":
            response = turns[idx + 1]["content"].strip()
        pairs.append((action, response))

    cycles = []
    covered = set()  # positions within `pairs` already inside a detected cycle

    for period in range(2, max_period + 1):
        if n < 2 * period:
            break
        i = 0
        while i <= n - 2 * period:
            if i in covered:
                i += 1
                continue

            pattern = pairs[i:i + period]

            reps = 1
            j = i + period
            while j + period <= n:
                if pairs[j:j + period] == pattern:
                    reps += 1
                    j += period
                else:
                    break

            if reps >= 2:
                total_len = reps * period
                cycle_start_pos = i
                cycle_end_pos = i + total_len - 1

                positions = set(range(cycle_start_pos, cycle_end_pos + 1))
                if positions.issubset(covered):
                    i += 1
                    continue

                cycle_turn_indices = [
                    agent_turn_indices[k]
                    for k in range(cycle_start_pos, cycle_end_pos + 1)
                ]
                pattern_actions = [p[0] for p in pattern]

                cycles.append({
                    "event_type": "cycle",
                    "turn_range": [cycle_turn_indices[0], cycle_turn_indices[-1]],
                    "period": period,
                    "repetitions": reps,
                    "pattern_actions": pattern_actions,
                    "cycle_turn_indices": cycle_turn_indices,
                    # First pass through the pattern = exploration;
                    # subsequent (k-1)*p positions = failures.
                    "failure_turn_indices": cycle_turn_indices[period:],
                    "steps_wasted": len(cycle_turn_indices) - period,
                })

                covered.update(positions)
                i = cycle_end_pos + 1
            else:
                i += 1

    return cycles


# ─── Process one transcript ──────────────────────────────────────────────

def process_transcript(filepath: Path, game: str, master_dir: Path = MASTER_DIR) -> dict:
    """Process a single transcript and return the annotation dict."""
    with open(filepath) as f:
        data = json.load(f)

    task = get_task_from_filename(filepath.name)
    error_set = load_error_set(game, task, master_dir=master_dir)

    turns = data.get("turns", [])

    # Detect tuple repeats
    failure_events, agent_turn_indices, error_turn_indices = detect_tuple_repeats(
        turns, error_set
    )

    # Detect cycles
    cycles = detect_cycles(turns, agent_turn_indices, max_period=10)

    # Compute summary
    total_agent_turns = len(agent_turn_indices)
    tuple_repeat_count = len(failure_events)

    # Steps wasted by cycles (only the non-first-pass turns)
    cycle_failure_turns = set()
    for c in cycles:
        cycle_failure_turns.update(c["failure_turn_indices"])

    # Steps wasted by tuple repeats
    tuple_repeat_turns = {e["turn_index"] for e in failure_events}

    # Union of all wasted turns
    all_wasted = tuple_repeat_turns | cycle_failure_turns
    total_steps_wasted = len(all_wasted)

    result = {
        "transcript_id": data.get("transcript_id", filepath.stem),
        "framework": data.get("framework", game),
        "game": data.get("game", task),
        "model": data.get("model", ""),
        "total_turns": len(turns),
        "total_agent_turns": total_agent_turns,
        "total_error_turns": len(error_turn_indices),
        "failure_events": failure_events,
        "cycles": cycles,
        "summary": {
            "tuple_repeat_failures": tuple_repeat_count,
            "cycles_detected": len(cycles),
            "total_steps_wasted": total_steps_wasted,
            "pct_agent_turns_wasted": (
                round(total_steps_wasted / total_agent_turns * 100, 1)
                if total_agent_turns > 0 else 0
            ),
        },
    }

    return result


# ─── Batch processing ────────────────────────────────────────────────────

def process_game(game: str, transcripts_dir: Path, output_dir: Path, master_dir: Path):
    """Process all transcripts for a game family."""
    game_dir = transcripts_dir / game
    if not game_dir.exists():
        print(f"  Skipping {game}: directory not found")
        return

    out_dir = output_dir / game
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(game_dir.glob("*.json"))
    total_failures = 0
    total_cycles = 0
    total_transcripts = 0
    transcripts_with_failures = 0

    for filepath in files:
        result = process_transcript(filepath, game, master_dir=master_dir)
        total_transcripts += 1

        n_failures = result["summary"]["tuple_repeat_failures"]
        n_cycles = result["summary"]["cycles_detected"]
        total_failures += n_failures
        total_cycles += n_cycles
        if n_failures > 0 or n_cycles > 0:
            transcripts_with_failures += 1

        # Write output
        out_path = out_dir / f"{filepath.stem}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    pct = (
        round(transcripts_with_failures / total_transcripts * 100, 1)
        if total_transcripts > 0 else 0
    )
    print(
        f"  {game}: {total_transcripts} transcripts, "
        f"{transcripts_with_failures} with failures ({pct}%), "
        f"{total_failures} tuple-repeat events, "
        f"{total_cycles} cycles detected"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Detect Rule Learning failures in game transcripts"
    )
    parser.add_argument("--game", type=str, help="Process only this game family")
    parser.add_argument("--file", type=str, help="Process a single transcript file")
    parser.add_argument("--transcripts-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument("--error-dir", type=Path, default=MASTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.file:
        filepath = Path(args.file)
        if not filepath.is_absolute():
            filepath = BASE / filepath
        # Infer game from path
        game = filepath.parent.name
        result = process_transcript(filepath, game, master_dir=args.error_dir)
        print(json.dumps(result, indent=2))
        return

    games = [args.game] if args.game else sorted(
        d.name for d in args.transcripts_dir.iterdir() if d.is_dir()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for game in games:
        print(f"\n{'=' * 60}")
        print(f"  {game.upper()}")
        print(f"{'=' * 60}")
        process_game(game, args.transcripts_dir, args.output_dir, args.error_dir)


if __name__ == "__main__":
    main()
