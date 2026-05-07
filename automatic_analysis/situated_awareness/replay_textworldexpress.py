#!/usr/bin/env python3
"""
Replay TextWorldExpress transcripts through the actual environment, extracting
ground-truth world state (admissible commands, look/inventory text) at every step.

Handles seed mismatch: if the transcript's seed doesn't reproduce the same
initial observation, searches test seeds for a match.

Usage:
    python analysis/situated_awareness/replay_textworldexpress.py --file transcripts_slim/textworldexpress/foo.json
    python analysis/situated_awareness/replay_textworldexpress.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import tales  # noqa: F401 — registers envs

BASE = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = BASE / "transcripts_slim" / "textworldexpress"
OUTPUT_DIR = BASE / "analysis_outputs" / "situated_awareness_replays" / "textworldexpress"


def normalize_obs(obs: str) -> str:
    """Normalize observation: collapse whitespace for comparison."""
    return " ".join(obs.split())


def extract_world_state(env, info: dict) -> dict:
    """
    Extract ground-truth world state from TextWorldExpress.

    Returns a dict with:
      - agent_at: str (room name from look text)
      - objects: dict (items visible in current location)
      - inventory: list of inventory items
      - receptacles: dict (empty)
      - admissible_commands: list of valid actions
      - open_receptacles: list (empty)
      - look: str (full look output)
      - inv: str (full inventory output)
    """
    admissible = info.get("admissible_commands", []) or info.get("validActions", []) or []

    look_text = info.get("look", "") or ""
    inv_text = info.get("inventory", "") or ""

    # Parse location from look text
    # TWX format: "You are in the <room>. In one part..."
    agent_at = "unknown"
    if look_text:
        import re
        m = re.match(r"You are in (?:the |a )([^.]+)\.", look_text)
        if m:
            agent_at = m.group(1).strip()

    # Parse inventory
    inventory = []
    if inv_text:
        for line in inv_text.split("\n"):
            line = line.strip()
            if line.startswith("a ") or line.startswith("an ") or line.startswith("the "):
                inventory.append(line)

    # Parse objects from look text
    objects = {}
    if look_text:
        for line in look_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Look for "a <container> that has <N> <items> on it"
            # or "a <item>"
            if stripped.startswith("a ") or stripped.startswith("an ") or stripped.startswith("the "):
                objects[stripped] = agent_at

    return {
        "agent_at": agent_at,
        "objects": objects,
        "inventory": inventory,
        "receptacles": {},
        "admissible_commands": list(admissible),
        "open_receptacles": [],
        "look": look_text,
        "inv": inv_text,
    }


def find_matching_seed(env, transcript_obs_norm: str, max_search: int = 1000) -> int | None:
    """
    Search test seeds to find one matching the transcript's initial obs.
    Returns the seed index or None.
    """
    inner = env.unwrapped
    num_seeds = min(len(inner.seeds), max_search)
    for seed_idx in range(num_seeds):
        obs, _ = env.reset(seed=seed_idx)
        if normalize_obs(obs) == transcript_obs_norm:
            return seed_idx
    return None


def replay_transcript(transcript_path: Path) -> dict:
    """Replay a single transcript through the TextWorldExpress environment."""
    with open(transcript_path) as f:
        transcript = json.load(f)

    game = transcript["game"]
    metadata = transcript.get("metadata", {})
    seed = metadata.get("seed", None)

    if seed is None:
        return {"error": "no seed in metadata", "transcript_id": transcript.get("transcript_id")}

    env_name = f"tales/{game}-v0"
    try:
        env = gym.make(env_name, disable_env_checker=True)
    except Exception as e:
        return {"error": f"failed to create env: {e}", "transcript_id": transcript.get("transcript_id")}

    obs, info = env.reset(seed=seed)

    transcript_turns = transcript.get("turns", [])
    if not transcript_turns or transcript_turns[0]["role"] != "environment":
        env.close()
        return {"error": "transcript has no initial env turn", "transcript_id": transcript.get("transcript_id")}

    transcript_init = normalize_obs(transcript_turns[0]["content"])
    env_init = normalize_obs(obs)

    # If seed doesn't match, search for the correct seed
    actual_seed = seed
    if transcript_init != env_init:
        found_seed = find_matching_seed(env, transcript_init)
        if found_seed is None:
            env.close()
            return {
                "error": "no matching seed found",
                "transcript_id": transcript.get("transcript_id"),
                "expected": transcript_turns[0]["content"].strip(),
                "got": obs.strip(),
            }
        actual_seed = found_seed
        obs, info = env.reset(seed=found_seed)

    # Extract initial state
    steps = []
    initial_state = extract_world_state(env, info)
    steps.append({
        "turn_index": 0,
        "role": "environment",
        "content": obs.strip(),
        "world_state": initial_state,
    })

    # Replay each agent action
    desync = False
    last_info = info
    for i, turn in enumerate(transcript_turns[1:], start=1):
        if turn["role"] == "agent":
            action = turn["content"].strip()

            # Replicate benchmark.py multi-line guard: the harness
            # skips env.step() entirely for actions containing newlines.
            multiline_skip = "\n" in action
            if multiline_skip:
                env_obs = "The game only allows one action per step."
                done = False
            else:
                env_obs, _, done, env_info = env.step(action)
                last_info = env_info

            # Replicate benchmark.py auto-reset on game over
            if done:
                reset_obs, reset_info = env.reset()
                last_info = reset_info

            steps.append({
                "turn_index": i,
                "role": "agent",
                "content": action,
            })

            # Check for desync (skip for multi-line actions)
            if not multiline_skip and i + 1 < len(transcript_turns) and transcript_turns[i + 1]["role"] == "environment":
                transcript_next = normalize_obs(transcript_turns[i + 1]["content"])
                env_next = normalize_obs(env_obs)

                if transcript_next != env_next:
                    desync = True
                    steps.append({
                        "turn_index": i + 1,
                        "role": "environment",
                        "content": env_obs.strip(),
                        "desync": True,
                        "transcript_content": transcript_turns[i + 1]["content"].strip(),
                    })
                    break

        elif turn["role"] == "environment":
            ws = extract_world_state(env, last_info)
            steps.append({
                "turn_index": i,
                "role": "environment",
                "content": turn["content"].strip(),
                "world_state": ws,
            })

    env.close()

    return {
        "transcript_id": transcript.get("transcript_id", transcript_path.stem),
        "game": game,
        "model": transcript.get("model", ""),
        "seed": actual_seed,
        "original_seed": seed,
        "score": transcript.get("score", 0),
        "max_score": transcript.get("max_score", 1),
        "total_turns": len(transcript_turns),
        "replayed_turns": len(steps),
        "desync": desync,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Single transcript file to replay")
    parser.add_argument("--all", action="store_true", help="Replay all textworldexpress transcripts")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = BASE / path
        print(f"Replaying: {path.name}")
        result = replay_transcript(path)
        out_path = OUTPUT_DIR / f"{path.stem}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
        elif result.get("desync"):
            print(f"  DESYNC at turn {result['replayed_turns']}/{result['total_turns']}")
        else:
            print(f"  OK: {result['replayed_turns']} turns replayed (seed={result['seed']})")
        return

    if args.all:
        files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
        ok = desync = errors = 0
        for i, path in enumerate(files):
            if i % 100 == 0:
                print(f"  [{i}/{len(files)}] {path.name[:60]}...")
            result = replay_transcript(path)
            out_path = OUTPUT_DIR / f"{path.stem}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            if result.get("error"):
                errors += 1
            elif result.get("desync"):
                desync += 1
            else:
                ok += 1
        print(f"\nDone: {ok} OK, {desync} desync, {errors} errors out of {len(files)}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
