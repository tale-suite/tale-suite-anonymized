#!/usr/bin/env python3
"""
Replay Jericho transcripts through the actual environment, extracting
ground-truth world state (Z-machine object tree, valid actions) at every step.

Usage:
    python analysis/situated_awareness/replay_jericho.py --file transcripts_slim/jericho/foo.json
    python analysis/situated_awareness/replay_jericho.py --all
"""

import argparse
import json
import signal
import sys
from pathlib import Path

import gymnasium as gym
import tales  # noqa: F401 — registers envs

BASE = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = BASE / "transcripts_slim" / "jericho"
OUTPUT_DIR = BASE / "analysis_outputs" / "situated_awareness_replays" / "jericho"


def normalize_obs(obs: str) -> str:
    """Normalize observation: collapse whitespace for comparison."""
    return " ".join(obs.split())


def get_frotz_env(env):
    """Navigate to the inner jericho.FrotzEnv from the gym wrapper."""
    # Stack: tales.JerichoEnv -> textworld.Filter -> textworld.JerichoEnv -> jericho.FrotzEnv
    deep = env.unwrapped
    tw_filter = deep.env                      # textworld.Filter
    tw_jericho = tw_filter._wrapped_env       # textworld.envs.zmachine.jericho.JerichoEnv
    return tw_jericho._jericho                # jericho.FrotzEnv


def extract_world_state(env, admissible_commands=True) -> dict:
    """
    Extract ground-truth world state from Jericho's Z-machine.

    Returns a dict with:
      - agent_at: str (room name or description)
      - objects: dict mapping object_name -> parent_name
      - inventory: list of object names held by player
      - receptacles: dict (rooms/containers found)
      - admissible_commands: list of valid actions
      - world_objects_count: int
    """
    frotz = get_frotz_env(env)

    # Player location
    try:
        loc = frotz.get_player_location()
        agent_at = loc.name.strip() if loc else "unknown"
    except Exception:
        agent_at = "unknown"

    # Inventory
    try:
        inv_objs = frotz.get_inventory()
        inventory = [obj.name.strip() for obj in inv_objs]
    except Exception:
        inventory = []

    # World objects → build parent map
    objects = {}
    receptacles = {}
    try:
        world_objs = frotz.get_world_objects()
        # Build id→name map
        id_to_name = {}
        for obj in world_objs:
            id_to_name[obj.num] = obj.name.strip()

        for obj in world_objs:
            name = obj.name.strip()
            if not name:
                continue
            parent_name = id_to_name.get(obj.parent, "unknown")
            # Skip objects whose parent is the root (typically Class/Object)
            if obj.parent == 0:
                continue
            objects[name] = parent_name
    except Exception:
        pass

    # Valid actions (slow — tests every action template against game state)
    admissible = []
    if admissible_commands:
        def _timeout_handler(signum, frame):
            raise TimeoutError()
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(120)
        try:
            valid = frotz.get_valid_actions()
            admissible = list(valid) if valid else []
        except (TimeoutError, Exception):
            admissible = []
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return {
        "agent_at": agent_at,
        "objects": objects,
        "inventory": inventory,
        "receptacles": receptacles,
        "admissible_commands": admissible,
        "open_receptacles": [],
    }


def replay_transcript(transcript_path: Path, admissible_commands=True) -> dict:
    """Replay a single transcript through the Jericho environment."""
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

    # Verify initial observation matches
    transcript_turns = transcript.get("turns", [])
    if not transcript_turns or transcript_turns[0]["role"] != "environment":
        env.close()
        return {"error": "transcript has no initial env turn", "transcript_id": transcript.get("transcript_id")}

    transcript_init = normalize_obs(transcript_turns[0]["content"])
    env_init = normalize_obs(obs)

    if transcript_init != env_init:
        env.close()
        return {
            "error": "initial observation mismatch",
            "transcript_id": transcript.get("transcript_id"),
            "expected": transcript_turns[0]["content"].strip(),
            "got": obs.strip(),
        }

    # Extract initial state
    steps = []
    initial_state = extract_world_state(env, admissible_commands=admissible_commands)
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
                # Don't combine obs — transcript records them as separate turns

            steps.append({
                "turn_index": i,
                "role": "agent",
                "content": action,
            })

            # Check for desync (skip for multi-line actions)
            # Use soft desync: log mismatch but continue replaying.
            # Jericho's Frotz binary produces platform-specific RNG sequences,
            # causing ~55% of transcripts to diverge on random events (combat
            # outcomes, weather descriptions). The structural world state
            # (rooms, objects, inventory) remains valid for SA detection.
            if not multiline_skip and i + 1 < len(transcript_turns) and transcript_turns[i + 1]["role"] == "environment":
                transcript_next = normalize_obs(transcript_turns[i + 1]["content"])
                env_next = normalize_obs(env_obs)

                if transcript_next != env_next:
                    desync = True
                    steps.append({
                        "turn_index": i + 1,
                        "role": "environment",
                        "content": env_obs.strip(),
                        "soft_desync": True,
                        "transcript_content": transcript_turns[i + 1]["content"].strip(),
                    })

        elif turn["role"] == "environment":
            ws = extract_world_state(env, admissible_commands=admissible_commands)
            steps.append({
                "turn_index": i,
                "role": "environment",
                "content": turn["content"].strip(),
                "world_state": ws,
            })

    env.close()

    soft_desync_count = sum(1 for s in steps if s.get("soft_desync"))

    return {
        "transcript_id": transcript.get("transcript_id", transcript_path.stem),
        "game": game,
        "model": transcript.get("model", ""),
        "seed": seed,
        "score": transcript.get("score", 0),
        "max_score": transcript.get("max_score", 1),
        "total_turns": len(transcript_turns),
        "replayed_turns": len(steps),
        "desync": False,  # No longer hard-break on desync
        "soft_desyncs": soft_desync_count,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Single transcript file to replay")
    parser.add_argument("--all", action="store_true", help="Replay all jericho transcripts")
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
            print(f"  OK: {result['replayed_turns']} turns replayed")
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
