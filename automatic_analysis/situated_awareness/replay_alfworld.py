#!/usr/bin/env python3
"""
Replay ALFWorld transcripts through the actual environment, extracting
ground-truth PDDL world state at every step.

Usage:
    python analysis/situated_awareness/replay_alfworld.py --file transcripts_slim/alfworld/foo.json
    python analysis/situated_awareness/replay_alfworld.py --all          # all alfworld transcripts
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import tales  # noqa: F401 — registers envs

BASE = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = BASE / "transcripts_slim" / "alfworld"
OUTPUT_DIR = BASE / "analysis_outputs" / "situated_awareness_replays" / "alfworld"


def extract_world_state(env):
    """
    Extract ground-truth world state from ALFWorld's PDDL backend.

    Returns a dict with:
      - agent_at: str (receptacle name the agent is at, or "start")
      - objects: dict mapping object_name -> receptacle_name
      - inventory: list of object names held by agent
      - receptacles: dict mapping receptacle_name -> location_id
      - admissible_commands: list of valid actions
      - open_receptacles: set of receptacle names that are open
    """
    deep = env.unwrapped.env.unwrapped  # PddlEnv
    state = deep.state
    facts = state["_facts"]
    entity_infos = state["_entity_infos"]

    def display_name(internal_id):
        info = entity_infos.get(internal_id)
        if info is None:
            return internal_id
        return str(info).strip()

    # Parse facts
    agent_location_id = None
    objects_in_receptacle = {}  # obj_name -> receptacle_name
    receptacle_locations = {}   # receptacle_name -> loc_id
    inventory = []
    open_receptacles = []
    agent_holding = []

    for f in facts:
        name = f.name
        args = f.arguments

        if name == "atlocation":
            # atlocation(agent, loc_id)
            agent_location_id = args[1].name

        elif name == "inreceptacle":
            # inreceptacle(obj, receptacle)
            obj_name = display_name(args[0].name)
            recep_name = display_name(args[1].name)
            objects_in_receptacle[obj_name] = recep_name

        elif name == "receptacleatlocation":
            # receptacleatlocation(receptacle, loc)
            recep_name = display_name(args[0].name)
            loc_id = args[1].name
            receptacle_locations[recep_name] = loc_id

        elif name == "holds":
            # holds(agent, obj)
            obj_name = display_name(args[1].name)
            agent_holding.append(obj_name)

        elif name == "isopened":
            recep_name = display_name(args[0].name)
            open_receptacles.append(recep_name)

    # Determine which receptacle the agent is at
    agent_at = "start"
    if agent_location_id:
        for recep_name, loc_id in receptacle_locations.items():
            if loc_id == agent_location_id:
                agent_at = recep_name
                break

    # admissible commands
    admissible = list(state.get("admissible_commands", []) or [])

    return {
        "agent_at": agent_at,
        "objects": objects_in_receptacle,
        "inventory": agent_holding,
        "receptacles": receptacle_locations,
        "open_receptacles": open_receptacles,
        "admissible_commands": admissible,
    }


def normalize_obs(obs: str) -> str:
    """Normalize observation for comparison: strip whitespace."""
    return obs.strip()


def replay_transcript(transcript_path: Path) -> dict:
    """
    Replay a single transcript through the ALFWorld environment.

    Returns a replay dict or None if the transcript can't be replayed.
    """
    with open(transcript_path) as f:
        transcript = json.load(f)

    game = transcript["game"]
    metadata = transcript.get("metadata", {})
    seed = metadata.get("seed", None)

    if seed is None:
        return {"error": "no seed in metadata", "transcript_id": transcript.get("transcript_id")}

    # Initialize environment
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

    transcript_init_obs = normalize_obs(transcript_turns[0]["content"])
    env_init_obs = normalize_obs(obs)

    if transcript_init_obs != env_init_obs:
        env.close()
        return {
            "error": "initial observation mismatch",
            "transcript_id": transcript.get("transcript_id"),
            "expected": transcript_init_obs,
            "got": env_init_obs,
        }

    # Extract initial state
    steps = []
    initial_state = extract_world_state(env)
    steps.append({
        "turn_index": 0,
        "role": "environment",
        "content": obs.strip(),
        "world_state": initial_state,
    })

    # Replay each agent action
    desync = False
    for i, turn in enumerate(transcript_turns[1:], start=1):
        if turn["role"] == "agent":
            action = turn["content"].strip()

            # Replicate benchmark.py multi-line guard: the harness
            # skips env.step() entirely for actions containing newlines.
            multiline_skip = "\n" in action
            if multiline_skip:
                env_obs = "The game only allows one action per step."
            else:
                env_obs, _, done, env_info = env.step(action)

            # Record agent action
            steps.append({
                "turn_index": i,
                "role": "agent",
                "content": action,
            })

            # Find the next environment response in the transcript
            if i + 1 < len(transcript_turns) and transcript_turns[i + 1]["role"] == "environment":
                transcript_obs = normalize_obs(transcript_turns[i + 1]["content"])
            else:
                transcript_obs = None

            env_obs_norm = normalize_obs(env_obs)

            # Check for desync (skip for multi-line actions — transcript
            # records stale feedback, not the guard message)
            if not multiline_skip and transcript_obs is not None and transcript_obs != env_obs_norm:
                desync = True
                steps.append({
                    "turn_index": i + 1,
                    "role": "environment",
                    "content": env_obs_norm,
                    "desync": True,
                    "transcript_content": transcript_obs,
                })
                break

        elif turn["role"] == "environment":
            # Extract world state after env response
            ws = extract_world_state(env)
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
        "seed": seed,
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
    parser.add_argument("--all", action="store_true", help="Replay all alfworld transcripts")
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
        ok = 0
        desync = 0
        errors = 0
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
