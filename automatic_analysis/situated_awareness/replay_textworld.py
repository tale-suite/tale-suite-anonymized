#!/usr/bin/env python3
"""
Replay TextWorld transcripts through the actual environment, extracting
ground-truth world state (facts, entities, admissible commands) at every step.

Usage:
    python analysis/situated_awareness/replay_textworld.py --file transcripts_slim/textworld/foo.json
    python analysis/situated_awareness/replay_textworld.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import textworld
import tales  # noqa: F401 — registers envs

BASE = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = BASE / "transcripts_slim" / "textworld"
OUTPUT_DIR = BASE / "analysis_outputs" / "situated_awareness_replays" / "textworld"


def normalize_obs(obs: str) -> str:
    """Normalize observation: collapse whitespace, strip move-count artifacts.

    TextWorld's facts=True can increment the internal move counter, causing
    game-over text like "in 6 turn(s)" instead of "in 5 turn(s)". We strip
    the turn-count phrase so this doesn't cause spurious desyncs.
    """
    import re
    text = " ".join(obs.split())
    text = re.sub(r"in \d+ turn\(s\)", "in N turn(s)", text)
    return text


def extract_world_state(env, info: dict) -> dict:
    """
    Extract ground-truth world state from TextWorld's info dict.

    Returns a dict with:
      - agent_at: str (room name)
      - objects: dict mapping object_name -> location (room or container)
      - inventory: list of object names held by agent
      - receptacles: dict (containers/supporters with their room)
      - admissible_commands: list of valid actions
      - open_receptacles: list of open container names
      - entities: list of all entity names
    """
    facts = info.get("facts", []) or []
    entities = info.get("entities", []) or []
    admissible = info.get("admissible_commands", []) or []

    agent_at = "unknown"
    objects_in = {}        # obj -> location
    inventory = []
    receptacles = {}       # container/supporter -> room
    open_receptacles = []

    for fact in facts:
        pred = fact.name
        args = fact.arguments

        if pred == "at":
            entity_name = args[0].name
            location = args[1].name
            if entity_name == "P":
                agent_at = location
            elif args[0].type in ("c", "s", "oven", "toaster", "stove", "bbq"):
                # Container or supporter
                receptacles[entity_name] = location
            else:
                objects_in[entity_name] = location

        elif pred == "in":
            obj = args[0].name
            container = args[1].name
            if container == "I":
                # "I" = player inventory
                inventory.append(obj)
            else:
                objects_in[obj] = container

        elif pred == "on":
            obj = args[0].name
            supporter = args[1].name
            objects_in[obj] = supporter

        elif pred == "inventory":
            # Not actually used in TW facts, but defensive
            pass

        elif pred == "open":
            open_receptacles.append(args[0].name)

        elif pred == "carrying":
            # carrying(P, obj)
            if len(args) >= 2:
                inventory.append(args[1].name)

    return {
        "agent_at": agent_at,
        "objects": objects_in,
        "inventory": inventory,
        "receptacles": receptacles,
        "open_receptacles": open_receptacles,
        "admissible_commands": list(admissible),
        "entities": list(entities),
    }


def replay_transcript(transcript_path: Path) -> dict:
    """Replay a single transcript through the TextWorld environment."""
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

    # Patch EnvInfos to request facts and admissible_commands.
    # NOTE: Do NOT request description=True or inventory=True — those cause
    # internal look/inventory commands that increment the move counter and desync.
    env.unwrapped.infos = textworld.EnvInfos(
        score=True, max_score=True, won=True, lost=True,
        feedback=True, moves=True, admissible_commands=True,
        facts=True, entities=True,
        extras=["walkthrough"],
    )

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

            # Replicate benchmark.py auto-reset on game over:
            # when done, reset the env so subsequent actions go to a fresh game.
            # The transcript records the game-over obs and post-reset obs as
            # separate turns, so we don't combine them here.
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
    parser.add_argument("--all", action="store_true", help="Replay all textworld transcripts")
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
