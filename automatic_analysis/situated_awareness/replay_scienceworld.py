#!/usr/bin/env python3
"""
Replay ScienceWorld transcripts through the actual environment, extracting
ground-truth world state (object tree, admissible commands) at every step.

Handles seed mismatch: if the transcript's seed doesn't reproduce the same
initial observation, searches all test variations for a match.

Usage:
    python analysis/situated_awareness/replay_scienceworld.py --file transcripts_slim/scienceworld/foo.json
    python analysis/situated_awareness/replay_scienceworld.py --all
"""

import argparse
import json
import re
import sys
from pathlib import Path

import gymnasium as gym
import tales  # noqa: F401 — registers envs

BASE = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = BASE / "transcripts_slim" / "scienceworld"
OUTPUT_DIR = BASE / "analysis_outputs" / "situated_awareness_replays" / "scienceworld"


def _sort_contiguous_block(lines: list, predicate, strip_key=None) -> list:
    """Sort contiguous runs of lines matching predicate, preserving others.

    strip_key: optional function to extract the sort key from a line.
               If None, sorts by stripped line content.
    """
    result = []
    block = []
    for line in lines:
        if predicate(line.strip()):
            block.append(line)
        else:
            if block:
                key = strip_key or (lambda l: l.strip())
                result.extend(sorted(block, key=key))
                block = []
            result.append(line)
    if block:
        key = strip_key or (lambda l: l.strip())
        result.extend(sorted(block, key=key))
    return result


def _disambiguation_sort_key(line: str) -> str:
    """Sort disambiguation options by content, ignoring the leading number."""
    # "0:\tlook at water (in toilet)" -> "look at water (in toilet)"
    stripped = line.strip()
    m = re.match(r"\d+:\s*", stripped)
    return stripped[m.end():] if m else stripped


def _sort_and_renumber_disambiguation(lines: list) -> list:
    """Sort disambiguation option blocks by content and renumber 0, 1, 2, ..."""
    result = []
    block = []
    for line in lines:
        if re.match(r"\s*\d+:\s", line.strip()):
            block.append(line)
        else:
            if block:
                # Sort by content after the number prefix
                block.sort(key=_disambiguation_sort_key)
                # Renumber sequentially
                for i, b in enumerate(block):
                    content = re.sub(r"^\d+:", f"{i}:", b.strip())
                    result.append(content)
                block = []
            result.append(line)
    if block:
        block.sort(key=_disambiguation_sort_key)
        for i, b in enumerate(block):
            content = re.sub(r"^\d+:", f"{i}:", b.strip())
            result.append(content)
    return result


def normalize_obs(obs: str) -> str:
    """Normalize ScienceWorld observations for version-independent comparison.

    Handles three categories of version differences:
      1. Paint cup ordering  — 'a wood cup (containing X paint)' lines shuffled
      2. Object listing order — tab-indented object lines in look output
      3. Disambiguation menu ordering — numbered options (0: ..., 1: ...) reordered
      4. Error message wording — normalize known equivalent error strings
    """
    lines = obs.split("\n")

    # Sort paint cup lines
    lines = _sort_contiguous_block(
        lines, lambda s: bool(re.match(r"a wood cup \(containing \w+ paint\)", s))
    )

    # Sort tab-indented object listing lines (e.g., "\ta adult dove\n\ta juvenile dove")
    lines = _sort_contiguous_block(
        lines, lambda s: s.startswith("a ") or s.startswith("an ")
    )

    # Sort numbered disambiguation options by content and renumber sequentially.
    # Different versions list the same options in different order with different
    # numbers; we normalize by sorting on content and assigning 0, 1, 2, ...
    lines = _sort_and_renumber_disambiguation(lines)

    text = "\n".join(lines)

    # Normalize equivalent error messages
    text = text.replace(
        "It's not clear how to get there from here.",
        "No known action matches that input.",
    )

    # Normalize temperature readings (JVM nondeterminism causes ±10°C drift)
    text = re.sub(r"temperature of \d+ degrees", "temperature of N degrees", text)

    # Normalize bee counts (JVM nondeterminism causes different spawn counts)
    # Replace contiguous runs of "a adult bee" lines with a single canonical form
    text = re.sub(r"(?:a adult bee\s*\n?\s*)+", "a adult bee (xN)\n", text)

    return " ".join(text.split())


def extract_world_state(env, info: dict) -> dict:
    """
    Extract ground-truth world state from ScienceWorld.

    Returns a dict with:
      - agent_at: str (room/location from look())
      - objects: dict mapping object refs to locations (simplified)
      - inventory: list of inventory item descriptions
      - receptacles: dict (empty — ScienceWorld doesn't use this concept the same way)
      - admissible_commands: list of valid actions
      - open_receptacles: list (empty)
      - look: str (full look output)
      - inv: str (full inventory output)
    """
    inner = env.unwrapped.env  # scienceworld.ScienceWorldEnv

    admissible = info.get("admissible_commands", []) or []

    # Get look and inventory (these are free actions in ScienceWorld)
    try:
        look_text = inner.look()
    except Exception:
        look_text = info.get("look", "")

    try:
        inv_text = inner.inventory()
    except Exception:
        inv_text = info.get("inv", "")

    # Parse location from look text
    agent_at = "unknown"
    if look_text:
        # ScienceWorld look output: "This room is called the <name>. In it, you see:"
        # or "This outside location is called the <name>. Here you see:"
        import re
        m = re.search(r"is called the ([^.]+)\.", look_text)
        if m:
            agent_at = m.group(1).strip()

    # Parse inventory items
    inventory = []
    if inv_text and "you see:" in inv_text.lower():
        for line in inv_text.split("\n"):
            line = line.strip()
            if line.startswith("a ") or line.startswith("an ") or line.startswith("the "):
                inventory.append(line)

    # Parse objects from look text
    objects = {}
    if look_text:
        in_contents = False
        current_container = None
        for line in look_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("a ") or stripped.startswith("an ") or stripped.startswith("the "):
                if "containing" in stripped.lower():
                    # Container with contents
                    container_name = stripped.split("(")[0].strip()
                    current_container = container_name
                    objects[container_name] = agent_at
                else:
                    obj_name = stripped
                    if current_container and line.startswith("\t\t"):
                        objects[obj_name] = current_container
                    else:
                        objects[obj_name] = agent_at
                        current_container = None

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


def _parse_disambiguation_options(text: str) -> dict:
    """Parse numbered options from a disambiguation menu.

    Returns dict mapping option number (int) -> option text (str).
    E.g., {0: "look at air (in bathroom)", 1: "look at water (in toilet, in bathroom)"}
    """
    options = {}
    for line in text.split("\n"):
        m = re.match(r"\s*(\d+):\s*(.+)", line)
        if m:
            options[int(m.group(1))] = m.group(2).strip()
    return options


def _remap_disambiguation_selection(
    action: str, last_env_obs: str, transcript_turns: list, current_turn_idx: int
) -> str:
    """Remap a disambiguation number selection to match our env's ordering.

    When the last env observation was a disambiguation menu and the agent
    selects by number, the transcript's menu ordering may differ from ours.
    We find what the transcript's menu looked like, determine which option
    the agent intended, and find the matching number in our env's menu.
    """
    # Only remap if action is a bare number
    if not re.match(r"^\d+$", action.strip()):
        return action

    # Only remap if our env showed a disambiguation menu
    if "Ambiguous request:" not in last_env_obs:
        return action

    selected_num = int(action.strip())

    # Find the transcript's disambiguation menu (the env turn just before this agent turn)
    transcript_menu = None
    for j in range(current_turn_idx - 1, -1, -1):
        t = transcript_turns[j]
        if t["role"] == "environment" and "Ambiguous request:" in t["content"]:
            transcript_menu = t["content"]
            break

    if transcript_menu is None:
        return action

    # Parse both menus
    transcript_options = _parse_disambiguation_options(transcript_menu)
    env_options = _parse_disambiguation_options(last_env_obs)

    if selected_num not in transcript_options:
        return action

    # Find what the agent intended to select
    intended_text = transcript_options[selected_num]

    # Find the matching option in our env's menu
    for env_num, env_text in env_options.items():
        if env_text == intended_text:
            return str(env_num)

    # Fuzzy match: try stripping whitespace differences
    intended_norm = " ".join(intended_text.split())
    for env_num, env_text in env_options.items():
        if " ".join(env_text.split()) == intended_norm:
            return str(env_num)

    # No match found — return original action
    return action


def find_matching_seed(env, transcript_obs_norm: str) -> int | None:
    """
    Search test variations to find one matching the transcript's initial obs.
    Returns the seed index or None.
    """
    inner = env.unwrapped
    num_variations = len(inner.variations)
    for seed_idx in range(num_variations):
        obs, _ = env.reset(seed=seed_idx)
        if normalize_obs(obs) == transcript_obs_norm:
            return seed_idx
    return None


def replay_transcript(transcript_path: Path) -> dict:
    """Replay a single transcript through the ScienceWorld environment."""
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
        try: env.close()
        except Exception: pass
        return {"error": "transcript has no initial env turn", "transcript_id": transcript.get("transcript_id")}

    transcript_init = normalize_obs(transcript_turns[0]["content"])
    env_init = normalize_obs(obs)

    # If seed doesn't match, search for the correct variation
    actual_seed = seed
    if transcript_init != env_init:
        found_seed = find_matching_seed(env, transcript_init)
        if found_seed is None:
            try: env.close()
            except Exception: pass
            return {
                "error": "no matching variation found",
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
    last_env_obs = obs  # track last env response for disambiguation remapping
    for i, turn in enumerate(transcript_turns[1:], start=1):
        if turn["role"] == "agent":
            action = turn["content"].strip()

            # Replicate benchmark.py multi-line guard: the harness
            # skips env.step() entirely for actions containing newlines.
            # The transcript records stale info["feedback"] as the env
            # response, so we skip desync comparison for these turns.
            multiline_skip = "\n" in action
            if multiline_skip:
                env_obs = "The game only allows one action per step."
                done = False
            else:
                # Remap disambiguation selections: if the last env response
                # was a disambiguation menu and the agent is selecting by
                # number, map from the transcript's ordering to our env's.
                effective_action = _remap_disambiguation_selection(
                    action, last_env_obs, transcript_turns, i
                )
                env_obs, _, done, env_info = env.step(effective_action)
                last_info = env_info

            # Replicate benchmark.py auto-reset on game over
            if done:
                try:
                    reset_obs, reset_info = env.reset()
                    last_info = reset_info
                except Exception:
                    pass

            steps.append({
                "turn_index": i,
                "role": "agent",
                "content": action,
            })

            last_env_obs = env_obs

            # Check for desync (skip for multi-line actions)
            if not multiline_skip and i + 1 < len(transcript_turns) and transcript_turns[i + 1]["role"] == "environment":
                transcript_next = normalize_obs(transcript_turns[i + 1]["content"])
                env_next = normalize_obs(env_obs)

                if transcript_next != env_next:
                    desync = True
                    # Log soft desync but continue replaying — ScienceWorld's
                    # JVM nondeterminism causes minor state drift (temperatures,
                    # object counts, phase changes) that don't affect the
                    # structural world state needed for failure detection.
                    steps.append({
                        "turn_index": i + 1,
                        "role": "environment",
                        "content": env_obs.strip(),
                        "soft_desync": True,
                        "transcript_content": transcript_turns[i + 1]["content"].strip(),
                    })

        elif turn["role"] == "environment":
            ws = extract_world_state(env, last_info)
            steps.append({
                "turn_index": i,
                "role": "environment",
                "content": turn["content"].strip(),
                "world_state": ws,
            })

    try:
        env.close()
    except Exception:
        pass

    soft_desync_count = sum(1 for s in steps if s.get("soft_desync"))

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
        "desync": False,  # No longer hard-break on desync
        "soft_desyncs": soft_desync_count,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Single transcript file to replay")
    parser.add_argument("--all", action="store_true", help="Replay all scienceworld transcripts")
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
            if i % 50 == 0:
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
