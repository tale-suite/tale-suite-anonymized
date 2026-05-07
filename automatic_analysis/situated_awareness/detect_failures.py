#!/usr/bin/env python3
"""
Unified situated awareness failure detector across all 5 frameworks.

Pipeline: replay transcript → extract world state → detect failures.

Usage:
    python analysis/situated_awareness/detect_failures.py --filter "opus-4.5_react-high"
    python analysis/situated_awareness/detect_failures.py --framework alfworld --filter "opus-4.5"
    python analysis/situated_awareness/detect_failures.py --file transcripts_slim/alfworld/foo.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = BASE / "transcripts_slim"
OUTPUT_DIR = BASE / "analysis_outputs" / "situated_awareness"

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import all replayers
from replay_alfworld import replay_transcript as replay_alfworld
from replay_textworld import replay_transcript as replay_textworld
from replay_jericho import replay_transcript as replay_jericho
from replay_scienceworld import replay_transcript as replay_scienceworld
from replay_textworldexpress import replay_transcript as replay_textworldexpress

REPLAYERS = {
    "alfworld": replay_alfworld,
    "textworld": replay_textworld,
    "jericho": replay_jericho,
    "scienceworld": replay_scienceworld,
    "textworldexpress": replay_textworldexpress,
}


# ─── Entity extraction from actions ──────────────────────────────────────

DIRECTIONS = {
    "north", "south", "east", "west", "northeast", "northwest",
    "southeast", "southwest", "up", "down", "forward", "back",
    "left", "right", "out", "in",
}

# Verbs whose arguments are NOT entity references
NON_ENTITY_VERBS = {
    "say", "answer", "learn", "memorize", "cast", "shout", "yell",
    "scream", "whisper", "sing", "chant", "pray", "think", "wait",
    "sleep", "wake", "save", "restore", "restart", "quit", "score",
    "verbose", "brief", "superbrief", "diagnose", "help", "hint",
    "undo", "again",
}


def extract_entity_references(action: str) -> list:
    """
    Extract entity names referenced in an action string.
    Splits on common prepositions, strips the leading verb.

    Filters out:
      - Non-entity verbs (say, answer, learn, cast, etc.)
      - Compass directions after 'go'
      - 'about' clause targets in ask/tell (topics, not entities)
    """
    action = action.strip()
    # Strip surrounding quotes from the whole action
    if action.startswith('"') and action.endswith('"'):
        return []

    words = action.split()
    if not words:
        return []

    verb = words[0].lower()

    # Skip verbs whose arguments are never entity references
    if verb in NON_ENTITY_VERBS:
        return []

    # Skip phrasal-verb posture commands (not entity references)
    two_word = " ".join(words[:2]).lower() if len(words) >= 2 else ""
    if two_word in ("stand up", "lie down", "sit down", "wake up", "take off"):
        return []

    # Handle phrasal verbs with particle: "turn on lamp" -> entity is "lamp"
    if two_word in ("turn on", "turn off", "pick up", "put down"):
        if len(words) >= 3:
            return [" ".join(words[2:])]
        return []

    # "look around", "look here" — not entity references
    if verb == "look" and len(words) == 2 and words[1].lower() in ("around", "here", "up", "down"):
        return []

    # For 'go', only extract non-direction targets
    if verb == "go":
        target = " ".join(words[1:]).strip().lower()
        # Strip "to " prefix: "go to kitchen" -> "kitchen"
        if target.startswith("to "):
            target = target[3:].strip()
        if target in DIRECTIONS:
            return []
        # Non-direction target (e.g., "go kitchen", "go backstage")
        return [target] if target else []

    entities = []

    # Split on prepositions, but treat 'about' specially
    parts = re.split(
        r'\b(?:from|to|with|in|on|into|using|at|through)\b',
        action, flags=re.IGNORECASE,
    )

    # For ask/tell/talk, only extract the person (first entity), not the topic
    if verb in ("ask", "tell", "talk"):
        # "ask angela about nikolai" -> extract "angela", skip "nikolai"
        # "talk to fishermen" -> extract "fishermen"
        before_about = re.split(r'\babout\b', action, flags=re.IGNORECASE)[0]
        about_words = before_about.strip().split()
        if len(about_words) >= 2:
            candidate = " ".join(about_words[1:]).strip()
            # Strip leading "to" — "talk to fishermen" -> "fishermen"
            if candidate.lower().startswith("to "):
                candidate = candidate[3:].strip()
            if candidate:
                entities.append(candidate)
        return entities

    # Treat walk/run/follow like go
    if verb in ("walk", "run", "follow"):
        target = " ".join(words[1:]).strip().lower()
        if target.startswith("to "):
            target = target[3:].strip()
        if target in DIRECTIONS:
            return []
        return [target] if target else []

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        pwords = part.split()
        if i == 0:
            if len(pwords) >= 2:
                candidate = " ".join(pwords[1:]).strip()
                if candidate:
                    entities.append(candidate)
        else:
            entities.append(part)

    return entities


# ─── Failure classification using world state ─────────────────────────────

def _is_reachable(obj_name: str, agent_at: str, objects: dict) -> bool:
    """Check if an object is reachable from agent's location via containment chain.

    Walks the parent chain: obj → parent → grandparent → ... until we reach
    the agent's location or exhaust the chain.  E.g., note → desk → cubicle
    means note is reachable when the agent is in the cubicle.
    """
    visited = set()
    current = obj_name
    for _ in range(10):  # max depth
        if current in visited:
            break
        visited.add(current)
        parent = objects.get(current)
        if parent is None:
            break
        if parent.lower() == agent_at.lower():
            return True
        current = parent
    return False


def classify_failed_action(
    action: str,
    world_state: dict,
    seen_entities: dict,
) -> list:
    """
    Given an action that was rejected by the environment, classify WHY
    by checking referenced entities against the ground-truth world state.
    """
    # Skip multi-line actions (agent prose, not real commands)
    if "\n" in action:
        return []

    entity_refs = extract_entity_references(action)
    if not entity_refs:
        return []

    agent_at = world_state.get("agent_at", "unknown")
    objects = world_state.get("objects", {})
    inventory = world_state.get("inventory", [])
    receptacles = world_state.get("receptacles", {})

    obj_lower = {o.lower(): o for o in objects}
    inv_lower = {o.lower() for o in inventory}
    recep_lower = {r.lower(): r for r in receptacles}

    results = []

    for ref in entity_refs:
        ref_l = ref.lower()

        # Known receptacle
        if ref_l in recep_lower:
            if agent_at.lower() != ref_l:
                results.append((
                    "scope_violation",
                    f"Agent is at '{agent_at}' but references receptacle '{recep_lower[ref_l]}'.",
                ))
            continue

        # Known object in the world
        if ref_l in obj_lower:
            obj_real = obj_lower[ref_l]
            obj_loc = objects[obj_real]
            # Skip Z-machine class/template objects and internal containers
            if obj_loc in ("Class", "Object", "unknown", "Nowhere", ""):
                continue
            # Skip objects in Z-machine internal containers (parenthesized names)
            if obj_loc.startswith("(") and obj_loc.endswith(")"):
                continue
            # Skip Z-machine direction objects (parented to "compass")
            if obj_loc.lower() == "compass":
                continue
            if obj_real in inventory:
                continue
            if obj_loc.lower() == agent_at.lower():
                continue
            # Check containment chain: obj might be inside something at agent's location
            if _is_reachable(obj_real, agent_at, objects):
                continue
            results.append((
                "scope_violation",
                f"Object '{obj_real}' is at '{obj_loc}', not at agent's location '{agent_at}'.",
            ))
            continue

        # In inventory
        if ref_l in inv_lower:
            continue

        # Previously seen but gone
        seen_lower = {k.lower(): k for k in seen_entities}
        if ref_l in seen_lower:
            results.append((
                "stale_reference",
                f"Entity '{ref}' was previously seen but no longer exists.",
            ))
            continue

        # Partial match — prefer objects reachable from agent's location
        partial_objs = [obj_lower[k] for k in obj_lower if ref_l in k or k in ref_l]
        partial_receps = [recep_lower[k] for k in recep_lower if ref_l in k or k in ref_l]

        if partial_objs:
            # Filter out Z-machine class templates, internal containers, and compass directions
            partial_objs = [o for o in partial_objs
                           if objects.get(o, "") not in ("Class", "Object", "unknown", "Nowhere", "compass", "")
                           and not (objects.get(o, "").startswith("(") and objects.get(o, "").endswith(")"))]
            if not partial_objs:
                continue  # Only class templates matched — not a real entity

            # Check if ANY partial match is reachable
            any_reachable = any(
                o in inventory
                or objects[o].lower() == agent_at.lower()
                or _is_reachable(o, agent_at, objects)
                for o in partial_objs
            )
            if any_reachable:
                continue  # At least one matching object is accessible

            obj_real = partial_objs[0]
            obj_loc = objects[obj_real]
            results.append((
                "scope_violation",
                f"Object '{obj_real}' is at '{obj_loc}', not at agent's location '{agent_at}'.",
            ))
        elif partial_receps:
            continue
        else:
            seen_partial = [k for k in seen_entities if ref_l in k.lower() or k.lower() in ref_l]
            if seen_partial:
                results.append((
                    "stale_reference",
                    f"Entity '{ref}' was previously seen but no longer exists.",
                ))
            else:
                results.append((
                    "hallucination",
                    f"Entity '{ref}' has never been observed and does not exist.",
                ))

    return results


def world_state_changed(ws_before: dict, ws_after: dict) -> bool:
    """Check if the world state changed between two snapshots."""
    if ws_before is None or ws_after is None:
        return True
    return (
        ws_before.get("agent_at") != ws_after.get("agent_at")
        or ws_before.get("objects") != ws_after.get("objects")
        or ws_before.get("inventory") != ws_after.get("inventory")
        or ws_before.get("open_receptacles") != ws_after.get("open_receptacles")
    )


# ─── Pathfinding detection ────────────────────────────────────────────────

def detect_oscillation(visit_history: list, new_location: str) -> bool:
    if len(visit_history) < 3:
        return False
    recent = [v[1] for v in visit_history[-3:]] + [new_location]
    return recent[-1] == recent[-3] and recent[-2] == recent[-4]


# ─── Main detection on replay output ─────────────────────────────────────

GRACE_STEPS = 10

def detect_on_replay(replay: dict) -> dict:
    """Run failure detection on a replay result dict."""
    if replay.get("error"):
        return {
            "transcript_id": replay.get("transcript_id"),
            "skipped": True,
            "reason": replay.get("error"),
        }

    game = replay.get("game", "")
    steps = replay["steps"]

    seen_entities = defaultdict(set)
    visit_history = []
    sa_failures = []
    pf_failures = []
    agent_step_count = 0
    current_ws = None

    for step_idx, step in enumerate(steps):
        if step["role"] == "environment" and "world_state" in step:
            current_ws = step["world_state"]

            for obj, loc in current_ws.get("objects", {}).items():
                seen_entities[obj].add(loc)
            for obj in current_ws.get("inventory", []):
                seen_entities[obj].add("inventory")
            for recep in current_ws.get("receptacles", {}):
                seen_entities[recep].add("world")

            agent_at = current_ws.get("agent_at", "unknown")
            if not visit_history or visit_history[-1][1] != agent_at:
                visit_history.append((step.get("turn_index", step_idx), agent_at))

        elif step["role"] == "agent" and current_ws is not None:
            action = step["content"]
            agent_step_count += 1

            next_ws = None
            if step_idx + 1 < len(steps) and steps[step_idx + 1]["role"] == "environment":
                next_ws = steps[step_idx + 1].get("world_state")

            admissible = set(current_ws.get("admissible_commands", []))
            action_is_valid = action.strip() in admissible
            action_was_rejected = not world_state_changed(current_ws, next_ws)

            # Situated Awareness detection
            # Skip if world state has no valid location (extraction failed)
            agent_at = current_ws.get("agent_at", "unknown")
            if not action_is_valid and action_was_rejected and agent_step_count > GRACE_STEPS and agent_at and agent_at != "unknown":
                classifications = classify_failed_action(
                    action, current_ws, seen_entities
                )
                for fail_type, detail in classifications:
                    sa_failures.append({
                        "type": fail_type,
                        "competency": "situated_awareness",
                        "turn_index": step.get("turn_index", step_idx),
                        "action": action,
                        "detail": detail,
                        "agent_at": current_ws.get("agent_at", "unknown"),
                    })

            # Pathfinding: oscillation
            if action_is_valid and not action_was_rejected:
                go_match = re.match(r"^go (?:to )?(.+)$", action, re.IGNORECASE)
                if go_match:
                    target = go_match.group(1).strip()
                    if detect_oscillation(visit_history, target):
                        pf_failures.append({
                            "type": "oscillation",
                            "competency": "path_finding",
                            "turn_index": step.get("turn_index", step_idx),
                            "action": action,
                            "detail": f"Oscillating: {[v[1] for v in visit_history[-3:]]} -> '{target}'.",
                        })

    total_agent_turns = sum(1 for s in steps if s["role"] == "agent")

    return {
        "transcript_id": replay.get("transcript_id"),
        "game": game,
        "model": replay.get("model"),
        "score": replay.get("score"),
        "max_score": replay.get("max_score"),
        "total_agent_turns": total_agent_turns,
        "soft_desyncs": replay.get("soft_desyncs", 0),
        "situated_awareness_failures": sa_failures,
        "pathfinding_failures": pf_failures,
        "summary": {
            "sa_total": len(sa_failures),
            "sa_scope_violation": sum(1 for f in sa_failures if f["type"] == "scope_violation"),
            "sa_hallucination": sum(1 for f in sa_failures if f["type"] == "hallucination"),
            "sa_stale_reference": sum(1 for f in sa_failures if f["type"] == "stale_reference"),
            "pf_total": len(pf_failures),
            "pf_oscillation": sum(1 for f in pf_failures if f["type"] == "oscillation"),
        },
    }


# ─── Pipeline: replay → detect ───────────────────────────────────────────

def process_transcript(transcript_path: Path, framework: str) -> dict:
    """Replay a transcript and run failure detection."""
    replayer = REPLAYERS[framework]
    # Jericho's get_valid_actions() is extremely slow; skip for now.
    # SA detection uses world objects/inventory/location instead.
    if framework == "jericho":
        replay = replayer(transcript_path, admissible_commands=False)
    else:
        replay = replayer(transcript_path)
    return detect_on_replay(replay)


def infer_framework(transcript_path: Path) -> str:
    """Infer framework from the parent directory name."""
    return transcript_path.parent.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Single transcript file")
    parser.add_argument("--framework", type=str, help="Limit to one framework")
    parser.add_argument("--filter", type=str, help="Filename substring filter (e.g. 'opus-4.5_react-high')")
    parser.add_argument("--transcripts-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = BASE / path
        fw = args.framework or infer_framework(path)
        print(f"Processing: {path.name} (framework={fw})")
        result = process_transcript(path, fw)
        out_path = args.output_dir / f"{path.stem}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result["summary"], indent=2))
        return

    # Batch mode
    frameworks = [args.framework] if args.framework else list(REPLAYERS.keys())
    all_files = []
    for fw in frameworks:
        fw_dir = args.transcripts_dir / fw
        if not fw_dir.exists():
            continue
        for path in sorted(fw_dir.glob("*.json")):
            if args.filter and args.filter not in path.name:
                continue
            all_files.append((path, fw))

    print(f"Processing {len(all_files)} transcripts across {frameworks}")

    totals = defaultdict(int)
    fw_totals = defaultdict(lambda: defaultdict(int))
    errors = 0

    for i, (path, fw) in enumerate(all_files):
        if i % 25 == 0:
            print(f"  [{i}/{len(all_files)}] {fw}/{path.name[:50]}...")
        try:
            result = process_transcript(path, fw)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR: {path.name}: {e}")
            continue

        out_path = args.output_dir / fw / f"{path.stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        if result.get("skipped"):
            totals["skipped"] += 1
        else:
            totals["processed"] += 1
            s = result["summary"]
            for key in s:
                totals[key] += s[key]
                fw_totals[fw][key] += s[key]
            fw_totals[fw]["processed"] += 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {totals['processed']} processed, {totals.get('skipped',0)} skipped, {errors} errors")
    print(f"{'='*70}")
    print(f"\n{'Framework':<20} {'Count':>6} {'SA':>5} {'Scope':>6} {'Halluc':>7} {'Stale':>6} {'PF':>5}")
    print("-" * 60)
    for fw in frameworks:
        ft = fw_totals.get(fw, {})
        n = ft.get("processed", 0)
        if n == 0:
            continue
        print(f"{fw:<20} {n:>6} {ft.get('sa_total',0):>5} {ft.get('sa_scope_violation',0):>6} {ft.get('sa_hallucination',0):>7} {ft.get('sa_stale_reference',0):>6} {ft.get('pf_total',0):>5}")

    ft = totals
    n = ft.get("processed", 0)
    print("-" * 60)
    print(f"{'TOTAL':<20} {n:>6} {ft.get('sa_total',0):>5} {ft.get('sa_scope_violation',0):>6} {ft.get('sa_hallucination',0):>7} {ft.get('sa_stale_reference',0):>6} {ft.get('pf_total',0):>5}")


if __name__ == "__main__":
    main()
