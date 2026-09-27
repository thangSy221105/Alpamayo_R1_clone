"""Deterministic action parser for Alpamayo-R1 reasoning traces.

The parser extracts driving intent from a CoC reasoning trace.  It is used for
alignment diagnostics and adaptive-alpha grouping; it is not a scene
understanding or safety classifier.

The audit convention is +x forward, +y left, metres, signed velocity, and
matched waypoint timestamps.  Direction rules bind left/right to the driving
verb, so "nudge left to pass the vehicle on the right" is correctly parsed as
a left nudge rather than an ambiguous direction.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from typing import Any


PRIMARY_PRIORITY = (
    "emergency_stop",
    "stop",
    "yield",
    "reverse",
    "u_turn",
    "lane_change",
    "merge",
    "pull_over",
    "overtake",
    "turn",
    "nudge",
    "avoid_obstacle",
    "park",
    "decelerate",
    "accelerate",
    "adjust_speed",
    "maintain_speed",
    "follow_lead",
    "distance_management",
    "keep_lane",
    "continue",
)

LATERAL_ACTIONS = {
    "lane_change",
    "merge",
    "overtake",
    "turn",
    "nudge",
    "avoid_obstacle",
    "pull_over",
}

ACTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "emergency_stop": (
        r"\bemergency\s+(?:brak\w*|stop)\b",
        r"\bhard\s+brak\w*\b",
        r"\bbrak\w*\s+hard\b",
        r"\bpanic\s+stop\b",
    ),
    "stop": (r"\bstop\b", r"\bcome\s+to\s+a\s+stop\b", r"\bhalt\b"),
    "yield": (
        r"\byield\b",
        r"\bgive\s+way\b",
        r"\bright\s+of\s+way\b",
        r"\blet\s+(?:the\s+)?[^.]{0,50}\s+pass\b",
        r"\bwait\s+for\s+(?:the\s+)?(?:pedestrian|vehicle|traffic)\b",
        r"\bwait\s+(?:due\s+to|for)\s+(?:the\s+)?(?:oncoming|approaching|crossing)\b",
    ),
    "reverse": (r"\breverse\b", r"\bback\s+up\b", r"\bbackward\b", r"\bback\s+into\b"),
    "u_turn": (r"\bu[- ]?turn\b", r"\bturn\s+around\b", r"\bturnaround\b"),
    "lane_change": (
        r"\blane\s+change\b",
        r"\bchange\s+lanes?\b",
        r"\bswitch\s+lanes?\b",
        r"\blane\s+shift\b",
    ),
    "merge": (
        r"\bmerge\b",
        r"\bjoin\s+(?:the\s+)?(?:lane|traffic)\b",
        r"\benter\s+(?:the\s+)?(?:lane|traffic)\b",
        r"\bre[- ]?enter\s+(?:the\s+)?lane\b",
        r"\bsplit\s+(?:to\s+the\s+)?(?:left|right)\b",
        r"\bfollow\s+the\s+(?:left|right)\s+branch\b",
        r"\btake\s+the\s+(?:left|right)\s+branch\b",
    ),
    "overtake": (
        r"\bovertak\w*\b",
        r"\bpass\s+(?:the\s+)?(?:vehicle|car|truck|bus|lead|traffic)\b",
        r"\bpass\s+on\s+(?:the\s+)?(?:left|right)\b",
        r"\bgo\s+around\s+(?:the\s+)?(?:vehicle|car|truck|bus)\b",
    ),
    "turn": (
        r"\bturn\b",
        r"\bmake\s+(?:a\s+)?(?:sharp\s+|gentle\s+)?turn\b",
        r"\btake\s+the\s+(?:left|right)\s+turn\b",
    ),
    "nudge": (
        r"\bnudge\b",
        r"\bshift\s+slightly\b",
        r"\bedge\s+(?:the\s+vehicle\s+)?over\b",
        r"\bmove\s+slightly\b",
        r"\bdrift\s+slightly\b",
    ),
    "avoid_obstacle": (
        r"\bavoid\b",
        r"\bevade\b",
        r"\bswerve\s+around\b",
        r"\bclear(?:ance)?\s+(?:from|of)\b",
        r"\b(?:increase|create|leave)\s+(?:more\s+)?clearance\b",
        r"\bkeep\s+away\s+from\b",
        r"\bmove\s+around\s+(?:the\s+)?(?:obstacle|cone|barrier|vehicle)\b",
    ),
    "pull_over": (
        r"\bpull\s+over\b",
        r"\bmove\s+to\s+the\s+shoulder\b",
        r"\bpull\s+to\s+the\s+side\b",
    ),
    "park": (
        r"\bpark(?:ing)?\s+(?:the\s+vehicle|the\s+car|in|at|near)\b",
        r"\bpark\s+(?:on|to)\s+(?:the\s+)?(?:left|right|side|shoulder)\b",
        r"\bpull\s+into\s+(?:a\s+)?parking\s+space\b",
        r"\bparking\s+maneuver\b",
    ),
    "accelerate": (
        r"\baccelerat\w*\b",
        r"\bspeed\s+up\b",
        r"\bincrease\s+(?:the\s+)?speed\b",
        r"\bpick\s+up\s+speed\b",
    ),
    "decelerate": (
        r"\bdecelerat\w*\b",
        r"\bslow\s+down\b",
        r"\breduce\s+(?:the\s+)?speed\b",
        r"\bbrak\w*\b",
    ),
    "adjust_speed": (
        r"\badjust\s+(?:the\s+)?speed\b",
        r"\bmodif\w*\s+(?:the\s+)?speed\b",
        r"\badapt\s+(?:the\s+)?speed\b",
        r"\bmatch\s+(?:the\s+)?speed\b",
    ),
    "maintain_speed": (
        r"\bmaintain\s+(?:a\s+)?(?:steady\s+)?speed\b",
        r"\bkeep\s+(?:a\s+)?(?:steady\s+)?speed\b",
        r"\bhold\s+(?:a\s+)?(?:steady\s+)?speed\b",
    ),
    "follow_lead": (
        r"\bfollow\s+the\s+(?:lead|vehicle\s+ahead|car\s+ahead)\b",
        r"\bkeep\s+up\s+with\s+traffic\b",
        r"\bmaintain\s+distance\s+behind\b",
        r"\bstay\s+behind\s+(?:the\s+)?(?:lead|vehicle|car)\b",
    ),
    "distance_management": (
        r"\bkeep\s+(?:a\s+)?(?:safe\s+)?distance\b",
        r"\bmaintain\s+(?:a\s+)?(?:safe\s+)?distance\b",
        r"\bfollowing\s+distance\b",
        r"\bcreate\s+(?:more\s+)?space\b",
    ),
    "keep_lane": (
        r"\bkeep\s+(?:the\s+)?lane\b",
        r"\bmaintain\s+(?:the\s+)?lane\b",
        r"\bstay\s+in\s+(?:the\s+)?lane\b",
        r"\bremain\s+in\s+(?:the\s+)?lane\b",
    ),
    "continue": (
        r"\bcontinue\s+driving\b",
        r"\bcontinue\s+forward\b",
        r"\bproceed\b",
        r"\bmove\s+forward\b",
        r"\bcarry\s+on\b",
        r"\bdrive\s+straight\b",
        r"\bgo\s+straight\b",
    ),
}

ACTION_DIRECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:nudge|steer|swerve|shift|move|edge|drift)\b[^.;,]{0,18}\b(left|right)(?:ward)?\b", "verb_then_direction"),
    (r"\b(?:turn|merge|split|overtake|pull\s+over)\b[^.;,]{0,18}\b(left|right)(?:ward)?\b", "verb_then_direction"),
    (r"\b(left|right)(?:ward)?\b[^.;,]{0,18}\b(?:turn|lane\s+change|merge|nudge|shift|swerve|pull\s+over)\b", "direction_then_verb"),
    (r"\b(?:change|switch|move)\b[^.;,]{0,18}\b(?:to\s+the\s+)?(left|right)\s+lane\b", "lane_direction"),
    (r"\b(?:pull\s+over|move\s+over)\b[^.;,]{0,18}\b(?:to\s+the\s+)?(left|right)\b", "side_direction"),
    (r"\b(?:increase|create|leave)\b[^.;,]{0,18}\bclearance\s+(?:to|on)\s+(left|right)\b", "clearance_direction"),
)


def _as_text(value: Any) -> str:
    """Convert a trace or a list-like trace to clean text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(item) for item in value).strip()
    text = str(value).strip()
    if not text:
        return ""

    # Alpamayo traces are often serialized as "['...']".  Parse only when
    # this is clearly a Python literal list/tuple; otherwise preserve text.
    if text[:1] in "[(" and text[-1:] in "])":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return " ".join(_as_text(item) for item in parsed).strip()

    return text.strip(" \t\n\r\"'")


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _direction(text: str) -> str:
    left = _has_any(
        text,
        (
            r"\bleft\b",
            r"leftward",
            r"to\s+the\s+left",
            r"on\s+the\s+left",
        ),
    )
    right = _has_any(
        text,
        (
            r"\bright\b",
            r"rightward",
            r"to\s+the\s+right",
            r"on\s+the\s+right",
        ),
    )
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    if left and right:
        return "ambiguous"
    return "none"


def _parse_reasoning_legacy(reasoning: Any) -> dict[str, Any]:
    """Extract action attributes from one reasoning trace.

    Returned fields:
      text, primary_action, secondary_actions, direction, speed_trend,
      lateral_trend, forward_motion, confidence, matched_rules, ambiguous.
    """
    text = _as_text(reasoning)
    lower = re.sub(r"\s+", " ", text.lower()).strip()

    result: dict[str, Any] = {
        "text": text,
        "primary_action": "unknown",
        "secondary_actions": [],
        "direction": "none",
        "speed_trend": "unknown",
        "lateral_trend": "unknown",
        "forward_motion": "unknown",
        "confidence": "low",
        "matched_rules": [],
        "ambiguous": False,
    }
    if not lower:
        result["matched_rules"].append("empty_reasoning")
        return result

    direction = _direction(lower)
    result["direction"] = direction
    if direction == "ambiguous":
        result["ambiguous"] = True
        result["matched_rules"].append("conflicting_left_right")

    # Detect explicit action concepts.  Patterns are intentionally phrase-
    # based so that "no critical agent" does not become an action.
    matches: set[str] = set()

    if _has_any(lower, (r"\bstop\b", r"come to a stop", r"near[- ]?stop")):
        matches.add("stop")
        result["speed_trend"] = "decrease_to_zero"
        result["forward_motion"] = "stop"
        result["matched_rules"].append("stop")

    if _has_any(lower, (r"\byield\b", r"give way", r"let .* pass")):
        matches.add("yield")
        result["speed_trend"] = "decrease"
        result["forward_motion"] = "slow_or_stop"
        result["matched_rules"].append("yield")

    if _has_any(lower, (r"lane change", r"change lanes", r"merge", r"move into .* lane")):
        matches.add("lane_change")
        result["lateral_trend"] = direction if direction in {"left", "right"} else "unknown"
        result["matched_rules"].append("lane_change")

    if _has_any(lower, (r"\bturn\b", r"make .* turn", r"take the .* turn")):
        matches.add("turn")
        result["lateral_trend"] = direction if direction in {"left", "right"} else "unknown"
        result["matched_rules"].append("turn")

    if _has_any(lower, (r"\bnudge\b", r"shift slightly", r"edge .* over")):
        matches.add("nudge")
        result["lateral_trend"] = direction if direction in {"left", "right"} else "unknown"
        result["matched_rules"].append("nudge")

    if _has_any(lower, (r"\baccelerat\w*\b", r"speed up", r"increase speed")):
        matches.add("accelerate")
        result["speed_trend"] = "increase"
        result["matched_rules"].append("accelerate")

    if _has_any(lower, (r"\bdecelerat\w*\b", r"slow down", r"reduce speed", r"brak\w*")):
        matches.add("decelerate")
        result["speed_trend"] = "decrease"
        result["matched_rules"].append("decelerate")

    if _has_any(lower, (r"adjust speed", r"modif\w* speed", r"adapt speed")):
        matches.add("adjust_speed")
        result["matched_rules"].append("adjust_speed")

    if _has_any(lower, (r"keep lane", r"maintain lane", r"stay in the lane")):
        matches.add("keep_lane")
        result["lateral_trend"] = "neutral"
        result["matched_rules"].append("keep_lane")

    # These phrases describe forward continuation, not acceleration.
    if _has_any(lower, (r"continue driving", r"proceed", r"move forward")):
        matches.add("continue")
        result["forward_motion"] = "continue"
        result["matched_rules"].append("continue")

    # Distance following is useful evidence but does not determine an exact
    # speed trend without a trajectory.
    if _has_any(lower, (r"keep distance", r"maintain a safe distance", r"follow distance")):
        result["matched_rules"].append("distance_management")
        if "decelerate" not in matches and "stop" not in matches:
            matches.add("adjust_speed")

    if direction in {"left", "right"} and any(action in matches for action in {"nudge", "turn", "lane_change"}):
        result["lateral_trend"] = direction

    # Stop is stronger than a generic deceleration phrase when both occur in
    # the same reasoning trace.
    if "stop" in matches:
        result["speed_trend"] = "decrease_to_zero"
        result["forward_motion"] = "stop"

    ordered = [action for action in PRIMARY_PRIORITY if action in matches]
    if ordered:
        result["primary_action"] = ordered[0]
        result["secondary_actions"] = ordered[1:]

    # Multiple unrelated primary actions are not collapsed silently.
    if len(ordered) > 2 or ("stop" in matches and "accelerate" in matches):
        result["ambiguous"] = True
        result["matched_rules"].append("multiple_actions")

    if result["primary_action"] != "unknown" and not result["ambiguous"]:
        result["confidence"] = "high" if direction != "ambiguous" else "medium"
    elif result["primary_action"] != "unknown":
        result["confidence"] = "medium"

    return result


def _is_negated(text: str, start: int) -> bool:
    """Detect a local negation immediately before an action phrase."""
    prefix = text[max(0, start - 24):start]
    return bool(
        re.search(
            r"(?:\bnot\b|\bdon't\b|\bdo not\b|\bnever\b|\bwithout\b)\s*$",
            prefix,
        )
    )


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _find_action_matches(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    matches: list[dict[str, Any]] = []
    negated: list[str] = []
    for action, patterns in ACTION_PATTERNS.items():
        seen_spans: set[tuple[int, int]] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                if _is_negated(text, match.start()):
                    negated.append(action)
                else:
                    matches.append(
                        {
                            "action": action,
                            "start": match.start(),
                            "end": match.end(),
                            "evidence": match.group(0),
                        }
                    )
    return matches, _unique(negated)


def _extract_direction(
    text: str,
    matches: list[dict[str, Any]],
) -> tuple[str, dict[str, str], list[str]]:
    """Bind left/right to a driving verb, not to an object in the scene."""
    local: list[tuple[str, int, str]] = []
    for pattern, rule_name in ACTION_DIRECTION_PATTERNS:
        for match in re.finditer(pattern, text):
            local.append((match.group(1), match.start(), rule_name))

    direction_candidates_by_action: dict[str, set[str]] = {}
    for item in matches:
        action = item["action"]
        if action not in LATERAL_ACTIONS:
            continue
        candidates = [
            (direction, abs(position - item["start"]))
            for direction, position, _ in local
            if abs(position - item["start"]) <= 42
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda value: value[1])
        nearest_distance = candidates[0][1]
        nearest = {
            direction
            for direction, distance in candidates
            if distance <= nearest_distance + 8
        }
        direction_candidates_by_action.setdefault(action, set()).update(nearest)

    directions_by_action: dict[str, str] = {}
    for action, candidates in direction_candidates_by_action.items():
        directions_by_action[action] = (
            next(iter(candidates)) if len(candidates) == 1 else "ambiguous"
        )

    if "ambiguous" in directions_by_action:
        return "ambiguous", directions_by_action, ["conflicting_action_directions"]
    local_directions = set(directions_by_action.values())
    if len(local_directions) == 1:
        return next(iter(local_directions)), directions_by_action, []
    if len(local_directions) > 1:
        return "ambiguous", directions_by_action, ["conflicting_action_directions"]

    # Fallback is used only when a lateral action exists.  A phrase such as
    # "vehicle on the right" alone must never create a vehicle direction.
    if any(item["action"] in LATERAL_ACTIONS for item in matches):
        generic = set(re.findall(r"\b(left|right)(?:ward)?\b", text))
        if len(generic) == 1:
            return next(iter(generic)), directions_by_action, []
        if len(generic) > 1:
            return "ambiguous", directions_by_action, ["conflicting_left_right"]
    return "none", directions_by_action, []


def parse_reasoning(reasoning: Any) -> dict[str, Any]:
    """Extract structured driving intent from one reasoning trace."""
    text = _as_text(reasoning)
    lower = re.sub(r"\s+", " ", text.lower()).strip()
    result: dict[str, Any] = {
        "text": text,
        "primary_action": "unknown",
        "secondary_actions": [],
        "action_components": [],
        "direction": "none",
        "speed_trend": "unknown",
        "lateral_trend": "unknown",
        "forward_motion": "unknown",
        "confidence": "low",
        "matched_rules": [],
        "negated_actions": [],
        "ambiguous": False,
    }
    if not lower:
        result["matched_rules"].append("empty_reasoning")
        return result

    matches, negated_actions = _find_action_matches(lower)
    result["negated_actions"] = negated_actions
    result["matched_rules"].extend(f"negated_{action}" for action in negated_actions)

    direction, action_directions, direction_rules = _extract_direction(lower, matches)
    result["direction"] = direction
    result["matched_rules"].extend(direction_rules)
    if direction == "ambiguous":
        result["ambiguous"] = True

    actions = {item["action"] for item in matches}
    for action in sorted(actions, key=PRIMARY_PRIORITY.index):
        result["action_components"].append(
            {
                "action": action,
                "direction": action_directions.get(action, "none"),
                "evidence": next(
                    item["evidence"] for item in matches if item["action"] == action
                ),
            }
        )
        result["matched_rules"].append(action)

    has_increase = "accelerate" in actions
    has_decrease = bool(actions & {"decelerate", "yield", "emergency_stop", "stop"})
    if "stop" in actions or "emergency_stop" in actions:
        result["speed_trend"] = "decrease_to_zero"
        result["forward_motion"] = "stop"
    elif has_increase and has_decrease:
        result["speed_trend"] = "conflicting"
        result["matched_rules"].append("conflicting_speed_trends")
        result["ambiguous"] = True
    elif has_increase:
        result["speed_trend"] = "increase"
    elif has_decrease:
        result["speed_trend"] = "decrease"
    elif "maintain_speed" in actions:
        result["speed_trend"] = "steady"
    elif "adjust_speed" in actions:
        result["speed_trend"] = "adjust"
    elif "distance_management" in actions:
        result["speed_trend"] = "adjust"

    if "reverse" in actions:
        result["forward_motion"] = "reverse"
    elif result["forward_motion"] == "unknown" and (
        {"continue", "overtake", "merge", "turn", "lane_change", "nudge"} & actions
    ):
        result["forward_motion"] = "continue"
    elif result["forward_motion"] == "unknown" and {"yield", "pull_over"} & actions:
        result["forward_motion"] = "slow_or_stop"

    if actions & LATERAL_ACTIONS:
        result["lateral_trend"] = (
            direction if direction in {"left", "right"} else "unknown"
        )
    elif "keep_lane" in actions:
        result["lateral_trend"] = "neutral"

    if {"keep_lane", "lane_change"} <= actions or {"keep_lane", "turn"} <= actions:
        result["matched_rules"].append("conflicting_lateral_actions")
        result["ambiguous"] = True
    if {"reverse", "continue"} <= actions or {"stop", "accelerate"} <= actions:
        result["matched_rules"].append("conflicting_forward_or_speed_actions")
        result["ambiguous"] = True

    ordered = [action for action in PRIMARY_PRIORITY if action in actions]
    if ordered:
        result["primary_action"] = ordered[0]
        result["secondary_actions"] = ordered[1:]
    elif negated_actions:
        result["matched_rules"].append("only_negated_action")

    if result["primary_action"] != "unknown" and not result["ambiguous"]:
        result["confidence"] = "high"
    elif result["primary_action"] != "unknown":
        result["confidence"] = "medium"

    result["matched_rules"] = _unique(result["matched_rules"])
    return result


def run_self_tests() -> None:
    cases = {
        "Nudge to the left to pass the parked vehicle on the right": (
            "nudge",
            "left",
            "unknown",
            False,
        ),
        "Turn right at the intersection": ("turn", "right", "unknown", False),
        "Accelerate after the light turns green": (
            "accelerate",
            "none",
            "increase",
            False,
        ),
        "Decelerate and stop at the stop line": (
            "stop",
            "none",
            "decrease_to_zero",
            False,
        ),
        "Keep lane to continue driving": ("keep_lane", "none", "unknown", False),
        "Therefore, make a sharp rightward lane change and accelerate.": (
            "lane_change",
            "right",
            "increase",
            False,
        ),
        "Adjust speed due to the road curvature.": (
            "adjust_speed",
            "none",
            "adjust",
            False,
        ),
        "Yield to the pedestrian and slow down.": (
            "yield",
            "none",
            "decrease",
            False,
        ),
        "Merge left into the open lane.": ("merge", "left", "unknown", False),
        "Reverse into the parking space.": ("reverse", "none", "unknown", False),
        "Pull over to the right shoulder.": ("pull_over", "right", "unknown", False),
        "Swerve right to avoid the obstacle.": (
            "avoid_obstacle",
            "right",
            "unknown",
            False,
        ),
        "Maintain a steady speed and keep the lane.": (
            "maintain_speed",
            "none",
            "steady",
            False,
        ),
        "Keep a safe distance from the lead vehicle.": (
            "distance_management",
            "none",
            "adjust",
            False,
        ),
        "Split to the right to take the freeway entrance ramp.": (
            "merge",
            "right",
            "unknown",
            False,
        ),
        "Wait due to oncoming vehicle.": ("yield", "none", "decrease", False),
    }
    for text, expected in cases.items():
        parsed = parse_reasoning(text)
        actual = (
            parsed["primary_action"],
            parsed["direction"],
            parsed["speed_trend"],
            parsed["ambiguous"],
        )
        assert actual == expected, (text, actual, expected)
    ambiguous = parse_reasoning("Turn left or turn right depending on traffic")
    assert ambiguous["ambiguous"] is True
    assert ambiguous["direction"] == "ambiguous"
    print(f"PASS: {len(cases) + 1} reasoning parser tests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reasoning", nargs="?", help="Reasoning text to parse")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if args.reasoning is None:
        parser.error("provide reasoning text or use --self-test")
    print(json.dumps(parse_reasoning(args.reasoning), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
