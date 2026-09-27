"""Deterministic reasoning--trajectory consistency evaluator.

This evaluator is intentionally independent of the old LM judgements.  It
parses the clean reasoning, reads the complete 64-waypoint trajectories, and
compares the action implied by each trajectory with that reasoning.

Conventions
-----------
* +x is forward and +y is left, in metres.
* Each waypoint is matched by index and timestamp.
* A consistency score is binary: 1.0 for ``consistent`` and 0.0 for
  ``inconsistent``.  ``partially_consistent`` and ``uncertain`` are reported
  separately and have no binary score (``None``).
* The thresholds below are deterministic audit heuristics, not ground-truth
  driving labels.  Lane geometry, objects, and the stop line are unavailable.

The JSONL writer is resumable: completed (clip_id, mode, alpha) keys are
skipped, and every completed record is flushed immediately.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from alpamayo_r1.reasoning_action_rules import parse_reasoning
except ModuleNotFoundError:  # direct execution from a source checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from alpamayo_r1.reasoning_action_rules import parse_reasoning


EXPECTED_RECORDS_PER_CLIP = 16  # 4 modes x 4 alpha values in this benchmark.
EPS = 1e-9


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 6) -> float | None:
    number = _float(value)
    return None if not math.isfinite(number) else round(number, digits)


def _median(values: list[float], default: float = 0.0) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.median(finite) if finite else default


def _waypoint_arrays(waypoints: Any) -> dict[str, list[float]]:
    if not isinstance(waypoints, list):
        raise ValueError("waypoints must be a list")
    fields = {"t_s": [], "x_m": [], "y_m": [], "velocity_mps": [],
              "acceleration_mps2": [], "curvature_inv_m": []}
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, dict):
            raise ValueError(f"waypoint {index} is not an object")
        for field in fields:
            if field not in waypoint:
                raise ValueError(f"waypoint {index} lacks {field}")
            fields[field].append(_float(waypoint[field]))
    return fields


def _validate_pair(clean: Any, guided: Any) -> tuple[dict[str, list[float]], dict[str, list[float]], list[str]]:
    errors: list[str] = []
    if not isinstance(clean, list) or not isinstance(guided, list):
        raise ValueError("clean_waypoints and guided_waypoints must be lists")
    if len(clean) != len(guided):
        errors.append(f"waypoint_count_mismatch:{len(clean)}!={len(guided)}")
    if len(clean) != 64 or len(guided) != 64:
        errors.append(f"expected_64_waypoints:clean={len(clean)},guided={len(guided)}")
    clean_a = _waypoint_arrays(clean)
    guided_a = _waypoint_arrays(guided)
    for index, (t_clean, t_guided) in enumerate(zip(clean_a["t_s"], guided_a["t_s"])):
        if not math.isfinite(t_clean) or not math.isfinite(t_guided):
            errors.append(f"nonfinite_timestamp:{index}")
        elif abs(t_clean - t_guided) > 1e-6:
            errors.append(f"timestamp_mismatch:{index}")
    for name in ("t_s", "x_m", "y_m"):
        if any(not math.isfinite(x) for x in clean_a[name] + guided_a[name]):
            errors.append(f"nonfinite_{name}")
    return clean_a, guided_a, sorted(set(errors))


def _trajectory_stats(values: dict[str, list[float]]) -> dict[str, Any]:
    x, y, v, acc, curv = (values[k] for k in ("x_m", "y_m", "velocity_mps", "acceleration_mps2", "curvature_inv_m"))
    y0 = y[0] if y else 0.0
    relative_y = [item - y0 for item in y]
    abs_v = [abs(item) for item in v]
    min_abs_index = min(range(len(abs_v)), key=abs_v.__getitem__) if abs_v else 0
    near_zero = [item <= 0.35 for item in abs_v]
    # A stop is supported by near-zero speed late in the horizon and a low
    # final speed.  This avoids calling a brief low-speed passage a stop.
    stop_like = bool(abs_v) and min(abs_v) <= 0.35 and (
        abs(v[-1]) <= 0.70 or (min_abs_index >= max(1, len(v) // 2) and sum(near_zero[min_abs_index:]) >= 2)
    )
    speed_delta = (v[-1] - v[0]) if v else 0.0
    median_acc = _median(acc)
    if stop_like:
        longitudinal = "stop"
    elif speed_delta >= 0.50 or median_acc >= 0.08:
        longitudinal = "accelerate"
    elif speed_delta <= -0.50 or median_acc <= -0.08:
        longitudinal = "decelerate"
    elif v and sum(item < -0.20 for item in v) >= max(2, len(v) // 2):
        longitudinal = "reverse"
    elif v:
        longitudinal = "steady"
    else:
        longitudinal = "unknown"

    peak_left = max(relative_y, default=0.0)
    peak_right = min(relative_y, default=0.0)
    endpoint_delta = relative_y[-1] if relative_y else 0.0
    if max(abs(peak_left), abs(peak_right)) < 0.05:
        lateral = "neutral"
    elif abs(peak_left) >= abs(peak_right):
        lateral = "left"
    else:
        lateral = "right"

    forward_delta = (x[-1] - x[0]) if x else 0.0
    forward_motion = "forward" if forward_delta > 0.5 else "not_forward"
    return {
        "longitudinal_trend": longitudinal,
        "lateral_trend": lateral,
        "forward_motion": forward_motion,
        "speed_start_mps": _round(v[0] if v else None),
        "speed_end_mps": _round(v[-1] if v else None),
        "speed_delta_mps": _round(speed_delta),
        "min_abs_speed_mps": _round(min(abs_v) if abs_v else None),
        "min_abs_speed_index": min_abs_index,
        "median_acceleration_mps2": _round(median_acc),
        "x_start_m": _round(x[0] if x else None),
        "x_end_m": _round(x[-1] if x else None),
        "y_start_m": _round(y[0] if y else None),
        "y_end_m": _round(y[-1] if y else None),
        "y_delta_m": _round(endpoint_delta),
        "peak_left_displacement_m": _round(peak_left),
        "peak_right_displacement_m": _round(peak_right),
        "max_abs_lateral_displacement_m": _round(max((abs(item) for item in relative_y), default=0.0)),
        "path_length_m": _round(sum(math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) for i in range(1, len(x)))),
    }


def _difference_stats(clean: dict[str, list[float]], guided: dict[str, list[float]]) -> dict[str, Any]:
    xy = [math.hypot(guided["x_m"][i] - clean["x_m"][i], guided["y_m"][i] - clean["y_m"][i]) for i in range(min(len(clean["x_m"]), len(guided["x_m"]))) ]
    fields = {
        "x_m": "max_abs_x_delta_m",
        "y_m": "max_abs_y_delta_m",
        "velocity_mps": "max_abs_speed_delta_mps",
        "acceleration_mps2": "max_abs_acceleration_delta_mps2",
        "curvature_inv_m": "max_abs_curvature_delta_inv_m",
    }
    result: dict[str, Any] = {
        "mean_xy_delta_m": _round(sum(xy) / len(xy) if xy else 0.0),
        "max_xy_delta_m": _round(max(xy, default=0.0)),
    }
    for source, target in fields.items():
        deltas = [abs(guided[source][i] - clean[source][i]) for i in range(min(len(clean[source]), len(guided[source])))]
        result[target] = _round(max(deltas, default=0.0))
    max_xy = max(xy, default=0.0)
    max_speed = result["max_abs_speed_delta_mps"] or 0.0
    if max_xy <= 0.01 and max_speed <= 0.02:
        relation = "near_identical"
    elif max_xy <= 0.05 and max_speed <= 0.10:
        relation = "small_change"
    else:
        relation = "changed"
    result["relation"] = relation
    return result


def _component_type(action: str) -> str:
    if action in {"stop", "emergency_stop", "yield", "decelerate", "accelerate", "adjust_speed", "maintain_speed"}:
        return "longitudinal"
    if action in {"reverse", "continue", "follow_lead", "distance_management"}:
        return "forward"
    if action in {"keep_lane", "lane_change", "merge", "overtake", "turn", "nudge", "avoid_obstacle", "pull_over", "u_turn", "park"}:
        return "lateral"
    return "unknown"


def _match_component(component: dict[str, Any], stats: dict[str, Any]) -> tuple[bool | None, str]:
    action = component.get("action", "unknown")
    direction = component.get("direction", "none")
    kind = _component_type(action)
    if action in {"stop", "emergency_stop"}:
        ok = stats["longitudinal_trend"] == "stop"
        return ok, f"reasoning={action}; trajectory={stats['longitudinal_trend']}"
    if action == "yield":
        ok = stats["longitudinal_trend"] in {"decelerate", "stop"}
        return ok, f"reasoning=yield; trajectory={stats['longitudinal_trend']}"
    if action == "reverse":
        ok = stats["longitudinal_trend"] == "reverse"
        return ok, f"reasoning=reverse; trajectory={stats['longitudinal_trend']}"
    if action == "accelerate":
        ok = stats["longitudinal_trend"] == "accelerate"
        return ok, f"reasoning=accelerate; speed_delta={stats['speed_delta_mps']} m/s"
    if action == "decelerate":
        ok = stats["longitudinal_trend"] in {"decelerate", "stop"}
        return ok, f"reasoning=decelerate; trajectory={stats['longitudinal_trend']}"
    if action == "adjust_speed":
        ok = stats["longitudinal_trend"] in {"accelerate", "decelerate", "stop"}
        return ok, f"reasoning=adjust_speed; trajectory={stats['longitudinal_trend']}"
    if action == "maintain_speed":
        ok = stats["longitudinal_trend"] == "steady"
        return ok, f"reasoning=maintain_speed; trajectory={stats['longitudinal_trend']}"
    if action in {"continue", "follow_lead", "distance_management"}:
        if action in {"follow_lead", "distance_management"}:
            return None, f"reasoning={action}; external lead distance is unavailable"
        ok = stats["forward_motion"] == "forward" and stats["longitudinal_trend"] not in {"reverse", "stop"}
        return ok, f"reasoning=continue; forward_motion={stats['forward_motion']}"
    if action == "keep_lane":
        ok = stats["lateral_trend"] == "neutral" and stats["forward_motion"] == "forward"
        return ok, f"reasoning=keep_lane; lateral={stats['lateral_trend']}"
    if kind == "lateral":
        if direction not in {"left", "right"}:
            return None, f"reasoning={action}; direction unavailable"
        # A nudge can be short; a lane change/turn requires a larger lateral
        # excursion.  The trend is still inferred from every waypoint.
        threshold = 0.05 if action == "nudge" else 0.30
        magnitude = max(stats["peak_left_displacement_m"] or 0.0, abs(stats["peak_right_displacement_m"] or 0.0))
        if magnitude < threshold:
            return None, f"reasoning={action} {direction}; lateral excursion={magnitude:.3f} m is too small"
        ok = stats["lateral_trend"] == direction
        return ok, f"reasoning={action} {direction}; trajectory lateral={stats['lateral_trend']}"
    return None, f"reasoning action {action} is not covered by deterministic trajectory rules"


def _evidence(stats: dict[str, Any], matches: list[dict[str, Any]]) -> str:
    component_text = "; ".join(f"{item['action']}={item['matched']} ({item['evidence']})" for item in matches)
    return (
        f"64 waypoint indices checked; v {stats['speed_start_mps']}→{stats['speed_end_mps']} m/s "
        f"(Δ={stats['speed_delta_mps']}), x {stats['x_start_m']}→{stats['x_end_m']} m, "
        f"y {stats['y_start_m']}→{stats['y_end_m']} m, peak lateral +{stats['peak_left_displacement_m']} / "
        f"{stats['peak_right_displacement_m']} m, longitudinal={stats['longitudinal_trend']}, "
        f"lateral={stats['lateral_trend']}; {component_text}"
    )


def _evaluate(reasoning: Any, stats: dict[str, Any]) -> dict[str, Any]:
    intent = parse_reasoning(reasoning)
    if intent.get("ambiguous") or intent.get("confidence") == "low":
        return {"label": "uncertain", "consistency_score": None, "intent": intent, "components": [], "evidence": "Reasoning is ambiguous or low-confidence."}
    components = intent.get("action_components") or [{"action": intent.get("primary_action", "unknown"), "direction": intent.get("direction", "none")}]
    matches: list[dict[str, Any]] = []
    for component in components:
        matched, evidence = _match_component(component, stats)
        matches.append({"action": component.get("action", "unknown"), "direction": component.get("direction", "none"), "matched": matched, "evidence": evidence})
    known = [item["matched"] for item in matches if item["matched"] is not None]
    if not known:
        label = "uncertain"
        score = None
    elif all(known) and len(known) == len(matches):
        label = "consistent"
        score = 1.0
    elif any(known) and all(item is not False for item in [m["matched"] for m in matches]):
        label = "partially_consistent"
        score = None
    elif any(known) and any(item is False for item in [m["matched"] for m in matches]):
        label = "partially_consistent"
        score = None
    else:
        label = "inconsistent"
        score = 0.0
    return {"label": label, "consistency_score": score, "intent": intent, "components": matches, "evidence": _evidence(stats, matches)}


def _audit_record(record: dict[str, Any]) -> dict[str, Any]:
    clean, guided, validation_errors = _validate_pair(record.get("clean_waypoints"), record.get("guided_waypoints"))
    clean_stats = _trajectory_stats(clean)
    guided_stats = _trajectory_stats(guided)
    difference = _difference_stats(clean, guided)
    u1 = _evaluate(record.get("clean_reasoning", ""), clean_stats)
    unew = _evaluate(record.get("clean_reasoning", ""), guided_stats)
    if validation_errors:
        test = "FAIL"
        status = "invalid_input"
    elif u1["label"] == "uncertain" or unew["label"] == "uncertain":
        test = "REVIEW"
        status = "insufficient_evidence"
    else:
        test = "PASS"
        status = "checked"
    return {
        "clip_id": record.get("clip_id"),
        "t0_us": record.get("t0_us"),
        "mode": record.get("mode"),
        "alpha": _float(record.get("alpha")),
        "clean_reasoning": record.get("clean_reasoning"),
        "u1": {"label": u1["label"], "consistency_score": u1["consistency_score"], "trajectory_intent": clean_stats, "evidence": u1["evidence"], "components": u1["components"]},
        "unew": {"label": unew["label"], "consistency_score": unew["consistency_score"], "trajectory_intent": guided_stats, "evidence": unew["evidence"], "components": unew["components"]},
        "trajectory_difference": difference,
        "validation_errors": validation_errors,
        "audit_status": status,
        "test": test,
        "source_lm_judgement_used": False,
    }


def _key(record: dict[str, Any]) -> tuple[str, str, float]:
    return str(record.get("clip_id")), str(record.get("mode")), _float(record.get("alpha"))


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                errors.append({"line": line_number, "error": "JSON record is not an object"})
    return rows, errors


def _summary(results: list[dict[str, Any]], input_errors: list[dict[str, Any]], excluded: list[str], duplicate_count: int, input_path: Path) -> dict[str, Any]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_config[f"{row['mode']}|alpha={row['alpha']:g}"].append(row)
    summary: dict[str, Any] = {
        "input": str(input_path),
        "source_lm_judgement_used": False,
        "conventions": {"x": "forward", "y": "left", "waypoint_matching": "same index and timestamp", "score": "1.0 only for consistent; 0.0 only for inconsistent; null otherwise"},
        "records_written": len(results),
        "clips_in_results": len({row["clip_id"] for row in results}),
        "excluded_incomplete_clips": excluded,
        "input_parse_errors": input_errors,
        "duplicate_records_skipped": duplicate_count,
        "by_configuration": {},
    }
    for config, rows in sorted(by_config.items()):
        labels_u1 = Counter(row["u1"]["label"] for row in rows)
        labels_unew = Counter(row["unew"]["label"] for row in rows)
        u1_scores = [row["u1"]["consistency_score"] for row in rows if row["u1"]["consistency_score"] is not None]
        unew_scores = [row["unew"]["consistency_score"] for row in rows if row["unew"]["consistency_score"] is not None]
        mean = lambda xs: round(sum(xs) / len(xs), 6) if xs else None
        transitions = Counter(f"{row['u1']['label']}→{row['unew']['label']}" for row in rows)
        summary["by_configuration"][config] = {
            "num_records": len(rows), "u1_label_counts": dict(labels_u1), "unew_label_counts": dict(labels_unew),
            "u1_binary_mean": mean(u1_scores), "unew_binary_mean": mean(unew_scores),
            "num_u1_scored": len(u1_scores), "num_unew_scored": len(unew_scores),
            "trajectory_relation_counts": dict(Counter(row["trajectory_difference"]["relation"] for row in rows)),
            "label_transitions": dict(transitions),
            "test_counts": dict(Counter(row["test"] for row in rows)),
        }
    return summary


def run_self_tests() -> None:
    def w(x: float, y: float, v: float) -> dict[str, float]:
        return {"t_s": x / 10.0, "x_m": x, "y_m": y, "velocity_mps": v, "acceleration_mps2": 0.0, "curvature_inv_m": 0.0}
    keep = [w(i, 0.01 * math.sin(i), 5.0) for i in range(64)]
    left = [w(i, 0.10 * min(i / 20, 1.0), 5.0) for i in range(64)]
    stop = [w(i, 0.0, max(0.0, 5.0 - i * 0.10)) for i in range(64)]
    assert _evaluate("Keep lane to continue driving.", _trajectory_stats(_waypoint_arrays(keep))) ["label"] == "consistent"
    assert _evaluate("Nudge left to clear the obstacle.", _trajectory_stats(_waypoint_arrays(left))) ["label"] == "consistent"
    assert _evaluate("Nudge right to clear the obstacle.", _trajectory_stats(_waypoint_arrays(left))) ["label"] == "inconsistent"
    assert _evaluate("Stop at the stop line.", _trajectory_stats(_waypoint_arrays(stop))) ["label"] == "consistent"
    clean, guided, errors = _validate_pair(keep, keep)
    assert not errors and _difference_stats(clean, guided)["relation"] == "near_identical"
    print("PASS: deterministic trajectory evaluator self-tests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=False, help="Input intervention JSONL")
    parser.add_argument("--output", required=False, help="Output audit JSONL")
    parser.add_argument("--summary", required=False, help="Output summary JSON")
    parser.add_argument("--include-incomplete", action="store_true", help="Include clips that do not have 16 config records")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required")
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    summary_path = Path(args.summary).expanduser() if args.summary else output_path.with_name(output_path.stem + "_summary.json")
    rows, input_errors = _load_jsonl(input_path)
    counts = Counter(str(row.get("clip_id")) for row in rows)
    incomplete = sorted(clip for clip, count in counts.items() if count != EXPECTED_RECORDS_PER_CLIP)
    allowed_clips = set(counts) if args.include_incomplete else set(counts) - set(incomplete)

    completed: set[tuple[str, str, float]] = set()
    if output_path.exists():
        existing, _ = _load_jsonl(output_path)
        completed = {_key(row) for row in existing}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    duplicate_count = 0
    seen: set[tuple[str, str, float]] = set()
    mode_total = len(allowed_clips)
    print(f"Input records: {len(rows)}; complete clips: {len(allowed_clips)}; excluded incomplete clips: {len(incomplete)}")
    with output_path.open("a", encoding="utf-8") as output:
        processed = 0
        for row in rows:
            clip_id = str(row.get("clip_id"))
            if clip_id not in allowed_clips:
                continue
            key = _key(row)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            if key in completed:
                continue
            try:
                audited = _audit_record(row)
            except Exception as exc:  # keep the batch resumable if one row is malformed
                audited = {"clip_id": row.get("clip_id"), "mode": row.get("mode"), "alpha": _float(row.get("alpha")), "audit_status": "evaluator_error", "test": "FAIL", "error": repr(exc), "source_lm_judgement_used": False}
            output.write(json.dumps(audited, ensure_ascii=False) + "\n")
            output.flush()
            results.append(audited)
            processed += 1
            print(f"[{processed}] clip={clip_id} mode={row.get('mode')} alpha={_float(row.get('alpha')):g} test={audited.get('test')} u1={audited.get('u1', {}).get('label')} unew={audited.get('unew', {}).get('label')}", flush=True)
    existing, _ = _load_jsonl(output_path)
    summary = _summary(existing, input_errors, incomplete if not args.include_incomplete else [], duplicate_count, input_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved audit results to {output_path}")
    print(f"Saved audit summary to {summary_path}")


if __name__ == "__main__":
    main()
