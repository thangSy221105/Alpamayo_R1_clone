"""Run a diverse batch of history-only NAVSIM windows and report ADE/FDE."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from alpamayo_r1.navsim_adapter import (
    DEFAULT_CAMERA_NAMES,
    DEFAULT_MAX_IMAGE_TIME_ERROR_S,
    DEFAULT_MAX_POSE_GAP_S,
    IMAGE_HISTORY_OFFSETS_S,
    _source_camera_name,
    load_navsim_sample,
)
from alpamayo_r1.navsim_smoke_utils import extract_first_reasoning, validate_trajectory_output


DEFAULT_EXCLUDED_SCENES = (
    "log-0019-scene-0001",
    "log-0019-scene-0002",
    "log-0019-scene-0005",
    "log-0019-scene-0008",
    "log-0019-scene-0012",
)


def _sensor_file_exists(sensor_root: Path, raw_path: Any) -> bool:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False
    relative = Path(raw_path)
    if not relative.is_absolute() and ".." in relative.parts:
        return False
    root = sensor_root.resolve()
    candidate = relative.resolve() if relative.is_absolute() else (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate.is_file()


def _has_usable_history_images(
    history_frames: list[Any],
    sensor_root: Path,
    camera_names: tuple[str, ...],
    max_image_time_error_s: float,
) -> bool:
    try:
        timestamps = np.asarray([int(frame["timestamp"]) for frame in history_frames], dtype=np.int64)
        if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0):
            return False
        anchor_timestamp = int(timestamps[-1])
        for camera_name in camera_names:
            for target_offset_s in IMAGE_HISTORY_OFFSETS_S:
                target_us = anchor_timestamp + int(round(target_offset_s * 1_000_000))
                selected_index = int(np.argmin(np.abs(timestamps - target_us)))
                error_s = abs(int(timestamps[selected_index]) - target_us) / 1_000_000.0
                if error_s > max_image_time_error_s:
                    return False
                cams = history_frames[selected_index].get("cams", {})
                source_camera_name = _source_camera_name(camera_name)
                camera = cams.get(source_camera_name) if isinstance(cams, dict) else None
                if not isinstance(camera, dict) or not _sensor_file_exists(
                    sensor_root, camera.get("data_path")
                ):
                    return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def load_navtest_filter(config_path: Path) -> dict[str, Any]:
    """Load the official NAVSIM v1.1 ``scene_filter/navtest.yaml`` contract."""
    try:
        import yaml
    except ImportError as exc:  # PyYAML is also a dependency of the NAVSIM/Hydra stack.
        raise RuntimeError("PyYAML is required to read the official navtest filter") from exc

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"NAVTEST filter must contain a YAML mapping: {config_path}")

    history = config.get("num_history_frames")
    future = config.get("num_future_frames")
    interval = config.get("frame_interval")
    logs = config.get("log_names")
    tokens = config.get("tokens")
    has_route = config.get("has_route")
    if (history, future, interval, has_route) != (4, 10, 1, True):
        raise ValueError(
            "Expected the official NAVSIM v1.1 navtest window contract "
            "(history=4, future=10, frame_interval=1, has_route=true); "
            f"got {(history, future, interval, has_route)!r}"
        )
    if not isinstance(logs, list) or not logs or not all(isinstance(x, str) for x in logs):
        raise ValueError("NAVTEST filter must contain a non-empty log_names list")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(x, str) for x in tokens):
        raise ValueError("NAVTEST filter must contain a non-empty anchor tokens list")
    return {
        "log_names": set(logs),
        "tokens": set(tokens),
        "history_frames": history,
        "future_frames": future,
        "frame_interval": interval,
        "has_route": has_route,
        "source": str(config_path.resolve()),
    }


def discover_candidates(
    metadata_paths: list[Path],
    sensor_root: Path,
    history_frames: int,
    future_frames: int,
    camera_names: tuple[str, ...],
    max_image_time_error_s: float,
    excluded_scene_names: set[str],
    navtest_filter: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    window_size = history_frames + future_frames
    candidates: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    matched_navtest_logs = 0
    matched_navtest_tokens: set[str] = set()

    for metadata_path in metadata_paths:
        with metadata_path.open("rb") as stream:
            frames = pickle.load(stream)
        if not isinstance(frames, list):
            raise ValueError(f"Metadata root must be a list: {metadata_path}")

        source_key = str(metadata_path)
        source_counts[source_key] = 0

        if navtest_filter is not None:
            if metadata_path.stem not in navtest_filter["log_names"]:
                del frames
                gc.collect()
                continue
            matched_navtest_logs += 1
            token_to_index = {
                str(frame["token"]): index
                for index, frame in enumerate(frames)
                if isinstance(frame, dict)
                and isinstance(frame.get("token"), str)
                and frame["token"] in navtest_filter["tokens"]
            }
            for anchor_token, anchor_index in sorted(
                token_to_index.items(), key=lambda item: item[1]
            ):
                matched_navtest_tokens.add(anchor_token)
                window_start = anchor_index - history_frames + 1
                future_end = anchor_index + future_frames + 1
                if window_start < 0 or future_end > len(frames):
                    continue
                history = frames[window_start : anchor_index + 1]
                anchor = history[-1]
                route_ids = anchor.get("roadblock_ids") if isinstance(anchor, dict) else None
                if navtest_filter["has_route"] and (
                    not isinstance(route_ids, (list, tuple)) or not route_ids
                ):
                    continue
                scene_name = str(
                    anchor.get("scene_name") or anchor.get("log_name") or metadata_path.stem
                )
                if not _has_usable_history_images(
                    history,
                    sensor_root,
                    camera_names,
                    max_image_time_error_s,
                ):
                    continue
                candidates.append(
                    {
                        "metadata_path": source_key,
                        "scene_index": anchor_index,
                        "scene_name": scene_name,
                        "anchor_index": anchor_index,
                        "anchor_token": anchor_token,
                        "window_start": window_start,
                        "navsim_split": "navtest",
                    }
                )
                source_counts[source_key] += 1
            del frames
            gc.collect()
            continue

        route_scene_index = 0
        for start in range(0, len(frames) - window_size + 1, window_size):
            window = frames[start : start + window_size]
            anchor = window[history_frames - 1]
            route_ids = anchor.get("roadblock_ids") if isinstance(anchor, dict) else None
            if not isinstance(route_ids, (list, tuple)) or not route_ids:
                continue

            scene_index = route_scene_index
            route_scene_index += 1
            scene_name = str(anchor.get("scene_name") or anchor.get("log_name") or "unknown")
            if scene_name in excluded_scene_names:
                continue
            if not _has_usable_history_images(
                window[:history_frames],
                sensor_root,
                camera_names,
                max_image_time_error_s,
            ):
                continue

            candidates.append(
                {
                    "metadata_path": source_key,
                    "scene_index": scene_index,
                    "scene_name": scene_name,
                    "anchor_index": start + history_frames - 1,
                    "window_start": start,
                }
            )
            source_counts[source_key] += 1

        # One representative per NAVSIM scene keeps the audit from spending
        # most of its budget on adjacent windows from the same short scenario.
        source_candidates = [item for item in candidates if item["metadata_path"] == source_key]
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in source_candidates:
            by_scene[item["scene_name"]].append(item)
        if source_candidates:
            candidates = [item for item in candidates if item["metadata_path"] != source_key]
            representatives = [
                sorted(items, key=lambda item: item["scene_index"])[(len(items) - 1) // 2]
                for items in by_scene.values()
            ]
            candidates.extend(representatives)
            source_counts[source_key] = len(representatives)

        del frames
        gc.collect()

    if navtest_filter is not None:
        if matched_navtest_logs == 0:
            raise ValueError(
                "None of the metadata pickle filenames match the official NAVTEST log_names. "
                "Use metadata from OpenScene's test split, not mini/trainval."
            )
        if not matched_navtest_tokens:
            raise ValueError(
                "No official NAVTEST anchor tokens were found in the supplied test metadata."
            )

    return candidates, source_counts


def _evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[(len(items) - 1) // 2]]
    positions = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[position] for position in positions]


def limit_candidates_per_scene(
    candidates: list[dict[str, Any]], max_samples_per_scene: int
) -> list[dict[str, Any]]:
    """Keep at most N evenly spaced anchor windows from each NAVSIM scene."""
    if max_samples_per_scene <= 0:
        raise ValueError("max_samples_per_scene must be positive")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        scene_key = (candidate["metadata_path"], candidate["scene_name"])
        grouped[scene_key].append(candidate)

    selected = []
    for items in grouped.values():
        items.sort(key=lambda item: item["scene_index"])
        selected.extend(_evenly_spaced(items, max_samples_per_scene))
    return sorted(selected, key=lambda item: (item["metadata_path"], item["scene_index"]))


def select_diverse_candidates(
    candidates: list[dict[str, Any]], max_samples: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["metadata_path"]].append(candidate)
    for items in grouped.values():
        items.sort(key=lambda item: item["scene_index"])

    sources = sorted(grouped)
    total = sum(len(grouped[source]) for source in sources)
    target = min(max_samples, total)
    if target == 0:
        return []

    if len(sources) > target:
        source_positions = (
            [round(i * (len(sources) - 1) / (target - 1)) for i in range(target)]
            if target > 1
            else [(len(sources) - 1) // 2]
        )
        sources = [sources[position] for position in source_positions]

    quotas = {source: 1 for source in sources}
    remaining = target - len(sources)
    while remaining > 0:
        progressed = False
        for source in sources:
            if quotas[source] < len(grouped[source]):
                quotas[source] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    selected = [
        candidate
        for source in sources
        for candidate in _evenly_spaced(grouped[source], quotas[source])
    ]
    return sorted(selected, key=lambda item: (item["metadata_path"], item["scene_index"]))


def load_excluded_anchor_tokens(paths: list[Path] | None) -> set[str]:
    """Read anchor IDs from JSONL result files or plain one-token-per-line files."""
    tokens: set[str] = set()
    for path in paths or []:
        with path.open("r", encoding="utf-8") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, dict):
                    token = value.get("anchor_token")
                    if token is not None and str(token).strip():
                        tokens.add(str(token).strip())
                elif line:
                    tokens.add(line)
    return tokens


def deduplicate_candidates_by_anchor(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep one deterministic candidate per NAVSIM anchor token."""
    ordered = sorted(
        candidates,
        key=lambda item: (str(item["metadata_path"]), int(item["scene_index"])),
    )
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicates = 0
    for candidate in ordered:
        token = candidate.get("anchor_token")
        if token is None or not str(token).strip():
            key = f"{candidate['metadata_path']}::{candidate['scene_index']}"
        else:
            key = str(token).strip()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(candidate)
    return unique, duplicates


def calculate_ade_fde(
    prediction_xyz: np.ndarray,
    ground_truth_xyz: np.ndarray,
    ground_truth_times_s: np.ndarray,
    waypoint_dt_s: float,
) -> dict[str, Any]:
    if prediction_xyz.ndim != 2 or prediction_xyz.shape[1] < 2:
        raise ValueError(f"Expected prediction shape [T, >=2], got {prediction_xyz.shape}")
    if ground_truth_xyz.ndim != 2 or ground_truth_xyz.shape[1] < 2:
        raise ValueError(f"Expected GT shape [N, >=2], got {ground_truth_xyz.shape}")
    if len(ground_truth_xyz) != len(ground_truth_times_s) or len(ground_truth_xyz) == 0:
        raise ValueError("Ground-truth positions and timestamps must have equal nonzero lengths")
    if not math.isfinite(waypoint_dt_s) or waypoint_dt_s <= 0:
        raise ValueError(f"Invalid prediction waypoint interval: {waypoint_dt_s}")

    prediction_times_s = np.arange(1, len(prediction_xyz) + 1, dtype=np.float64) * waypoint_dt_s
    if (
        np.any(~np.isfinite(ground_truth_times_s))
        or ground_truth_times_s[0] < prediction_times_s[0] - 1e-6
        or ground_truth_times_s[-1] > prediction_times_s[-1] + 1e-6
    ):
        raise ValueError(
            "Prediction does not cover the GT time window: "
            f"pred=[{prediction_times_s[0]:.6f},{prediction_times_s[-1]:.6f}], "
            f"GT=[{ground_truth_times_s[0]:.6f},{ground_truth_times_s[-1]:.6f}]"
        )

    predicted_at_gt_times = np.column_stack(
        [
            np.interp(ground_truth_times_s, prediction_times_s, prediction_xyz[:, axis])
            for axis in range(2)
        ]
    )
    errors_m = np.linalg.norm(predicted_at_gt_times - ground_truth_xyz[:, :2], axis=1)
    return {
        "horizon_s": float(ground_truth_times_s[-1]),
        "sample_count": int(len(errors_m)),
        "ground_truth_time_offsets_s": ground_truth_times_s.tolist(),
        "ade_m": float(np.mean(errors_m)),
        "fde_m": float(errors_m[-1]),
        "per_sample_error_m": errors_m.tolist(),
        "comparison": "linear interpolation of predicted XY at each observed NAVSIM future-pose timestamp",
    }


def infer_one(
    data: dict[str, Any],
    model: Any,
    helper: Any,
    processor: Any,
    torch: Any,
    max_generation_length: int,
    seed: int,
) -> dict[str, Any]:
    if data["future_used_for_model_input"]:
        raise RuntimeError("Refusing evaluation: NAVSIM future was marked as model input")
    ground_truth = data.get("ground_truth_future")
    if not isinstance(ground_truth, dict) or ground_truth.get("used_for_model_input") is not False:
        raise RuntimeError("Missing explicitly separated, evaluator-only NAVSIM future GT")

    image_frames = torch.from_numpy(data["image_frames"])
    messages = helper.create_message(image_frames.flatten(0, 1))
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": tokenized,
            "ego_history_xyz": torch.from_numpy(data["ego_history_xyz"]),
            "ego_history_rot": torch.from_numpy(data["ego_history_rot"]),
        },
        "cuda",
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra, _sampled_action = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=max_generation_length,
            return_extra=True,
            return_action=True,
        )

    validate_trajectory_output(pred_xyz, pred_rot)
    prediction = pred_xyz.detach().float().cpu().numpy()[0, 0, 0]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Model returned NaN/Inf trajectory values")

    gt_xyz = np.asarray(ground_truth["ego_xyz"], dtype=np.float64)
    gt_times = np.asarray(ground_truth["time_offsets_s"], dtype=np.float64)
    metrics = calculate_ade_fde(
        prediction_xyz=prediction,
        ground_truth_xyz=gt_xyz,
        ground_truth_times_s=gt_times,
        waypoint_dt_s=float(model.action_space.dt),
    )
    cot = extract_first_reasoning(extra.get("cot") if hasattr(extra, "get") else None)

    return {
        "status": "ok",
        "scene_name": data["scene_name"],
        "log_name": data.get("log_name"),
        "navsim_split": data["navsim_split"],
        "anchor_token": data["anchor_token"],
        "anchor_timestamp_us": data["anchor_timestamp_us"],
        "reasoning": cot,
        "camera_names": data["camera_names"],
        "image_frame_indices": data["image_frame_indices"],
        "image_time_offsets_s": data["image_time_offsets_s"],
        "history_offsets_s": data["history_offsets_s"],
        "coordinate_frame": ground_truth["coordinate_frame"],
        "future_used_for_model_input": False,
        "future_frames_deserialized": data["future_frames_deserialized"],
        "predicted_waypoint_count": int(len(prediction)),
        "prediction_waypoint_dt_s": float(model.action_space.dt),
        "ground_truth_frame_indices": ground_truth["frame_indices"],
        "ground_truth_xy_m": gt_xyz[:, :2].tolist(),
        "prediction_xy_m": prediction[:, :2].tolist(),
        "ade_fde": metrics,
        "camera_mapping_note": data["camera_mapping_note"],
        "smoke_test_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-pkl", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--navtest-filter-yaml",
        type=Path,
        required=True,
        help=(
            "Required official NAVSIM v1.1 scene_filter/navtest.yaml; metadata is restricted "
            "to its test log_names and exact anchor tokens (frame_interval=1)."
        ),
    )
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--max-samples-per-scene",
        type=int,
        default=None,
        help="Keep at most this many evenly spaced NAVSIM anchor windows per scene.",
    )
    parser.add_argument(
        "--exclude-anchor-tokens-file",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="JSONL result or text file of anchor tokens to skip; may be passed repeatedly.",
    )
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERA_NAMES))
    parser.add_argument("--model", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--max-image-time-error-s", type=float, default=DEFAULT_MAX_IMAGE_TIME_ERROR_S)
    parser.add_argument("--max-pose-gap-s", type=float, default=DEFAULT_MAX_POSE_GAP_S)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--exclude-scene-names",
        default=",".join(DEFAULT_EXCLUDED_SCENES),
        help="Comma-separated scenes already smoke-tested; set '' to include all scenes.",
    )
    args = parser.parse_args()
    if args.max_samples <= 0 or args.min_samples <= 0 or args.min_samples > args.max_samples:
        raise ValueError("Require 0 < --min-samples <= --max-samples")
    if args.max_samples_per_scene is not None and args.max_samples_per_scene <= 0:
        raise ValueError("--max-samples-per-scene must be positive")
    if args.history_frames < 2 or args.future_frames < 1:
        raise ValueError("Need at least two history frames and one future frame")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    camera_names = tuple(name.strip() for name in args.cameras.split(",") if name.strip())
    if len(camera_names) not in (3, 4) or len(set(camera_names)) != len(camera_names):
        raise ValueError("--cameras must name three or four distinct NAVSIM cameras")
    metadata_paths = [path.resolve() for path in args.metadata_pkl]
    missing_metadata = [str(path) for path in metadata_paths if not path.is_file()]
    if missing_metadata:
        raise FileNotFoundError(f"Metadata pickle(s) not found: {missing_metadata}")
    if not args.sensor_root.is_dir():
        raise NotADirectoryError(f"Sensor root does not exist: {args.sensor_root}")
    output_path = args.output_jsonl.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; choose another path or pass --overwrite: {output_path}")

    excluded = {name.strip() for name in args.exclude_scene_names.split(",") if name.strip()}
    filter_path = args.navtest_filter_yaml.resolve()
    if not filter_path.is_file():
        raise FileNotFoundError(f"NAVTEST filter YAML not found: {filter_path}")
    navtest_filter = load_navtest_filter(filter_path)
    args.history_frames = navtest_filter["history_frames"]
    args.future_frames = navtest_filter["future_frames"]
    print(
        f"Dataset split: NAVSIM v1.1 navtest; official filter={filter_path}; "
        f"logs={len(navtest_filter['log_names'])}, anchor_tokens={len(navtest_filter['tokens'])}"
    )
    candidates, source_counts = discover_candidates(
        metadata_paths=metadata_paths,
        sensor_root=args.sensor_root,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        camera_names=camera_names,
        max_image_time_error_s=args.max_image_time_error_s,
        excluded_scene_names=excluded,
        navtest_filter=navtest_filter,
    )
    excluded_anchor_tokens = load_excluded_anchor_tokens(args.exclude_anchor_tokens_file)
    if excluded_anchor_tokens:
        before_exclusion = len(candidates)
        candidates = [
            item
            for item in candidates
            if str(item.get("anchor_token", "")).strip() not in excluded_anchor_tokens
        ]
        print(
            f"Excluded {before_exclusion - len(candidates)} candidate(s) matching "
            f"{len(excluded_anchor_tokens)} previously evaluated anchor token(s)."
        )
    candidates, duplicate_anchor_count = deduplicate_candidates_by_anchor(candidates)
    if duplicate_anchor_count:
        print(f"Removed {duplicate_anchor_count} duplicate candidate anchor(s).")
    source_counts = {
        source: sum(item["metadata_path"] == source for item in candidates)
        for source in source_counts
    }
    if args.max_samples_per_scene is not None:
        candidates = limit_candidates_per_scene(candidates, args.max_samples_per_scene)
        source_counts = {
            source: sum(item["metadata_path"] == source for item in candidates)
            for source in source_counts
        }
        print(
            f"Candidate windows after per-scene cap ({args.max_samples_per_scene}): "
            f"{len(candidates)} across "
            f"{len({(item['metadata_path'], item['scene_name']) for item in candidates})} scenes"
        )
    selected = select_diverse_candidates(candidates, args.max_samples)
    if len(selected) < args.min_samples:
        raise RuntimeError(
            f"Only {len(selected)} usable NAVSIM anchor windows were selected; "
            f"need at least {args.min_samples}. Add metadata/sensor shards or lower --min-samples."
        )

    print(f"Metadata files: {len(metadata_paths)}")
    print(f"Usable anchor windows after route/image checks: {len(candidates)}")
    print(f"Selected windows: {len(selected)} (limit={args.max_samples})")
    print("Usable windows by metadata file:")
    for source, count in sorted(source_counts.items()):
        if count:
            print(f"  {source}: {count}")
    print("Selected scene windows:")
    for item in selected:
        print(
            f"  {Path(item['metadata_path']).name} scene_index={item['scene_index']} "
            f"scene={item['scene_name']} anchor_frame={item['anchor_index']} "
            f"anchor_token={item.get('anchor_token', 'legacy-window')}"
        )

    import torch

    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run in the GPU AR1 environment")
    print(f"Loading model once: {args.model}", flush=True)
    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        grouped[item["metadata_path"]].append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    successful_metrics: list[dict[str, float]] = []
    attempted = 0
    failed = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output_stream:
        for source in sorted(grouped):
            with Path(source).open("rb") as stream:
                metadata_frames = pickle.load(stream)
            for item in sorted(grouped[source], key=lambda entry: entry["scene_index"]):
                attempted += 1
                prefix = f"[{attempted}/{len(selected)}] {item['scene_name']} index={item['scene_index']}"
                try:
                    data = load_navsim_sample(
                        metadata_path=source,
                        sensor_root=args.sensor_root,
                        scene_index=None if item.get("anchor_token") else item["scene_index"],
                        anchor_token=item.get("anchor_token"),
                        history_frame_count=args.history_frames,
                        future_frame_count=args.future_frames,
                        require_route=True,
                        camera_names=camera_names,
                        max_image_time_error_s=args.max_image_time_error_s,
                        max_pose_gap_s=args.max_pose_gap_s,
                        metadata_frames=metadata_frames,
                    )
                    result = infer_one(
                        data=data,
                        model=model,
                        helper=helper,
                        processor=processor,
                        torch=torch,
                        max_generation_length=args.max_generation_length,
                        seed=args.seed,
                    )
                    result["metadata_file"] = Path(source).name
                    result["source_scene_index"] = item["scene_index"]
                    result["dataset_split"] = item.get("navsim_split", "unfiltered")
                    successful_metrics.append(result["ade_fde"])
                    print(
                        f"{prefix} ADE={result['ade_fde']['ade_m']:.4f}m "
                        f"FDE={result['ade_fde']['fde_m']:.4f}m "
                        f"reasoning={result['reasoning']!r}",
                        flush=True,
                    )
                except Exception as exc:  # Record per-window failures and continue the batch.
                    failed += 1
                    result = {
                        "status": "error",
                        "metadata_file": Path(source).name,
                        "source_scene_index": item["scene_index"],
                        "scene_name": item["scene_name"],
                        "anchor_index": item["anchor_index"],
                        "anchor_token": item.get("anchor_token"),
                        "dataset_split": item.get("navsim_split", "unfiltered"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "future_used_for_model_input": False,
                    }
                    print(f"{prefix} ERROR {type(exc).__name__}: {exc}", flush=True)
                output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_stream.flush()
                if "data" in locals():
                    del data
                gc.collect()
                torch.cuda.empty_cache()
            del metadata_frames
            gc.collect()

    successful = len(successful_metrics)
    summary: dict[str, Any] = {
        "dataset_split": "navtest" if navtest_filter is not None else "unfiltered",
        "navtest_filter": navtest_filter["source"] if navtest_filter is not None else None,
        "selected_windows": len(selected),
        "attempted": attempted,
        "successful": successful,
        "failed": failed,
        "future_used_for_model_input": False,
        "metric_horizon": "NAVSIM 10-frame future window (nominally 5 seconds)",
        "ade_mean_m": float(np.mean([metric["ade_m"] for metric in successful_metrics])) if successful else None,
        "ade_median_m": float(np.median([metric["ade_m"] for metric in successful_metrics])) if successful else None,
        "fde_mean_m": float(np.mean([metric["fde_m"] for metric in successful_metrics])) if successful else None,
        "fde_median_m": float(np.median([metric["fde_m"] for metric in successful_metrics])) if successful else None,
        "output_jsonl": str(output_path),
        "limitations": [
            "Inference smoke/evaluation batch only; not an official NAVSIM score.",
            "Fourth view is a centered CAM_F0 digital crop, resized while "
            "preserving aspect ratio; it is not a calibrated "
            "30-degree or physical tele camera.",
            "OpenScene test poses are nominally about 2 Hz; model history poses are interpolated to 10 Hz by the adapter.",
            "When dataset_split is navtest, samples are restricted to the official NAVSIM v1.1 log/token filter.",
        ],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("BATCH SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary saved: {summary_path}")


if __name__ == "__main__":
    main()
