"""Generate paired reasoning-intervention trajectories for selected NAVSIM anchors.

Writes the selected intervention modes at the selected guidance alphas
(one row per mode/alpha pair for each anchor).
Future NAVSIM poses are used only for ADE/FDE and never enter model inputs.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from alpamayo_r1.navsim_adapter import DEFAULT_CAMERA_NAMES, load_navsim_sample

MODES = ("no_reasoning", "noisy", "cross_scene", "opposite_action")
DEFAULT_ALPHAS = (0.0, 0.5, 1.0, 2.0)
NOISE_SUFFIX = " Unrelated note: decorative roadside signs are visible."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_noisy_cache(path: Path, strategy: str) -> dict[str, dict[str, str]]:
    cached: dict[str, dict[str, str]] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                token = str(record.get("anchor_token", record.get("clip_id", ""))).strip()
                noisy_reasoning = str(record["noisy_reasoning"]).strip()
                if token and noisy_reasoning and record.get("strategy") == strategy:
                    cached[token] = {
                        "primary_action": str(record.get("primary_action", "unknown")),
                        "noisy_reasoning": noisy_reasoning,
                        "generation_model": str(record.get("generation_model", "")),
                        "strategy": strategy,
                    }
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return cached


def append_noisy_cache(path: Path, anchor_token: str, noise: dict[str, str]) -> None:
    record = {"anchor_token": anchor_token, **noise}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def heuristic_action_conflict(clean_reasoning: str) -> dict[str, str]:
    text = clean_reasoning.lower()
    if any(word in text for word in ("stop", "yield", "decelerate", "slow")):
        return {"primary_action": "stop_or_slow", "noisy_reasoning": "Continue forward and accelerate."}
    if "left" in text:
        return {"primary_action": "leftward_motion", "noisy_reasoning": "Nudge right while continuing forward."}
    if "right" in text:
        return {"primary_action": "rightward_motion", "noisy_reasoning": "Nudge left while continuing forward."}
    if any(word in text for word in ("accelerate", "proceed")):
        return {"primary_action": "accelerate_or_proceed", "noisy_reasoning": "Decelerate and prepare to stop."}
    return {"primary_action": "unknown", "noisy_reasoning": "Make a rightward lane change and accelerate."}


def lm_action_conflict(client: Any, model_name: str, clean_reasoning: str) -> dict[str, str]:
    prompt = (
        "Create the opposite driving action for a robustness test. "
        "Map LEFT to RIGHT and RIGHT to LEFT. "
        "Map ACCELERATE to DECELERATE and DECELERATE to ACCELERATE. "
        "Use a concrete short action, not a refusal such as do not, avoid, or stay. "
        "Both fields must describe the same opposite action. "
        "Return exactly two JSON fields: primary_action and noisy_reasoning. "
        "Keep every value under 12 words.\n\n"
        f"Input action: {clean_reasoning}"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "none"},
    )
    raw = response.choices[0].message.content if hasattr(response, "choices") else response
    if isinstance(raw, list):
        raw = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in raw)
    cleaned = str(raw or "").strip().removeprefix("```json").removesuffix("```").strip()
    try:
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end <= start:
                raise
            payload = json.loads(cleaned[start : end + 1])
        noisy_reasoning = str(payload["noisy_reasoning"]).strip()
        if not noisy_reasoning:
            raise ValueError("empty noisy_reasoning")
        return {
            "primary_action": str(payload.get("primary_action", "unknown")),
            "noisy_reasoning": noisy_reasoning,
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if cleaned:
            return {"primary_action": "lm_text_fallback", "noisy_reasoning": cleaned}
        print(f"Warning: noise LM returned invalid JSON ({exc}); using heuristic conflict.", flush=True)
        return heuristic_action_conflict(clean_reasoning)


def load_navtest_filter(path: Path) -> tuple[set[str], set[str]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to validate NAVTEST anchors") from exc
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"NAVTEST filter is not a mapping: {path}")
    logs, tokens = config.get("log_names"), config.get("tokens")
    if not isinstance(logs, list) or not isinstance(tokens, list):
        raise ValueError("NAVTEST filter must contain log_names and tokens lists")
    return set(map(str, logs)), set(map(str, tokens))


def index_metadata(
    metadata_root: Path, wanted_tokens: set[str], allowed_logs: set[str]
) -> dict[str, tuple[Path, int]]:
    found: dict[str, tuple[Path, int]] = {}
    paths = sorted(metadata_root.rglob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No metadata .pkl files under {metadata_root}")
    for path in paths:
        if path.stem not in allowed_logs:
            continue
        with path.open("rb") as stream:
            frames = pickle.load(stream)
        if not isinstance(frames, list):
            raise ValueError(f"Metadata pickle is not a frame list: {path}")
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            token = str(frame.get("token", ""))
            if token not in wanted_tokens:
                continue
            if token in found:
                raise ValueError(f"Anchor token appears in multiple metadata files: {token}")
            found[token] = (path, frame_index)
        del frames
    missing = sorted(wanted_tokens - found.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} selected anchor(s) not found in metadata; first missing: {missing[:5]}"
        )
    return found


def find_cross_scene_source(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    scene = str(rows[index].get("scene_name", ""))
    for offset in range(1, len(rows)):
        candidate = rows[(index + offset) % len(rows)]
        if str(candidate.get("scene_name", "")) != scene:
            return candidate
    raise ValueError("cross_scene requires at least two distinct NAVSIM scenes")


def perturbation(mode: str, clean: str, cross_scene: str, noisy_reasoning: str) -> str:
    if mode == "no_reasoning":
        return ""
    if mode == "noisy":
        return noisy_reasoning
    if mode == "cross_scene":
        return cross_scene
    if mode == "opposite_action":
        return clean + " Therefore, make a sharp rightward lane change and accelerate."
    raise ValueError(f"Unsupported intervention mode: {mode}")


def make_model_inputs(data: dict[str, Any], processor: Any, helper: Any, torch: Any, reasoning: str):
    frames = torch.from_numpy(np.asarray(data["image_frames"]))
    messages = helper.create_message(frames.flatten(0, 1), forced_reasoning=reasoning)
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return helper.to_device(
        {
            "tokenized_data": tokenized,
            "ego_history_xyz": torch.from_numpy(np.asarray(data["ego_history_xyz"])),
            "ego_history_rot": torch.from_numpy(np.asarray(data["ego_history_rot"])),
        },
        "cuda",
    )


def calculate_ade_fde(
    prediction_xyz: np.ndarray, gt_xyz: np.ndarray, gt_times_s: np.ndarray, waypoint_dt_s: float
) -> dict[str, Any]:
    prediction_times = np.arange(1, len(prediction_xyz) + 1, dtype=np.float64) * waypoint_dt_s
    if gt_times_s[0] < prediction_times[0] - 1e-6 or gt_times_s[-1] > prediction_times[-1] + 1e-6:
        raise ValueError("Predicted trajectory does not cover the NAVSIM GT window")
    predicted_at_gt = np.column_stack(
        [np.interp(gt_times_s, prediction_times, prediction_xyz[:, axis]) for axis in range(2)]
    )
    errors = np.linalg.norm(predicted_at_gt - gt_xyz[:, :2], axis=1)
    return {
        "horizon_s": float(gt_times_s[-1]),
        "sample_count": int(len(errors)),
        "ade_m": float(errors.mean()),
        "fde_m": float(errors[-1]),
        "per_sample_error_m": errors.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--navtest-filter-yaml", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True, help="Merged sensor root containing all selected NAVTEST images")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERA_NAMES))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES),
        default=list(MODES),
        help="Intervention modes to run; all four are used by default.",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument(
        "--noisy-strategy",
        choices=["irrelevant", "heuristic_conflict", "lm_conflict"],
        default="irrelevant",
        help="How mode=noisy is generated; lm_conflict uses LM API for an opposite driving action.",
    )
    parser.add_argument("--noise-text", default=NOISE_SUFFIX)
    parser.add_argument("--noisy-model", default=None, help="LM model for noise; defaults to --lm-model.")
    parser.add_argument("--lm-model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument(
        "--noisy-cache",
        type=Path,
        default=None,
        help="Cache of LM-generated noisy reasonings; defaults beside the output JSONL.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Use a small subset for a smoke run")
    parser.add_argument("--preflight-only", action="store_true", help="Check every selected sample's metadata and images, without loading the model")
    parser.add_argument("--model", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.metadata_root.is_dir() or not args.sensor_root.is_dir():
        raise NotADirectoryError("metadata-root and sensor-root must both exist")
    if not args.navtest_filter_yaml.is_file():
        raise FileNotFoundError(args.navtest_filter_yaml)
    if len(args.cameras) not in (3, 4) or len(set(args.cameras)) != len(args.cameras):
        raise ValueError("Choose three or four distinct NAVSIM cameras")
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("Do not repeat intervention modes")
    if args.noisy_strategy == "lm_conflict" and "noisy" not in args.modes:
        raise ValueError("--noisy-strategy lm_conflict requires --modes to include noisy")
    if len(args.alphas) != 4 or len(set(args.alphas)) != 4 or any(alpha < 0 for alpha in args.alphas):
        raise ValueError("Provide four distinct nonnegative alpha values")

    source_rows = read_jsonl(args.input_jsonl)
    rows = source_rows
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("Input contains no selected samples")
    tokens = [str(row.get("anchor_token", "")).strip() for row in rows]
    if any(not token for token in tokens) or len(tokens) != len(set(tokens)):
        raise ValueError("Every input row must have a unique, non-empty anchor_token")
    if any(str(row.get("status", "ok")) != "ok" for row in rows):
        raise ValueError("Input contains a non-successful source record")
    if any(not isinstance(row.get("reasoning"), str) or not row["reasoning"].strip() for row in rows):
        raise ValueError("Every sample must have non-empty clean reasoning")
    allowed_logs, official_tokens = load_navtest_filter(args.navtest_filter_yaml)
    if "cross_scene" in args.modes and len({str(row.get("scene_name", "")) for row in source_rows}) < 2:
        raise ValueError("cross_scene requires at least two distinct scenes in the input JSONL")
    outside_filter = sorted(set(tokens) - official_tokens)
    if outside_filter:
        raise ValueError(f"{len(outside_filter)} input anchor(s) are outside the NAVTEST filter")

    output_path = args.output_jsonl.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; choose another path or pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    noisy_cache_path = (args.noisy_cache or output_path.with_name(output_path.stem + "_noisy_reasonings.jsonl")).resolve()
    noisy_cache_path.parent.mkdir(parents=True, exist_ok=True)
    token_to_metadata = index_metadata(args.metadata_root, set(tokens), allowed_logs)
    metadata_groups: dict[Path, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        metadata_path, _ = token_to_metadata[str(row["anchor_token"])]
        metadata_groups[metadata_path].append((row_index, row))

    if args.preflight_only:
        checked = 0
        for metadata_path in sorted(metadata_groups):
            with metadata_path.open("rb") as stream:
                metadata_frames = pickle.load(stream)
            for _, row in metadata_groups[metadata_path]:
                data = load_navsim_sample(
                    metadata_path=metadata_path,
                    sensor_root=args.sensor_root,
                    anchor_token=str(row["anchor_token"]),
                    history_frame_count=4,
                    future_frame_count=10,
                    require_route=True,
                    camera_names=tuple(args.cameras),
                    metadata_frames=metadata_frames,
                )
                if data["future_used_for_model_input"]:
                    raise RuntimeError(f"Future data leaked into model input for {row['anchor_token']}")
                checked += 1
                del data
            del metadata_frames
        print(
            f"PREFLIGHT PASS: {checked} NAVTEST samples; metadata, selected camera images, "
            "history and separate future-GT structure verified.",
            flush=True,
        )
        return

    noisy_cache: dict[str, dict[str, str]] = {}
    if "noisy" in args.modes:
        noisy_cache = load_noisy_cache(noisy_cache_path, args.noisy_strategy)
        if args.noisy_strategy == "irrelevant":
            for row in rows:
                token = str(row["anchor_token"])
                noisy_cache.setdefault(
                    token,
                    {
                        "primary_action": "not_applicable",
                        "noisy_reasoning": str(row["reasoning"]).strip() + args.noise_text,
                        "generation_model": "",
                        "strategy": args.noisy_strategy,
                    },
                )
        elif args.noisy_strategy == "heuristic_conflict":
            for row in rows:
                token = str(row["anchor_token"])
                if token not in noisy_cache:
                    noise = heuristic_action_conflict(str(row["reasoning"]).strip())
                    noisy_cache[token] = {
                        **noise,
                        "generation_model": "heuristic",
                        "strategy": args.noisy_strategy,
                    }
                    append_noisy_cache(noisy_cache_path, token, noisy_cache[token])
        elif args.noisy_strategy == "lm_conflict":
            missing_noise = [row for row in rows if str(row["anchor_token"]) not in noisy_cache]
            if missing_noise:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError(
                        "Install OpenAI client in the AR1 environment before LM noise generation."
                    ) from exc
                api_key = os.getenv("OPENAI_API_KEY") or getpass.getpass(
                    "Enter LM API key for noisy-reasoning generation (input hidden): "
                )
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY or an interactive LM API key is required")
                client_kwargs: dict[str, Any] = {"api_key": api_key}
                if args.base_url:
                    client_kwargs["base_url"] = args.base_url
                lm_client = OpenAI(**client_kwargs)
                noise_model = args.noisy_model or args.lm_model
                print(
                    f"Generating {len(missing_noise)} noisy reasoning(s) with LM API model={noise_model}; "
                    f"cache={noisy_cache_path}",
                    flush=True,
                )
                for index, row in enumerate(missing_noise, 1):
                    token = str(row["anchor_token"])
                    noise = lm_action_conflict(lm_client, noise_model, str(row["reasoning"]).strip())
                    noisy_cache[token] = {
                        **noise,
                        "generation_model": noise_model,
                        "strategy": args.noisy_strategy,
                    }
                    append_noisy_cache(noisy_cache_path, token, noisy_cache[token])
                    print(f"[noise {index}/{len(missing_noise)}] {token}", flush=True)
            else:
                print(f"Reusing cached LM noisy reasonings: {len(rows)}", flush=True)

    import torch

    from alpamayo_r1 import helper
    from alpamayo_r1.evaluate_reasoning_intervention import (
        action_summary,
        action_values,
        clamp_action_to_physical_bounds,
        dynamic_waypoints,
        trajectory_summary,
    )
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run in the GPU AR1 environment")
    print(
        f"Samples={len(rows)} modes={args.modes} alphas={args.alphas}; "
        f"expected records={len(rows) * len(args.modes) * len(args.alphas)}",
        flush=True,
    )
    print(f"Loading model once: {args.model}", flush=True)
    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for metadata_path in sorted(metadata_groups):
            with metadata_path.open("rb") as stream:
                metadata_frames = pickle.load(stream)
            for row_index, row in metadata_groups[metadata_path]:
                sample_index = row_index + 1
                token = str(row["anchor_token"])
                scene = str(row.get("scene_name", ""))
                clean_reasoning = row["reasoning"].strip()
                cross_source = (
                    find_cross_scene_source(source_rows, row_index)
                    if "cross_scene" in args.modes
                    else None
                )
                cross_reasoning = (
                    str(cross_source["reasoning"]).strip() if cross_source is not None else ""
                )
                noise_metadata = noisy_cache.get(token)
                data = load_navsim_sample(
                    metadata_path=metadata_path,
                    sensor_root=args.sensor_root,
                    anchor_token=token,
                    history_frame_count=4,
                    future_frame_count=10,
                    require_route=True,
                    camera_names=tuple(args.cameras),
                    metadata_frames=metadata_frames,
                )
                if data["future_used_for_model_input"]:
                    raise RuntimeError(f"Future data leaked into model input for {token}")

                history_xyz = torch.from_numpy(np.asarray(data["ego_history_xyz"])).to("cuda")[:, -1]
                history_rot = torch.from_numpy(np.asarray(data["ego_history_rot"])).to("cuda")[:, -1]
                clean_inputs = make_model_inputs(data, processor, helper, torch, clean_reasoning)
                seed = args.seed + sample_index
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    clean_xyz, _clean_rot, u1 = model.sample_trajectory_from_forced_reasoning(clean_inputs)
                clean_xyz_np = clean_xyz.detach().float().cpu().numpy()[0]
                clean_xy = clean_xyz_np[:, :2]
                gt = data["ground_truth_future"]
                gt_xyz = np.asarray(gt["ego_xyz"], dtype=np.float64)
                gt_times = np.asarray(gt["time_offsets_s"], dtype=np.float64)

                for mode in args.modes:
                    altered_reasoning = perturbation(
                        mode,
                        clean_reasoning,
                        cross_reasoning,
                        noise_metadata["noisy_reasoning"] if mode == "noisy" and noise_metadata else "",
                    )
                    altered_inputs = make_model_inputs(data, processor, helper, torch, altered_reasoning)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        perturbed_xyz, _perturbed_rot, u2 = model.sample_trajectory_from_forced_reasoning(altered_inputs)
                    perturbed_xyz_np = perturbed_xyz.detach().float().cpu().numpy()[0]
                    perturbed_xy = perturbed_xyz_np[:, :2]

                    for alpha in args.alphas:
                        guided_action, saturation_rate = clamp_action_to_physical_bounds(
                            u1 + float(alpha) * (u1 - u2), model.action_space
                        )
                        with torch.inference_mode():
                            guided_xyz, _guided_rot = model.action_space.action_to_traj(
                                guided_action, history_xyz, history_rot
                            )
                        guided_xyz_np = guided_xyz.detach().float().cpu().numpy()[0]
                        guided_xy = guided_xyz_np[:, :2]
                        record = {
                            "status": "ok",
                            "dataset_split": "navtest",
                            "scene_name": scene,
                            "anchor_token": token,
                            "clip_id": token,
                            "clean_reasoning": clean_reasoning,
                            "mode": mode,
                            "alpha": float(alpha),
                            "perturbed_reasoning": altered_reasoning,
                            "noisy_strategy": args.noisy_strategy if mode == "noisy" else None,
                            "noisy_primary_action": (
                                noise_metadata.get("primary_action")
                                if mode == "noisy" and noise_metadata
                                else None
                            ),
                            "noisy_generation_model": (
                                noise_metadata.get("generation_model")
                                if mode == "noisy" and noise_metadata
                                else None
                            ),
                            "cross_scene_source_anchor_token": (
                                str(cross_source["anchor_token"])
                                if mode == "cross_scene" and cross_source is not None
                                else None
                            ),
                            "formula": "u_new = u1 + alpha * (u1 - u2), with physical acceleration/curvature clipping",
                            "camera_names": data["camera_names"],
                            "future_used_for_model_input": False,
                            "clean_action_u1": action_summary(u1),
                            "perturbed_action_u2": action_summary(u2),
                            "guided_action_u_new_values": action_values(guided_action),
                            "control_saturation_rate_after_guidance": float(saturation_rate),
                            "clean_trajectory": trajectory_summary(clean_xy),
                            "perturbed_trajectory": trajectory_summary(perturbed_xy),
                            "guided_trajectory": trajectory_summary(guided_xy),
                            "clean_waypoints": dynamic_waypoints(
                                clean_xy, u1, model.action_space, history_xyz, history_rot
                            ),
                            "perturbed_waypoints": dynamic_waypoints(
                                perturbed_xy, u2, model.action_space, history_xyz, history_rot
                            ),
                            "guided_waypoints": dynamic_waypoints(
                                guided_xy, guided_action, model.action_space, history_xyz, history_rot
                            ),
                            "ground_truth_future": {
                                "coordinate_frame": gt["coordinate_frame"],
                                "used_for_model_input": False,
                                "time_offsets_s": gt_times.tolist(),
                                "ego_xyz": gt_xyz.tolist(),
                            },
                            "ade_fde": calculate_ade_fde(
                                guided_xyz_np, gt_xyz, gt_times, float(model.action_space.dt)
                            ),
                        }
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                print(
                    f"[{sample_index}/{len(rows)}] {scene} {token}: "
                    f"{len(args.modes) * len(args.alphas)} records",
                    flush=True,
                )
            del metadata_frames

    record_count = sum(1 for line in output_path.open("r", encoding="utf-8") if line.strip())
    expected_count = len(rows) * len(args.modes) * len(args.alphas)
    if record_count != expected_count:
        raise RuntimeError(f"Wrote {record_count} records; expected {expected_count}")
    summary = {
        "dataset_split": "navtest",
        "source_jsonl": str(args.input_jsonl.resolve()),
        "samples": len(rows),
        "modes": list(args.modes),
        "alphas": list(args.alphas),
        "noisy_strategy": args.noisy_strategy if "noisy" in args.modes else None,
        "noisy_model": (args.noisy_model or args.lm_model)
        if "noisy" in args.modes and args.noisy_strategy == "lm_conflict"
        else None,
        "noisy_cache": str(noisy_cache_path) if "noisy" in args.modes else None,
        "records": record_count,
        "future_used_for_model_input": False,
        "output_jsonl": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
