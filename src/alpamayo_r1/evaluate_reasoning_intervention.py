# SPDX-License-Identifier: Apache-2.0
"""Paired reasoning-intervention evaluation for Alpamayo-R1.

For each identical camera/egomotion input this script:
  1. lets AR1 generate its normal Chain-of-Causation (CoC);
  2. decodes an action conditioned on that *forced copy* of the CoC (u1);
  3. decodes another action conditioned on a perturbed CoC (u2); and
  4. evaluates u_new = u1 + alpha * (u1 - u2), in normalized unicycle-control
     space, after physically clipping acceleration and curvature.

This is an intervention/sensitivity probe, not an accuracy benchmark: it never
uses the ground-truth future trajectory.  It answers whether changing the CoC
can causally alter the action expert's output under otherwise identical input.

Example:
    python src/alpamayo_r1/evaluate_reasoning_intervention.py \
      --clip-ids-file test_clip_ids_100.txt \
      --output reasoning_intervention.jsonl \
      --modes no_reasoning noisy cross_scene \
      --alphas 0 0.25 0.5 1.0
"""

from __future__ import annotations

import argparse
from datetime import datetime
import getpass
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alpamayo_r1 import helper
from alpamayo_r1.evaluate_reasoning_action import (
    build_judge_prompt,
    judge_with_lm,
)
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_NOISE_TEXT = " Unrelated note: decorative roadside signs are visible."


def read_clip_ids(path: str | None) -> list[str]:
    if path is None:
        return [DEFAULT_CLIP_ID]
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    ids = [clip_id for clip_id in ids if clip_id and not clip_id.startswith("#")]
    if not ids:
        raise ValueError(f"No clip IDs found in {path}")
    return ids


def as_reasoning(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(item) for item in value)
    return str(value)


def trajectory_summary(xy: np.ndarray) -> dict[str, Any]:
    displacement = xy[-1] - xy[0]
    return {
        "start_xy_m": np.round(xy[0], 3).tolist(),
        "end_xy_m": np.round(xy[-1], 3).tolist(),
        "displacement_xy_m": np.round(displacement, 3).tolist(),
        "final_lateral_y_m": round(float(xy[-1, 1]), 3),
        "max_abs_lateral_y_m": round(float(np.abs(xy[:, 1]).max()), 3),
        "path_length_m": round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 3),
    }


def action_summary(action: torch.Tensor) -> dict[str, float]:
    """Compact control statistics for an action tensor shaped [1, 64, 2]."""
    values = action.detach().float().cpu().numpy()[0]
    return {
        "mean_normalized_accel": round(float(values[:, 0].mean()), 5),
        "mean_normalized_curvature": round(float(values[:, 1].mean()), 5),
        "max_abs_normalized_accel": round(float(np.abs(values[:, 0]).max()), 5),
        "max_abs_normalized_curvature": round(float(np.abs(values[:, 1]).max()), 5),
    }


def action_values(action: torch.Tensor) -> list[list[float]]:
    """Return all normalized [acceleration, curvature] controls for one sample."""
    return np.round(action.detach().float().cpu().numpy()[0], 6).tolist()


def dynamic_waypoints(
    xy: np.ndarray,
    action: torch.Tensor,
    action_space: Any,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
) -> list[dict[str, float]]:
    """Attach physical acceleration, curvature, and velocity to each waypoint."""
    action_physical = action.detach().float()
    accel = action_physical[..., 0] * action_space.accel_std + action_space.accel_mean
    curvature = (
        action_physical[..., 1] * action_space.curvature_std
        + action_space.curvature_mean
    )
    t0_states = action_space.estimate_t0_states(history_xyz, history_rot)
    velocity = t0_states["v"].float().unsqueeze(-1) + torch.cumsum(
        accel * action_space.dt, dim=-1
    )
    accel_np = accel.cpu().numpy()[0]
    curvature_np = curvature.cpu().numpy()[0]
    velocity_np = velocity.cpu().numpy()[0]
    return [
        {
            "t_s": round((index + 1) * action_space.dt, 2),
            "x_m": round(float(point[0]), 3),
            "y_m": round(float(point[1]), 3),
            "velocity_mps": round(float(velocity_np[index]), 3),
            "acceleration_mps2": round(float(accel_np[index]), 3),
            "curvature_inv_m": round(float(curvature_np[index]), 5),
        }
        for index, point in enumerate(xy)
    ]


def clamp_action_to_physical_bounds(action: torch.Tensor, action_space: Any) -> tuple[torch.Tensor, float]:
    """Clip physical a/kappa bounds, then return to the model's normalized space."""
    accel = action[..., 0]
    curvature = action[..., 1]
    accel_physical = accel * action_space.accel_std + action_space.accel_mean
    curvature_physical = curvature * action_space.curvature_std + action_space.curvature_mean
    accel_clamped = accel_physical.clamp(*action_space.accel_bounds)
    curvature_clamped = curvature_physical.clamp(*action_space.curvature_bounds)
    saturation_rate = float(
        ((accel_physical != accel_clamped) | (curvature_physical != curvature_clamped))
        .float()
        .mean()
        .item()
    )
    normalized = torch.stack(
        [
            (accel_clamped - action_space.accel_mean) / action_space.accel_std,
            (curvature_clamped - action_space.curvature_mean) / action_space.curvature_std,
        ],
        dim=-1,
    )
    return normalized, saturation_rate


def make_model_inputs(
    data: dict[str, Any], processor: Any, device: str, forced_reasoning: str | None
) -> dict[str, Any]:
    messages = helper.create_message(
        data["image_frames"].flatten(0, 1), forced_reasoning=forced_reasoning
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        device,
    )


def load_reasoning_cache(path: Path) -> dict[str, str]:
    cached: dict[str, str] = {}
    if not path.exists():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            cached[str(record["clip_id"])] = str(record["reasoning"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return cached


def append_reasoning_cache(path: Path, clip_id: str, reasoning: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"clip_id": clip_id, "reasoning": reasoning}, ensure_ascii=False) + "\n")


def load_noisy_cache(path: Path) -> dict[str, dict[str, str]]:
    cached: dict[str, dict[str, str]] = {}
    if not path.exists():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            cached[str(record["clip_id"])] = {
                "noisy_reasoning": str(record["noisy_reasoning"]),
                "primary_action": str(record.get("primary_action", "")),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return cached


def append_noisy_cache(path: Path, clip_id: str, noise: dict[str, str]) -> None:
    record = {"clip_id": clip_id, **noise}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def heuristic_action_conflict(clean_reasoning: str) -> dict[str, str]:
    """Create a deterministic opposite action when no noise-generation LM is used."""
    reasoning_lower = clean_reasoning.lower()
    if any(word in reasoning_lower for word in ("stop", "yield", "decelerate", "slow")):
        return {
            "primary_action": "stop_or_slow",
            "noisy_reasoning": "Continue forward and accelerate.",
        }
    if "left" in reasoning_lower:
        return {
            "primary_action": "leftward_motion",
            "noisy_reasoning": "Nudge right while continuing forward.",
        }
    if "right" in reasoning_lower:
        return {
            "primary_action": "rightward_motion",
            "noisy_reasoning": "Nudge left while continuing forward.",
        }
    if any(word in reasoning_lower for word in ("accelerate", "proceed")):
        return {
            "primary_action": "accelerate_or_proceed",
            "noisy_reasoning": "Decelerate and prepare to stop.",
        }
    return {
        "primary_action": "unknown",
        "noisy_reasoning": "Make a rightward lane change and accelerate.",
    }


def lm_action_conflict(client: Any, model_name: str, clean_reasoning: str) -> dict[str, str]:
    """Ask an LM for one concise action contradiction, without scene hallucinations."""
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
    if isinstance(response, str):
        raw_output = response
    elif hasattr(response, "choices"):
        raw_output = response.choices[0].message.content
    else:
        raw_output = response["choices"][0]["message"]["content"]
    if isinstance(raw_output, list):
        raw_output = " ".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block)
            for block in raw_output
        )
    raw_output = str(raw_output or "")
    cleaned = raw_output.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            # Recover a JSON object if the proxy wrapped it in extra text.
            object_start = cleaned.find("{")
            object_end = cleaned.rfind("}")
            if object_start < 0 or object_end <= object_start:
                raise
            payload = json.loads(cleaned[object_start : object_end + 1])
        noisy_reasoning = str(payload["noisy_reasoning"]).strip()
        if not noisy_reasoning:
            raise ValueError("empty noisy_reasoning")
        return {
            "primary_action": str(payload.get("primary_action", "unknown")),
            "noisy_reasoning": noisy_reasoning,
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if cleaned:
            # If the model returned a plain-text imperative rather than JSON,
            # preserve that generated text as the intervention instead of
            # discarding it.  The result records the fallback source.
            return {
                "primary_action": "lm_text_fallback",
                "noisy_reasoning": cleaned,
            }
        # Some OpenAI-compatible proxies ignore response_format and return an
        # empty/non-JSON answer.  Keep the benchmark running with a deterministic
        # contradiction rather than silently using malformed text as reasoning.
        print(
            "Warning: noise LM returned invalid JSON "
            f"({exc}); falling back to heuristic_conflict.",
            flush=True,
        )
        return heuristic_action_conflict(clean_reasoning)


def existing_keys(path: Path, overwrite: bool) -> set[tuple[str, str, float]]:
    if overwrite or not path.exists():
        return set()
    keys: set[tuple[str, str, float]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            keys.add((str(record["clip_id"]), str(record["mode"]), float(record["alpha"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return keys


def build_perturbation(
    mode: str,
    clean_reasoning: str,
    cross_reasoning: str | None,
    noisy_reasoning: str,
) -> str:
    if mode == "no_reasoning":
        return ""
    if mode == "noisy":
        return noisy_reasoning
    if mode == "cross_scene":
        if not cross_reasoning:
            raise ValueError("cross_scene needs at least two clips with cached reasoning traces.")
        return cross_reasoning
    if mode == "opposite_action":
        return clean_reasoning + " Therefore, make a sharp rightward lane change and accelerate."
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-ids-file", default=None)
    parser.add_argument("--output", default="reasoning_intervention.jsonl")
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["no_reasoning", "noisy", "cross_scene", "opposite_action"],
        default=["no_reasoning", "noisy", "cross_scene"],
    )
    parser.add_argument("--noise-text", default=DEFAULT_NOISE_TEXT)
    parser.add_argument(
        "--noisy-strategy",
        choices=["irrelevant", "heuristic_conflict", "lm_conflict"],
        default="irrelevant",
        help="How the noisy mode is generated; lm_conflict creates an action contradiction and caches it.",
    )
    parser.add_argument(
        "--noisy-model",
        default=None,
        help="Optional LM model for lm_conflict. Defaults to --lm-model.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lm-model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument(
        "--positive-y-direction",
        choices=["left", "right"],
        default="left",
        help="Ego-frame direction represented by positive y; verify this for the dataset before use.",
    )
    parser.add_argument(
        "--skip-lm",
        action="store_true",
        help="Only run numerical intervention metrics; do not call the LM judge.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if any(alpha < 0 for alpha in args.alphas):
        raise ValueError("alpha must be non-negative")
    clip_ids = read_clip_ids(args.clip_ids_file)
    if "cross_scene" in args.modes and len(clip_ids) < 2:
        raise ValueError("cross_scene intervention needs at least two clip IDs.")
    if args.noisy_strategy == "lm_conflict" and args.skip_lm:
        raise ValueError("lm_conflict requires an LM client; remove --skip-lm.")

    lm_client = None
    if not args.skip_lm:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the evaluator dependency with: uv pip install --python .venv/bin/python openai"
            ) from exc
        api_key = os.getenv("OPENAI_API_KEY") or getpass.getpass(
            "Enter LM API key (shared by noise LM and judge, input hidden): "
        )
        if not api_key:
            raise RuntimeError("An LM API key is required unless --skip-lm is used.")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        lm_client = OpenAI(**client_kwargs)

    device = "cuda"
    output_path = Path(args.output)
    reasoning_path = output_path.with_name(output_path.stem + "_reasonings.jsonl")
    noisy_path = output_path.with_name(output_path.stem + "_noisy_reasonings.jsonl")
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        reasoning_path.unlink(missing_ok=True)
        noisy_path.unlink(missing_ok=True)

    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    avdi = None

    # First obtain and persist one normal CoC per clip.  This pass makes an
    # actual different-scene rationale available for every cross_scene pair.
    reasonings = load_reasoning_cache(reasoning_path)
    noisy_cache = load_noisy_cache(noisy_path)
    missing = [clip_id for clip_id in clip_ids if clip_id not in reasonings]
    print(f"Generating/resuming {len(missing)} normal reasoning trace(s)...")
    for index, clip_id in enumerate(missing, start=1):
        print(f"[reasoning {index}/{len(missing)}] {clip_id}")
        data = load_physical_aiavdataset(clip_id, t0_us=args.t0_us, avdi=avdi)
        normal_inputs = make_model_inputs(data, processor, device, forced_reasoning=None)
        torch.cuda.manual_seed_all(args.seed + index)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            _xyz, _rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=normal_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
        reasoning = as_reasoning(extra["cot"][0])
        reasonings[clip_id] = reasoning
        append_reasoning_cache(reasoning_path, clip_id, reasoning)

    completed = existing_keys(output_path, args.overwrite)
    total = len(clip_ids) * len(args.modes) * len(args.alphas)
    print(f"Running paired interventions; {total - len(completed)} result(s) remain...")
    with output_path.open("a", encoding="utf-8") as output_file:
        for clip_index, clip_id in enumerate(clip_ids):
            clip_expected_keys = {
                (clip_id, mode, float(alpha))
                for mode in args.modes
                for alpha in args.alphas
            }
            if clip_expected_keys.issubset(completed):
                print(f"[clip {clip_index + 1}/{len(clip_ids)}] {clip_id} (already complete; skip)")
                continue
            print(f"[clip {clip_index + 1}/{len(clip_ids)}] {clip_id}")
            clip_started = time.perf_counter()
            data = load_physical_aiavdataset(clip_id, t0_us=args.t0_us, avdi=avdi)
            clean_reasoning = reasonings[clip_id]
            alternate_id = clip_ids[(clip_index + 1) % len(clip_ids)]
            cross_reasoning = reasonings.get(alternate_id)
            noise_metadata: dict[str, str] | None = None
            if "noisy" in args.modes:
                if args.noisy_strategy == "irrelevant":
                    noise_metadata = {
                        "primary_action": "not_applicable",
                        "noisy_reasoning": clean_reasoning + args.noise_text,
                    }
                elif clip_id in noisy_cache:
                    noise_metadata = noisy_cache[clip_id]
                elif args.noisy_strategy == "heuristic_conflict":
                    noise_metadata = heuristic_action_conflict(clean_reasoning)
                    append_noisy_cache(noisy_path, clip_id, noise_metadata)
                    noisy_cache[clip_id] = noise_metadata
                else:
                    assert lm_client is not None
                    noise_metadata = lm_action_conflict(
                        lm_client,
                        args.noisy_model or args.lm_model,
                        clean_reasoning,
                    )
                    append_noisy_cache(noisy_path, clip_id, noise_metadata)
                    noisy_cache[clip_id] = noise_metadata

            # Both u1 and u2 use the same forced-CoC action-expert path.  Reset
            # the seed before each decode so their diffusion initial noise matches.
            clean_inputs = make_model_inputs(data, processor, device, clean_reasoning)
            torch.cuda.manual_seed_all(args.seed + clip_index)
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                clean_xyz, _clean_rot, u1 = model.sample_trajectory_from_forced_reasoning(clean_inputs)

            history_xyz = clean_inputs["ego_history_xyz"][:, -1]
            history_rot = clean_inputs["ego_history_rot"][:, -1]
            clean_xy = clean_xyz.detach().cpu().numpy()[0, :, :2]
            clean_lm_judgement: dict[str, Any] | None = None

            for mode_index, mode in enumerate(args.modes):
                perturbed_reasoning = build_perturbation(
                    mode,
                    clean_reasoning,
                    cross_reasoning,
                    noise_metadata["noisy_reasoning"] if noise_metadata is not None else "",
                )
                # alpha=0 is exactly the clean action.  If the caller requests
                # only alpha=0, avoid the second VLM/action-expert decode.  If
                # positive alphas are also requested, one perturbed decode is
                # shared by all of them.
                if any(alpha != 0.0 for alpha in args.alphas):
                    perturbed_inputs = make_model_inputs(data, processor, device, perturbed_reasoning)
                    torch.cuda.manual_seed_all(args.seed + clip_index)
                    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                        perturbed_xyz, _perturbed_rot, u2 = model.sample_trajectory_from_forced_reasoning(
                            perturbed_inputs
                        )
                    perturbed_xy = perturbed_xyz.detach().cpu().numpy()[0, :, :2]
                else:
                    u2 = u1
                    perturbed_xy = clean_xy
                control_delta_l2 = float(torch.linalg.vector_norm(u1 - u2, dim=-1).mean().item())

                for alpha in args.alphas:
                    key = (clip_id, mode, float(alpha))
                    if key in completed:
                        continue
                    guided_raw = u1 + alpha * (u1 - u2)
                    guided_action, saturation_rate = clamp_action_to_physical_bounds(
                        guided_raw, model.action_space
                    )
                    theoretical_guidance_change = float(
                        torch.linalg.vector_norm(alpha * (u1 - u2), dim=-1).mean().item()
                    )
                    actual_guidance_change = float(
                        torch.linalg.vector_norm(guided_action - u1, dim=-1).mean().item()
                    )
                    with torch.inference_mode():
                        guided_xyz, _guided_rot = model.action_space.action_to_traj(
                            guided_action, history_xyz, history_rot
                        )
                    guided_xy = guided_xyz.detach().cpu().numpy()[0, :, :2]
                    lm_judgement = None
                    if lm_client is not None:
                        if alpha == 0.0:
                            # alpha=0 always produces u_new=u1, independently
                            # of the perturbation mode.  Reuse one LM judgement
                            # so identical trajectories cannot receive different
                            # scores merely because the API was called twice.
                            if clean_lm_judgement is None:
                                clean_lm_judgement = judge_with_lm(
                                    lm_client,
                                    args.lm_model,
                                    build_judge_prompt(
                                        clean_reasoning,
                                        guided_xy,
                                        args.positive_y_direction,
                                    ),
                                )
                            lm_judgement = clean_lm_judgement
                        else:
                            lm_judgement = judge_with_lm(
                                lm_client,
                                args.lm_model,
                                build_judge_prompt(
                                    clean_reasoning,
                                    guided_xy,
                                    args.positive_y_direction,
                                ),
                            )
                    record = {
                        "clip_id": clip_id,
                        "t0_us": args.t0_us,
                        "mode": mode,
                        "alpha": alpha,
                        "noisy_strategy": args.noisy_strategy if mode == "noisy" else None,
                        "noisy_primary_action": (
                            noise_metadata["primary_action"] if mode == "noisy" and noise_metadata else None
                        ),
                        "clean_reasoning": clean_reasoning,
                        "perturbed_reasoning": perturbed_reasoning,
                        "cross_scene_source_clip_id": alternate_id if mode == "cross_scene" else None,
                        "formula": "u_new = u1 + alpha * (u1 - u2), with physical a/kappa clipping",
                        "clean_action_u1": action_summary(u1),
                        "perturbed_action_u2": action_summary(u2),
                        "clean_action_u1_values": action_values(u1),
                        "perturbed_action_u2_values": action_values(u2),
                        "guided_action_u_new_values": action_values(guided_action),
                        "mean_l2_control_delta_u1_u2": round(control_delta_l2, 6),
                        "mean_l2_theoretical_guidance_change": round(
                            theoretical_guidance_change, 6
                        ),
                        "mean_l2_guidance_change_u1": round(actual_guidance_change, 6),
                        "control_saturation_rate_after_guidance": round(saturation_rate, 6),
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
                            guided_xy,
                            guided_action,
                            model.action_space,
                            history_xyz,
                            history_rot,
                        ),
                        "lm_judgement_guided_action": lm_judgement,
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed.add(key)
                    lm_score = (
                        lm_judgement["consistency_score"]
                        if lm_judgement is not None
                        else None
                    )
                    lm_label = (
                        lm_judgement["label"] if lm_judgement is not None else None
                    )
                    print(
                        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                        f"clip={clip_id} mode={mode} alpha={alpha:g} "
                        f"lm_score={lm_score} label={lm_label} "
                        f"clip_elapsed_s={time.perf_counter() - clip_started:.1f}",
                        flush=True,
                    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    by_mode_alpha: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        mode = str(record["mode"])
        alpha_key = str(record["alpha"])
        by_mode_alpha.setdefault(mode, {}).setdefault(alpha_key, []).append(record)

    def summarize_alpha(items: list[dict[str, Any]]) -> dict[str, Any]:
        lm_scores = [
            item["lm_judgement_guided_action"]["consistency_score"]
            for item in items
            if item.get("lm_judgement_guided_action") is not None
        ]
        return {
            "num_records": len(items),
            "mean_l2_control_delta_u1_u2": round(
                float(np.mean([item["mean_l2_control_delta_u1_u2"] for item in items])), 6
            ),
            "mean_control_saturation_rate": round(
                float(np.mean([item["control_saturation_rate_after_guidance"] for item in items])), 6
            ),
            "mean_l2_theoretical_guidance_change": round(
                float(np.mean([item["mean_l2_theoretical_guidance_change"] for item in items])), 6
            ),
            "mean_l2_guidance_change_u1": round(
                float(np.mean([item["mean_l2_guidance_change_u1"] for item in items])), 6
            ),
            "mean_lm_consistency_score_guided": (
                round(float(np.mean(lm_scores)), 6) if lm_scores else None
            ),
        }

    summary = {
        "num_records": len(records),
        "note": (
            "Intervention sensitivity only; no ground-truth future trajectory is used. "
            "Large control shifts show action sensitivity, not better driving."
        ),
        "by_mode": {
            mode: {
                "by_alpha": {
                    alpha: summarize_alpha(items)
                    for alpha, items in sorted(
                        alpha_groups.items(), key=lambda pair: float(pair[0])
                    )
                }
            }
            for mode, alpha_groups in by_mode_alpha.items()
        },
        "per_case_results": str(output_path),
        "reasoning_cache": str(reasoning_path),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved intervention results to {output_path}")
    print(f"Saved intervention summary to {summary_path}")


if __name__ == "__main__":
    main()
