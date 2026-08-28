# SPDX-License-Identifier: Apache-2.0
"""Evaluate consistency between Alpamayo reasoning and predicted actions.

The script runs Alpamayo on clip IDs from the Physical AI AV dataset, summarizes
the predicted BEV trajectory, and asks an LM to judge whether the action is
consistent with the generated reasoning. Ground truth is intentionally not used
for this consistency score.

Example:
    python src/alpamayo_r1/evaluate_reasoning_action.py \
        --clip-ids-file test_clip_ids.txt \
        --output reasoning_action_eval.jsonl
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "consistency_score": {
            "type": "number",
            "description": "Score from 0.0 (inconsistent) to 1.0 (fully consistent).",
        },
        "label": {
            "type": "string",
            "enum": ["consistent", "partially_consistent", "inconsistent", "uncertain"],
        },
        "intended_action": {"type": "string"},
        "trajectory_evidence": {"type": "string"},
        "uncertainty": {"type": "string"},
    },
    "required": [
        "consistency_score",
        "label",
        "intended_action",
        "trajectory_evidence",
        "uncertainty",
    ],
    "additionalProperties": False,
}


def read_clip_ids(path: str | None) -> list[str]:
    if path is None:
        return ["030c760c-ae38-49aa-9ad8-f5650a545d26"]
    clip_ids = [line.strip() for line in Path(path).read_text().splitlines()]
    clip_ids = [clip_id for clip_id in clip_ids if clip_id and not clip_id.startswith("#")]
    if not clip_ids:
        raise ValueError(f"No clip IDs found in {path}")
    return clip_ids


def trajectory_summary(xy: np.ndarray, name: str) -> dict[str, Any]:
    displacement = xy[-1] - xy[0]
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return {
        "name": name,
        "num_waypoints": int(len(xy)),
        "start_xy_m": np.round(xy[0], 3).tolist(),
        "end_xy_m": np.round(xy[-1], 3).tolist(),
        "displacement_xy_m": np.round(displacement, 3).tolist(),
        "final_lateral_y_m": round(float(xy[-1, 1]), 3),
        "max_abs_lateral_y_m": round(float(np.abs(xy[:, 1]).max()), 3),
        "path_length_m": round(float(segment_lengths.sum()), 3),
    }


def trajectory_kinematics(xy: np.ndarray, dt_s: float = 0.1) -> dict[str, Any]:
    """Derive compact motion evidence from regularly sampled XY waypoints."""
    segment_speeds = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt_s
    return {
        "waypoint_dt_s": dt_s,
        "segment_speed_mps": np.round(segment_speeds, 3).tolist(),
        "min_segment_speed_mps": round(float(segment_speeds.min()), 3),
        "max_segment_speed_mps": round(float(segment_speeds.max()), 3),
        "mean_segment_speed_mps": round(float(segment_speeds.mean()), 3),
        "final_segment_speed_mps": round(float(segment_speeds[-1]), 3),
    }


def timed_waypoints(xy: np.ndarray, dt_s: float = 0.1) -> list[dict[str, float]]:
    """Return every future waypoint; waypoint zero is the state at t=dt."""
    return [
        {
            "t_s": round((index + 1) * dt_s, 2),
            "x_m": round(float(point[0]), 3),
            "y_m": round(float(point[1]), 3),
        }
        for index, point in enumerate(xy)
    ]


def build_judge_prompt(
    reasoning: str,
    pred_xy: np.ndarray,
    positive_y_direction: str = "left",
) -> str:
    if positive_y_direction not in {"left", "right"}:
        raise ValueError("positive_y_direction must be 'left' or 'right'")
    payload = {
        "reasoning": reasoning,
        "coordinate_note": (
            "The trajectory is expressed in the ego BEV frame. Positive x is forward. "
            f"Positive y is {positive_y_direction}; negative y is "
            f"{'right' if positive_y_direction == 'left' else 'left'}. "
            "Every waypoint is sampled at a fixed 0.1-second interval."
        ),
        "predicted_trajectory": trajectory_summary(pred_xy, "prediction"),
        "trajectory_waypoints": timed_waypoints(pred_xy),
        "trajectory_kinematics": trajectory_kinematics(pred_xy),
    }
    return (
        "Evaluate whether the predicted driving action is consistent with the reasoning. "
        "Judge action-reasoning alignment, not whether the reasoning is factually correct about the scene. "
        "Do not judge whether the action is correct for the real scene because no ground-truth trajectory "
        "or scene annotation is provided. Use all timed waypoints and kinematics: for a stop, inspect low "
        "segment speeds and whether forward position stops changing; for turn/lane actions, inspect the "
        "lateral path using the supplied y-direction convention. Do not invent obstacle evidence. "
        "The consistency_score is a continuous alignment score, not evaluator confidence. "
        "Use these as soft guidelines: consistent usually 0.75-1.0, "
        "partially_consistent usually 0.30-0.74, inconsistent usually 0.0-0.29, "
        "and uncertain usually 0.25-0.70. The ranges are not exact requirements, "
        "but the score must agree semantically with the label: do not give an "
        "inconsistent trajectory a high consistency score or a clearly consistent "
        "trajectory a very low score. "
        "Return exactly one JSON object with these fields: "
        "consistency_score (number from 0.0 to 1.0), label (one of consistent, "
        "partially_consistent, inconsistent, uncertain), intended_action (string), "
        "trajectory_evidence (string), and uncertainty (string). "
        "Do not use fields named consistent or explanation.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def normalize_judgement(raw_output: str) -> dict[str, Any]:
    """Validate the LM response and explicitly normalize the proxy's legacy format."""
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    payload = json.loads(cleaned)

    required = {
        "consistency_score",
        "label",
        "intended_action",
        "trajectory_evidence",
        "uncertainty",
    }
    if required.issubset(payload):
        score = payload["consistency_score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            raise ValueError("consistency_score must be a number in [0, 1]")
        if payload["label"] not in {
            "consistent",
            "partially_consistent",
            "inconsistent",
            "uncertain",
        }:
            raise ValueError("Invalid consistency label")
        if not all(isinstance(payload[field], str) for field in required - {"consistency_score", "label"}):
            raise ValueError("Judgement text fields must be strings")
        return payload

    if set(payload) >= {"consistent", "explanation"} and isinstance(payload["consistent"], bool):
        # The xah.io proxy may return this legacy format despite the JSON request.
        is_consistent = payload["consistent"]
        return {
            "consistency_score": 1.0 if is_consistent else 0.0,
            "label": "consistent" if is_consistent else "inconsistent",
            "intended_action": "Not provided by legacy response",
            "trajectory_evidence": str(payload["explanation"]),
            "uncertainty": (
                "Legacy proxy schema was mapped from consistent/explanation; "
                "the numeric score is a binary mapping."
            ),
        }

    raise ValueError(
        "LM response does not match the required schema and is not a recognized legacy response"
    )


def judge_with_lm(client: Any, model_name: str, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Return only one valid JSON object matching the requested fields.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    if isinstance(response, str):
        raw_output = response
    elif hasattr(response, "choices"):
        raw_output = response.choices[0].message.content
    else:
        raw_output = response["choices"][0]["message"]["content"]
    return normalize_judgement(raw_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-ids-file", default=None)
    parser.add_argument("--output", default="reasoning_action_eval.jsonl")
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--lm-model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument(
        "--positive-y-direction",
        choices=["left", "right"],
        default="left",
        help="Ego-frame direction represented by positive y; verify this for the dataset before use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing output file and evaluate all clips again.",
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the evaluator dependency with: pip install openai") from exc

    api_key = os.getenv("OPENAI_API_KEY") or getpass.getpass(
        "Enter LM API key (input hidden): "
    )
    if not api_key:
        raise RuntimeError("An LM API key is required.")
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    clip_ids = read_clip_ids(args.clip_ids_file)
    avdi = None
    device = "cuda"

    existing_results: list[dict[str, Any]] = []
    processed_clip_ids: set[str] = set()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as existing_file:
            for line_number, line in enumerate(existing_file, start=1):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    clip_id = str(result["clip_id"])
                    result["lm_judgement"]["consistency_score"]
                    result["lm_judgement"]["label"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    print(f"Warning: ignoring malformed result at line {line_number}: {exc}")
                    continue
                if clip_id not in processed_clip_ids:
                    existing_results.append(result)
                    processed_clip_ids.add(clip_id)

    clips_to_run = [clip_id for clip_id in clip_ids if clip_id not in processed_clip_ids]
    print(
        f"Evaluating {len(clips_to_run)} new clip(s); "
        f"resuming with {len(existing_results)} existing result(s)..."
    )

    all_results = list(existing_results)
    output_mode = "w" if args.overwrite else "a"
    with output_path.open(output_mode, encoding="utf-8") as output_file:
        for index, clip_id in enumerate(clips_to_run, start=1):
            print(f"[{index}/{len(clips_to_run)}] {clip_id}")
            data = load_physical_aiavdataset(clip_id, t0_us=args.t0_us, avdi=avdi)
            if avdi is None:
                # The loader creates the interface internally when avdi=None. Keeping
                # this explicit makes the first sample compatible with the public API.
                avdi = None

            messages = helper.create_message(data["image_frames"].flatten(0, 1))
            model = getattr(main, "_model", None)
            if model is None:
                model = AlpamayoR1.from_pretrained(
                    "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
                ).to(device)
                model.eval()
                main._model = model

            processor = getattr(main, "_processor", None)
            if processor is None:
                processor = helper.get_processor(model.tokenizer)
                main._processor = processor

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = helper.to_device(
                {
                    "tokenized_data": inputs,
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                },
                device,
            )

            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=args.num_traj_samples,
                    max_generation_length=256,
                    return_extra=True,
                )

            reasoning = extra["cot"][0]
            if isinstance(reasoning, (list, tuple, np.ndarray)):
                reasoning = " ".join(str(item) for item in reasoning)
            else:
                reasoning = str(reasoning)
            pred_xy_all = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]
            pred_xy = pred_xy_all[0]
            judge = judge_with_lm(
                client,
                args.lm_model,
                build_judge_prompt(reasoning, pred_xy, args.positive_y_direction),
            )

            result = {
                "clip_id": clip_id,
                "t0_us": args.t0_us,
                "reasoning": reasoning,
                "prediction": trajectory_summary(pred_xy, "prediction"),
                "lm_judgement": judge,
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            all_results.append(result)

    all_judgements = [result["lm_judgement"] for result in all_results]
    scores = [float(judgement["consistency_score"]) for judgement in all_judgements]
    labels = [str(judgement["label"]) for judgement in all_judgements]
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    summary = {
        "num_clips": len(all_results),
        "mean_consistency_score": round(float(np.mean(scores)), 4),
        "label_counts": label_counts,
        "label_rates": {
            label: round(count / len(labels), 4) for label, count in label_counts.items()
        },
        "per_clip_results": args.output,
    }
    summary_path = str(Path(args.output).with_name(Path(args.output).stem + "_summary.json"))
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved per-clip results to {args.output}")
    print(f"Mean consistency score: {summary['mean_consistency_score']:.4f}")
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
