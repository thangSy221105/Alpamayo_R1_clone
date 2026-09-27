"""Run one NAVSIM/OpenScene sample through Alpamayo-R1 (inference smoke test only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alpamayo_r1.navsim_adapter import (
    DEFAULT_CAMERA_NAMES,
    DEFAULT_MAX_IMAGE_TIME_ERROR_S,
    DEFAULT_MAX_POSE_GAP_S,
    load_navsim_sample,
)
from alpamayo_r1.navsim_smoke_utils import extract_first_reasoning, validate_trajectory_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The adapter does not download NAVSIM data. If the Alpamayo checkpoint or Qwen3-VL "
            "processor is not cached, Hugging Face may download it unless --offline is set. "
            "Run only in a GPU environment with sufficient disk and VRAM."
        ),
    )
    parser.add_argument("--metadata-pkl", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Zero-based index among complete NAVSIM windows that pass the route filter.",
    )
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERA_NAMES),
        help="Three or four comma-separated NAVSIM camera names, camera-major order.",
    )
    parser.add_argument("--model", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument(
        "--max-image-time-error-s", type=float, default=DEFAULT_MAX_IMAGE_TIME_ERROR_S
    )
    parser.add_argument("--max-pose-gap-s", type=float, default=DEFAULT_MAX_POSE_GAP_S)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require cached model/processor files; do not contact Hugging Face.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.offline:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this smoke test in the GPU AR1 environment")
    camera_names = tuple(name.strip() for name in args.cameras.split(",") if name.strip())
    data = load_navsim_sample(
        metadata_path=args.metadata_pkl,
        sensor_root=args.sensor_root,
        scene_index=args.scene_index,
        history_frame_count=args.history_frames,
        future_frame_count=args.future_frames,
        require_route=True,
        camera_names=camera_names,
        max_image_time_error_s=args.max_image_time_error_s,
        max_pose_gap_s=args.max_pose_gap_s,
    )

    print("NAVSIM sample prepared:")
    for key in (
        "scene_name",
        "navsim_split",
        "anchor_index",
        "anchor_token",
        "anchor_timestamp_us",
        "camera_names",
        "image_frame_indices",
        "image_time_offsets_s",
        "image_selection_error_s",
        "history_offsets_s",
        "coordinate_frame",
        "camera_mapping_note",
        "future_used_for_model_input",
        "future_frames_deserialized",
        "source_pose_median_rate_hz",
        "source_camera_rate_hz_estimate",
        "ego_history_interpolated",
    ):
        print(f"  {key}: {data[key]}")
    if data["camera_mapping_note"].startswith("CAM_L1 is a left-side"):
        print("  WARNING: CAM_L1 is not equivalent to AR1 front-tele.")
    print(f"  image_frames shape: {tuple(data['image_frames'].shape)}")
    print(f"  ego_history_xyz shape: {tuple(data['ego_history_xyz'].shape)}")
    print(f"  ego_history_rot shape: {tuple(data['ego_history_rot'].shape)}")
    ground_truth = data["ground_truth_future"]
    if ground_truth is None:
        raise RuntimeError("NAVSIM scene split did not produce a future ground-truth window")
    print(
        "  GT future frames: "
        f"{len(ground_truth['frame_indices'])}, "
        f"t=[{ground_truth['time_offsets_s'][0]:.3f}, "
        f"{ground_truth['time_offsets_s'][-1]:.3f}]s after anchor"
    )
    print(
        "  GT first/last xyz in anchor frame: "
        f"{ground_truth['ego_xyz'][0]} / {ground_truth['ego_xyz'][-1]}"
    )
    print("  GT future used for model input: False")
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    if not args.offline:
        print(
            "WARNING: uncached checkpoint/processor assets may download from Hugging Face; "
            "use --offline to require local cache only."
        )

    image_frames = torch.from_numpy(data["image_frames"])
    messages = helper.create_message(image_frames.flatten(0, 1))
    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
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

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra, sampled_action = (
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=args.max_generation_length,
                return_extra=True,
                return_action=True,
            )
        )

    xyz_shape, rot_shape = validate_trajectory_output(pred_xyz, pred_rot)
    prediction = pred_xyz.detach().float().cpu()
    rotation_prediction = pred_rot.detach().float().cpu()
    action_normalized = sampled_action.detach().float().cpu()[0, 0, 0].numpy()
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Model returned NaN/Inf trajectory values")
    if not torch.isfinite(rotation_prediction).all():
        raise RuntimeError("Model returned NaN/Inf rotation values")
    if not np.isfinite(action_normalized).all():
        raise RuntimeError("Model returned NaN/Inf action controls")

    action_space = model.action_space
    accel_mean = float(action_space.accel_mean.detach().float().cpu().item())
    accel_std = float(action_space.accel_std.detach().float().cpu().item())
    curvature_mean = float(action_space.curvature_mean.detach().float().cpu().item())
    curvature_std = float(action_space.curvature_std.detach().float().cpu().item())
    acceleration_mps2 = action_normalized[:, 0] * accel_std + accel_mean
    curvature_inv_m = action_normalized[:, 1] * curvature_std + curvature_mean
    history_xyz_device = model_inputs["ego_history_xyz"]
    history_rot_device = model_inputs["ego_history_rot"]
    initial_states = action_space.estimate_t0_states(history_xyz_device, history_rot_device)
    initial_speed_mps = float(initial_states["v"].detach().float().cpu().reshape(-1)[0].item())
    velocity_mps = initial_speed_mps + np.cumsum(acceleration_mps2 * action_space.dt)
    action_physical = np.column_stack((acceleration_mps2, curvature_inv_m))

    cot = extract_first_reasoning(extra.get("cot") if hasattr(extra, "get") else None)
    print("\nSMOKE RESULT: inference completed")
    print("pred_xyz shape:", xyz_shape)
    print("pred_rot shape:", rot_shape)
    print("reasoning/CoC:", cot if cot is not None else "<not returned>")
    print("first waypoint xyz:", prediction[0, 0, 0, 0].tolist())
    print("last waypoint xyz:", prediction[0, 0, 0, -1].tolist())
    print(f"initial speed from history: {initial_speed_mps:.3f} m/s")
    print(
        "u columns: a_norm, curvature_norm, acceleration_mps2, "
        "curvature_inv_m, velocity_mps"
    )
    for index, (control, physical, speed) in enumerate(
        zip(action_normalized, action_physical, velocity_mps), start=1
    ):
        print(
            f"{index:02d} t={index * action_space.dt:.1f}s "
            f"u=[{control[0]: .5f}, {control[1]: .5f}] "
            f"a={physical[0]: .4f} m/s^2 "
            f"kappa={physical[1]: .6f} 1/m "
            f"v={speed:.3f} m/s"
        )
    print("NOTE: this is an input/inference smoke test, not a NAVSIM score or ADE evaluation.")

    if args.output_json:
        result = {
            "scene_name": data["scene_name"],
            "navsim_split": data["navsim_split"],
            "anchor_index": data["anchor_index"],
            "anchor_token": data["anchor_token"],
            "anchor_timestamp_us": data["anchor_timestamp_us"],
            "camera_names": data["camera_names"],
            "image_frame_indices": data["image_frame_indices"],
            "image_timestamps_us": data["image_timestamps_us"],
            "image_time_offsets_s": data["image_time_offsets_s"],
            "history_offsets_s": data["history_offsets_s"],
            "future_used_for_model_input": False,
            "future_frames_deserialized": data["future_frames_deserialized"],
            "ground_truth_future": data["ground_truth_future"],
            "history_interpolation": data["history_interpolation"],
            "ego_history_interpolated": data["ego_history_interpolated"],
            "source_pose_median_rate_hz": data["source_pose_median_rate_hz"],
            "source_camera_rate_hz_estimate": data["source_camera_rate_hz_estimate"],
            "source_rate_note": data["source_rate_note"],
            "ar1_reference_camera_rate_hz": data["ar1_reference_camera_rate_hz"],
            "fourth_camera_equivalent_to_ar1_front_tele": data[
                "fourth_camera_equivalent_to_ar1_front_tele"
            ],
            "camera_mapping_note": data["camera_mapping_note"],
            "image_time_error_s": data["image_time_error_s"],
            "image_selection_error_s": data["image_selection_error_s"],
            "pred_xyz": prediction[0, 0, 0].tolist(),
            "u_normalized": action_normalized.tolist(),
            "u_physical_acceleration_curvature": action_physical.tolist(),
            "initial_speed_mps": initial_speed_mps,
            "velocity_mps": velocity_mps.tolist(),
            "action_dt_s": action_space.dt,
            "reasoning": cot,
            "smoke_test_only": True,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
