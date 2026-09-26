"""Run one NAVSIM/OpenScene sample through Alpamayo-R1 (inference smoke test only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--anchor-index", type=int, default=20)
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERA_NAMES),
        help="Exactly four comma-separated NAVSIM camera names, camera-major order.",
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
        anchor_index=args.anchor_index,
        camera_names=camera_names,
        max_image_time_error_s=args.max_image_time_error_s,
        max_pose_gap_s=args.max_pose_gap_s,
    )

    print("NAVSIM sample prepared:")
    for key in (
        "scene_name",
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
    if data["camera_mapping_note"].startswith("CAM_L1"):
        print("  WARNING: fourth view CAM_L1 is not equivalent to AR1 front-tele.")
    print(f"  image_frames shape: {tuple(data['image_frames'].shape)}")
    print(f"  ego_history_xyz shape: {tuple(data['ego_history_xyz'].shape)}")
    print(f"  ego_history_rot shape: {tuple(data['ego_history_rot'].shape)}")
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
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )

    xyz_shape, rot_shape = validate_trajectory_output(pred_xyz, pred_rot)
    prediction = pred_xyz.detach().float().cpu()
    rotation_prediction = pred_rot.detach().float().cpu()
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Model returned NaN/Inf trajectory values")
    if not torch.isfinite(rotation_prediction).all():
        raise RuntimeError("Model returned NaN/Inf rotation values")
    cot = extract_first_reasoning(extra.get("cot") if hasattr(extra, "get") else None)
    print("\nSMOKE RESULT: inference completed")
    print("pred_xyz shape:", xyz_shape)
    print("pred_rot shape:", rot_shape)
    print("reasoning/CoC:", cot if cot is not None else "<not returned>")
    print("first waypoint xyz:", prediction[0, 0, 0, 0].tolist())
    print("last waypoint xyz:", prediction[0, 0, 0, -1].tolist())
    print("NOTE: this is an input/inference smoke test, not a NAVSIM score or ADE evaluation.")

    if args.output_json:
        result = {
            "scene_name": data["scene_name"],
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
            "reasoning": cot,
            "smoke_test_only": True,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
