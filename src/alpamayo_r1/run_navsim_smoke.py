"""Run one NAVSIM/OpenScene sample through Alpamayo-R1 (inference smoke test only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.navsim_adapter import DEFAULT_CAMERA_NAMES, load_navsim_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this smoke test in the GPU AR1 environment")
    camera_names = tuple(name.strip() for name in args.cameras.split(",") if name.strip())
    data = load_navsim_sample(
        metadata_path=args.metadata_pkl,
        sensor_root=args.sensor_root,
        anchor_index=args.anchor_index,
        camera_names=camera_names,
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
        "future_loaded",
    ):
        print(f"  {key}: {data[key]}")
    print(f"  image_frames shape: {tuple(data['image_frames'].shape)}")
    print(f"  ego_history_xyz shape: {tuple(data['ego_history_xyz'].shape)}")
    print(f"  ego_history_rot shape: {tuple(data['ego_history_rot'].shape)}")
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    messages = helper.create_message(data["image_frames"].flatten(0, 1))
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
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )

    torch.cuda.manual_seed_all(42)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )

    prediction = pred_xyz.detach().float().cpu()
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Model returned NaN/Inf trajectory values")
    cot = extra.get("cot")
    print("\nSMOKE RESULT: inference completed")
    print("pred_xyz shape:", tuple(prediction.shape))
    print("pred_rot shape:", tuple(pred_rot.shape))
    print("reasoning/CoC:", cot[0] if cot else "<not returned>")
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
            "future_loaded": False,
            "pred_xyz": prediction[0, 0, 0].tolist(),
            "reasoning": str(cot[0]) if cot else None,
            "smoke_test_only": True,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
