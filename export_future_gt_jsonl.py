"""Export compact future ego trajectories for ADE evaluation.

The output contains only the 64-step future ground truth per clip; camera
frames and model outputs are deliberately not written.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    return list(dict.fromkeys(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpamayo-root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--clip-ids-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.alpamayo_root / "src"))
    import numpy as np
    import physical_ai_av
    import scipy.spatial.transform as spt

    clip_ids = read_ids(args.clip_ids_file)
    if args.max_clips is not None:
        clip_ids = clip_ids[: args.max_clips]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if args.output.exists() and not args.overwrite:
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing[str(row["clip_id"])] = row

    pending = [clip_id for clip_id in clip_ids if clip_id not in existing]
    print(f"Requested={len(clip_ids)} existing={len(existing)} pending={len(pending)}", flush=True)

    results = dict(existing)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    history_steps = 16
    time_step_us = 100_000

    for index, clip_id in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {clip_id}", flush=True)
        egomotion = avdi.get_clip_feature(
            clip_id,
            avdi.features.LABELS.EGOMOTION,
            maybe_stream=True,
        )
        history_timestamps = args.t0_us + np.arange(
            -(history_steps - 1) * time_step_us,
            time_step_us // 2,
            time_step_us,
            dtype=np.int64,
        )
        future_timestamps = args.t0_us + np.arange(
            time_step_us,
            (64 + 0.5) * time_step_us,
            time_step_us,
            dtype=np.int64,
        )
        history = egomotion(history_timestamps)
        future = egomotion(future_timestamps)
        history_xyz = history.pose.translation
        future_xyz = future.pose.translation
        history_quat = history.pose.rotation.as_quat()
        t0_rot_inv = spt.Rotation.from_quat(history_quat[-1]).inv()
        future = t0_rot_inv.apply(future_xyz - history_xyz[-1])
        row = {
            "clip_id": clip_id,
            "t0_us": int(args.t0_us),
            "future_frame": "ar1_ego",
            "future_waypoint_count": int(future.shape[0]),
            "ego_future_xyz": future.astype(float).tolist(),
        }
        results[clip_id] = row
        del egomotion, history, future_xyz, history_xyz, history_quat, future
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        with args.output.open("w", encoding="utf-8") as f:
            for item in (results[clip] for clip in clip_ids if clip in results):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} records to {args.output}", flush=True)


if __name__ == "__main__":
    main()
