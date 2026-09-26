"""Minimal NAVSIM/OpenScene adapter for an Alpamayo-R1 inference smoke test.

This is an input-compatibility adapter, not a benchmark loader. NAVSIM mini is
sampled at 2 Hz and does not provide a forward-tele camera matching AR1's
training cameras. The default fourth view is therefore an explicitly marked
side-view substitute; do not use its scores for a like-for-like evaluation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_CAMERA_NAMES = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_L1")
IMAGE_HISTORY_OFFSETS_S = (-1.5, -1.0, -0.5, 0.0)


def _load_scene(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("rb") as stream:
        scene = pickle.load(stream)
    if not isinstance(scene, list) or not scene or not all(isinstance(x, dict) for x in scene):
        raise ValueError(f"Expected a non-empty list of frame dictionaries: {metadata_path}")
    return scene


def _resolve_sensor_path(sensor_root: Path, relative_path: str) -> Path:
    root = sensor_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Camera path escapes sensor root: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing NAVSIM camera image: {path}")
    return path


def load_navsim_sample(
    metadata_path: str | Path,
    sensor_root: str | Path,
    anchor_index: int,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    num_history_steps: int = 16,
    history_step_s: float = 0.1,
) -> dict[str, Any]:
    """Load one NAVSIM anchor into the image/ego-history shape AR1 expects.

    Images are returned camera-major as ``(N_camera, 4, 3, H, W)``. The four
    image timestamps use NAVSIM's available 2 Hz history (about 1.5 s span).
    Ego poses are interpolated to 10 Hz over the same 1.5 s span, then expressed
    in the anchor ego frame. No future annotations or ground truth are loaded.
    NAVSIM quaternion input follows ``pyquaternion.Quaternion`` ordering
    ``[w, x, y, z]`` and is reordered for SciPy.
    """
    metadata_path = Path(metadata_path)
    sensor_root = Path(sensor_root)
    if len(camera_names) != 4:
        raise ValueError("Smoke adapter expects four camera streams; pass exactly four --cameras names")
    if num_history_steps < 2 or history_step_s <= 0:
        raise ValueError("num_history_steps must be >= 2 and history_step_s must be positive")

    frames = _load_scene(metadata_path)
    if anchor_index < 0 or anchor_index >= len(frames):
        raise IndexError(f"anchor_index={anchor_index} outside scene with {len(frames)} frames")

    timestamps_us = np.asarray([int(frame["timestamp"]) for frame in frames], dtype=np.int64)
    if np.any(np.diff(timestamps_us) <= 0):
        raise ValueError("NAVSIM metadata timestamps must be strictly increasing")
    anchor_timestamp_us = int(timestamps_us[anchor_index])
    history_offsets_s = -np.arange(num_history_steps - 1, -1, -1, dtype=np.float64) * history_step_s
    history_times_s = history_offsets_s
    if history_times_s[0] < (timestamps_us[0] - anchor_timestamp_us) * 1e-6:
        raise ValueError("Selected anchor does not have enough past ego-pose history")

    translations = np.asarray(
        [frame["ego2global_translation"] for frame in frames], dtype=np.float64
    )
    quaternions_wxyz = np.asarray(
        [frame["ego2global_rotation"] for frame in frames], dtype=np.float64
    )
    if translations.shape != (len(frames), 3) or quaternions_wxyz.shape != (len(frames), 4):
        raise ValueError("Unexpected NAVSIM ego pose or quaternion shape")

    # Use relative seconds to retain precision for large absolute microsecond timestamps.
    scene_times_s = (timestamps_us - anchor_timestamp_us).astype(np.float64) * 1e-6
    rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]])
    anchor_rotation = rotations[anchor_index]
    anchor_translation = translations[anchor_index]

    history_translation_global = np.stack(
        [np.interp(history_times_s, scene_times_s, translations[:, axis]) for axis in range(3)],
        axis=-1,
    )
    history_rotation_global = Slerp(scene_times_s, rotations)(history_times_s)
    history_translation_local = anchor_rotation.inv().apply(
        history_translation_global - anchor_translation
    )
    history_rotation_local = (anchor_rotation.inv() * history_rotation_global).as_matrix()

    # Choose actual past/current camera frames nearest the desired offsets; never select future.
    past_indices = np.arange(anchor_index + 1, dtype=np.int64)
    past_times_s = scene_times_s[past_indices]
    selected_indices: list[int] = []
    selected_deltas_s: list[float] = []
    for offset_s in IMAGE_HISTORY_OFFSETS_S:
        local_index = int(np.argmin(np.abs(past_times_s - offset_s)))
        selected_index = int(past_indices[local_index])
        selected_indices.append(selected_index)
        selected_deltas_s.append(float(past_times_s[local_index] - offset_s))
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("Could not select four distinct historical NAVSIM camera frames")

    camera_tensors: list[torch.Tensor] = []
    for camera_name in camera_names:
        images: list[torch.Tensor] = []
        for frame_index in selected_indices:
            camera = frames[frame_index].get("cams", {}).get(camera_name)
            if not isinstance(camera, dict) or not camera.get("data_path"):
                raise KeyError(f"Missing cams[{camera_name!r}].data_path at frame {frame_index}")
            image_path = _resolve_sensor_path(sensor_root, str(camera["data_path"]))
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
        camera_tensors.append(torch.stack(images, dim=0))

    image_frames = torch.stack(camera_tensors, dim=0)
    return {
        "scene_name": str(frames[anchor_index].get("scene_name", metadata_path.stem)),
        "anchor_index": anchor_index,
        "anchor_token": frames[anchor_index].get("token"),
        "anchor_timestamp_us": anchor_timestamp_us,
        "image_frame_indices": selected_indices,
        "image_timestamps_us": [int(timestamps_us[i]) for i in selected_indices],
        "image_time_offsets_s": [float(scene_times_s[i]) for i in selected_indices],
        "image_selection_error_s": selected_deltas_s,
        "camera_names": list(camera_names),
        "image_frames": image_frames,
        "ego_history_xyz": torch.from_numpy(history_translation_local.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0),
        "ego_history_rot": torch.from_numpy(history_rotation_local.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0),
        "history_offsets_s": history_offsets_s.tolist(),
        "future_loaded": False,
        "coordinate_frame": "anchor ego frame; source NAVSIM ego2global poses",
        "camera_mapping_note": (
            "CAM_L0/F0/R0 approximate AR1 cross-left/front-wide/cross-right; "
            "CAM_L1 is a side-view substitute because NAVSIM mini has no matching front-tele view."
        ),
    }
