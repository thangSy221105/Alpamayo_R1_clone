"""Prepare a strictly history-only NAVSIM/OpenScene sample for Alpamayo-R1.

NAVSIM metadata stores ego poses at roughly 2 Hz.  The model input history is
sampled at 10 Hz by interpolating those poses; these are not 10 Hz observations.
Images and poses after the requested anchor are never inspected or returned.
"""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_CAMERA_NAMES = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_L1")
IMAGE_HISTORY_OFFSETS_S = (-1.5, -1.0, -0.5, 0.0)
AR1_HISTORY_STEPS = 16
AR1_HISTORY_STEP_S = 0.1
AR1_REFERENCE_CAMERA_RATE_HZ = 10.0
DEFAULT_MAX_IMAGE_TIME_ERROR_S = 0.05
DEFAULT_MAX_POSE_GAP_S = 0.55
_QUAT_EPS = 1e-8


def _context(frame_index: int, frame: Any, field: str) -> str:
    token = frame.get("token") if isinstance(frame, dict) else None
    return f"frame_index={frame_index}, token={token!r}, field={field!r}"


def _fail(frame_index: int, frame: Any, field: str, detail: str) -> ValueError:
    return ValueError(f"Invalid NAVSIM history ({_context(frame_index, frame, field)}): {detail}")


def _timestamp_us(frame: Any, frame_index: int) -> int:
    field = "timestamp"
    if not isinstance(frame, dict) or field not in frame:
        raise _fail(frame_index, frame, field, "required integer microsecond timestamp is missing")
    value = frame[field]
    numeric_types = (int, np.integer, float, np.floating)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numeric_types):
        detail = f"expected integer microseconds, got {type(value).__name__}"
        raise _fail(frame_index, frame, field, detail)
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        detail = f"cannot convert to integer microseconds: {exc}"
        raise _fail(frame_index, frame, field, detail) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        detail = f"expected finite integer microseconds, got {value!r}"
        raise _fail(frame_index, frame, field, detail)
    return int(numeric)


def _vector(frame: Any, frame_index: int, field: str, length: int) -> np.ndarray:
    if not isinstance(frame, dict) or field not in frame:
        raise _fail(frame_index, frame, field, "required field is missing")
    try:
        value = np.asarray(frame[field], dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _fail(frame_index, frame, field, f"cannot convert to numeric array: {exc}") from exc
    if value.shape != (length,):
        raise _fail(frame_index, frame, field, f"expected shape ({length},), got {value.shape}")
    if not np.isfinite(value).all():
        raise _fail(frame_index, frame, field, "contains NaN or infinity")
    if field == "ego2global_rotation":
        norm = float(np.linalg.norm(value))
        if norm <= _QUAT_EPS:
            raise _fail(frame_index, frame, field, "quaternion norm must be nonzero")
        value = value / norm
    return value


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.asarray([0.0, *vector], dtype=np.float64)
    conjugate = q * np.asarray([1.0, -1.0, -1.0, -1.0])
    return _quat_multiply(_quat_multiply(q, pure), conjugate)[1:]


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - fraction) * theta) / sin_theta * q0
        + math.sin(fraction * theta) / sin_theta * q1
    )


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _resolve_sensor_path(
    sensor_root: Path, raw_path: Any, frame: Any, frame_index: int, camera: str
) -> Path:
    field = f"cams.{camera}.data_path"
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise _fail(frame_index, frame, field, f"expected non-empty path, got {raw_path!r}")
    relative = Path(raw_path)
    if not relative.is_absolute() and ".." in relative.parts:
        raise _fail(frame_index, frame, field, "relative path traversal ('..') is not allowed")
    root = sensor_root.resolve()
    candidate = relative.resolve() if relative.is_absolute() else (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        detail = f"resolved path escapes sensor root: {candidate}"
        raise _fail(frame_index, frame, field, detail) from exc
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Missing NAVSIM image ({_context(frame_index, frame, field)}): "
            f"file does not exist under sensor root: {candidate}"
        )
    return candidate


def _history_offsets() -> np.ndarray:
    # 16 samples at 10 Hz: -1.5, -1.4, ..., 0.0 seconds.
    return np.linspace(-(AR1_HISTORY_STEPS - 1) * AR1_HISTORY_STEP_S, 0.0, AR1_HISTORY_STEPS)


def _interpolate_history(
    timestamps_us: np.ndarray,
    translations: np.ndarray,
    quaternions_wxyz: np.ndarray,
    target_timestamps_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_xyz = np.empty((len(target_timestamps_us), 3), dtype=np.float64)
    for target_index, target in enumerate(target_timestamps_us):
        right = int(np.searchsorted(timestamps_us, target, side="left"))
        if right < len(timestamps_us) and timestamps_us[right] == target:
            sampled_xyz[target_index] = translations[right]
            continue
        if right == 0 or right >= len(timestamps_us):
            raise ValueError(
                "History timestamp "
                f"{int(target)} is outside the source pose range; extrapolation is forbidden"
            )
        left = right - 1
        fraction = float(
            (target - timestamps_us[left]) / (timestamps_us[right] - timestamps_us[left])
        )
        sampled_xyz[target_index] = translations[left] + fraction * (
            translations[right] - translations[left]
        )

    # NAVSIM stores [w, x, y, z]; SciPy Rotation/Slerp expects [x, y, z, w].
    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError:  # Minimal CPU test environments may omit this declared dependency.
        sampled_quat = np.empty((len(target_timestamps_us), 4), dtype=np.float64)
        for target_index, target in enumerate(target_timestamps_us):
            right = int(np.searchsorted(timestamps_us, target, side="left"))
            if right < len(timestamps_us) and timestamps_us[right] == target:
                sampled_quat[target_index] = quaternions_wxyz[right]
                continue
            left = right - 1
            fraction = float(
                (target - timestamps_us[left]) / (timestamps_us[right] - timestamps_us[left])
            )
            sampled_quat[target_index] = _quat_slerp(
                quaternions_wxyz[left], quaternions_wxyz[right], fraction
            )
        return sampled_xyz, sampled_quat
    source_times_s = timestamps_us.astype(np.float64) / 1_000_000.0
    target_times_s = target_timestamps_us.astype(np.float64) / 1_000_000.0
    source_rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]])
    sampled_rotations = Slerp(source_times_s, source_rotations)(target_times_s)
    sampled_xyzw = sampled_rotations.as_quat()
    sampled_quat = sampled_xyzw[:, [3, 0, 1, 2]]
    return sampled_xyz, sampled_quat


def load_navsim_sample(
    metadata_path: str | Path,
    sensor_root: str | Path,
    anchor_index: int,
    camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
    max_image_time_error_s: float = DEFAULT_MAX_IMAGE_TIME_ERROR_S,
    max_pose_gap_s: float = DEFAULT_MAX_POSE_GAP_S,
) -> dict[str, Any]:
    """Load one NAVSIM anchor using historical poses and images only.

    Returned arrays are NumPy arrays so this adapter can be tested without
    importing the model stack. The smoke runner converts them to torch tensors.
    NAVSIM pose quaternions are scalar-first ``[w, x, y, z]``.
    """
    if not math.isfinite(max_image_time_error_s) or max_image_time_error_s < 0:
        raise ValueError("max_image_time_error_s must be finite and non-negative")
    if not math.isfinite(max_pose_gap_s) or max_pose_gap_s <= 0:
        raise ValueError("max_pose_gap_s must be finite and positive")
    if len(camera_names) != 4 or len(set(camera_names)) != 4:
        raise ValueError(f"Exactly four distinct camera names are required, got {camera_names!r}")

    metadata_path = Path(metadata_path)
    sensor_root = Path(sensor_root)
    with metadata_path.open("rb") as stream:
        frames = pickle.load(stream)  # The pickle container is deserialized as a whole.
    if not isinstance(frames, list) or not frames:
        detail = f"NAVSIM metadata root must be a non-empty list; got {type(frames).__name__}"
        raise ValueError(detail)
    if not isinstance(anchor_index, int) or anchor_index < 0 or anchor_index >= len(frames):
        detail = f"anchor_index={anchor_index} outside metadata range [0, {len(frames) - 1}]"
        raise ValueError(detail)

    # Do not validate or index any future frame fields; only the list length is
    # used to disclose that the pickle container included later records.
    history_frames = frames[: anchor_index + 1]
    if len(history_frames) < 2:
        raise ValueError("Insufficient historical frames: at least two poses are required")

    timestamps = np.asarray(
        [_timestamp_us(frame, i) for i, frame in enumerate(history_frames)], dtype=np.int64
    )
    duplicate = np.flatnonzero(np.diff(timestamps) == 0)
    if duplicate.size:
        i = int(duplicate[0] + 1)
        detail = f"duplicate timestamp {int(timestamps[i])} us"
        raise _fail(i, history_frames[i], "timestamp", detail)
    decreasing = np.flatnonzero(np.diff(timestamps) < 0)
    if decreasing.size:
        i = int(decreasing[0] + 1)
        detail = f"timestamps are not increasing after frame {i - 1}"
        raise _fail(i, history_frames[i], "timestamp", detail)

    translations = np.stack(
        [_vector(frame, i, "ego2global_translation", 3) for i, frame in enumerate(history_frames)]
    )
    quaternions = np.stack(
        [_vector(frame, i, "ego2global_rotation", 4) for i, frame in enumerate(history_frames)]
    )
    anchor = history_frames[-1]
    anchor_timestamp = int(timestamps[-1])
    history_offsets = _history_offsets()
    target_timestamps = anchor_timestamp + np.rint(history_offsets * 1_000_000).astype(np.int64)
    if target_timestamps[0] < timestamps[0]:
        raise ValueError(
            "Insufficient historical coverage: "
            f"requested start={int(target_timestamps[0])} us, "
            f"earliest pose={int(timestamps[0])} us, "
            f"anchor_index={anchor_index}"
        )

    # Check gaps in the pose segment used to interpolate the requested history.
    start_bracket = max(0, int(np.searchsorted(timestamps, target_timestamps[0], side="right")) - 1)
    relevant_gaps = np.diff(timestamps[start_bracket:])
    if relevant_gaps.size:
        gap_index = int(np.argmax(relevant_gaps))
        largest_gap_s = float(relevant_gaps[gap_index]) / 1_000_000.0
        if largest_gap_s > max_pose_gap_s:
            left_index = start_bracket + gap_index
            right_index = left_index + 1
            raise ValueError(
                "NAVSIM source pose gap exceeds limit: "
                f"frames={left_index}->{right_index}, "
                f"tokens={history_frames[left_index].get('token')!r}->"
                f"{history_frames[right_index].get('token')!r}, "
                f"gap={largest_gap_s:.6f}s, limit={max_pose_gap_s:.6f}s"
            )

    sampled_global_xyz, sampled_global_quat = _interpolate_history(
        timestamps, translations, quaternions, target_timestamps
    )
    anchor_xyz = translations[-1]
    anchor_q = quaternions[-1]
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:  # Match scipy Rotation behavior for lightweight CPU-only tests.
        anchor_q_inv = anchor_q * np.asarray([1.0, -1.0, -1.0, -1.0])
        ego_xyz = np.stack([_quat_rotate(anchor_q_inv, p - anchor_xyz) for p in sampled_global_xyz])
        ego_quat = np.stack([_quat_multiply(anchor_q_inv, q) for q in sampled_global_quat])
        ego_rot = np.stack([_quat_to_matrix(q / np.linalg.norm(q)) for q in ego_quat])
    else:
        anchor_rotation = Rotation.from_quat(anchor_q[[1, 2, 3, 0]])
        ego_xyz = anchor_rotation.inv().apply(sampled_global_xyz - anchor_xyz)
        sampled_rotations = Rotation.from_quat(sampled_global_quat[:, [1, 2, 3, 0]])
        ego_rot = (anchor_rotation.inv() * sampled_rotations).as_matrix().astype(np.float32)
    if not np.allclose(ego_xyz[-1], np.zeros(3), atol=1e-5):
        raise ArithmeticError(f"Anchor-frame position sanity check failed: {ego_xyz[-1].tolist()}")
    if not np.allclose(ego_rot[-1], np.eye(3), atol=1e-5):
        raise ArithmeticError(f"Anchor-frame rotation sanity check failed: {ego_rot[-1].tolist()}")
    # Eliminate floating-point noise only after checking the transform itself.
    ego_xyz[-1] = 0.0
    ego_rot[-1] = np.eye(3, dtype=np.float32)

    camera_arrays: list[np.ndarray] = []
    image_indices: list[list[int]] = []
    image_timestamps: list[list[int]] = []
    image_offsets: list[list[float]] = []
    image_signed_errors: list[list[float]] = []
    image_abs_errors: list[list[float]] = []
    expected_hw: tuple[int, int] | None = None

    for camera_name in camera_names:
        camera_images: list[np.ndarray] = []
        camera_indices: list[int] = []
        camera_times: list[int] = []
        camera_offsets: list[float] = []
        camera_errors: list[float] = []
        camera_abs_errors: list[float] = []
        for target_offset in IMAGE_HISTORY_OFFSETS_S:
            target_timestamp = anchor_timestamp + int(round(target_offset * 1_000_000))
            selected_index = int(np.argmin(np.abs(timestamps - target_timestamp)))
            signed_error_s = (int(timestamps[selected_index]) - target_timestamp) / 1_000_000.0
            abs_error_s = abs(signed_error_s)
            frame = history_frames[selected_index]
            if abs_error_s > max_image_time_error_s:
                raise ValueError(
                    "NAVSIM image timing error exceeds limit: "
                    f"camera={camera_name}, target_offset={target_offset:+.3f}s, "
                    f"selected_frame_index={selected_index}, token={frame.get('token')!r}, "
                    f"selected_offset="
                    f"{(timestamps[selected_index] - anchor_timestamp) / 1e6:+.6f}s, "
                    f"absolute_error={abs_error_s:.6f}s, limit={max_image_time_error_s:.6f}s"
                )
            cams = frame.get("cams") if isinstance(frame, dict) else None
            if not isinstance(cams, dict) or camera_name not in cams:
                raise _fail(selected_index, frame, f"cams.{camera_name}", "camera entry is missing")
            camera_record = cams[camera_name]
            if not isinstance(camera_record, dict):
                raise _fail(
                    selected_index,
                    frame,
                    f"cams.{camera_name}",
                    "camera entry must be a mapping",
                )
            path = _resolve_sensor_path(
                sensor_root, camera_record.get("data_path"), frame, selected_index, camera_name
            )
            try:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                detail = f"cannot decode {path}: {exc}"
                raise _fail(selected_index, frame, f"cams.{camera_name}.data_path", detail) from exc
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                detail = f"expected RGB HWC image, got {rgb.shape}"
                raise _fail(selected_index, frame, f"cams.{camera_name}.data_path", detail)
            hw = (int(rgb.shape[0]), int(rgb.shape[1]))
            if expected_hw is None:
                expected_hw = hw
            elif hw != expected_hw:
                raise _fail(
                    selected_index,
                    frame,
                    f"cams.{camera_name}.data_path",
                    f"image resolution mismatch: expected HxW={expected_hw}, got {hw} at {path}",
                )
            camera_images.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            camera_indices.append(selected_index)
            camera_times.append(int(timestamps[selected_index]))
            offset_s = float((timestamps[selected_index] - anchor_timestamp) / 1_000_000.0)
            camera_offsets.append(offset_s)
            camera_errors.append(float(signed_error_s))
            camera_abs_errors.append(float(abs_error_s))
        camera_arrays.append(np.stack(camera_images, axis=0))
        image_indices.append(camera_indices)
        image_timestamps.append(camera_times)
        image_offsets.append(camera_offsets)
        image_signed_errors.append(camera_errors)
        image_abs_errors.append(camera_abs_errors)

    if camera_names == DEFAULT_CAMERA_NAMES and camera_names[3] == "CAM_L1":
        warnings.warn(
            "Default fourth view CAM_L1 is a left-side camera, not an AR1 front-tele equivalent.",
            UserWarning,
            stacklevel=2,
        )
    image_frames = np.ascontiguousarray(np.stack(camera_arrays, axis=0), dtype=np.uint8)
    gaps_s = np.diff(timestamps).astype(np.float64) / 1_000_000.0
    source_rate_hz = float(1.0 / np.median(gaps_s)) if gaps_s.size else None
    scene_name = anchor.get("scene_name") or anchor.get("log_name")

    return {
        "scene_name": scene_name,
        "log_name": anchor.get("log_name"),
        "anchor_index": anchor_index,
        "anchor_token": anchor.get("token"),
        "anchor_timestamp_us": anchor_timestamp,
        "camera_names": list(camera_names),
        "image_frames": image_frames,
        "image_frame_indices": image_indices,
        "image_timestamps_us": image_timestamps,
        "image_target_offsets_s": [list(IMAGE_HISTORY_OFFSETS_S) for _ in camera_names],
        "image_time_offsets_s": image_offsets,
        "image_time_error_s": image_signed_errors,
        "image_selection_error_s": image_abs_errors,
        "ego_history_xyz": ego_xyz.astype(np.float32)[None, None, ...],
        "ego_history_rot": ego_rot.astype(np.float32)[None, None, ...],
        "history_offsets_s": history_offsets.astype(np.float32).tolist(),
        "coordinate_frame": "anchor ego frame; x-forward, y-left, z-up",
        "history_interpolation": "linear translation + quaternion SLERP from NAVSIM ego poses",
        "ego_history_interpolated": True,
        "source_pose_median_rate_hz": source_rate_hz,
        "source_camera_rate_hz_estimate": source_rate_hz,
        "source_rate_note": (
            "Estimated from median historical frame interval; "
            "OpenScene mini is nominally about 2 Hz."
        ),
        "ar1_reference_camera_rate_hz": AR1_REFERENCE_CAMERA_RATE_HZ,
        # NAVSIM has no camera that is asserted to be the AR1 front-tele view.
        "fourth_camera_equivalent_to_ar1_front_tele": False,
        "camera_mapping_note": (
            "CAM_L1 is not equivalent to the AR1 front-tele camera."
            if camera_names[3] == "CAM_L1"
            else "Camera names are passed as selected; no equivalence is inferred."
        ),
        "future_frames_deserialized": len(frames) > anchor_index + 1,
        "future_used_for_model_input": False,
    }
