"""Prepare a strictly history-only NAVSIM/OpenScene sample for Alpamayo-R1.

NAVSIM metadata stores ego poses at roughly 2 Hz.  The model input history is
sampled at 10 Hz by interpolating those poses; these are not 10 Hz observations.
Images and poses after the requested anchor are never inspected or returned.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# NAVSIM has no optical front-tele camera. The crop remains available as an
# explicit option, while the default fourth view is CAM_L1.
PSEUDO_TELE_CAMERA_NAME = "CAM_F0_TELE_CROP"
# Center on the NAVSIM camera optical axis; double the 558x314 crop dimensions.
PSEUDO_TELE_NAVSIM_ROI = (0.209375, 0.2093, 0.790625, 0.7907)
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


def _source_camera_name(camera_name: str) -> str:
    """Resolve derived camera views to their underlying NAVSIM sensor."""
    if camera_name == PSEUDO_TELE_CAMERA_NAME:
        return "CAM_F0"
    return camera_name


def _pseudo_tele_crop_bounds(width: int, height: int) -> tuple[int, int, int, int]:
    """Return an optical-axis-centered CAM_F0 crop with source aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(width, height)}")

    x0_rel, y0_rel, x1_rel, y1_rel = PSEUDO_TELE_NAVSIM_ROI
    center_x = (x0_rel + x1_rel) / 2.0
    center_y = (y0_rel + y1_rel) / 2.0
    crop_width = min(width, max(1, int(round((x1_rel - x0_rel) * width))))
    # Keep the crop's pixel aspect equal to the source image; do not stretch it.
    crop_height = min(height, max(1, int(round(crop_width * height / width))))
    x0 = int(round(center_x * width - crop_width / 2.0))
    y0 = int(round(center_y * height - crop_height / 2.0))
    x0 = min(width - crop_width, max(0, x0))
    y0 = min(height - crop_height, max(0, y0))
    return x0, y0, crop_width, crop_height


def _make_pseudo_tele_view(rgb: np.ndarray) -> np.ndarray:
    """Make a digital pseudo-tele view using a centered CAM_F0 region.

    This is a fixed center-region crop, not a calibrated 30-degree camera or
    an optical telephoto view. It is centered on the optical axis and resized
    to the model's input shape while preserving the source aspect ratio.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for the CAM_F0 pseudo-tele view; "
            "install opencv-python-headless in the AR1 environment"
        ) from exc

    height, width = rgb.shape[:2]
    x0, y0, crop_width, crop_height = _pseudo_tele_crop_bounds(width, height)
    crop = rgb[y0 : y0 + crop_height, x0 : x0 + crop_width]
    if crop.size == 0:
        raise ValueError("CAM_F0 centered pseudo-tele ROI produced an empty crop")
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)


def _history_offsets() -> np.ndarray:
    # 16 samples at 10 Hz: -1.5, -1.4, ..., 0.0 seconds.
    return np.linspace(-(AR1_HISTORY_STEPS - 1) * AR1_HISTORY_STEP_S, 0.0, AR1_HISTORY_STEPS)


def _select_navsim_window(
    frames: list[Any],
    scene_index: int,
    history_frame_count: int,
    future_frame_count: int,
    require_route: bool,
) -> tuple[int, list[Any], list[Any]]:
    """Select a NAVSIM SceneFilter-style non-overlapping history/future window.

    NAVSIM's ``frame_interval: null`` means ``num_history_frames +
    num_future_frames``. Its route filter checks ``roadblock_ids`` on the final
    history frame (the prediction anchor), then uses the frames after that as
    ground truth.
    """
    if history_frame_count < 2:
        raise ValueError("NAVSIM split requires at least two history frames")
    if future_frame_count < 1:
        raise ValueError("NAVSIM split requires at least one future frame")
    if scene_index < 0:
        raise ValueError("scene_index must be non-negative")

    window_size = history_frame_count + future_frame_count
    valid_windows: list[tuple[int, list[Any]]] = []
    for start in range(0, len(frames) - window_size + 1, window_size):
        window = frames[start : start + window_size]
        if len(window) != window_size:
            continue
        if require_route:
            anchor_frame = window[history_frame_count - 1]
            route_ids = anchor_frame.get("roadblock_ids") if isinstance(anchor_frame, dict) else None
            if not isinstance(route_ids, (list, tuple)) or not route_ids:
                continue
        valid_windows.append((start, window))

    if scene_index >= len(valid_windows):
        raise ValueError(
            "Requested NAVSIM scene window is unavailable: "
            f"scene_index={scene_index}, eligible_windows={len(valid_windows)}, "
            f"window_size={window_size}, require_route={require_route}"
        )

    start, window = valid_windows[scene_index]
    return start, window[:history_frame_count], window[history_frame_count:]


def _select_navsim_window_by_anchor_token(
    frames: list[Any],
    anchor_token: str,
    history_frame_count: int,
    future_frame_count: int,
    require_route: bool,
) -> tuple[int, list[Any], list[Any]]:
    """Select the exact sliding NAVSIM window named by its official anchor token."""
    if history_frame_count < 2 or future_frame_count < 1:
        raise ValueError("NAVSIM token window needs at least two history and one future frame")
    matches = [
        index
        for index, frame in enumerate(frames)
        if isinstance(frame, dict) and str(frame.get("token")) == anchor_token
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one NAVSIM anchor token {anchor_token!r}; found {len(matches)}"
        )

    anchor_index = matches[0]
    window_start = anchor_index - history_frame_count + 1
    future_end = anchor_index + future_frame_count + 1
    if window_start < 0 or future_end > len(frames):
        raise ValueError(
            "Official NAVSIM token does not have the configured history/future window: "
            f"anchor_index={anchor_index}, history={history_frame_count}, "
            f"future={future_frame_count}, frames={len(frames)}"
        )

    history = frames[window_start : anchor_index + 1]
    future = frames[anchor_index + 1 : future_end]
    if require_route:
        route_ids = history[-1].get("roadblock_ids") if isinstance(history[-1], dict) else None
        if not isinstance(route_ids, (list, tuple)) or not route_ids:
            raise ValueError(f"NAVSIM anchor token {anchor_token!r} has no route")
    return window_start, history, future


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
    anchor_index: int | None = None,
    camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
    max_image_time_error_s: float = DEFAULT_MAX_IMAGE_TIME_ERROR_S,
    max_pose_gap_s: float = DEFAULT_MAX_POSE_GAP_S,
    scene_index: int | None = None,
    history_frame_count: int = 4,
    future_frame_count: int = 10,
    require_route: bool = True,
    metadata_frames: list[Any] | None = None,
    anchor_token: str | None = None,
) -> dict[str, Any]:
    """Load NAVSIM history for inference and keep its future as separate GT.

    Returned arrays are NumPy arrays so this adapter can be tested without
    importing the model stack. The smoke runner converts history arrays to
    torch tensors. ``scene_index`` selects the legacy non-overlapping window;
    ``anchor_token`` selects an exact official NAVSIM-filter token using a
    sliding window (the NAVSIM v1.1 navtest protocol). ``anchor_index`` remains
    available for isolated adapter tests and backward compatibility.

    NAVSIM pose quaternions are scalar-first ``[w, x, y, z]``. Future frames
    are transformed to the anchor ego frame and returned only under
    ``ground_truth_future``; they are never used to construct model inputs.
    """
    if not math.isfinite(max_image_time_error_s) or max_image_time_error_s < 0:
        raise ValueError("max_image_time_error_s must be finite and non-negative")
    if not math.isfinite(max_pose_gap_s) or max_pose_gap_s <= 0:
        raise ValueError("max_pose_gap_s must be finite and positive")
    if len(camera_names) not in (3, 4) or len(set(camera_names)) != len(camera_names):
        raise ValueError(
            "Three or four distinct camera names are required, "
            f"got {camera_names!r}"
        )

    metadata_path = Path(metadata_path)
    sensor_root = Path(sensor_root)
    if metadata_frames is None:
        with metadata_path.open("rb") as stream:
            frames = pickle.load(stream)  # The pickle container is deserialized as a whole.
    else:
        # Batch evaluation can reuse one deserialized metadata sequence across
        # several non-overlapping windows instead of unpickling it per sample.
        frames = metadata_frames
    if not isinstance(frames, list) or not frames:
        detail = f"NAVSIM metadata root must be a non-empty list; got {type(frames).__name__}"
        raise ValueError(detail)

    navsim_split: dict[str, Any] | None = None
    ground_truth_frames: list[Any] = []
    if sum(value is not None for value in (scene_index, anchor_index, anchor_token)) > 1:
        raise ValueError("Pass only one of scene_index, anchor_index, or anchor_token")

    if anchor_token is not None:
        window_start, history_frames, ground_truth_frames = _select_navsim_window_by_anchor_token(
            frames,
            anchor_token=anchor_token,
            history_frame_count=history_frame_count,
            future_frame_count=future_frame_count,
            require_route=require_route,
        )
        anchor_index = window_start + history_frame_count - 1
        history_source_indices = list(range(window_start, anchor_index + 1))
        future_source_indices = list(
            range(anchor_index + 1, anchor_index + 1 + len(ground_truth_frames))
        )
        navsim_split = {
            "protocol": "NAVSIM v1.1 navtest SceneFilter token window",
            "anchor_token": anchor_token,
            "prediction_anchor_frame_index": anchor_index,
            "window_start_frame_index": window_start,
            "history_frame_count": history_frame_count,
            "future_frame_count": future_frame_count,
            "frame_interval": 1,
            "require_route": require_route,
            "history_source_frame_indices": history_source_indices,
            "future_ground_truth_source_frame_indices": future_source_indices,
        }
    elif scene_index is not None:
        if anchor_index is not None:
            raise ValueError("Pass scene_index or anchor_index, not both")
        window_start, history_frames, ground_truth_frames = _select_navsim_window(
            frames,
            scene_index=scene_index,
            history_frame_count=history_frame_count,
            future_frame_count=future_frame_count,
            require_route=require_route,
        )
        anchor_index = window_start + history_frame_count - 1
        history_source_indices = list(range(window_start, anchor_index + 1))
        future_source_indices = list(
            range(anchor_index + 1, anchor_index + 1 + len(ground_truth_frames))
        )
        navsim_split = {
            "protocol": "NAVSIM SceneFilter non-overlapping window",
            "scene_index": scene_index,
            "window_index_in_source": window_start // (history_frame_count + future_frame_count),
            "window_start_frame_index": window_start,
            "prediction_anchor_frame_index": anchor_index,
            "history_frame_count": history_frame_count,
            "future_frame_count": future_frame_count,
            "frame_interval": history_frame_count + future_frame_count,
            "require_route": require_route,
            "route_ids_at_anchor": (
                list(history_frames[-1].get("roadblock_ids", []))
                if isinstance(history_frames[-1], dict)
                else []
            ),
            "history_source_frame_indices": history_source_indices,
            "future_ground_truth_source_frame_indices": future_source_indices,
        }
    else:
        if anchor_index is None or not isinstance(anchor_index, int):
            raise ValueError("Provide a valid scene_index or anchor_index")
        if anchor_index < 0 or anchor_index >= len(frames):
            detail = f"anchor_index={anchor_index} outside metadata range [0, {len(frames) - 1}]"
            raise ValueError(detail)
        # Legacy path retained for unit tests; the smoke runner does not use it.
        history_frames = frames[: anchor_index + 1]
        history_source_indices = list(range(anchor_index + 1))
        future_source_indices = []

    if len(history_frames) < 2:
        raise ValueError("Insufficient historical frames: at least two poses are required")

    timestamps = np.asarray(
        [
            _timestamp_us(frame, history_source_indices[i])
            for i, frame in enumerate(history_frames)
        ],
        dtype=np.int64,
    )
    duplicate = np.flatnonzero(np.diff(timestamps) == 0)
    if duplicate.size:
        i = int(duplicate[0] + 1)
        detail = f"duplicate timestamp {int(timestamps[i])} us"
        raise _fail(history_source_indices[i], history_frames[i], "timestamp", detail)
    decreasing = np.flatnonzero(np.diff(timestamps) < 0)
    if decreasing.size:
        i = int(decreasing[0] + 1)
        detail = f"timestamps are not increasing after frame {i - 1}"
        raise _fail(history_source_indices[i], history_frames[i], "timestamp", detail)

    translations = np.stack(
        [
            _vector(frame, history_source_indices[i], "ego2global_translation", 3)
            for i, frame in enumerate(history_frames)
        ]
    )
    quaternions = np.stack(
        [
            _vector(frame, history_source_indices[i], "ego2global_rotation", 4)
            for i, frame in enumerate(history_frames)
        ]
    )
    anchor = history_frames[-1]
    anchor_timestamp = int(timestamps[-1])
    if navsim_split is not None:
        # NAVSIM poses are nominally 2 Hz but their microsecond timestamps are
        # not perfectly periodic. Resample the 16 AR1 history points over the
        # exact observed 4-frame span, avoiding extrapolation by sub-ms jitter.
        observed_history_start_s = float((timestamps[0] - anchor_timestamp) / 1_000_000.0)
        history_offsets = np.linspace(
            observed_history_start_s, 0.0, AR1_HISTORY_STEPS, dtype=np.float64
        )
    else:
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
                f"frames={history_source_indices[left_index]}->"
                f"{history_source_indices[right_index]}, "
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

    ground_truth_future: dict[str, Any] | None = None
    if ground_truth_frames:
        future_timestamps = np.asarray(
            [
                _timestamp_us(frame, future_source_indices[i])
                for i, frame in enumerate(ground_truth_frames)
            ],
            dtype=np.int64,
        )
        combined_timestamps = np.concatenate((timestamps[-1:], future_timestamps))
        invalid_future_order = np.flatnonzero(np.diff(combined_timestamps) <= 0)
        if invalid_future_order.size:
            bad_local_index = int(invalid_future_order[0])
            bad_source_index = future_source_indices[bad_local_index]
            raise _fail(
                bad_source_index,
                ground_truth_frames[bad_local_index],
                "timestamp",
                "future ground-truth timestamps must strictly follow the prediction anchor",
            )
        future_gaps = np.diff(combined_timestamps)
        oversized_future_gap = np.flatnonzero(future_gaps > max_pose_gap_s * 1_000_000)
        if oversized_future_gap.size:
            gap_index = int(oversized_future_gap[0])
            bad_source_index = future_source_indices[gap_index]
            gap_s = float(future_gaps[gap_index]) / 1_000_000.0
            raise _fail(
                bad_source_index,
                ground_truth_frames[gap_index],
                "timestamp",
                f"future frame gap={gap_s:.6f}s exceeds limit={max_pose_gap_s:.6f}s",
            )

        future_translations = np.stack(
            [
                _vector(frame, future_source_indices[i], "ego2global_translation", 3)
                for i, frame in enumerate(ground_truth_frames)
            ]
        )
        future_quaternions = np.stack(
            [
                _vector(frame, future_source_indices[i], "ego2global_rotation", 4)
                for i, frame in enumerate(ground_truth_frames)
            ]
        )
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:  # Match the history transform fallback.
            anchor_q_inv = anchor_q * np.asarray([1.0, -1.0, -1.0, -1.0])
            future_xyz_ego = np.stack(
                [_quat_rotate(anchor_q_inv, p - anchor_xyz) for p in future_translations]
            )
            future_ego_quat = np.stack(
                [_quat_multiply(anchor_q_inv, q) for q in future_quaternions]
            )
            future_rot_ego = np.stack(
                [_quat_to_matrix(q / np.linalg.norm(q)) for q in future_ego_quat]
            )
        else:
            anchor_rotation = Rotation.from_quat(anchor_q[[1, 2, 3, 0]])
            future_rotations = Rotation.from_quat(future_quaternions[:, [1, 2, 3, 0]])
            future_xyz_ego = anchor_rotation.inv().apply(future_translations - anchor_xyz)
            future_rot_ego = (anchor_rotation.inv() * future_rotations).as_matrix()

        ground_truth_future = {
            "source": "NAVSIM SceneFilter future frames after the history anchor",
            "used_for_model_input": False,
            "frame_indices": future_source_indices,
            "timestamps_us": future_timestamps.tolist(),
            "time_offsets_s": ((future_timestamps - anchor_timestamp) / 1_000_000.0).tolist(),
            "ego_xyz": future_xyz_ego.astype(np.float32).tolist(),
            "ego_heading_rad": np.arctan2(future_rot_ego[:, 1, 0], future_rot_ego[:, 0, 0])
            .astype(np.float32)
            .tolist(),
            "coordinate_frame": "prediction-anchor ego frame; x-forward, y-left, z-up",
        }

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
                    f"selected_frame_index={history_source_indices[selected_index]}, "
                    f"token={frame.get('token')!r}, "
                    f"selected_offset="
                    f"{(timestamps[selected_index] - anchor_timestamp) / 1e6:+.6f}s, "
                    f"absolute_error={abs_error_s:.6f}s, limit={max_image_time_error_s:.6f}s"
                )
            source_camera_name = _source_camera_name(camera_name)
            cams = frame.get("cams") if isinstance(frame, dict) else None
            if not isinstance(cams, dict) or source_camera_name not in cams:
                raise _fail(
                    selected_index,
                    frame,
                    f"cams.{source_camera_name}",
                    "source camera entry is missing",
                )
            camera_record = cams[source_camera_name]
            if not isinstance(camera_record, dict):
                raise _fail(
                    selected_index,
                    frame,
                    f"cams.{source_camera_name}",
                    "camera entry must be a mapping",
                )
            path = _resolve_sensor_path(
                sensor_root,
                camera_record.get("data_path"),
                frame,
                history_source_indices[selected_index],
                source_camera_name,
            )
            try:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                detail = f"cannot decode {path}: {exc}"
                raise _fail(
                    history_source_indices[selected_index],
                    frame,
                    f"cams.{source_camera_name}.data_path",
                    detail,
                ) from exc
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                detail = f"expected RGB HWC image, got {rgb.shape}"
                raise _fail(
                    history_source_indices[selected_index],
                    frame,
                    f"cams.{camera_name}.data_path",
                    detail,
                )
            if camera_name == PSEUDO_TELE_CAMERA_NAME:
                try:
                    rgb = _make_pseudo_tele_view(rgb)
                except Exception as exc:
                    raise _fail(
                        history_source_indices[selected_index],
                        frame,
                        f"derived_view.{camera_name}",
                        str(exc),
                    ) from exc
            hw = (int(rgb.shape[0]), int(rgb.shape[1]))
            if expected_hw is None:
                expected_hw = hw
            elif hw != expected_hw:
                raise _fail(
                    history_source_indices[selected_index],
                    frame,
                    f"cams.{source_camera_name}.data_path",
                    f"image resolution mismatch: expected HxW={expected_hw}, got {hw} at {path}",
                )
            camera_images.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            camera_indices.append(history_source_indices[selected_index])
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

    image_frames = np.ascontiguousarray(np.stack(camera_arrays, axis=0), dtype=np.uint8)
    gaps_s = np.diff(timestamps).astype(np.float64) / 1_000_000.0
    source_rate_hz = float(1.0 / np.median(gaps_s)) if gaps_s.size else None
    scene_name = anchor.get("scene_name") or anchor.get("log_name")

    return {
        "scene_name": scene_name,
        "log_name": anchor.get("log_name"),
        "anchor_index": anchor_index,
        "navsim_split": navsim_split,
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
            "OpenScene poses are nominally about 2 Hz."
        ),
        "ar1_reference_camera_rate_hz": AR1_REFERENCE_CAMERA_RATE_HZ,
        # NAVSIM has no camera that is asserted to be the AR1 front-tele view.
        "fourth_camera_equivalent_to_ar1_front_tele": False,
        "camera_mapping_note": (
            "Fourth view CAM_F0_TELE_CROP uses the user's manually marked center ROI "
            "from CAM_F0, expanded vertically to preserve aspect ratio; it is a digital "
            "pseudo-tele crop, not a calibrated 30-degree or optical tele camera."
            if PSEUDO_TELE_CAMERA_NAME in camera_names
            else (
                "CAM_L1 is a left-side camera, not an asserted AR1 front-tele equivalent."
                if "CAM_L1" in camera_names
                else "Camera names are passed as selected; no equivalence is inferred."
            )
        ),
        "future_frames_deserialized": len(frames) > anchor_index + 1,
        "future_used_for_model_input": False,
        "ground_truth_future": ground_truth_future,
    }
