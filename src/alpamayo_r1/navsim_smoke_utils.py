"""Small, dependency-light helpers for the NAVSIM inference smoke runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_first_reasoning(value: Any) -> str | None:
    """Extract a usable reasoning string without stringifying containers.

    Supports nested Python containers and NumPy-like values exposing
    ``tolist()``. Empty, missing, and unsupported values return ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace").strip()
        return text or None
    if isinstance(value, Mapping):
        for key in ("reasoning", "cot", "text", "content"):
            if key in value:
                result = extract_first_reasoning(value[key])
                if result is not None:
                    return result
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            result = extract_first_reasoning(item)
            if result is not None:
                return result
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            converted = tolist()
        except Exception:
            return None
        if converted is value:
            return None
        return extract_first_reasoning(converted)
    return None


def validate_trajectory_output(
    pred_xyz: Any, pred_rot: Any
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate the trajectory tensor rank/shape contract and return shapes."""
    try:
        xyz_shape = tuple(int(dim) for dim in pred_xyz.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("pred_xyz must expose a numeric shape") from exc
    if len(xyz_shape) != 5:
        raise ValueError(f"pred_xyz must have shape [B, sets, samples, T, 3], got {xyz_shape}")
    if any(dim <= 0 for dim in xyz_shape) or xyz_shape[-1] != 3:
        raise ValueError(f"pred_xyz dimensions must be positive and end in 3, got {xyz_shape}")

    try:
        rot_shape = tuple(int(dim) for dim in pred_rot.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("pred_rot must expose a numeric shape") from exc
    expected_rot_shape = xyz_shape[:-1] + (3, 3)
    if rot_shape != expected_rot_shape:
        raise ValueError(f"pred_rot must have shape {expected_rot_shape}, got {rot_shape}")
    if any(dim <= 0 for dim in rot_shape):
        raise ValueError(f"pred_rot dimensions must be positive, got {rot_shape}")
    return xyz_shape, rot_shape
