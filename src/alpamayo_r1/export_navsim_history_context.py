"""Export compact, strictly history-only NAVSIM context for selected anchors.

The exporter scans OpenScene metadata pickle files at the data source, matches
exact anchor tokens from an evaluation JSONL, and writes only per-frame context
at or before each anchor. It never serializes future frames or evaluator GT.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


FRAME_FIELDS = (
    "token",
    "frame_idx",
    "timestamp",
    "ego2global_translation",
    "ego2global_rotation",
    "ego_dynamic_state",
    "roadblock_ids",
    "traffic_lights",
    "driving_command",
    "anns",
)


def _jsonable(value: Any) -> Any:
    """Convert common NAVSIM/NumPy annotation values to JSON-safe objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def load_anchor_rows(input_jsonl: Path) -> list[dict[str, Any]]:
    """Read result rows and retain one row per exact NAVSIM anchor token."""
    rows_by_token: dict[str, dict[str, Any]] = {}
    with input_jsonl.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                row = json.loads(line)
            else:
                row = {"anchor_token": line.strip()}
            token = str(row.get("anchor_token", "")).strip()
            if not token or token == "None":
                raise ValueError(f"Missing anchor_token at {input_jsonl}:{line_number}")
            if token in rows_by_token:
                # Noise/mode result files can repeat anchors; their context is identical.
                continue
            rows_by_token[token] = row
    if not rows_by_token:
        raise ValueError(f"No anchor_token records found in {input_jsonl}")
    return list(rows_by_token.values())


def _compact_frame(frame: dict[str, Any], source_index: int, anchor_timestamp_us: int) -> dict[str, Any]:
    timestamp_us = int(frame["timestamp"])
    if timestamp_us > anchor_timestamp_us:
        raise ValueError("Internal error: attempted to export a post-anchor frame")
    compact = {field: _jsonable(frame[field]) for field in FRAME_FIELDS if field in frame}
    compact["source_frame_index"] = source_index
    compact["timestamp_us"] = timestamp_us
    compact["time_offset_from_anchor_s"] = (timestamp_us - anchor_timestamp_us) / 1_000_000.0
    return compact


def build_context_record(
    anchor_row: dict[str, Any],
    frames: list[Any],
    history_seconds: float = 5.0,
    model_history_frames: int = 4,
) -> dict[str, Any] | None:
    """Build one context payload. Returns None when the token is absent."""
    token = str(anchor_row["anchor_token"])
    matches = [
        index
        for index, frame in enumerate(frames)
        if isinstance(frame, dict) and str(frame.get("token")) == token
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Anchor token {token!r} occurs {len(matches)} times in one metadata file")

    anchor_index = matches[0]
    anchor = frames[anchor_index]
    if "timestamp" not in anchor:
        raise ValueError(f"Anchor token {token!r} has no timestamp")
    anchor_timestamp_us = int(anchor["timestamp"])
    cutoff_us = anchor_timestamp_us - int(round(history_seconds * 1_000_000))

    history: list[dict[str, Any]] = []
    for index in range(anchor_index + 1):
        frame = frames[index]
        if not isinstance(frame, dict) or "timestamp" not in frame:
            continue
        timestamp_us = int(frame["timestamp"])
        # The index bound and timestamp bound both prevent future leakage.
        if cutoff_us <= timestamp_us <= anchor_timestamp_us:
            history.append(_compact_frame(frame, index, anchor_timestamp_us))

    if not history or str(history[-1].get("token")) != token:
        raise ValueError(f"Anchor {token!r} was not the final exported history frame")
    history_timestamps = [frame["timestamp_us"] for frame in history]
    if any(right <= left for left, right in zip(history_timestamps, history_timestamps[1:])):
        raise ValueError(f"History timestamps are not strictly increasing for anchor {token!r}")
    if any(frame["timestamp_us"] > anchor_timestamp_us for frame in history):
        raise ValueError(f"Future timestamp leaked into context for {token!r}")

    protocol_tokens = [str(frame["token"]) for frame in history[-model_history_frames :]]
    return {
        "anchor_token": token,
        "scene_name": anchor_row.get("scene_name", anchor.get("scene_name")),
        "log_name": anchor_row.get("log_name", anchor.get("log_name")),
        "reasoning": anchor_row.get("reasoning"),
        "anchor_timestamp_us": anchor_timestamp_us,
        "context_window_seconds": history_seconds,
        "history_frame_count": len(history),
        "model_protocol_history_frame_count": len(protocol_tokens),
        "history_start_timestamp_us": history[0]["timestamp_us"],
        "available_history_span_s": (
            anchor_timestamp_us - history[0]["timestamp_us"]
        ) / 1_000_000.0,
        "model_protocol_history_tokens": protocol_tokens,
        "history_frames": history,
        "future_data_exported": False,
        "future_gt_exported": False,
        "driving_command_note": "Route/navigation context only; not a ground-truth action label.",
        "context_source": "OpenScene metadata; exact anchor-token match; timestamps <= anchor only",
    }


def export_context(
    input_jsonl: Path,
    metadata_root: Path,
    output_jsonl: Path,
    history_seconds: float = 5.0,
    model_history_frames: int = 4,
) -> dict[str, int]:
    if history_seconds <= 0 or model_history_frames < 1:
        raise ValueError("history_seconds and model_history_frames must be positive")
    anchor_rows = load_anchor_rows(input_jsonl)
    rows_by_token = {str(row["anchor_token"]): row for row in anchor_rows}
    contexts: dict[str, dict[str, Any]] = {}
    metadata_files = sorted(metadata_root.rglob("*.pkl"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata .pkl files found under {metadata_root}")

    for metadata_path in metadata_files:
        with metadata_path.open("rb") as stream:
            frames = pickle.load(stream)
        if not isinstance(frames, list):
            raise ValueError(f"Expected a list in {metadata_path}, got {type(frames).__name__}")
        matched_indices: dict[str, int] = {}
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            token = str(frame.get("token", ""))
            if token not in rows_by_token:
                continue
            if token in matched_indices:
                raise ValueError(f"Anchor token {token!r} occurs more than once in {metadata_path}")
            matched_indices[token] = frame_index
        for token in matched_indices:
            if token in contexts:
                raise ValueError(f"Anchor token {token!r} occurs in multiple metadata files")
            anchor_row = rows_by_token[token]
            context = build_context_record(
                anchor_row,
                frames,
                history_seconds=history_seconds,
                model_history_frames=model_history_frames,
            )
            if context is not None:
                context["metadata_source_file"] = metadata_path.name
                contexts[token] = context

    missing = sorted(set(rows_by_token) - set(contexts))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Could not match {len(missing)} anchor token(s); first: {preview}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for token in rows_by_token:
            stream.write(json.dumps(contexts[token], ensure_ascii=False, allow_nan=False) + "\n")
    return {"anchors": len(contexts), "metadata_files_scanned": len(metadata_files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Result JSONL with anchor_token fields, or a plain text file with one token per line",
    )
    parser.add_argument("--metadata-root", type=Path, required=True, help="Root containing NAVSIM metadata .pkl files")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--history-seconds", type=float, default=5.0)
    parser.add_argument("--model-history-frames", type=int, default=4)
    args = parser.parse_args()
    summary = export_context(
        args.input_jsonl,
        args.metadata_root,
        args.output_jsonl,
        history_seconds=args.history_seconds,
        model_history_frames=args.model_history_frames,
    )
    print(json.dumps({**summary, "output_jsonl": str(args.output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
