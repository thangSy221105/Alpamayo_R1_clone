"""Hard-link only selected NAVSIM history images into one contained sensor root.

This prepares camera inputs for an existing selected-anchor JSONL. It never
collects images from the future ground-truth window.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path, PurePosixPath

import yaml


def relative_image_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Invalid camera data_path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"Unsafe camera data_path: {raw!r}")
    return Path(*path.parts)


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--navtest-filter-yaml", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=["CAM_L0", "CAM_F0", "CAM_R0", "CAM_L1"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.sensor_root.is_dir() or not args.metadata_root.is_dir():
        raise NotADirectoryError("Metadata root and merged sensor root must exist")
    if len(args.cameras) != len(set(args.cameras)):
        raise ValueError("Camera names must be distinct")
    with args.input_jsonl.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    tokens = [str(row.get("anchor_token", "")).strip() for row in rows]
    if not tokens or any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Input must contain unique nonempty anchor tokens")
    config = yaml.safe_load(args.navtest_filter_yaml.read_text(encoding="utf-8"))
    allowed_logs = set(map(str, config["log_names"]))
    official_tokens = set(map(str, config["tokens"]))
    if not set(tokens) <= official_tokens:
        raise ValueError("Input contains tokens outside the official NAVTEST filter")

    wanted = set(tokens)
    found: set[str] = set()
    image_paths: set[Path] = set()
    for metadata_path in sorted(args.metadata_root.rglob("*.pkl")):
        if metadata_path.stem not in allowed_logs:
            continue
        with metadata_path.open("rb") as stream:
            frames = pickle.load(stream)
        if not isinstance(frames, list):
            raise ValueError(f"Expected frame list in {metadata_path}")
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            token = str(frame.get("token", ""))
            if token not in wanted:
                continue
            if token in found:
                raise ValueError(f"Duplicate anchor token in metadata: {token}")
            if index < 3 or index + 10 >= len(frames):
                raise ValueError(f"Incomplete history/future window for anchor {token}")
            found.add(token)
            for history_frame in frames[index - 3 : index + 1]:
                cameras = history_frame.get("cams")
                if not isinstance(cameras, dict):
                    raise ValueError(f"Missing camera metadata for anchor {token}")
                for camera in args.cameras:
                    entry = cameras.get(camera)
                    if not isinstance(entry, dict):
                        raise ValueError(f"Missing {camera} metadata for anchor {token}")
                    image_paths.add(relative_image_path(entry.get("data_path")))
        del frames
    if found != wanted:
        raise ValueError(f"Missing {len(wanted - found)} anchor(s) in metadata: {sorted(wanted - found)[:5]}")

    merged = args.sensor_root.resolve()
    source_roots = sorted(
        root.resolve()
        for root in args.probe_root.glob("*/openscene-v1.1/sensor_blobs/test")
        if root.is_dir() and root.resolve() != merged
    )
    if not source_roots:
        raise FileNotFoundError("No source camera roots found under probe root")
    counts: dict[str, int] = defaultdict(int)
    missing: list[str] = []
    for relative in sorted(image_paths):
        dest = merged / relative
        if not within(dest.parent, merged):
            raise ValueError(f"Destination parent escapes merged sensor root: {dest.parent}")
        if dest.exists():
            if not dest.is_file() or not within(dest, merged):
                raise ValueError(f"Destination is not a contained regular file: {dest}")
            counts["already_present"] += 1
            continue
        if dest.is_symlink():
            raise ValueError(f"Broken destination symlink: {dest}")
        source = next(
            (candidate for root in source_roots
             if (candidate := root / relative).is_file() and within(candidate, root)),
            None,
        )
        if source is None:
            missing.append(relative.as_posix())
            continue
        if os.stat(source).st_dev != os.stat(merged).st_dev:
            raise OSError(f"Source and merged root are on different filesystems: {source}")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not within(dest.parent, merged):
                raise ValueError(f"Destination parent escapes merged sensor root: {dest.parent}")
            os.link(source, dest)
        counts["would_link" if args.dry_run else "linked"] += 1
    print(json.dumps({
        "anchors": len(tokens),
        "unique_history_images": len(image_paths),
        "source_roots": list(map(str, source_roots)),
        **counts,
        "missing_from_downloads": len(missing),
        "first_missing": missing[:20],
        "dry_run": args.dry_run,
    }, indent=2))
    if missing:
        raise SystemExit("Required history images are absent from downloaded sources")


if __name__ == "__main__":
    main()
