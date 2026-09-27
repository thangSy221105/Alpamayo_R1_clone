import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from alpamayo_r1.navsim_adapter import (
    DEFAULT_CAMERA_NAMES,
    _select_navsim_window_by_anchor_token,
)
from alpamayo_r1.run_navsim_batch import (
    calculate_ade_fde,
    deduplicate_candidates_by_anchor,
    discover_candidates,
    load_excluded_anchor_tokens,
    limit_candidates_per_scene,
    select_diverse_candidates,
)


class NavsimBatchRunnerTest(unittest.TestCase):
    def test_ade_fde_interpolates_at_observed_gt_times(self):
        prediction_times = np.arange(1, 65, dtype=np.float64) * 0.1
        prediction = np.column_stack((prediction_times, prediction_times * 2.0, prediction_times * 0.0))
        gt_times = np.asarray([0.5, 1.0, 2.5, 5.0001])
        gt = np.column_stack((gt_times, gt_times * 2.0, gt_times * 0.0))

        metrics = calculate_ade_fde(prediction, gt, gt_times, waypoint_dt_s=0.1)

        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(metrics["ade_m"], 0.0, places=10)
        self.assertAlmostEqual(metrics["fde_m"], 0.0, places=10)

    def test_candidate_selection_spreads_samples_across_sources(self):
        candidates = [
            {"metadata_path": "a.pkl", "scene_index": index}
            for index in range(8)
        ] + [
            {"metadata_path": "b.pkl", "scene_index": index}
            for index in range(4)
        ]

        selected = select_diverse_candidates(candidates, max_samples=6)

        self.assertEqual(len(selected), 6)
        source_counts = {}
        for item in selected:
            source_counts[item["metadata_path"]] = source_counts.get(item["metadata_path"], 0) + 1
        self.assertEqual(source_counts, {"a.pkl": 3, "b.pkl": 3})
        self.assertEqual(len({(item["metadata_path"], item["scene_index"]) for item in selected}), 6)

    def test_per_scene_limit_keeps_evenly_spaced_anchor_windows(self):
        candidates = [
            {
                "metadata_path": "a.pkl",
                "scene_name": "scene-1",
                "scene_index": index,
            }
            for index in range(5)
        ] + [
            {
                "metadata_path": "a.pkl",
                "scene_name": "scene-2",
                "scene_index": index,
            }
            for index in (5, 6)
        ]

        selected = limit_candidates_per_scene(candidates, max_samples_per_scene=2)

        self.assertEqual(
            [(item["scene_name"], item["scene_index"]) for item in selected],
            [("scene-1", 0), ("scene-1", 4), ("scene-2", 5), ("scene-2", 6)],
        )

    def test_deduplicate_candidates_keeps_one_per_anchor_token(self):
        candidates = [
            {"metadata_path": "a.pkl", "scene_index": 3, "anchor_token": "same"},
            {"metadata_path": "a.pkl", "scene_index": 8, "anchor_token": "same"},
            {"metadata_path": "b.pkl", "scene_index": 2, "anchor_token": "other"},
        ]

        unique, duplicate_count = deduplicate_candidates_by_anchor(candidates)

        self.assertEqual(duplicate_count, 1)
        self.assertEqual([item["anchor_token"] for item in unique], ["same", "other"])
        self.assertEqual(unique[0]["scene_index"], 3)

    def test_excluded_anchor_tokens_accept_jsonl_and_plain_text(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "results.jsonl"
            text_path = Path(directory) / "tokens.txt"
            jsonl_path.write_text(
                '{"anchor_token":"token-a"}\n{"anchor_token":"token-b"}\n',
                encoding="utf-8",
            )
            text_path.write_text("token-c\ntoken-a\n", encoding="utf-8")

            tokens = load_excluded_anchor_tokens([jsonl_path, text_path])

        self.assertEqual(tokens, {"token-a", "token-b", "token-c"})

    def test_navtest_token_selects_exact_sliding_history_and_future(self):
        frames = [
            {"token": f"token-{index}", "roadblock_ids": ["route"] if index == 8 else []}
            for index in range(19)
        ]

        start, history, future = _select_navsim_window_by_anchor_token(
            frames,
            anchor_token="token-8",
            history_frame_count=4,
            future_frame_count=10,
            require_route=True,
        )

        self.assertEqual(start, 5)
        self.assertEqual([frame["token"] for frame in history], [f"token-{i}" for i in range(5, 9)])
        self.assertEqual([frame["token"] for frame in future], [f"token-{i}" for i in range(9, 19)])

    def test_navtest_discovery_uses_official_logs_and_anchor_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sensor_root = root / "sensors"
            sensor_root.mkdir()
            metadata_path = root / "test-log.pkl"
            frames = []
            for index in range(19):
                cams = {}
                for camera_name in DEFAULT_CAMERA_NAMES:
                    relative = Path("test-log") / camera_name / f"{index}.jpg"
                    image_path = sensor_root / relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(b"test")
                    cams[camera_name] = {"data_path": relative.as_posix()}
                frames.append(
                    {
                        "token": f"token-{index}",
                        "timestamp": index * 500_000,
                        "scene_name": "test-log-scene",
                        "log_name": "test-log",
                        "roadblock_ids": ["route"] if index == 8 else [],
                        "cams": cams,
                    }
                )
            with metadata_path.open("wb") as stream:
                pickle.dump(frames, stream)

            candidates, counts = discover_candidates(
                metadata_paths=[metadata_path],
                sensor_root=sensor_root,
                history_frames=4,
                future_frames=10,
                camera_names=DEFAULT_CAMERA_NAMES,
                max_image_time_error_s=0.05,
                excluded_scene_names={"old-mini-scene"},
                navtest_filter={
                    "log_names": {"test-log"},
                    "tokens": {"token-8", "not-present"},
                    "has_route": True,
                },
            )

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["anchor_token"], "token-8")
            self.assertEqual(candidates[0]["anchor_index"], 8)
            self.assertEqual(counts[str(metadata_path)], 1)


if __name__ == "__main__":
    unittest.main()
