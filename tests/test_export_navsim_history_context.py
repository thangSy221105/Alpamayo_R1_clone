from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from alpamayo_r1.export_navsim_history_context import (
    build_context_record,
    export_context,
    load_anchor_rows,
)


class ExportNavsimHistoryContextTests(unittest.TestCase):
    def make_frames(self) -> list[dict]:
        frames = []
        for index in range(15):
            frames.append(
                {
                    "token": f"token-{index}",
                    "frame_idx": index,
                    "timestamp": 1_000_000 + index * 500_000,
                    "ego2global_translation": np.asarray([index, 0.0, 0.0]),
                    "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
                    "ego_dynamic_state": [float(index), 0.0, 0.0, 0.0],
                    "roadblock_ids": ["route-1"],
                    "traffic_lights": [{"status": "GREEN"}],
                    "driving_command": "STRAIGHT",
                    "anns": {
                        "track_tokens": [f"track-{index}"],
                        "gt_velocity_3d": np.asarray([[1.0, 0.0, 0.0]]),
                    },
                    "cams": {"CAM_F0": {"data_path": f"future-or-history-{index}.jpg"}},
                }
            )
        return frames

    def test_context_is_exact_anchor_history_and_excludes_later_frames(self):
        anchor_row = {"anchor_token": "token-10", "reasoning": "Keep distance."}
        context = build_context_record(anchor_row, self.make_frames(), history_seconds=2.0)

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context["anchor_timestamp_us"], 6_000_000)
        self.assertEqual(
            [frame["token"] for frame in context["history_frames"]],
            [f"token-{i}" for i in range(6, 11)],
        )
        self.assertTrue(all(frame["timestamp_us"] <= 6_000_000 for frame in context["history_frames"]))
        self.assertNotIn("cams", context["history_frames"][0])
        self.assertEqual(context["history_frames"][-1]["anns"]["track_tokens"], ["track-10"])
        self.assertEqual(context["model_protocol_history_tokens"], [f"token-{i}" for i in range(7, 11)])
        self.assertFalse(context["future_data_exported"])

    def test_export_matches_anchor_tokens_and_writes_compact_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata_root = root / "meta"
            metadata_root.mkdir()
            with (metadata_root / "scene.pkl").open("wb") as stream:
                pickle.dump(self.make_frames(), stream)
            input_jsonl = root / "results.jsonl"
            input_jsonl.write_text(
                json.dumps({"anchor_token": "token-10", "reasoning": "Keep distance."}) + "\n",
                encoding="utf-8",
            )
            output_jsonl = root / "context.jsonl"

            summary = export_context(input_jsonl, metadata_root, output_jsonl, history_seconds=2.0)

            self.assertEqual(summary["anchors"], 1)
            record = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["anchor_token"], "token-10")
            self.assertEqual(record["history_frames"][-1]["token"], "token-10")
            self.assertFalse(record["future_gt_exported"])

    def test_plain_anchor_token_list_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            token_file = Path(temp) / "tokens.txt"
            token_file.write_text("token-a\ntoken-b\ntoken-a\n", encoding="utf-8")
            rows = load_anchor_rows(token_file)
            self.assertEqual([row["anchor_token"] for row in rows], ["token-a", "token-b"])


if __name__ == "__main__":
    unittest.main()
