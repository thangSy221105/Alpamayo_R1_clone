from __future__ import annotations

import unittest

import numpy as np

from alpamayo_r1.navsim_smoke_utils import extract_first_reasoning, validate_trajectory_output


class ReasoningExtractionTests(unittest.TestCase):
    def test_string_bytes_and_nested_lists(self):
        self.assertEqual(extract_first_reasoning("  keep lane  "), "keep lane")
        self.assertEqual(extract_first_reasoning(b"slow down"), "slow down")
        self.assertEqual(extract_first_reasoning([None, [["turn left"], "ignored"]]), "turn left")

    def test_numpy_array(self):
        value = np.asarray([[["yield to traffic"]]], dtype=object)
        self.assertEqual(extract_first_reasoning(value), "yield to traffic")

    def test_missing_none_empty_and_nested_mapping(self):
        self.assertIsNone(extract_first_reasoning(None))
        self.assertIsNone(extract_first_reasoning({}))
        self.assertIsNone(extract_first_reasoning([]))
        self.assertIsNone(extract_first_reasoning("   "))
        self.assertIsNone(extract_first_reasoning({"cot": None}))
        self.assertIsNone(extract_first_reasoning(np.asarray([], dtype=object)))
        nested = {"cot": [[{"text": "brake gently"}]]}
        self.assertEqual(extract_first_reasoning(nested), "brake gently")


class OutputShapeTests(unittest.TestCase):
    def test_expected_shapes(self):
        xyz = np.zeros((1, 1, 2, 64, 3), dtype=np.float32)
        rot = np.zeros((1, 1, 2, 64, 3, 3), dtype=np.float32)
        self.assertEqual(
            validate_trajectory_output(xyz, rot),
            ((1, 1, 2, 64, 3), (1, 1, 2, 64, 3, 3)),
        )

    def test_bad_xyz_and_rotation_shapes_fail(self):
        with self.assertRaisesRegex(ValueError, "pred_xyz must have shape"):
            validate_trajectory_output(np.zeros((64, 3)), np.zeros((64, 3, 3)))
        with self.assertRaisesRegex(ValueError, "pred_rot must have shape"):
            validate_trajectory_output(np.zeros((1, 1, 1, 64, 3)), np.zeros((1, 64, 3, 3)))
        with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
            xyz = np.zeros((1, 1, 1, 0, 3))
            rot = np.zeros((1, 1, 1, 0, 3, 3))
            validate_trajectory_output(xyz, rot)


if __name__ == "__main__":
    unittest.main()
