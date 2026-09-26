from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from alpamayo_r1.navsim_adapter import DEFAULT_CAMERA_NAMES, load_navsim_sample


class NavsimAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sensor_root = self.root / "sensor_blobs"
        self.sensor_root.mkdir()
        self.metadata_path = self.root / "metadata.pkl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_frames(
        self,
        timestamps=(1_000_000, 1_500_000, 2_000_000, 2_500_000),
        pose_fn=None,
        resolution_fn=None,
    ) -> list[dict]:
        frames = []
        for frame_index, timestamp in enumerate(timestamps):
            if pose_fn is None:
                translation = [0.0, 0.0, 0.0]
                rotation = [1.0, 0.0, 0.0, 0.0]
            else:
                translation, rotation = pose_fn(timestamp / 1_000_000.0)
            cams = {}
            for camera_index, camera in enumerate(DEFAULT_CAMERA_NAMES):
                h, w = (2, 3) if resolution_fn is None else resolution_fn(camera, frame_index)
                rel_path = Path("scene") / camera / f"{frame_index}.png"
                path = self.sensor_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                marker = np.zeros((h, w, 3), dtype=np.uint8)
                marker[..., 0] = camera_index
                marker[..., 1] = frame_index
                Image.fromarray(marker, mode="RGB").save(path)
                cams[camera] = {"data_path": rel_path.as_posix()}
            frames.append(
                {
                    "token": f"tok-{frame_index}",
                    "timestamp": timestamp,
                    "scene_name": "scene-test",
                    "log_name": "log-test",
                    "ego2global_translation": translation,
                    "ego2global_rotation": rotation,
                    "cams": cams,
                }
            )
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        return frames

    def load(self, **kwargs):
        defaults = {
            "metadata_path": self.metadata_path,
            "sensor_root": self.sensor_root,
            "anchor_index": 3,
        }
        defaults.update(kwargs)
        return load_navsim_sample(**defaults)

    def test_regular_2hz_shapes_order_and_default_camera_warning(self):
        self.make_frames()
        with self.assertWarnsRegex(UserWarning, "not an AR1 front-tele equivalent"):
            result = self.load()
        self.assertEqual(result["image_frames"].shape, (4, 4, 3, 2, 3))
        self.assertEqual(result["image_frames"].dtype, np.uint8)
        self.assertTrue(result["image_frames"].flags.c_contiguous)
        self.assertEqual(result["ego_history_xyz"].shape, (1, 1, 16, 3))
        self.assertEqual(result["ego_history_rot"].shape, (1, 1, 16, 3, 3))
        self.assertEqual(result["image_frame_indices"], [[0, 1, 2, 3]] * 4)
        self.assertEqual(result["history_offsets_s"][0], -1.5)
        self.assertEqual(result["history_offsets_s"][-1], 0.0)
        self.assertTrue(np.allclose(result["ego_history_xyz"][0, 0, -1], 0.0))
        self.assertTrue(np.allclose(result["ego_history_rot"][0, 0, -1], np.eye(3)))
        self.assertAlmostEqual(result["source_camera_rate_hz_estimate"], 2.0)
        self.assertEqual(result["future_used_for_model_input"], False)
        self.assertFalse(result["fourth_camera_equivalent_to_ar1_front_tele"])

    def test_camera_major_time_major_markers(self):
        self.make_frames()
        result = self.load()
        images = result["image_frames"]
        for camera_index in range(4):
            for time_index in range(4):
                self.assertEqual(
                    images[camera_index, time_index, :, 0, 0].tolist(),
                    [camera_index, time_index, 0],
                )

    def test_stationary_pose_history(self):
        self.make_frames()
        result = self.load()
        self.assertTrue(np.allclose(result["ego_history_xyz"], 0.0))
        self.assertTrue(
            np.allclose(result["ego_history_rot"], np.broadcast_to(np.eye(3), (1, 1, 16, 3, 3)))
        )

    def test_straight_motion_is_interpolated_into_anchor_frame(self):
        self.make_frames(
            pose_fn=lambda t: ([2.0 * (t - 1.0), 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        )
        result = self.load()
        x = result["ego_history_xyz"][0, 0, :, 0]
        self.assertTrue(np.allclose(x, np.linspace(-3.0, 0.0, 16), atol=1e-6))

    def test_yaw_90_degrees_rotates_global_motion_to_anchor_x(self):
        half = np.sqrt(0.5)
        self.make_frames(
            pose_fn=lambda t: ([0.0, t - 1.0, 0.0], [half, 0.0, 0.0, half])
        )
        result = self.load()
        xyz = result["ego_history_xyz"][0, 0]
        self.assertTrue(np.allclose(xyz[:, 0], np.linspace(-1.5, 0.0, 16), atol=1e-6))
        self.assertTrue(np.allclose(xyz[:, 1:], 0.0, atol=1e-6))
        self.assertTrue(
            np.allclose(
                result["ego_history_rot"][0, 0],
                np.broadcast_to(np.eye(3), (16, 3, 3)),
                atol=1e-6,
            )
        )

    def test_quaternion_slerp_interpolates_yaw_in_scalar_first_order(self):
        def pose(t):
            yaw = (t - 1.0) / 1.5 * (np.pi / 2.0)
            quat = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            return [t - 1.0, 0.0, 0.0], quat

        self.make_frames(pose_fn=pose)
        result = self.load()
        # At t=1.7s the global yaw is 42 degrees; anchor yaw is 90 degrees.
        relative_yaw = np.deg2rad(-48.0)
        expected = np.asarray(
            [
                [np.cos(relative_yaw), -np.sin(relative_yaw), 0.0],
                [np.sin(relative_yaw), np.cos(relative_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.assertTrue(np.allclose(result["ego_history_rot"][0, 0, 7], expected, atol=1e-5))

    def test_duplicate_timestamps_fail_with_frame_context(self):
        self.make_frames(timestamps=(1_000_000, 1_500_000, 1_500_000, 2_500_000))
        with self.assertRaisesRegex(ValueError, r"frame_index=2.*tok-2.*timestamp.*duplicate"):
            self.load()

    def test_missing_timestamp_fails_with_context(self):
        frames = self.make_frames()
        del frames[1]["timestamp"]
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        with self.assertRaisesRegex(ValueError, r"frame_index=1.*tok-1.*timestamp"):
            self.load()

    def test_bad_translation_shape_fails_with_context(self):
        frames = self.make_frames()
        frames[2]["ego2global_translation"] = [1.0, 2.0]
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        message = r"frame_index=2.*tok-2.*ego2global_translation.*expected shape \(3,\)"
        with self.assertRaisesRegex(ValueError, message):
            self.load()

    def test_zero_quaternion_fails_with_context(self):
        frames = self.make_frames()
        frames[1]["ego2global_rotation"] = [0.0, 0.0, 0.0, 0.0]
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        message = r"frame_index=1.*tok-1.*ego2global_rotation.*norm must be nonzero"
        with self.assertRaisesRegex(ValueError, message):
            self.load()

    def test_insufficient_history_fails(self):
        self.make_frames(timestamps=(1_500_000, 2_000_000, 2_500_000))
        with self.assertRaisesRegex(ValueError, "Insufficient historical coverage"):
            self.load(anchor_index=2)

    def test_irregular_sampling_rejects_image_timing_error(self):
        self.make_frames(timestamps=(1_000_000, 1_300_000, 2_100_000, 2_500_000))
        with self.assertRaisesRegex(ValueError, "image timing error.*camera=CAM_L0"):
            self.load(max_pose_gap_s=1.0)

    def test_large_pose_gap_is_not_silently_interpolated(self):
        self.make_frames(timestamps=(1_000_000, 1_300_000, 2_100_000, 2_500_000))
        message = r"pose gap exceeds limit.*gap=0.800000s.*limit=0.550000s"
        with self.assertRaisesRegex(ValueError, message):
            self.load(max_image_time_error_s=1.0)

    def test_malformed_future_frame_after_anchor_is_not_validated(self):
        frames = self.make_frames()
        frames.append({"timestamp": "broken future record"})
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        result = self.load()
        self.assertTrue(result["future_frames_deserialized"])
        self.assertFalse(result["future_used_for_model_input"])

    def test_missing_camera_has_frame_and_token_context(self):
        frames = self.make_frames()
        del frames[0]["cams"]["CAM_L0"]
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        with self.assertRaisesRegex(ValueError, r"frame_index=0.*tok-0.*cams.CAM_L0"):
            self.load()

    def test_relative_path_traversal_is_rejected(self):
        frames = self.make_frames()
        frames[0]["cams"]["CAM_L0"]["data_path"] = "../outside.png"
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        with self.assertRaisesRegex(ValueError, "path traversal"):
            self.load()

    def test_absolute_path_inside_sensor_root_is_accepted(self):
        frames = self.make_frames()
        image_path = self.sensor_root / "scene" / "CAM_L0" / "0.png"
        frames[0]["cams"]["CAM_L0"]["data_path"] = str(image_path.resolve())
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        result = self.load()
        self.assertEqual(result["image_frames"].shape[0], 4)

    def test_absolute_path_outside_sensor_root_is_rejected(self):
        frames = self.make_frames()
        outside = self.root / "outside.png"
        Image.new("RGB", (2, 3)).save(outside)
        frames[0]["cams"]["CAM_L0"]["data_path"] = str(outside.resolve())
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        with self.assertRaisesRegex(ValueError, "escapes sensor root"):
            self.load()

    def test_missing_image_has_file_not_found_and_context(self):
        frames = self.make_frames()
        frames[0]["cams"]["CAM_L0"]["data_path"] = "scene/CAM_L0/missing.png"
        with self.metadata_path.open("wb") as stream:
            pickle.dump(frames, stream)
        message = r"frame_index=0.*tok-0.*CAM_L0.*does not exist"
        with self.assertRaisesRegex(FileNotFoundError, message):
            self.load()

    def test_resolution_mismatch_reports_camera(self):
        self.make_frames(
            resolution_fn=lambda camera, index: (4, 5)
            if camera == "CAM_F0" and index == 0
            else (2, 3)
        )
        with self.assertRaisesRegex(ValueError, r"CAM_F0.*resolution mismatch"):
            self.load()


if __name__ == "__main__":
    unittest.main()
