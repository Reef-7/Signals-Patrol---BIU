import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dts_simulation.array_geometry import default_simulation_microphone_coordinates
from dts_simulation.config import (
    DEFAULT_LOCALIZATION_ARRAY_IDS,
    DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS,
    DEFAULT_TEST_TONE_FREQUENCY_HZ,
)
from dts_simulation.localization import localize_source
from dts_simulation.models import ArrayPose, ArraySampleMatrix, MicrophoneArrayConfig
from dts_simulation.music_doa import MusicDoaEstimator, MusicScanConfig
from dts_simulation.signal_generator import generate_narrowband_array_samples

PIPELINE_SOURCE_XYZ = (2.0, 1.2, 2.5)
PIPELINE_POSITION_TOLERANCE_METERS = 0.15
PIPELINE_NOISY_POSITION_TOLERANCE_METERS = 0.2


def default_array_poses() -> tuple[ArrayPose, ...]:
    return tuple(
        ArrayPose(array_id=array_id, position_xyz=position)
        for array_id, position in zip(
            DEFAULT_LOCALIZATION_ARRAY_IDS, DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS
        )
    )


def configured_array(pose: ArrayPose) -> MicrophoneArrayConfig:
    return MicrophoneArrayConfig(
        array_id=pose.array_id,
        position=pose.position_xyz,
        orientation=(0.0, 0.0, 0.0),
        microphone_count=8,
        microphone_coordinates=default_simulation_microphone_coordinates(),
    )


def direction_from_pose_to_source(
    pose: ArrayPose, source_xyz: tuple[float, float, float]
) -> tuple[float, float]:
    dx = source_xyz[0] - pose.position_xyz[0]
    dy = source_xyz[1] - pose.position_xyz[1]
    dz = source_xyz[2] - pose.position_xyz[2]
    azimuth_degrees = math.degrees(math.atan2(dy, dx)) % 360.0
    elevation_degrees = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return azimuth_degrees, elevation_degrees


def add_deterministic_noise(
    samples: ArraySampleMatrix, standard_deviation: float
) -> ArraySampleMatrix:
    rng = np.random.default_rng(20260527)
    sample_array = np.asarray(samples.samples, dtype=float)
    noisy = sample_array + rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=sample_array.shape,
    )
    return ArraySampleMatrix(
        samples=tuple(tuple(float(value) for value in row) for row in noisy)
    )


def estimate_directions_for_source(
    poses: tuple[ArrayPose, ...],
    source_xyz: tuple[float, float, float],
    noise_standard_deviation: float = 0.0,
):
    estimator = MusicDoaEstimator(
        frequency_hz=DEFAULT_TEST_TONE_FREQUENCY_HZ,
        scan_config=MusicScanConfig(
            azimuth_step_degrees=2.0,
            elevation_step_degrees=2.0,
        ),
    )
    estimates = []
    for pose in poses:
        configuration = configured_array(pose)
        azimuth_degrees, elevation_degrees = direction_from_pose_to_source(
            pose, source_xyz
        )
        samples = generate_narrowband_array_samples(
            configuration,
            azimuth_degrees=azimuth_degrees,
            elevation_degrees=elevation_degrees,
        )
        if noise_standard_deviation > 0.0:
            samples = add_deterministic_noise(samples, noise_standard_deviation)
        estimates.append(estimator.estimate(samples, configuration))
    return tuple(estimates)


class AlgorithmicPipelineTests(unittest.TestCase):
    def test_music_estimates_feed_localization_for_three_synthetic_arrays(self) -> None:
        poses = default_array_poses()
        estimates = estimate_directions_for_source(poses, PIPELINE_SOURCE_XYZ)
        result = localize_source(poses, estimates, PIPELINE_SOURCE_XYZ)

        self.assertTrue(all(estimate.valid for estimate in estimates))
        self.assertTrue(result.valid)
        self.assertEqual(result.used_array_ids, DEFAULT_LOCALIZATION_ARRAY_IDS)
        self.assertLessEqual(
            result.error_distance,
            PIPELINE_POSITION_TOLERANCE_METERS,
            "Pipeline tolerance reflects the 2-degree MUSIC test scan grid.",
        )
        self.assertGreater(result.quality_score, 0.0)
        self.assertLessEqual(result.quality_score, 1.0)

    def test_pipeline_tolerates_mild_deterministic_narrowband_noise(self) -> None:
        poses = default_array_poses()
        estimates = estimate_directions_for_source(
            poses,
            PIPELINE_SOURCE_XYZ,
            noise_standard_deviation=0.01,
        )
        result = localize_source(poses, estimates, PIPELINE_SOURCE_XYZ)

        self.assertTrue(all(estimate.valid for estimate in estimates))
        self.assertTrue(result.valid)
        self.assertLessEqual(
            result.error_distance,
            PIPELINE_NOISY_POSITION_TOLERANCE_METERS,
        )
        self.assertGreater(result.residual_error, 0.0)


if __name__ == "__main__":
    unittest.main()
