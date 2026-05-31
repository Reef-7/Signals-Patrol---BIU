import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dts_simulation.config import (
    DEFAULT_LOCALIZATION_ARRAY_IDS,
    DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS,
)
from dts_simulation.localization import localize_source
from dts_simulation.models import ArrayPose, DirectionEstimate

POSITION_TOLERANCE_METERS = 1.0e-9
NOISY_POSITION_TOLERANCE_METERS = 0.35


def default_poses() -> tuple[ArrayPose, ...]:
    return tuple(
        ArrayPose(array_id=array_id, position_xyz=position)
        for array_id, position in zip(
            DEFAULT_LOCALIZATION_ARRAY_IDS, DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS
        )
    )


def estimate_toward(
    array_id: str,
    origin: tuple[float, float, float],
    target: tuple[float, float, float],
    quality_score: float | None = 1.0,
    valid: bool = True,
    azimuth_offset_degrees: float = 0.0,
    elevation_offset_degrees: float = 0.0,
) -> DirectionEstimate:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    dz = target[2] - origin[2]
    horizontal = math.hypot(dx, dy)
    azimuth = (math.degrees(math.atan2(dy, dx)) + azimuth_offset_degrees) % 360.0
    elevation = math.degrees(math.atan2(dz, horizontal)) + elevation_offset_degrees
    return DirectionEstimate(
        array_id=array_id,
        azimuth_degrees=azimuth,
        elevation_degrees=elevation,
        quality_score=quality_score,
        valid=valid,
    )


def estimates_for_target(
    poses: tuple[ArrayPose, ...],
    target: tuple[float, float, float],
) -> tuple[DirectionEstimate, ...]:
    return tuple(
        estimate_toward(pose.array_id, pose.position_xyz, target) for pose in poses
    )


class RayLocalizationTests(unittest.TestCase):
    def test_perfect_three_ray_intersection_returns_known_source(self) -> None:
        poses = default_poses()
        target = (2.0, 1.2, 2.5)
        result = localize_source(poses, estimates_for_target(poses, target), target)

        self.assertTrue(result.valid)
        self.assertEqual(result.used_array_ids, DEFAULT_LOCALIZATION_ARRAY_IDS)
        for actual, expected in zip(result.estimated_xyz, target):
            self.assertAlmostEqual(actual, expected, delta=POSITION_TOLERANCE_METERS)
        self.assertAlmostEqual(result.residual_error, 0.0, delta=POSITION_TOLERANCE_METERS)
        self.assertAlmostEqual(result.error_distance, 0.0, delta=POSITION_TOLERANCE_METERS)

    def test_slightly_noisy_directions_return_near_source_with_residual(self) -> None:
        poses = default_poses()
        target = (2.0, 1.2, 2.5)
        estimates = (
            estimate_toward("array-0", poses[0].position_xyz, target, azimuth_offset_degrees=1.0),
            estimate_toward("array-1", poses[1].position_xyz, target, elevation_offset_degrees=-1.0),
            estimate_toward("array-2", poses[2].position_xyz, target, azimuth_offset_degrees=-1.0),
        )
        result = localize_source(poses, estimates, target)

        self.assertTrue(result.valid)
        self.assertGreater(result.residual_error, 0.0)
        self.assertLess(result.error_distance, NOISY_POSITION_TOLERANCE_METERS)

    def test_invalid_direction_is_ignored_when_two_valid_rays_remain(self) -> None:
        poses = default_poses()
        target = (2.0, 1.2, 2.5)
        estimates = list(estimates_for_target(poses, target))
        estimates[2] = DirectionEstimate(
            array_id="array-2",
            azimuth_degrees=0.0,
            elevation_degrees=0.0,
            quality_score=1.0,
            valid=False,
        )
        result = localize_source(poses, tuple(estimates), target)

        self.assertTrue(result.valid)
        self.assertEqual(result.used_array_ids, ("array-0", "array-1"))
        self.assertAlmostEqual(result.error_distance, 0.0, delta=POSITION_TOLERANCE_METERS)

    def test_fewer_than_two_valid_rays_returns_invalid(self) -> None:
        poses = default_poses()
        target = (2.0, 1.2, 2.5)
        estimate = estimate_toward("array-0", poses[0].position_xyz, target)
        result = localize_source(poses, (estimate,), target)

        self.assertFalse(result.valid)
        self.assertIn("fewer than two", result.status_message)
        self.assertEqual(result.quality_score, 0.0)

    def test_nearly_parallel_rays_return_invalid(self) -> None:
        poses = (
            ArrayPose(array_id="array-0", position_xyz=(0.0, 0.0, 0.0)),
            ArrayPose(array_id="array-1", position_xyz=(0.0, 1.0, 0.0)),
            ArrayPose(array_id="array-2", position_xyz=(0.0, 2.0, 0.0)),
        )
        estimates = tuple(
            DirectionEstimate(
                array_id=pose.array_id,
                azimuth_degrees=0.0,
                elevation_degrees=0.0,
                quality_score=1.0,
                valid=True,
            )
            for pose in poses
        )
        result = localize_source(poses, estimates)

        self.assertFalse(result.valid)
        self.assertIn("parallel", result.status_message)

    def test_quality_score_weights_influence_noisy_result(self) -> None:
        poses = default_poses()
        target = (2.0, 1.2, 2.5)
        biased = (
            estimate_toward("array-0", poses[0].position_xyz, target, quality_score=1.0),
            estimate_toward("array-1", poses[1].position_xyz, target, quality_score=1.0),
            estimate_toward(
                "array-2",
                poses[2].position_xyz,
                target,
                quality_score=0.05,
                azimuth_offset_degrees=10.0,
            ),
        )
        unweighted = tuple(
            DirectionEstimate(
                array_id=estimate.array_id,
                azimuth_degrees=estimate.azimuth_degrees,
                elevation_degrees=estimate.elevation_degrees,
                quality_score=1.0,
                valid=estimate.valid,
            )
            for estimate in biased
        )

        weighted_result = localize_source(poses, biased, target)
        unweighted_result = localize_source(poses, unweighted, target)

        self.assertTrue(weighted_result.valid)
        self.assertTrue(unweighted_result.valid)
        self.assertLess(weighted_result.error_distance, unweighted_result.error_distance)


if __name__ == "__main__":
    unittest.main()
