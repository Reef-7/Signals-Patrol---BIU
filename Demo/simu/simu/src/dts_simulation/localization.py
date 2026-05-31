"""Bounded multi-array 3D ray localization.

The solver estimates the point minimizing weighted squared perpendicular
distance to valid DOA rays. It is simulation/math localization only and does
not perform hardware calibration, graphics, launcher behavior, or real-world
validation.
"""

from math import cos, radians, sin

import numpy as np

from .config import (
    LOCALIZATION_CONDITION_LIMIT,
    LOCALIZATION_MIN_USABLE_RAYS,
    LOCALIZATION_PARALLEL_DOT_THRESHOLD,
)
from .models import ArrayPose, DirectionEstimate, LocalizationResult, Vector3D


def direction_estimate_to_unit_vector(estimate: DirectionEstimate) -> Vector3D:
    """Convert contract azimuth/elevation degrees to a 3D unit vector."""
    azimuth = radians(estimate.azimuth_degrees)
    elevation = radians(estimate.elevation_degrees)
    return (
        cos(elevation) * cos(azimuth),
        cos(elevation) * sin(azimuth),
        sin(elevation),
    )


def localize_source(
    array_poses: tuple[ArrayPose, ...],
    direction_estimates: tuple[DirectionEstimate, ...],
    true_position: Vector3D | None = None,
) -> LocalizationResult:
    pose_by_id = {pose.array_id: pose for pose in array_poses}
    usable_rays = tuple(
        _ray_from_estimate(estimate, pose_by_id)
        for estimate in direction_estimates
        if estimate.valid and estimate.array_id in pose_by_id
    )
    usable_rays = tuple(ray for ray in usable_rays if ray is not None)

    if len(usable_rays) < LOCALIZATION_MIN_USABLE_RAYS:
        return _invalid_result(
            "fewer than two valid matched rays",
            direction_estimates,
            true_position,
        )
    if _rays_are_nearly_parallel(tuple(ray[1] for ray in usable_rays)):
        return _invalid_result(
            "rays are nearly parallel or ill-conditioned",
            direction_estimates,
            true_position,
        )

    matrix = np.zeros((3, 3), dtype=float)
    vector = np.zeros(3, dtype=float)
    identity = np.eye(3, dtype=float)
    for origin, direction, weight, _estimate in usable_rays:
        projection = identity - np.outer(direction, direction)
        matrix += weight * projection
        vector += weight * projection @ origin

    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number) or condition_number > LOCALIZATION_CONDITION_LIMIT:
        return _invalid_result(
            "ray system is ill-conditioned",
            direction_estimates,
            true_position,
        )

    estimated = np.linalg.solve(matrix, vector)
    residual_error = _mean_weighted_perpendicular_distance(estimated, usable_rays)
    quality_score = float(np.clip(1.0 / (1.0 + residual_error), 0.0, 1.0))
    true_xyz = tuple(float(value) for value in true_position) if true_position is not None else None
    error_distance = None
    if true_position is not None:
        error_distance = float(np.linalg.norm(estimated - np.asarray(true_position, dtype=float)))

    used_estimates = tuple(ray[3] for ray in usable_rays)
    used_array_ids = tuple(estimate.array_id for estimate in used_estimates)
    return LocalizationResult(
        estimated_xyz=tuple(float(value) for value in estimated),
        contributing_direction_estimates=used_estimates,
        used_array_ids=used_array_ids,
        valid=True,
        status_message="ok",
        residual_error=residual_error,
        quality_score=quality_score,
        true_xyz=true_xyz,
        error_distance=error_distance,
    )


def _ray_from_estimate(
    estimate: DirectionEstimate, pose_by_id: dict[str, ArrayPose]
) -> tuple[np.ndarray, np.ndarray, float, DirectionEstimate] | None:
    pose = pose_by_id.get(estimate.array_id)
    if pose is None:
        return None
    origin = np.asarray(pose.position_xyz, dtype=float)
    direction = np.asarray(direction_estimate_to_unit_vector(estimate), dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return None
    direction = direction / norm
    weight = estimate.quality_score if estimate.quality_score is not None and estimate.quality_score > 0 else 1.0
    return origin, direction, float(weight), estimate


def _rays_are_nearly_parallel(directions: tuple[np.ndarray, ...]) -> bool:
    for first_index, first in enumerate(directions):
        for second in directions[first_index + 1 :]:
            if abs(float(np.dot(first, second))) < LOCALIZATION_PARALLEL_DOT_THRESHOLD:
                return False
    return True


def _mean_weighted_perpendicular_distance(
    estimated: np.ndarray,
    rays: tuple[tuple[np.ndarray, np.ndarray, float, DirectionEstimate], ...],
) -> float:
    weighted_distance_sum = 0.0
    weight_sum = 0.0
    for origin, direction, weight, _estimate in rays:
        offset = estimated - origin
        perpendicular = offset - np.dot(offset, direction) * direction
        weighted_distance_sum += weight * float(np.linalg.norm(perpendicular))
        weight_sum += weight
    if weight_sum <= 0.0:
        return 0.0
    return weighted_distance_sum / weight_sum


def _invalid_result(
    status_message: str,
    direction_estimates: tuple[DirectionEstimate, ...],
    true_position: Vector3D | None,
) -> LocalizationResult:
    true_xyz = tuple(float(value) for value in true_position) if true_position is not None else None
    return LocalizationResult(
        estimated_xyz=(0.0, 0.0, 0.0),
        contributing_direction_estimates=tuple(
            estimate for estimate in direction_estimates if estimate.valid
        ),
        used_array_ids=(),
        valid=False,
        status_message=status_message,
        residual_error=None,
        quality_score=0.0,
        true_xyz=true_xyz,
        error_distance=None,
    )
