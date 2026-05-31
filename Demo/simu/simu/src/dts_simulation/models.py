"""Validated domain contracts; algorithm behavior is intentionally excluded."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .config import ARRAY_COUNT, ARRAY_SAMPLE_SHAPE, MICROPHONES_PER_ARRAY, SYSTEM_SAMPLE_SHAPE

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
Vector3D = tuple[float, float, float]
SampleRow = tuple[float, ...]


def _require_finite(values: tuple[float, ...], field_name: str) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError(f"{field_name} must contain finite numeric values")


@dataclass(frozen=True, slots=True)
class ArraySampleMatrix:
    """Raw samples for one array, represented as microphone x sample."""

    samples: tuple[SampleRow, ...]

    def __post_init__(self) -> None:
        expected_microphones, expected_samples = ARRAY_SAMPLE_SHAPE
        if len(self.samples) != expected_microphones:
            raise ValueError(
                f"array samples must have {expected_microphones} microphone rows"
            )
        if any(len(row) != expected_samples for row in self.samples):
            raise ValueError(f"each microphone row must contain {expected_samples} samples")

    @property
    def shape(self) -> tuple[int, int]:
        return ARRAY_SAMPLE_SHAPE


@dataclass(frozen=True, slots=True)
class SystemSampleTensor:
    """Raw input for one activation window, represented as array x microphone x sample."""

    arrays: tuple[ArraySampleMatrix, ...]

    def __post_init__(self) -> None:
        if len(self.arrays) != ARRAY_COUNT:
            raise ValueError(f"system samples must contain {ARRAY_COUNT} arrays")

    @property
    def shape(self) -> tuple[int, int, int]:
        return SYSTEM_SAMPLE_SHAPE


@dataclass(frozen=True, slots=True)
class MicrophoneArrayConfig:
    """World placement and 3D microphone geometry supplied for DOA estimation."""

    array_id: str
    position: Vector3D
    orientation: Vector3D
    microphone_count: int
    microphone_coordinates: tuple[Vector3D, ...]

    def __post_init__(self) -> None:
        if not self.array_id:
            raise ValueError("array_id must not be empty")
        if self.microphone_count != MICROPHONES_PER_ARRAY:
            raise ValueError(f"microphone_count must be {MICROPHONES_PER_ARRAY}")
        if len(self.microphone_coordinates) != self.microphone_count:
            raise ValueError("microphone_coordinates must match microphone_count")
        _require_finite(self.position, "position")
        _require_finite(self.orientation, "orientation")
        for coordinates in self.microphone_coordinates:
            _require_finite(coordinates, "microphone_coordinates")


@dataclass(frozen=True, slots=True)
class DirectionEstimate:
    """One array's 3D DOA output: azimuth from +x and elevation from x-y.

    A future estimator must return normalized azimuth in [0, 360), elevation in
    [-90, 90], and treat quality_score as a simulation-only peak-dominance or
    equivalent deterministic metric in [0, 1].
    """

    array_id: str
    azimuth_degrees: float
    elevation_degrees: float
    quality_score: float | None = None
    valid: bool = False

    def __post_init__(self) -> None:
        if not self.array_id:
            raise ValueError("array_id must not be empty")
        _require_finite((self.azimuth_degrees,), "azimuth_degrees")
        _require_finite((self.elevation_degrees,), "elevation_degrees")
        if self.quality_score is not None:
            _require_finite((self.quality_score,), "quality_score")


@dataclass(frozen=True, slots=True)
class ArrayPose:
    """World-space pose for one array's localization ray origin.

    DirectionEstimate values are treated as world-relative until orientation
    rotation support is explicitly implemented.
    """

    array_id: str
    position_xyz: Point3D
    yaw_degrees: float | None = None
    pitch_degrees: float | None = None
    roll_degrees: float | None = None

    def __post_init__(self) -> None:
        if not self.array_id:
            raise ValueError("array_id must not be empty")
        _require_finite(self.position_xyz, "position_xyz")
        optional_angles = (
            ("yaw_degrees", self.yaw_degrees),
            ("pitch_degrees", self.pitch_degrees),
            ("roll_degrees", self.roll_degrees),
        )
        for field_name, value in optional_angles:
            if value is not None:
                _require_finite((value,), field_name)


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Later-stage 3D localization output contract."""

    estimated_xyz: Point3D
    contributing_direction_estimates: tuple[DirectionEstimate, ...]
    used_array_ids: tuple[str, ...]
    valid: bool = False
    status_message: str = ""
    residual_error: float | None = None
    quality_score: float | None = None
    true_xyz: Point3D | None = None
    error_distance: float | None = None

    def __post_init__(self) -> None:
        _require_finite(self.estimated_xyz, "estimated_xyz")
        if self.true_xyz is not None:
            _require_finite(self.true_xyz, "true_xyz")
        if self.residual_error is not None:
            _require_finite((self.residual_error,), "residual_error")
            if self.residual_error < 0:
                raise ValueError("residual_error must not be negative")
        if self.quality_score is not None:
            _require_finite((self.quality_score,), "quality_score")
        if self.error_distance is not None:
            _require_finite((self.error_distance,), "error_distance")
            if self.error_distance < 0:
                raise ValueError("error_distance must not be negative")
