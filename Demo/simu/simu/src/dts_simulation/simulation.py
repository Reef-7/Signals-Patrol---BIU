"""Single-shot non-graphical DTS scenario runner."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot

import numpy as np

from .array_geometry import default_simulation_microphone_coordinates
from .config import (
    ARRAY_SAMPLE_SHAPE,
    DEFAULT_LOCALIZATION_ARRAY_IDS,
    DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS,
    DEFAULT_TEST_TONE_FREQUENCY_HZ,
    MICROPHONES_PER_ARRAY,
    SYSTEM_SAMPLE_SHAPE,
)
from .launcher import VirtualLauncher
from .localization import localize_source
from .models import (
    ArrayPose,
    ArraySampleMatrix,
    DirectionEstimate,
    LocalizationResult,
    MicrophoneArrayConfig,
    Point3D,
)
from .music_doa import MusicDoaEstimator, MusicScanConfig
from .signal_generator import generate_narrowband_array_samples

DEFAULT_SCENARIO_SOURCE_XYZ: Point3D = (2.0, 1.2, 2.5)
DEFAULT_SCENARIO_SCAN_STEP_DEGREES = 2.0
DEFAULT_SCENARIO_NOISE_SEED = 20260527


@dataclass(frozen=True, slots=True)
class ScenarioInput:
    """Simulation-only input for one bounded synthetic DTS scenario."""

    source_xyz: Point3D = DEFAULT_SCENARIO_SOURCE_XYZ
    array_poses: tuple[ArrayPose, ...] = ()
    tone_frequency_hz: float = DEFAULT_TEST_TONE_FREQUENCY_HZ
    noise_standard_deviation: float = 0.0
    scan_step_degrees: float = DEFAULT_SCENARIO_SCAN_STEP_DEGREES

    def resolved_array_poses(self) -> tuple[ArrayPose, ...]:
        if self.array_poses:
            return self.array_poses
        return default_array_poses()


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Simulation-only output for one full algorithmic scenario run."""

    true_source_xyz: Point3D
    direction_estimates: tuple[DirectionEstimate, ...]
    localization_result: LocalizationResult
    virtual_aim_target_xyz: Point3D | None
    valid: bool
    status_message: str

    @property
    def error_distance(self) -> float | None:
        return self.localization_result.error_distance


def default_array_poses() -> tuple[ArrayPose, ...]:
    """Return the default simulation-only three-array layout."""
    return tuple(
        ArrayPose(array_id=array_id, position_xyz=position)
        for array_id, position in zip(
            DEFAULT_LOCALIZATION_ARRAY_IDS, DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS
        )
    )


def run_synthetic_scenario(
    scenario_input: ScenarioInput | None = None,
) -> ScenarioResult:
    """Run one synthetic narrowband scenario through DOA and localization.

    This is a non-graphical, simulation-only data-flow runner. It does not
    model broadband drone audio, animate motion, fire a launcher, or control
    hardware.
    """
    scenario = scenario_input or ScenarioInput()
    array_poses = scenario.resolved_array_poses()
    estimator = MusicDoaEstimator(
        frequency_hz=scenario.tone_frequency_hz,
        scan_config=MusicScanConfig(
            azimuth_step_degrees=scenario.scan_step_degrees,
            elevation_step_degrees=scenario.scan_step_degrees,
        ),
    )
    rng = np.random.default_rng(DEFAULT_SCENARIO_NOISE_SEED)
    direction_estimates = []
    for pose in array_poses:
        configuration = _configuration_for_pose(pose)
        azimuth_degrees, elevation_degrees = _direction_from_pose_to_source(
            pose, scenario.source_xyz
        )
        samples = generate_narrowband_array_samples(
            configuration,
            azimuth_degrees=azimuth_degrees,
            elevation_degrees=elevation_degrees,
            frequency_hz=scenario.tone_frequency_hz,
        )
        if scenario.noise_standard_deviation > 0.0:
            samples = _add_deterministic_noise(
                samples, scenario.noise_standard_deviation, rng
            )
        direction_estimates.append(estimator.estimate(samples, configuration))

    localization_result = localize_source(
        array_poses, tuple(direction_estimates), scenario.source_xyz
    )
    virtual_launcher = VirtualLauncher()
    virtual_aim_target_xyz = None
    if localization_result.valid:
        virtual_aim_target_xyz = localization_result.estimated_xyz
        virtual_launcher.aim_at_xyz(virtual_aim_target_xyz)

    status_message = "ok" if localization_result.valid else localization_result.status_message
    return ScenarioResult(
        true_source_xyz=scenario.source_xyz,
        direction_estimates=tuple(direction_estimates),
        localization_result=localization_result,
        virtual_aim_target_xyz=virtual_launcher.target_xyz,
        valid=localization_result.valid,
        status_message=status_message,
    )


def _configuration_for_pose(pose: ArrayPose) -> MicrophoneArrayConfig:
    return MicrophoneArrayConfig(
        array_id=pose.array_id,
        position=pose.position_xyz,
        orientation=(0.0, 0.0, 0.0),
        microphone_count=MICROPHONES_PER_ARRAY,
        microphone_coordinates=default_simulation_microphone_coordinates(),
    )


def _direction_from_pose_to_source(
    pose: ArrayPose, source_xyz: Point3D
) -> tuple[float, float]:
    dx = source_xyz[0] - pose.position_xyz[0]
    dy = source_xyz[1] - pose.position_xyz[1]
    dz = source_xyz[2] - pose.position_xyz[2]
    azimuth_degrees = degrees(atan2(dy, dx)) % 360.0
    elevation_degrees = degrees(atan2(dz, hypot(dx, dy)))
    return azimuth_degrees, elevation_degrees


def _add_deterministic_noise(
    samples: ArraySampleMatrix,
    standard_deviation: float,
    rng: np.random.Generator,
) -> ArraySampleMatrix:
    sample_array = np.asarray(samples.samples, dtype=float)
    noisy_samples = sample_array + rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=sample_array.shape,
    )
    return ArraySampleMatrix(
        samples=tuple(tuple(float(value) for value in row) for row in noisy_samples)
    )


def _format_xyz(point: Point3D) -> str:
    return f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"


def _format_optional_xyz(point: Point3D | None) -> str:
    if point is None:
        return "None"
    return _format_xyz(point)


def main() -> int:
    result = run_synthetic_scenario()
    localization = result.localization_result

    print("DTS-Simulation source project ready")
    print(f"Array sample matrix shape: {ARRAY_SAMPLE_SHAPE}")
    print(f"System raw sample tensor shape: {SYSTEM_SAMPLE_SHAPE}")
    print(
        "MUSIC DOA: one-array 3D narrowband frequency-domain estimator implemented "
        f"for synthetic validation tones such as {DEFAULT_TEST_TONE_FREQUENCY_HZ:g} Hz"
    )
    print("3D localization: bounded ray least-squares solver implemented for known array poses")
    print(f"Scenario arrays: {len(result.direction_estimates)}")
    print(f"True source xyz: {_format_xyz(result.true_source_xyz)}")
    print(f"Estimated source xyz: {_format_xyz(localization.estimated_xyz)}")
    print(f"Error distance: {result.error_distance:.3f} m")
    print(f"Virtual aim target xyz: {_format_optional_xyz(result.virtual_aim_target_xyz)}")
    for estimate in result.direction_estimates:
        print(
            f"{estimate.array_id}: azimuth={estimate.azimuth_degrees:.1f} deg, "
            f"elevation={estimate.elevation_degrees:.1f} deg, "
            f"valid={estimate.valid}"
        )
    print("Broadband drone handling and multi-bin fusion: deferred")
    print("Graphical simulation and launcher/Nerf behavior: deferred")
    print("Hardware control: not implemented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
