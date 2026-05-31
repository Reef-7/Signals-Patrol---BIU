import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dts_simulation.array_geometry import (
    default_simulation_microphone_coordinates,
    validate_array_configurations,
)
from dts_simulation.config import (
    ARRAY_SAMPLE_SHAPE,
    AZIMUTH_MAX_DEGREES_EXCLUSIVE,
    AZIMUTH_MIN_DEGREES,
    DEFAULT_ARRAY_LAYER_HALF_WIDTH_METERS,
    DEFAULT_ARRAY_LAYER_SPACING_METERS,
    DEFAULT_UPPER_LAYER_ROTATION_DEGREES,
    DEFAULT_LOCALIZATION_ARRAY_IDS,
    DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS,
    ELEVATION_MAX_DEGREES,
    ELEVATION_MIN_DEGREES,
    FREQUENCY_DOMAIN_TRANSFORM,
    INITIAL_NARROWBAND_VALIDATION,
    MUSIC_AZIMUTH_SCAN_STEP_DEGREES,
    MUSIC_ELEVATION_SCAN_STEP_DEGREES,
    QUALITY_SCORE_MAX,
    QUALITY_SCORE_MIN,
    SAMPLE_DURATION_SECONDS,
    SAMPLE_RATE_HZ,
    SPEED_OF_SOUND_METERS_PER_SECOND,
    SYSTEM_SAMPLE_SHAPE,
    WORLD_DIMENSIONS,
)
from dts_simulation.launcher import VirtualLauncher
from dts_simulation.localization import direction_estimate_to_unit_vector, localize_source
from dts_simulation.models import (
    ArrayPose,
    ArraySampleMatrix,
    DirectionEstimate,
    LocalizationResult,
    MicrophoneArrayConfig,
)
from dts_simulation.music_doa import MusicDoaEstimator
from dts_simulation.signal_generator import (
    generate_silent_array_samples,
    generate_silent_system_samples,
)


def configured_array(array_id: str) -> MicrophoneArrayConfig:
    return MicrophoneArrayConfig(
        array_id=array_id,
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        microphone_count=8,
        microphone_coordinates=default_simulation_microphone_coordinates(),
    )


class SampleContractTests(unittest.TestCase):
    def test_one_array_sample_matrix_has_confirmed_shape(self) -> None:
        samples = generate_silent_array_samples()
        self.assertEqual(samples.shape, (8, 48000))
        self.assertEqual(samples.shape, ARRAY_SAMPLE_SHAPE)

    def test_system_sample_tensor_has_confirmed_shape(self) -> None:
        samples = generate_silent_system_samples()
        self.assertEqual(samples.shape, (3, 8, 48000))
        self.assertEqual(samples.shape, SYSTEM_SAMPLE_SHAPE)

    def test_invalid_array_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArraySampleMatrix(samples=((0.0,) * 48000,) * 7)


class DoaConventionTests(unittest.TestCase):
    def test_sampling_and_world_conventions_are_declared(self) -> None:
        self.assertEqual(WORLD_DIMENSIONS, 3)
        self.assertEqual(SAMPLE_RATE_HZ, 48000)
        self.assertEqual(SAMPLE_DURATION_SECONDS, 1)
        self.assertEqual(SPEED_OF_SOUND_METERS_PER_SECOND, 343.0)

    def test_geometry_and_scan_conventions_are_declared(self) -> None:
        self.assertEqual(DEFAULT_ARRAY_LAYER_HALF_WIDTH_METERS, 0.045)
        self.assertEqual(DEFAULT_ARRAY_LAYER_SPACING_METERS, 0.045)
        self.assertEqual(DEFAULT_UPPER_LAYER_ROTATION_DEGREES, 45.0)
        self.assertEqual(MUSIC_AZIMUTH_SCAN_STEP_DEGREES, 1.0)
        self.assertEqual(MUSIC_ELEVATION_SCAN_STEP_DEGREES, 1.0)
        self.assertEqual((AZIMUTH_MIN_DEGREES, AZIMUTH_MAX_DEGREES_EXCLUSIVE), (0.0, 360.0))
        self.assertEqual((ELEVATION_MIN_DEGREES, ELEVATION_MAX_DEGREES), (-90.0, 90.0))

    def test_default_simulation_geometry_uses_two_square_layers(self) -> None:
        coordinates = default_simulation_microphone_coordinates()
        self.assertEqual(len(coordinates), 8)
        lower_z_values = {coordinate[2] for coordinate in coordinates[:4]}
        upper_z_values = {coordinate[2] for coordinate in coordinates[4:]}
        self.assertEqual(lower_z_values, {-0.0225})
        self.assertEqual(upper_z_values, {0.0225})
        self.assertEqual(coordinates[0], (0.045, 0.045, -0.0225))
        self.assertEqual(coordinates[4], (0.045 * 2 ** 0.5, 0.0, 0.0225))

    def test_frequency_domain_music_contract_is_declared(self) -> None:
        self.assertEqual(FREQUENCY_DOMAIN_TRANSFORM, "FFT/STFT")
        self.assertTrue(INITIAL_NARROWBAND_VALIDATION)

    def test_quality_score_contract_uses_normalized_range(self) -> None:
        self.assertEqual((QUALITY_SCORE_MIN, QUALITY_SCORE_MAX), (0.0, 1.0))


class BoundaryContractTests(unittest.TestCase):
    def test_three_unique_array_configurations_are_accepted(self) -> None:
        configurations = tuple(configured_array(f"array-{number}") for number in range(3))
        validate_array_configurations(configurations)

    def test_music_contract_rejects_silent_input_as_invalid(self) -> None:
        estimate = MusicDoaEstimator().estimate(
            generate_silent_array_samples(), configured_array("array-0")
        )
        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.quality_score, 0.0)

    def test_localization_requires_at_least_two_valid_matched_rays(self) -> None:
        pose = ArrayPose(array_id="array-0", position_xyz=(0.0, 0.0, 0.0))
        estimate = DirectionEstimate(
            array_id="array-0",
            azimuth_degrees=0.0,
            elevation_degrees=0.0,
            quality_score=None,
            valid=True,
        )
        result = localize_source((pose,), (estimate,))
        self.assertFalse(result.valid)
        self.assertIn("fewer than two", result.status_message)

    def test_localization_result_preserves_later_stage_contract(self) -> None:
        estimate = DirectionEstimate(
            array_id="array-0",
            azimuth_degrees=15.0,
            elevation_degrees=10.0,
            quality_score=0.5,
            valid=True,
        )
        result = LocalizationResult(
            estimated_xyz=(1.0, 2.0, 3.0),
            true_xyz=(1.5, 2.0, 3.0),
            error_distance=0.5,
            residual_error=0.25,
            quality_score=0.75,
            used_array_ids=("array-0",),
            valid=True,
            status_message="contract test",
            contributing_direction_estimates=(estimate,),
        )
        self.assertEqual(result.contributing_direction_estimates, (estimate,))
        self.assertEqual(result.used_array_ids, ("array-0",))
        self.assertEqual(result.residual_error, 0.25)
        self.assertEqual(result.error_distance, 0.5)

    def test_virtual_launcher_only_tracks_a_target(self) -> None:
        launcher = VirtualLauncher()
        launcher.aim_at((1.0, 2.0))
        self.assertEqual(launcher.target_xy, (1.0, 2.0))


class LocalizationContractTests(unittest.TestCase):
    def test_direction_estimate_to_unit_vector_convention(self) -> None:
        cases = (
            (0.0, 0.0, (1.0, 0.0, 0.0)),
            (90.0, 0.0, (0.0, 1.0, 0.0)),
            (0.0, 90.0, (0.0, 0.0, 1.0)),
            (0.0, -90.0, (0.0, 0.0, -1.0)),
        )
        for azimuth, elevation, expected in cases:
            with self.subTest(azimuth=azimuth, elevation=elevation):
                estimate = DirectionEstimate(
                    array_id="array-0",
                    azimuth_degrees=azimuth,
                    elevation_degrees=elevation,
                    valid=True,
                )
                actual = direction_estimate_to_unit_vector(estimate)
                for actual_value, expected_value in zip(actual, expected):
                    self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_array_pose_contract_accepts_optional_orientation(self) -> None:
        pose = ArrayPose(
            array_id="array-0",
            position_xyz=(1.0, 2.0, 3.0),
            yaw_degrees=10.0,
            pitch_degrees=20.0,
            roll_degrees=30.0,
        )
        self.assertEqual(pose.position_xyz, (1.0, 2.0, 3.0))

    def test_default_localization_layout_is_triangular_and_simulation_only(self) -> None:
        self.assertEqual(DEFAULT_LOCALIZATION_ARRAY_IDS, ("array-0", "array-1", "array-2"))
        self.assertEqual(len(DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS), 3)
        self.assertEqual(DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS[0], (0.0, 0.0, 0.0))
        self.assertNotEqual(
            DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS[1],
            DEFAULT_LOCALIZATION_ARRAY_POSITIONS_METERS[2],
        )


if __name__ == "__main__":
    unittest.main()
