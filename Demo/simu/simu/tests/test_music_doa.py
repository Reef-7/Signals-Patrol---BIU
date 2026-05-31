import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dts_simulation.array_geometry import default_simulation_microphone_coordinates
import numpy as np

from dts_simulation.config import (
    DEFAULT_TEST_TONE_FREQUENCY_HZ,
    MUSIC_AZIMUTH_SCAN_STEP_DEGREES,
    MUSIC_ELEVATION_SCAN_STEP_DEGREES,
    MUSIC_VALID_QUALITY_THRESHOLD,
    SAMPLE_RATE_HZ,
)
from dts_simulation.models import MicrophoneArrayConfig
from dts_simulation.music_doa import MusicDoaEstimator, MusicScanConfig
from dts_simulation.signal_generator import generate_narrowband_array_samples

ANGULAR_TOLERANCE_DEGREES = 3.0


def angular_difference_degrees(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def configured_array() -> MicrophoneArrayConfig:
    return MicrophoneArrayConfig(
        array_id="array-0",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        microphone_count=8,
        microphone_coordinates=default_simulation_microphone_coordinates(),
    )


class Narrowband3DMusicTests(unittest.TestCase):
    def test_default_scan_is_one_degree_and_tests_use_coarser_grid(self) -> None:
        default_scan = MusicScanConfig()
        test_scan = MusicScanConfig(azimuth_step_degrees=2.0, elevation_step_degrees=2.0)

        self.assertEqual(default_scan.azimuth_step_degrees, MUSIC_AZIMUTH_SCAN_STEP_DEGREES)
        self.assertEqual(default_scan.elevation_step_degrees, MUSIC_ELEVATION_SCAN_STEP_DEGREES)
        self.assertEqual((test_scan.azimuth_step_degrees, test_scan.elevation_step_degrees), (2.0, 2.0))

    def test_scan_steps_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            MusicScanConfig(azimuth_step_degrees=0.0)
        with self.assertRaises(ValueError):
            MusicScanConfig(elevation_step_degrees=-1.0)

    def test_expected_frequency_selects_nearest_fft_bin(self) -> None:
        estimator = MusicDoaEstimator(frequency_hz=DEFAULT_TEST_TONE_FREQUENCY_HZ)
        frequency_bins = np.fft.rfftfreq(48000, d=1.0 / SAMPLE_RATE_HZ)
        spectrum = np.zeros((8, frequency_bins.shape[0]), dtype=complex)
        selected = estimator._select_frequency_bin(spectrum, frequency_bins)

        self.assertEqual(frequency_bins[selected], DEFAULT_TEST_TONE_FREQUENCY_HZ)

    def test_dominant_frequency_selection_ignores_dc(self) -> None:
        estimator = MusicDoaEstimator()
        frequency_bins = np.array([0.0, 10.0, 20.0])
        spectrum = np.array(
            [
                [1000.0 + 0j, 1.0 + 0j, 3.0 + 0j],
                [1000.0 + 0j, 2.0 + 0j, 4.0 + 0j],
            ]
        )
        selected = estimator._select_frequency_bin(spectrum, frequency_bins)

        self.assertEqual(selected, 2)

    def test_weak_input_is_invalid(self) -> None:
        configuration = configured_array()
        samples = generate_narrowband_array_samples(
            configuration,
            azimuth_degrees=90.0,
            elevation_degrees=20.0,
            amplitude=0.0,
        )
        estimate = MusicDoaEstimator(frequency_hz=DEFAULT_TEST_TONE_FREQUENCY_HZ).estimate(
            samples, configuration
        )

        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.quality_score, 0.0)

    def test_known_synthetic_3d_directions(self) -> None:
        configuration = configured_array()
        estimator = MusicDoaEstimator(
            frequency_hz=DEFAULT_TEST_TONE_FREQUENCY_HZ,
            scan_config=MusicScanConfig(
                azimuth_step_degrees=2.0, elevation_step_degrees=2.0
            ),
        )
        cases = (
            (0.0, 0.0),
            (90.0, 20.0),
            (180.0, -20.0),
            (270.0, 30.0),
        )

        for expected_azimuth, expected_elevation in cases:
            with self.subTest(azimuth=expected_azimuth, elevation=expected_elevation):
                samples = generate_narrowband_array_samples(
                    configuration,
                    azimuth_degrees=expected_azimuth,
                    elevation_degrees=expected_elevation,
                )
                estimate = estimator.estimate(samples, configuration)
                self.assertTrue(estimate.valid)
                self.assertLessEqual(
                    angular_difference_degrees(
                        estimate.azimuth_degrees, expected_azimuth
                    ),
                    ANGULAR_TOLERANCE_DEGREES,
                )
                self.assertLessEqual(
                    abs(estimate.elevation_degrees - expected_elevation),
                    ANGULAR_TOLERANCE_DEGREES,
                )
                self.assertIsNotNone(estimate.quality_score)
                self.assertGreaterEqual(estimate.quality_score, 0.0)
                self.assertLessEqual(estimate.quality_score, 1.0)
                self.assertGreaterEqual(
                    estimate.quality_score, MUSIC_VALID_QUALITY_THRESHOLD
                )

    def test_wraparound_and_high_elevation_edge_cases(self) -> None:
        configuration = configured_array()
        estimator = MusicDoaEstimator(
            frequency_hz=DEFAULT_TEST_TONE_FREQUENCY_HZ,
            scan_config=MusicScanConfig(
                azimuth_step_degrees=2.0, elevation_step_degrees=2.0
            ),
        )
        cases = (
            (358.0, 10.0),
            (45.0, 60.0),
            (315.0, -40.0),
        )

        for expected_azimuth, expected_elevation in cases:
            with self.subTest(azimuth=expected_azimuth, elevation=expected_elevation):
                samples = generate_narrowband_array_samples(
                    configuration,
                    azimuth_degrees=expected_azimuth,
                    elevation_degrees=expected_elevation,
                )
                estimate = estimator.estimate(samples, configuration)
                self.assertTrue(estimate.valid)
                self.assertLessEqual(
                    angular_difference_degrees(
                        estimate.azimuth_degrees, expected_azimuth
                    ),
                    ANGULAR_TOLERANCE_DEGREES,
                )
                self.assertLessEqual(
                    abs(estimate.elevation_degrees - expected_elevation),
                    ANGULAR_TOLERANCE_DEGREES,
                )
                self.assertGreaterEqual(estimate.quality_score, 0.0)
                self.assertLessEqual(estimate.quality_score, 1.0)

    def test_dominant_bin_mode_estimates_bin_aligned_tone(self) -> None:
        configuration = configured_array()
        samples = generate_narrowband_array_samples(
            configuration,
            azimuth_degrees=120.0,
            elevation_degrees=30.0,
        )
        estimator = MusicDoaEstimator(
            scan_config=MusicScanConfig(
                azimuth_step_degrees=2.0, elevation_step_degrees=2.0
            ),
        )
        estimate = estimator.estimate(samples, configuration)

        self.assertTrue(estimate.valid)
        self.assertLessEqual(
            angular_difference_degrees(estimate.azimuth_degrees, 120.0),
            ANGULAR_TOLERANCE_DEGREES,
        )
        self.assertLessEqual(abs(estimate.elevation_degrees - 30.0), ANGULAR_TOLERANCE_DEGREES)
        self.assertGreaterEqual(estimate.quality_score, MUSIC_VALID_QUALITY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
