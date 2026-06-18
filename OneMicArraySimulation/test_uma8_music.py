import contextlib
import io
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

import uma8_music


class UMA8MusicTests(unittest.TestCase):
    def test_import_does_not_start_recording(self):
        self.assertTrue(hasattr(uma8_music, "UMA8MUSICEstimator"))

    def test_estimator_uses_seven_microphones(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        self.assertEqual(estimator.num_mics, 7)
        self.assertEqual(estimator.mic_coords.shape, (7, 2))

    def test_mock_audio_source_produces_predictable_window_shape(self):
        window = uma8_music.make_self_check_signal()
        source = uma8_music.MockAudioSource([window])
        self.assertEqual(source.read_window().shape[0], 7)
        self.assertGreaterEqual(source.read_window if False else window.shape[1], 512)

    def test_invalid_audio_window_dimension_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "2D matrix"):
            uma8_music.validate_audio_window(np.zeros((7, 512, 1)))

    def test_invalid_audio_window_microphone_count_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "7 microphone rows"):
            uma8_music.validate_audio_window(np.zeros((8, 512)))

    def test_live_source_constructs_without_opening_hardware(self):
        source = uma8_music.LiveSoundDeviceAudioSource(window_duration=0.1)
        self.assertEqual(source.channels, 7)
        self.assertEqual(source.fs, 48000)

    def test_streaming_source_constructs_without_opening_hardware(self):
        source = uma8_music.StreamingSoundDeviceAudioSource(window_duration=0.05)
        self.assertEqual(source.channels, 7)
        self.assertEqual(source.fs, 48000)
        source.close()

    def test_audio_ring_buffer_returns_latest_window(self):
        buffer = uma8_music.AudioRingBuffer(channels=7, capacity_samples=520)
        buffer.append(np.tile(np.arange(300, dtype=np.float32), (7, 1)))
        buffer.append(np.tile(np.arange(300, 530, dtype=np.float32), (7, 1)))

        window = buffer.latest_window(512)

        self.assertEqual(window.shape, (7, 512))
        np.testing.assert_array_equal(window[0], np.arange(18, 530, dtype=np.float32))

    def test_smoothing_moves_toward_new_estimate(self):
        smoother = uma8_music.DirectionSmoother(alpha=0.5)
        first = smoother.update(uma8_music.DirectionEstimate(azimuth=10.0, elevation=20.0))
        second = smoother.update(uma8_music.DirectionEstimate(azimuth=50.0, elevation=40.0))
        self.assertEqual(first.azimuth, 10.0)
        self.assertEqual(second.azimuth, 30.0)
        self.assertEqual(second.elevation, 30.0)

    def test_smoothing_handles_azimuth_wraparound(self):
        smoother = uma8_music.DirectionSmoother(alpha=0.5)
        smoother.update(uma8_music.DirectionEstimate(azimuth=350.0, elevation=0.0))
        second = smoother.update(uma8_music.DirectionEstimate(azimuth=10.0, elevation=0.0))
        self.assertEqual(second.azimuth, 0.0)

    def test_select_target_frequencies_stays_inside_configured_band(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        signal = uma8_music.make_directional_tone(
            azimuth=40.0,
            elevation=10.0,
            estimator=estimator,
            frequency=1000.0,
        )[0]

        freqs = estimator.select_target_frequencies(
            signal,
            band_min=900.0,
            band_max=1100.0,
            max_frequencies=3,
        )

        self.assertGreaterEqual(len(freqs), 1)
        self.assertTrue(all(900.0 <= freq <= 1100.0 for freq in freqs))

    def test_estimate_direction_reports_confidence_and_frequency_bins(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        window = uma8_music.make_directional_tone(
            azimuth=45.0,
            elevation=20.0,
            estimator=estimator,
            frequency=1000.0,
        )

        estimate = uma8_music.estimate_direction_from_audio_matrix(
            window,
            band_min=900.0,
            band_max=1100.0,
            max_frequencies=2,
        )

        self.assertIsNotNone(estimate.confidence)
        self.assertGreaterEqual(estimate.confidence, 0.0)
        self.assertLessEqual(estimate.confidence, 1.0)
        self.assertGreaterEqual(len(estimate.frequencies), 1)

    def test_azimuth_only_music_fixes_elevation_at_zero(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        window = uma8_music.make_directional_tone(
            azimuth=45.0,
            elevation=20.0,
            estimator=estimator,
            frequency=1000.0,
        )

        estimate = uma8_music.estimate_direction_from_audio_matrix(
            window,
            band_min=900.0,
            band_max=1100.0,
            max_frequencies=1,
            azimuth_only=True,
        )

        self.assertAlmostEqual(estimate.azimuth, 45.0, delta=2.0)
        self.assertEqual(estimate.elevation, 0.0)
        self.assertIsNotNone(estimate.confidence)

    def test_experimental_elevation_follows_synthetic_delay_geometry(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        window = uma8_music.make_directional_tone(
            azimuth=60.0,
            elevation=30.0,
            estimator=estimator,
            frequency=1000.0,
        )

        estimate = uma8_music.estimate_direction_from_audio_matrix(
            window,
            band_min=900.0,
            band_max=1100.0,
            max_frequencies=1,
            azimuth_only=True,
        )

        self.assertAlmostEqual(estimate.azimuth, 60.0, delta=2.0)
        self.assertIsNotNone(estimate.experimental_elevation)
        self.assertAlmostEqual(estimate.experimental_elevation, 30.0, delta=3.0)
        self.assertGreaterEqual(estimate.experimental_elevation_valid_mics, 4)
        self.assertGreater(estimate.experimental_elevation_confidence, 0.5)

    def test_experimental_elevation_rejects_perpendicular_denominators(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        window = uma8_music.make_directional_tone(
            azimuth=0.0,
            elevation=20.0,
            estimator=estimator,
            frequency=1000.0,
        )

        elevation = estimator.estimate_experimental_elevation(
            window,
            music_azimuth=0.0,
            band_min=900.0,
            band_max=1100.0,
            target_frequency=1000.0,
        )

        self.assertGreaterEqual(elevation.valid_mics, 4)
        self.assertLessEqual(elevation.valid_mics, 6)
        self.assertTrue(all(0.0 <= candidate <= 180.0 for candidate in elevation.candidates))

    def test_multifrequency_consensus_rejects_direction_outlier(self):
        estimator = uma8_music.UMA8MUSICEstimator()

        def fake_run_music(
            array_data,
            target_freq,
            az_step=2,
            el_step=2,
            azimuth_only=False,
        ):
            del array_data, az_step, el_step, azimuth_only
            return {
                1000.0: (40.0, 20.0),
                1100.0: (43.0, 22.0),
                2200.0: (170.0, 50.0),
            }[target_freq]

        estimator.run_music = fake_run_music
        consensus = estimator.run_music_frequency_consensus(
            np.zeros((7, 512)),
            [1000.0, 1100.0, 2200.0],
            weights=[1.0, 1.0, 1.0],
            consensus_degrees=10.0,
        )

        self.assertAlmostEqual(consensus.azimuth, 41.5, delta=0.2)
        self.assertEqual(consensus.frequencies, (1000.0, 1100.0))
        self.assertAlmostEqual(consensus.agreement, 2 / 3, places=3)

    def test_no_smoothing_tracking_reports_raw_current_estimate(self):
        estimates = iter(
            [
                uma8_music.DirectionEstimate(azimuth=10.0, elevation=5.0),
                uma8_music.DirectionEstimate(azimuth=80.0, elevation=15.0),
            ]
        )
        output = uma8_music.DirectionConsoleOutput()
        clock = {"now": 0.0}

        def monotonic():
            return clock["now"]

        def sleep(seconds):
            clock["now"] += seconds

        with contextlib.redirect_stdout(io.StringIO()):
            uma8_music.run_direction_loop(
                source=lambda: next(estimates),
                output=output,
                duration=0.2,
                interval=0.1,
                smoothing_alpha=None,
                sleep=sleep,
                monotonic=monotonic,
            )

        self.assertEqual(output.commands[-1].azimuth, 80.0)

    def test_too_short_window_duration_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least 512 samples"):
            uma8_music.SimulatedAudioSource(window_duration=0.001)

    def test_direction_console_output_records_estimates(self):
        output = uma8_music.DirectionConsoleOutput()
        estimate = uma8_music.DirectionEstimate(
            azimuth=25.0,
            elevation=15.0,
            experimental_elevation=22.0,
            experimental_elevation_valid_mics=4,
            experimental_elevation_confidence=0.5,
        )
        with contextlib.redirect_stdout(io.StringIO()) as text:
            output.report(estimate)
        self.assertEqual(output.commands, [estimate])
        self.assertIn("experimental_elevation=22.0 deg", text.getvalue())

    def test_simulated_tracking_loop_runs_without_hardware(self):
        estimates = iter(
            [
                uma8_music.DirectionEstimate(azimuth=10.0, elevation=5.0),
                uma8_music.DirectionEstimate(azimuth=20.0, elevation=10.0),
            ]
        )
        output = uma8_music.DirectionConsoleOutput()
        clock = {"now": 0.0}

        def monotonic():
            return clock["now"]

        def sleep(seconds):
            clock["now"] += seconds

        with contextlib.redirect_stdout(io.StringIO()):
            frames = uma8_music.run_tracking_loop(
                source=lambda: next(estimates),
                controller=output,
                duration=0.2,
                interval=0.1,
                sleep=sleep,
                monotonic=monotonic,
            )

        self.assertEqual(frames, 2)
        self.assertEqual(len(output.commands), 2)

    def test_tracking_can_consume_mock_audio_source_without_sounddevice(self):
        window = uma8_music.make_self_check_signal()
        source = uma8_music.MockAudioSource([window, window])
        direction_source = uma8_music.audio_window_direction_source(source)
        output = uma8_music.DirectionConsoleOutput()
        clock = {"now": 0.0}

        def monotonic():
            return clock["now"]

        def sleep(seconds):
            clock["now"] += seconds

        with contextlib.redirect_stdout(io.StringIO()):
            frames = uma8_music.run_tracking_loop(
                source=direction_source,
                controller=output,
                duration=0.2,
                interval=0.1,
                sleep=sleep,
                monotonic=monotonic,
            )

        self.assertEqual(frames, 2)
        self.assertEqual(len(output.commands), 2)

    def test_audio_source_can_use_azimuth_only_music(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        window = uma8_music.make_directional_tone(
            azimuth=90.0,
            elevation=25.0,
            estimator=estimator,
            frequency=1000.0,
        )
        source = uma8_music.MockAudioSource([window])
        direction_source = uma8_music.audio_window_direction_source(
            source,
            band_min=900.0,
            band_max=1100.0,
            max_frequencies=1,
            azimuth_only=True,
        )

        estimate = direction_source()

        self.assertAlmostEqual(estimate.azimuth, 90.0, delta=2.0)
        self.assertEqual(estimate.elevation, 0.0)

    def test_self_check_runs_without_hardware(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            uma8_music.run_self_check()
        self.assertIn("Self-check completed without microphone input.", output.getvalue())

    def test_simulated_tracking_runs_without_hardware_for_short_duration(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            frames = uma8_music.run_simulated_tracking(duration=0.01, interval=0.01)
        self.assertGreaterEqual(frames, 1)
        self.assertIn("DIRECTION", output.getvalue())

    def test_visualizer_can_update_in_noninteractive_backend(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        visualizer = uma8_music.DirectionVisualizer(estimator.mic_coords)
        try:
            visualizer.update(
                uma8_music.DirectionEstimate(
                    azimuth=45.0,
                    elevation=20.0,
                    frequency=1000.0,
                    reference_azimuth=50.0,
                )
            )
            self.assertEqual(visualizer.azimuth_history, [45.0])
        finally:
            visualizer.close()

    def test_3d_visualizer_can_show_experimental_elevation(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        visualizer = uma8_music.DirectionVisualizer(
            estimator.mic_coords,
            show_history=False,
            view_mode="3d",
        )
        try:
            visualizer.update(
                uma8_music.DirectionEstimate(
                    azimuth=45.0,
                    elevation=0.0,
                    frequency=1000.0,
                    experimental_elevation=30.0,
                    experimental_elevation_valid_mics=4,
                    experimental_elevation_confidence=0.7,
                )
            )
            self.assertEqual(visualizer.azimuth_history, [45.0])
            self.assertEqual(visualizer.view_mode, "3d")
        finally:
            visualizer.close()

    def test_visualizer_rejects_unknown_view_mode(self):
        estimator = uma8_music.UMA8MUSICEstimator()
        with self.assertRaisesRegex(ValueError, "view_mode"):
            uma8_music.DirectionVisualizer(estimator.mic_coords, view_mode="sideways")


if __name__ == "__main__":
    unittest.main()
