import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dts_simulation.playback import (
    default_source_path,
    export_playback_gif,
    generate_playback_frames,
)


class PlaybackGenerationTests(unittest.TestCase):
    def test_default_source_path_is_deterministic_and_non_degenerate(self) -> None:
        path = default_source_path(5)

        self.assertEqual(path, default_source_path(5))
        self.assertEqual(len(path), 5)
        self.assertEqual(path[0], (1.0, 0.8, 1.6))
        self.assertEqual(path[-1], (3.0, 2.5, 3.0))
        self.assertGreater(len({point[0] for point in path}), 1)
        self.assertGreater(len({point[1] for point in path}), 1)
        self.assertGreater(len({point[2] for point in path}), 1)

    def test_playback_generation_creates_png_sequence_under_outputs_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "outputs" / "playback"
            result = generate_playback_frames(
                frame_count=3,
                output_directory=output_directory,
                duration_seconds=0.2,
            )

            self.assertEqual(result.output_directory, output_directory)
            self.assertEqual(len(result.frames), 3)
            self.assertTrue(result.all_frames_valid)
            self.assertIsNotNone(result.max_error_distance)
            self.assertIsNotNone(result.mean_error_distance)
            for expected_index, frame in enumerate(result.frames):
                self.assertEqual(frame.frame_index, expected_index)
                self.assertEqual(frame.output_path.parent, output_directory)
                self.assertEqual(frame.output_path.name, f"frame_{expected_index:03d}.png")
                self.assertIn("outputs", frame.output_path.parts)
                self.assertIn("playback", frame.output_path.parts)
                self.assertTrue(frame.output_path.exists())
                self.assertGreater(frame.output_path.stat().st_size, 0)

    def test_gif_export_creates_non_empty_ignored_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs"
            playback_result = generate_playback_frames(
                frame_count=3,
                output_directory=output_root / "playback",
                duration_seconds=0.2,
            )
            gif_path = output_root / "dts_playback.gif"

            gif_result = export_playback_gif(playback_result, gif_path)

            self.assertEqual(gif_result.output_path, gif_path)
            self.assertEqual(gif_result.frame_count, 3)
            self.assertTrue(gif_result.succeeded)
            self.assertIn("outputs", gif_result.output_path.parts)
            self.assertTrue(gif_result.output_path.exists())
            self.assertGreater(gif_result.output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
