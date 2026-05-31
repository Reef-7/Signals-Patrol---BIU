import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dts_simulation.simulation import run_synthetic_scenario
from dts_simulation.visualization import render_static_scene


class StaticVisualizationTests(unittest.TestCase):
    def test_static_renderer_creates_output_file(self) -> None:
        scenario_result = run_synthetic_scenario()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "scene.png"
            rendered_path = render_static_scene(scenario_result, output_path)

            self.assertEqual(rendered_path, output_path)
            self.assertTrue(rendered_path.exists())
            self.assertGreater(rendered_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
