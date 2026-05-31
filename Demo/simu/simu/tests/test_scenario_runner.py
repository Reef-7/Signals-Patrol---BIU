import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dts_simulation.config import DEFAULT_LOCALIZATION_ARRAY_IDS
from dts_simulation.simulation import (
    DEFAULT_SCENARIO_SOURCE_XYZ,
    ScenarioInput,
    run_synthetic_scenario,
)

SCENARIO_POSITION_TOLERANCE_METERS = 0.15
SCENARIO_NOISY_POSITION_TOLERANCE_METERS = 0.2


class ScenarioRunnerTests(unittest.TestCase):
    def test_default_synthetic_scenario_returns_valid_result_and_aim_target(self) -> None:
        result = run_synthetic_scenario()

        self.assertTrue(result.valid)
        self.assertEqual(result.true_source_xyz, DEFAULT_SCENARIO_SOURCE_XYZ)
        self.assertEqual(len(result.direction_estimates), 3)
        self.assertEqual(
            result.localization_result.used_array_ids,
            DEFAULT_LOCALIZATION_ARRAY_IDS,
        )
        self.assertLessEqual(
            result.error_distance,
            SCENARIO_POSITION_TOLERANCE_METERS,
        )
        self.assertEqual(
            result.virtual_aim_target_xyz,
            result.localization_result.estimated_xyz,
        )
        self.assertTrue(all(estimate.valid for estimate in result.direction_estimates))

    def test_synthetic_scenario_supports_deterministic_mild_noise(self) -> None:
        result = run_synthetic_scenario(
            ScenarioInput(noise_standard_deviation=0.01)
        )

        self.assertTrue(result.valid)
        self.assertLessEqual(
            result.error_distance,
            SCENARIO_NOISY_POSITION_TOLERANCE_METERS,
        )
        self.assertEqual(
            result.virtual_aim_target_xyz,
            result.localization_result.estimated_xyz,
        )


if __name__ == "__main__":
    unittest.main()
