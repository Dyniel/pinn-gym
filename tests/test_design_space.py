from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.design_space import (
    CURVE_POINTS,
    TARGET_ENERGY_J,
    integrate_curve_energy_j,
    pseudo_response,
    sample_design,
    sample_designs,
)


class DesignSpaceTests(unittest.TestCase):
    def test_sampled_designs_respect_printability_floor(self) -> None:
        designs = sample_designs(100, seed=123)
        self.assertEqual(len(designs), 100)
        for design in designs:
            self.assertGreaterEqual(design.min_feature_mm, 0.5)
            self.assertGreaterEqual(design.wall_thickness_mm, design.min_feature_mm)
            self.assertGreater(design.cell_size_mm, 4.0)
            self.assertLess(design.cell_size_mm, 10.0)

    def test_pseudo_response_has_positive_curve_and_consistent_energy(self) -> None:
        design = sample_design(random.Random(7))
        response = pseudo_response(design)
        curve = response["curve_force_N"]
        disp = response["curve_displacement_mm"]
        self.assertEqual(len(curve), CURVE_POINTS)
        self.assertTrue(all(force >= 0 for force in curve))
        self.assertGreater(response["mass_g"], 1.0)
        self.assertGreater(response["energy_abs_J"], 0.5)
        self.assertLessEqual(response["failure_probability"], 1.0)
        integrated = integrate_curve_energy_j(disp, curve)
        self.assertAlmostEqual(integrated, response["energy_abs_J"], delta=0.05 * max(1.0, response["energy_abs_J"]))

    def test_design_space_can_produce_viable_bootstrap_candidates(self) -> None:
        viable = 0
        for design in sample_designs(500, seed=99):
            response = pseudo_response(design)
            if response["energy_abs_J"] >= TARGET_ENERGY_J and response["failure_probability"] < 0.45:
                viable += 1
        self.assertGreater(viable, 10)


if __name__ == "__main__":
    unittest.main()
