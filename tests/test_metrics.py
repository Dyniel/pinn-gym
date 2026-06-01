from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.metrics import (
    best_feasible_regret,
    best_selected_feasible_mass,
    force_curve_error_metrics,
    force_curve_metrics,
    integrate_energy_j,
    nrmse,
    physical_violation_rate,
    precision_at_k,
    relative_best_feasible_regret,
    rmse,
)


class MetricsTests(unittest.TestCase):
    def test_force_curve_metrics_for_constant_plateau(self) -> None:
        disp = [0.0, 10.0, 20.0, 30.0, 40.0]
        force = [100.0, 100.0, 100.0, 100.0, 100.0]
        metrics = force_curve_metrics(disp, force, mass_g=10.0, target_energy_j=4.0, target_stroke_mm=40.0)
        self.assertAlmostEqual(metrics["absorbed_energy_J"], 4.0)
        self.assertAlmostEqual(metrics["specific_energy_absorption_J_g"], 0.4)
        self.assertAlmostEqual(metrics["mean_crushing_force_N"], 100.0)
        self.assertAlmostEqual(metrics["crush_force_efficiency"], 1.0)
        self.assertAlmostEqual(metrics["plateau_force_cv"], 0.0)

    def test_integrate_energy_respects_limit(self) -> None:
        self.assertAlmostEqual(integrate_energy_j([0.0, 40.0], [100.0, 100.0], limit_mm=20.0), 2.0)

    def test_nrmse(self) -> None:
        self.assertGreater(nrmse([0.0, 1.0], [0.0, 2.0]), 0.0)

    def test_curve_error_metrics(self) -> None:
        metrics = force_curve_error_metrics(
            [0.0, 20.0, 40.0],
            [100.0, 100.0, 100.0],
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [80.0, 80.0, 80.0, 80.0, 80.0],
        )
        self.assertAlmostEqual(metrics["curve_rmse_N"], 20.0)
        self.assertAlmostEqual(metrics["energy_integral_abs_error_J"], 0.8)
        self.assertAlmostEqual(rmse([1.0, 3.0], [1.0, 1.0]), 2.0**0.5)

    def test_ranking_metrics(self) -> None:
        rows = [
            {"e": 30.0, "p": 3000.0, "c": 41.0, "r": 0.2},
            {"e": 20.0, "p": 3000.0, "c": 41.0, "r": 0.2},
            {"e": 31.0, "p": 3600.0, "c": 41.0, "r": 0.2},
            {"e": 31.0, "p": 3000.0, "c": 35.0, "r": 0.2},
        ]
        self.assertAlmostEqual(
            physical_violation_rate(rows, energy_key="e", peak_key="p", crush_key="c", risk_key="r", max_risk=0.5),
            0.75,
        )
        self.assertAlmostEqual(precision_at_k([True, False, True], 2), 0.5)
        self.assertAlmostEqual(best_feasible_regret([12.0, 10.0], [False, True], [9.0, 11.0]), 1.0)
        self.assertAlmostEqual(best_selected_feasible_mass([12.0, 10.0], [False, True]), 10.0)
        self.assertAlmostEqual(relative_best_feasible_regret([12.0, 10.0], [False, True], [9.0, 11.0]), 1.0 / 9.0)


if __name__ == "__main__":
    unittest.main()
