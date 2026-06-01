from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.sr_review import load_loss_weight_grid, parse_int_list, summarize_metric_rows


class SRReviewTests(unittest.TestCase):
    def test_parse_int_list(self) -> None:
        self.assertEqual(parse_int_list("1, 2,3"), [1, 2, 3])

    def test_inline_loss_weight_grid(self) -> None:
        specs = load_loss_weight_grid("base:0.05,0.2,0.1,0.05,0.02;energy:0.05,0.5,0,0,0")
        self.assertEqual([spec.name for spec in specs], ["base", "energy"])
        self.assertAlmostEqual(specs[1].energy_weight, 0.5)
        self.assertAlmostEqual(specs[1].peak_weight, 0.0)

    def test_summarize_metric_rows(self) -> None:
        rows = [
            {"seed": 1, "preset": "pa12", "scope": "pooled", "method": "pinn_full", "mean_curve_nrmse": "0.4", "regret_at_10_g": "inf"},
            {"seed": 2, "preset": "pa12", "scope": "pooled", "method": "pinn_full", "mean_curve_nrmse": "0.6", "regret_at_10_g": "1.0"},
        ]
        summary = summarize_metric_rows(rows, group_keys=("preset", "scope", "method"))
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(float(summary[0]["mean_curve_nrmse_mean"]), 0.5)
        self.assertEqual(summary[0]["regret_at_10_g_finite_n"], 1)
        self.assertTrue(math.isfinite(float(summary[0]["regret_at_10_g_mean"])))


if __name__ == "__main__":
    unittest.main()
