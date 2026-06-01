from __future__ import annotations

import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.design_space import design_to_candidate_row, sample_design
from pinn_gym.core.materials import MaterialCard
from pinn_gym.core.physics import (
    IMPACT_ENERGY_J,
    ImpactConfig,
    PhysicsConfig,
    evaluate_candidate_physics,
    run_dynamic_impact,
    run_physics_gate,
    run_sensitivity_sweep,
)


class PhysicsGateTests(unittest.TestCase):
    def test_layered_crush_fem_returns_positive_metrics(self) -> None:
        design = sample_design(random.Random(11))
        row = design.to_row()
        result, sim = evaluate_candidate_physics(row, config=PhysicsConfig(layers=16, steps=48))
        self.assertGreater(result.physics_energy_50mm_J, 0.0)
        self.assertGreater(result.physics_sea_J_g, 0.0)
        self.assertGreaterEqual(result.physics_cfe, 0.0)
        self.assertGreater(result.physics_peak_force_N, 0.0)
        self.assertGreater(result.physics_collapse_mm, 0.0)
        self.assertEqual(len(sim["displacement_mm"]), 48)
        self.assertLess(IMPACT_ENERGY_J, 40.0)

    def test_dynamic_impact_returns_energy_balance(self) -> None:
        design = sample_design(random.Random(13))
        impact, payload = run_dynamic_impact(
            design,
            physics_config=PhysicsConfig(layers=12, steps=32),
            impact_config=ImpactConfig(dt_s=5e-5, max_time_s=0.006),
            rank=1,
        )
        self.assertGreater(impact.impact_initial_ke_J, 20.0)
        self.assertGreaterEqual(impact.impact_sea_J_g, 0.0)
        self.assertGreaterEqual(impact.impact_absorbed_J, 0.0)
        self.assertIn("history", payload)

    def test_material_card_changes_mass_and_response(self) -> None:
        design = sample_design(random.Random(17))
        row = design.to_row()
        base, _ = evaluate_candidate_physics(row, config=PhysicsConfig(layers=12, steps=32))
        dense = MaterialCard(
            material_name="dense smoke material",
            density_g_cm3=2.02,
            compressive_plateau_strength_MPa=70.0,
            compressive_yield_strength_MPa=70.0,
        )
        changed, _ = evaluate_candidate_physics(row, config=PhysicsConfig(layers=12, steps=32, material=dense))
        self.assertGreater(changed.mass_g, 1.5 * base.mass_g)
        self.assertEqual(changed.material_name, "dense smoke material")

    def test_physics_gate_writes_ranked_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            top_csv = Path(tmp) / "top.csv"
            rows = []
            for rank, seed in enumerate([1, 2, 3], start=1):
                row = design_to_candidate_row(sample_design(random.Random(seed)), curve_points=16)
                row["rank"] = str(rank)
                row["score"] = str(30.0 + rank)
                rows.append(row)
            fieldnames = ["rank", "score"] + [key for key in rows[0].keys() if key not in {"rank", "score"}]
            with top_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            summary = run_physics_gate(top_csv, Path(tmp) / "physics", top_n=3, config=PhysicsConfig(layers=12, steps=32), write_curves=1)
            self.assertEqual(summary["evaluated"], 3)
            self.assertTrue((Path(tmp) / "physics" / "physics_candidates.csv").exists())

    def test_sensitivity_sweep_writes_uncertainty_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            top_csv = Path(tmp) / "top.csv"
            row = design_to_candidate_row(sample_design(random.Random(21)), curve_points=16)
            row["rank"] = "1"
            row["score"] = "1.0"
            fieldnames = ["rank", "score"] + [key for key in row.keys() if key not in {"rank", "score"}]
            with top_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            summary = run_sensitivity_sweep(top_csv, Path(tmp) / "sensitivity", top_n=1, config=PhysicsConfig(layers=8, steps=24))
            self.assertEqual(summary["evaluated_designs"], 1)
            self.assertTrue((Path(tmp) / "sensitivity" / "sensitivity_sweep.csv").exists())


if __name__ == "__main__":
    unittest.main()
