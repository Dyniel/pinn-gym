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
from pinn_gym.core.physics_bakeoff import PhysicsBakeoffConfig, run_physics_bakeoff


class PhysicsBakeoffTests(unittest.TestCase):
    def test_physics_bakeoff_writes_consensus_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            top_csv = Path(tmp) / "top.csv"
            rows = []
            for rank, seed in enumerate([31, 32], start=1):
                row = design_to_candidate_row(sample_design(random.Random(seed)), curve_points=12)
                row["rank"] = str(rank)
                row["score"] = str(100.0 + rank)
                row["mass_g_mean"] = row["mass_g"]
                rows.append(row)
            fieldnames = ["rank", "score"] + [key for key in rows[0].keys() if key not in {"rank", "score"}]
            with top_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            out_dir = Path(tmp) / "bakeoff"
            config = PhysicsBakeoffConfig(
                layers=8,
                steps=24,
                write_curves=1,
                models=("homogenized_ga", "buckling_stack"),
            )
            summary = run_physics_bakeoff(top_csv, out_dir, top_n=2, config=config)
            self.assertEqual(summary["evaluated"], 2)
            self.assertTrue((out_dir / "physics_bakeoff_candidates.csv").exists())
            self.assertTrue((out_dir / "robust_candidates.csv").exists())
            self.assertTrue(any((out_dir / "curves").glob("*.csv")))

            with (out_dir / "physics_bakeoff_candidates.csv").open(newline="", encoding="utf-8") as f:
                out_rows = list(csv.DictReader(f))
            self.assertIn("consensus_energy_min_J", out_rows[0])
            self.assertIn("homogenized_ga_peak_force_N", out_rows[0])
            self.assertIn("buckling_stack_crush_mm", out_rows[0])


if __name__ == "__main__":
    unittest.main()
