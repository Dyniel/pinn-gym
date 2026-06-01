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
from pinn_gym.core.refinement import ActiveRefinementConfig, run_active_refinement


class ActiveRefinementTests(unittest.TestCase):
    def test_active_refinement_writes_local_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            top_csv = Path(tmp) / "top.csv"
            rows = []
            for rank, seed in enumerate([31, 32], start=1):
                row = design_to_candidate_row(sample_design(random.Random(seed)), curve_points=16)
                row["rank"] = str(rank)
                row["score"] = str(rank)
                rows.append(row)
            fieldnames = ["rank", "score"] + [key for key in rows[0].keys() if key not in {"rank", "score"}]
            with top_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            summary = run_active_refinement(
                top_csv,
                Path(tmp) / "refine",
                config=ActiveRefinementConfig(max_seeds=2, variants_per_seed=2, iterations=1),
            )
            self.assertGreater(summary["refined_rows"], 0)
            self.assertTrue((Path(tmp) / "refine" / "active_refinement_candidates.csv").exists())


if __name__ == "__main__":
    unittest.main()
