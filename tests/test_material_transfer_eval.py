from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.design_space import design_to_candidate_row, sample_design
from pinn_gym.core.material_transfer_eval import evaluate_material_transfer_run


class MaterialTransferEvalTests(unittest.TestCase):
    def test_evaluate_material_transfer_run_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = design_to_candidate_row(sample_design(__import__("random").Random(7)))
            candidate["rank"] = "1"
            top_csv = root / "top.csv"
            self._write_rows(top_csv, [candidate])
            for preset in ("pa12", "pla"):
                physics_dir = root / preset / "physics_gate"
                dynamic_dir = root / preset / "dynamic_impact"
                physics_dir.mkdir(parents=True)
                dynamic_dir.mkdir(parents=True)
                (physics_dir / "summary.json").write_text(f'{{"top_csv": "{top_csv}"}}', encoding="utf-8")
                feasible = preset == "pa12"
                self._write_rows(
                    physics_dir / "physics_candidates.csv",
                    [
                        {
                            "rank": "1",
                            "topology": candidate["topology"],
                            "mass_g": "20.0",
                            "physics_energy_usable_J": "35.0" if feasible else "10.0",
                            "physics_peak_force_N": "1000.0",
                            "physics_collapse_mm": "42.0" if feasible else "20.0",
                            "physics_failure_risk": "0.1",
                            "physics_survives_gate": str(feasible),
                            "physics_score": "1.0",
                        }
                    ],
                )
                self._write_rows(
                    dynamic_dir / "dynamic_impact_candidates.csv",
                    [
                        {
                            "rank": "1",
                            "topology": candidate["topology"],
                            "mass_g": "20.0",
                            "impact_absorbed_J": "30.0" if feasible else "5.0",
                            "impact_peak_force_N": "1000.0",
                            "impact_max_displacement_mm": "34.0",
                            "impact_failure_risk": "0.1",
                            "impact_survives": str(feasible),
                            "impact_score": "1.0",
                        }
                    ],
                )

            summary = evaluate_material_transfer_run(root, root / "metrics", top_csv=top_csv, precision_ks=[1])
            self.assertEqual(summary["transfer_pairs"], 2)
            self.assertTrue((root / "metrics" / "curve_metrics.csv").exists())
            self.assertTrue((root / "metrics" / "transfer_metrics.csv").exists())

    def _write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
