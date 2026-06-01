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
from pinn_gym.core.rom_guided import build_rom_feedback_dataset, select_candidates_by_physics, select_finalist_shortlist


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top_row(rank: int, seed: int) -> dict[str, object]:
    row = design_to_candidate_row(sample_design(random.Random(seed)), curve_points=12)
    row.update(
        {
            "rank": str(rank),
            "score": str(10.0 + rank),
            "mass_g_mean": "28.0",
            "energy_abs_J_lcb": "36.0",
            "force_peak_N_mean": "1450.0",
            "force_plateau_N_mean": "1100.0",
            "collapse_displacement_mm_mean": "44.0",
            "failure_probability_mean": "0.12",
            "peak_plateau_ratio_mean": "1.32",
            "progressive_crush_score_mean": "0.72",
        }
    )
    return row


def _physics_row(rank: int, score: float) -> dict[str, object]:
    return {
        "rank": str(rank),
        "physics_energy_usable_J": "32.0",
        "physics_peak_force_N": "3100.0",
        "physics_collapse_mm": "42.0",
        "physics_failure_risk": "0.25",
        "physics_survives_gate": "True",
        "physics_score": str(score),
    }


def _dynamic_row(rank: int, score: float) -> dict[str, object]:
    return {
        "rank": str(rank),
        "impact_absorbed_J": "30.0",
        "impact_peak_force_N": "3250.0",
        "impact_max_displacement_mm": "41.0",
        "impact_crush_distance_pass": "True",
        "impact_survives": "True",
        "impact_failure_risk": "0.28",
        "impact_score": str(score),
    }


class RomGuidedTests(unittest.TestCase):
    def test_feedback_dataset_joins_prior_gate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_a"
            _write_csv(run_dir / "postanalysis" / "top_candidates.csv", [_top_row(1, 1), _top_row(2, 2)])
            _write_csv(run_dir / "physics_gate" / "physics_candidates.csv", [_physics_row(1, 20.0)])
            _write_csv(run_dir / "dynamic_impact" / "dynamic_impact_candidates.csv", [_dynamic_row(1, 21.0), _dynamic_row(2, 40.0)])

            out_csv = Path(tmp) / "feedback" / "rom_feedback.csv"
            summary = build_rom_feedback_dataset([run_dir], out_csv, top_n=10)
            self.assertEqual(summary["rows"], 2)
            self.assertTrue(out_csv.exists())

            with out_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["source_run"], "run_a")
            self.assertIn("physics_energy_usable_J", rows[0])
            self.assertIn("impact_absorbed_J", rows[0])

    def test_select_candidates_by_physics_keeps_original_rows_with_coarse_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            top_csv = Path(tmp) / "top.csv"
            physics_csv = Path(tmp) / "physics.csv"
            out_csv = Path(tmp) / "selected" / "top_candidates.csv"
            _write_csv(top_csv, [_top_row(1, 11), _top_row(2, 12), _top_row(3, 13)])
            _write_csv(physics_csv, [_physics_row(1, 50.0), _physics_row(2, 15.0), _physics_row(3, 30.0)])

            summary = select_candidates_by_physics(top_csv, physics_csv, out_csv, top_n=2)
            self.assertEqual(summary["rows"], 2)
            with out_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["rank"] for row in rows], ["2", "3"])
            self.assertEqual(rows[0]["coarse_physics_score"], "15.0")

    def test_select_finalist_shortlist_requires_physics_and_dynamic_survival(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            top_csv = Path(tmp) / "top.csv"
            physics_csv = Path(tmp) / "physics.csv"
            dynamic_csv = Path(tmp) / "dynamic.csv"
            out_csv = Path(tmp) / "shortlist" / "top_candidates.csv"
            row1 = _top_row(1, 21)
            row1["mass_g_mean"] = "28.0"
            row2 = _top_row(2, 22)
            row2["mass_g_mean"] = "26.0"
            row3 = _top_row(3, 23)
            row3["mass_g_mean"] = "25.0"
            _write_csv(top_csv, [row1, row2, row3])
            _write_csv(physics_csv, [_physics_row(1, 50.0), _physics_row(2, 30.0), _physics_row(3, 10.0)])
            dyn1 = _dynamic_row(1, 40.0)
            dyn2 = _dynamic_row(2, 20.0)
            dyn3 = _dynamic_row(3, 10.0)
            dyn3["impact_survives"] = "False"
            dyn3["impact_absorbed_J"] = "20.0"
            _write_csv(dynamic_csv, [dyn1, dyn2, dyn3])

            summary = select_finalist_shortlist(top_csv, physics_csv, dynamic_csv, out_csv, top_n=5)
            self.assertEqual(summary["rows"], 2)
            with out_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["rank"] for row in rows], ["2", "1"])
            self.assertIn("shortlist_dynamic_crush_distance_pass", rows[0])


if __name__ == "__main__":
    unittest.main()
