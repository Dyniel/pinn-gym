from __future__ import annotations

import csv
import json
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
from pinn_gym.core.evidence import EvidenceConfig, build_evidence_stack


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class EvidenceStackTests(unittest.TestCase):
    def test_evidence_stack_joins_available_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            top_row = design_to_candidate_row(sample_design(random.Random(7)), curve_points=16)
            top_row.update(
                {
                    "rank": "1",
                    "score": "12.0",
                    "mass_g_mean": "24.0",
                    "energy_abs_J_lcb": "36.0",
                    "force_peak_N_mean": "1200.0",
                    "failure_probability_mean": "0.1",
                    "collapse_displacement_mm_mean": "42.0",
                }
            )
            _write_csv(run_dir / "postanalysis" / "top_candidates.csv", [top_row])
            _write_csv(
                run_dir / "physics_gate" / "physics_candidates.csv",
                [
                    {
                        "rank": "1",
                        "topology": top_row["topology"],
                        "physics_energy_usable_J": "34.0",
                        "physics_peak_force_N": "3000.0",
                        "physics_collapse_mm": "43.0",
                        "physics_impact_stop_mm": "41.0",
                        "physics_failure_risk": "0.2",
                        "physics_score": "20.0",
                    }
                ],
            )
            _write_csv(
                run_dir / "dynamic_impact" / "dynamic_impact_candidates.csv",
                [
                    {
                        "rank": "1",
                        "topology": top_row["topology"],
                        "impact_absorbed_J": "29.5",
                        "impact_peak_force_N": "3100.0",
                        "impact_max_displacement_mm": "41.0",
                        "impact_failure_risk": "0.2",
                        "impact_energy_balance_error_J": "0.1",
                        "impact_score": "21.0",
                    }
                ],
            )
            _write_csv(
                run_dir / "explicit_vector_impact" / "explicit_vector_impact_candidates.csv",
                [
                    {
                        "rank": "1",
                        "topology": top_row["topology"],
                        "impact_absorbed_J": "29.0",
                        "peak_contact_force_N": "3200.0",
                        "max_indenter_displacement_mm": "20.0",
                        "damage_proxy": "0.5",
                        "failed_springs_fraction": "0.02",
                        "energy_balance_error_J": "0.2",
                        "explicit_fem_score": "22.0",
                    }
                ],
            )
            _write_csv(
                run_dir / "sensitivity" / "sensitivity_sweep.csv",
                [
                    {
                        "rank": "1",
                        "topology": top_row["topology"],
                        "scenario": "nominal",
                        "energy_margin_J": "1.0",
                        "peak_force_N": "3000.0",
                        "failure_risk": "0.2",
                        "survives_physics": "True",
                        "impact_absorbed_J": "29.0",
                        "impact_risk": "0.2",
                        "survives_impact": "True",
                    }
                ],
            )
            summary = build_evidence_stack(run_dir, config=EvidenceConfig(min_hex_elements=0))
            self.assertEqual(summary["evaluated"], 1)
            self.assertTrue((run_dir / "evidence" / "candidate_evidence_stack.csv").exists())

    def test_dynamic_stop_before_40mm_is_not_bottom_out_when_crush_capacity_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            top_row = design_to_candidate_row(sample_design(random.Random(17)), curve_points=16)
            top_row.update(
                {
                    "rank": "1",
                    "score": "12.0",
                    "mass_g_mean": "26.0",
                    "energy_abs_J_lcb": "36.0",
                    "force_peak_N_mean": "1200.0",
                    "failure_probability_mean": "0.1",
                    "collapse_displacement_mm_mean": "42.0",
                }
            )
            _write_csv(run_dir / "postanalysis" / "top_candidates.csv", [top_row])
            _write_csv(
                run_dir / "physics_gate" / "physics_candidates.csv",
                [
                    {
                        "rank": "1",
                        "topology": top_row["topology"],
                        "physics_energy_usable_J": "36.0",
                        "physics_peak_force_N": "1500.0",
                        "physics_collapse_mm": "42.0",
                        "physics_impact_stop_mm": "35.0",
                        "physics_failure_risk": "0.2",
                        "physics_score": "20.0",
                    }
                ],
            )
            _write_csv(
                run_dir / "dynamic_impact" / "dynamic_impact_candidates.csv",
                [
                    {
                        "rank": "1",
                        "topology": top_row["topology"],
                        "impact_absorbed_J": "30.0",
                        "impact_peak_force_N": "1500.0",
                        "impact_max_displacement_mm": "36.0",
                        "impact_crush_distance_pass": "True",
                        "impact_failure_risk": "0.2",
                        "impact_energy_balance_error_J": "0.1",
                        "impact_score": "21.0",
                    }
                ],
            )
            _write_csv(
                run_dir / "sensitivity" / "sensitivity_sweep.csv",
                [
                    {
                        "rank": "1",
                        "topology": top_row["topology"],
                        "scenario": "nominal",
                        "energy_margin_J": "3.0",
                        "peak_force_N": "1500.0",
                        "failure_risk": "0.2",
                        "survives_physics": "True",
                        "impact_absorbed_J": "30.0",
                        "impact_risk": "0.2",
                        "survives_impact": "True",
                    }
                ],
            )
            _write_csv(
                run_dir / "hex_fem" / "hex_fem_candidates.csv",
                [
                    {
                        "rank": "1",
                        "elements": "12",
                        "reaction_force_N": "1000.0",
                    }
                ],
            )
            (run_dir / "postanalysis").mkdir(parents=True, exist_ok=True)
            (run_dir / "postanalysis" / "mesh_quality.json").write_text(
                json.dumps(
                    [
                        {
                            "path": str(run_dir / "postanalysis" / "stl" / "rank_001_test.stl"),
                            "exists": True,
                            "within_envelope": True,
                            "watertight_by_edges": True,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            summary = build_evidence_stack(run_dir)
            self.assertEqual(summary["candidate_decision_pass"], 1)
            with (run_dir / "evidence" / "candidate_evidence_stack.csv").open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["dynamic_conservative_pass"], "True")
            self.assertEqual(row["dynamic_available_crush_pass"], "True")
            self.assertEqual(row["failure_reasons"], "")


if __name__ == "__main__":
    unittest.main()
