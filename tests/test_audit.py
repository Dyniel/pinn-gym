from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.audit import audit_generated_candidates, summarize_checks
from pinn_gym.core.design_space import sample_designs, write_candidates_csv


class AuditTests(unittest.TestCase):
    def test_generated_candidate_audit_passes_on_valid_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            write_candidates_csv(path, sample_designs(64, seed=42), curve_points=16)
            checks = audit_generated_candidates(path)
            summary = summarize_checks(checks)
            self.assertEqual(summary["counts"]["error"], 0)
            self.assertGreaterEqual(summary["counts"]["ok"], 3)


if __name__ == "__main__":
    unittest.main()
