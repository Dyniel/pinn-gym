from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.sr_benchmark import (
    SRBuildConfig,
    SREvalConfig,
    build_sr_dataset,
    evaluate_sr_run,
)


class SRBenchmarkSmokeTests(unittest.TestCase):
    def test_build_then_evaluate_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = root / "datasets"
            payload = build_sr_dataset(
                datasets,
                config=SRBuildConfig(presets=("pa12", "tpu"), train_n=4, eval_n=4, seed=7, layers=14, steps=18),
            )
            self.assertEqual(len(payload["datasets"]), 2)
            for preset in ("pa12", "tpu"):
                self.assertTrue((datasets / preset / "train.csv").exists())
                self.assertTrue((datasets / preset / "eval.csv").exists())
                scales = json.loads((datasets / preset / "scales.json").read_text())
                self.assertGreater(scales["force_scale_N"], 0.0)
            eval_payload = evaluate_sr_run(
                datasets,
                checkpoint_root=None,
                out_dir=root / "eval",
                presets=["pa12", "tpu"],
                config=SREvalConfig(precision_ks=(1, 2)),
                include_transfer=False,
            )
            methods = {row["method"] for row in eval_payload["rows"]}
            self.assertIn("random", methods)
            self.assertIn("pseudo_bootstrap", methods)
            self.assertIn("oracle_upper_bound", methods)
            self.assertTrue((root / "eval" / "method_metrics.csv").exists())
            with (root / "eval" / "method_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            presets_in_csv = {row["preset"] for row in rows}
            self.assertEqual(presets_in_csv, {"pa12", "tpu"})
            self.assertIn("best_selected_feasible_mass_at_1_g", rows[0])
            self.assertIn("regret_at_1_g", rows[0])
            self.assertIn("relative_regret_at_1", rows[0])


if __name__ == "__main__":
    unittest.main()
