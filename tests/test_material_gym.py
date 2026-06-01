from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.material_gym import (
    MaterialGymBuildConfig,
    MaterialGymCompareConfig,
    build_material_gym_datasets,
    compare_material_gym,
)


class MaterialGymTests(unittest.TestCase):
    def test_build_and_compare_tiny_material_gym(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = root / "datasets"
            build = build_material_gym_datasets(
                datasets,
                config=MaterialGymBuildConfig(presets=("pa12",), train_n=2, eval_n=2, seed=11, layers=12, steps=16),
            )
            self.assertEqual(len(build["datasets"]), 1)
            self.assertTrue((datasets / "pa12" / "train.csv").exists())
            compare = compare_material_gym(
                datasets,
                root / "comparison",
                config=MaterialGymCompareConfig(precision_ks=(1, 2)),
            )
            methods = {row["method"] for row in compare["rows"]}
            self.assertIn("random", methods)
            self.assertIn("pseudo_bootstrap", methods)
            self.assertTrue((root / "comparison" / "method_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
