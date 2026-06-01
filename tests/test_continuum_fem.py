from __future__ import annotations

import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.continuum_fem import (
    HexFEMConfig,
    calibrate_pa12_from_curve,
    compare_fem_to_reference,
    run_hex_fem_compression,
)
from pinn_gym.core.design_space import sample_design


class ContinuumHexFEMTests(unittest.TestCase):
    def test_hex_fem_smoke(self) -> None:
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("numpy/scipy is not installed")
        design = sample_design(random.Random(55))
        result, payload = run_hex_fem_compression(
            design,
            config=HexFEMConfig(resolution=6, displacement_mm=1.0, load_steps=2, relax_iterations=4, enable_self_contact=False),
            rank=1,
        )
        self.assertGreater(result.nodes, 0)
        self.assertGreater(result.gauss_points, 0)
        self.assertGreaterEqual(result.strain_energy_J, 0.0)
        self.assertIn("curve", payload)

    def test_hex_fem_empty_mesh_result(self) -> None:
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy is not installed")
        design = sample_design(random.Random(56))
        with patch("pinn_gym.core.continuum_fem._build_hex_mesh", return_value=(np.empty((0, 3)), np.empty((0, 8), dtype=int), 0.0)):
            result, payload = run_hex_fem_compression(
                design,
                config=HexFEMConfig(resolution=6, displacement_mm=1.0),
                rank=2,
            )
        self.assertEqual(result.nodes, 0)
        self.assertEqual(result.elements, 0)
        self.assertEqual(result.gauss_points, 0)
        self.assertEqual(result.damage_proxy, 1.0)
        self.assertEqual(payload["curve"], [])

    def test_material_calibration_and_compare_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            curve = Path(tmp) / "curve.csv"
            with curve.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["displacement_mm", "force_N"])
                writer.writeheader()
                for d, force in [(0.0, 0.0), (1.0, 120.0), (10.0, 700.0), (20.0, 760.0)]:
                    writer.writerow({"displacement_mm": d, "force_N": force})
            card = Path(tmp) / "pa12.json"
            payload = calibrate_pa12_from_curve(curve, card)
            self.assertTrue(card.exists())
            self.assertIn("material", payload)

            fem = Path(tmp) / "fem.csv"
            ref = Path(tmp) / "ref.csv"
            for path, scale in [(fem, 1.0), (ref, 1.1)]:
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["reaction_force_N", "strain_energy_J"])
                    writer.writeheader()
                    writer.writerow({"reaction_force_N": 100.0 * scale, "strain_energy_J": 0.5 * scale})
            summary = compare_fem_to_reference(fem, ref, Path(tmp) / "cmp")
            self.assertTrue(Path(summary["out_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
