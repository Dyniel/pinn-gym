from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.design_space import sample_design
from pinn_gym.core.solid_fem import ExplicitImpactFEMConfig, VectorFEMConfig, run_explicit_vector_impact, run_vector_voxel_fem


class VectorSolidFEMTests(unittest.TestCase):
    def test_vector_voxel_fem_smoke(self) -> None:
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("numpy/scipy is not installed")
        design = sample_design(random.Random(44))
        result = run_vector_voxel_fem(design, config=VectorFEMConfig(resolution=10, displacement_mm=1.0, cg_maxiter=500))
        self.assertGreater(result.solid_voxels, 0)
        self.assertGreater(result.vector_dofs, 0)
        self.assertGreaterEqual(result.reaction_force_N, 0.0)
        self.assertGreaterEqual(result.max_von_mises_MPa, 0.0)
        self.assertLessEqual(result.damage_proxy, 1.0)

    def test_explicit_vector_impact_smoke(self) -> None:
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("numpy/scipy is not installed")
        design = sample_design(random.Random(45))
        result, payload = run_explicit_vector_impact(
            design,
            config=ExplicitImpactFEMConfig(resolution=8, max_time_s=2.0e-5, dt_s=2.0e-6),
            rank=1,
        )
        self.assertGreater(result.solid_voxels, 0)
        self.assertGreater(result.impact_initial_ke_J, 20.0)
        self.assertGreaterEqual(result.impact_absorbed_J, 0.0)
        self.assertIn("history", payload)


if __name__ == "__main__":
    unittest.main()
