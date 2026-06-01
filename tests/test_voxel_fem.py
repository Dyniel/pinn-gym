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
from pinn_gym.core.voxel_fem import VoxelFEMConfig, run_scalar_voxel_fem


class VoxelFEMTests(unittest.TestCase):
    def test_scalar_voxel_fem_smoke(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("scipy is not installed")
        design = sample_design(random.Random(12))
        result = run_scalar_voxel_fem(design, config=VoxelFEMConfig(resolution=12, cg_maxiter=200))
        self.assertGreater(result.solid_voxels, 0)
        self.assertGreaterEqual(result.voxel_stiffness_N_per_mm, 0.0)
        self.assertGreater(result.top_contact_voxels, 0)
        self.assertGreater(result.bottom_contact_voxels, 0)


if __name__ == "__main__":
    unittest.main()
