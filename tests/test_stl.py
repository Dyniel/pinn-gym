from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.design_space import sample_design
from pinn_gym.core.mesh_quality import audit_stl_mesh
from pinn_gym.core.stl import export_design_stl


class StlTests(unittest.TestCase):
    def test_box_fallback_stl_is_written(self) -> None:
        design = sample_design(random.Random(5))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.stl"
            export_design_stl(design, path, backend="box")
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("solid"))
            self.assertIn("facet normal", text)
            self.assertIn("endsolid", text)

    def test_implicit_stl_is_closed_inside_envelope(self) -> None:
        try:
            import numpy  # noqa: F401
            import skimage  # noqa: F401
        except Exception:
            self.skipTest("implicit STL dependencies are not installed")
        design = sample_design(random.Random(8))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.stl"
            export_design_stl(design, path, backend="implicit", resolution=48)
            quality = audit_stl_mesh(path)
            self.assertTrue(quality.within_envelope)
            self.assertEqual(quality.edge_count_not_two, 0)

    def test_voxelized_implicit_stl_is_closed_inside_envelope(self) -> None:
        try:
            import numpy  # noqa: F401
            import skimage  # noqa: F401
        except Exception:
            self.skipTest("implicit STL dependencies are not installed")
        design = sample_design(random.Random(9))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.stl"
            export_design_stl(design, path, backend="voxel", resolution=48)
            quality = audit_stl_mesh(path)
            self.assertTrue(quality.within_envelope)
            self.assertEqual(quality.edge_count_not_two, 0)


if __name__ == "__main__":
    unittest.main()
