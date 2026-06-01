from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.mesh_quality import audit_stl_mesh


TETRA_STL = """solid tetra
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 1 0 0
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 0 0 1
    endloop
  endfacet
  facet normal 1 1 1
    outer loop
      vertex 1 0 0
      vertex 0 0 1
      vertex 0 1 0
    endloop
  endfacet
endsolid tetra
"""


class MeshQualityTests(unittest.TestCase):
    def test_closed_tetra_is_watertight_by_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tetra.stl"
            path.write_text(TETRA_STL, encoding="utf-8")
            quality = audit_stl_mesh(path)
            self.assertEqual(quality.facets, 4)
            self.assertEqual(quality.edge_count_not_two, 0)
            self.assertTrue(quality.watertight_by_edges)
            self.assertTrue(quality.within_envelope)


if __name__ == "__main__":
    unittest.main()
