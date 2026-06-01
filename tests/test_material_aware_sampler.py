from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.design_space import TOPOLOGIES, estimate_relative_density
from pinn_gym.core.material_aware_sampler import (
    derive_pool_spec,
    sample_material_aware_designs,
)
from pinn_gym.core.materials import preset_material_cards


class MaterialAwareSamplerTests(unittest.TestCase):
    def test_pool_spec_centers_shift_with_yield(self) -> None:
        cards = preset_material_cards()
        tpu_spec = derive_pool_spec(cards["tpu"])
        pa_cf_spec = derive_pool_spec(cards["pa_cf"])
        # Softer material needs denser lattice to reach impact energy
        self.assertGreater(tpu_spec.relative_density_center, pa_cf_spec.relative_density_center)
        # Bounds are sane
        for spec in (tpu_spec, pa_cf_spec):
            self.assertLess(spec.relative_density_center, 0.70)
            self.assertGreater(spec.relative_density_center, 0.05)
            self.assertGreater(spec.wall_max_mm, spec.wall_min_mm)

    def test_sampled_designs_span_density_band(self) -> None:
        cards = preset_material_cards()
        for preset, card in cards.items():
            designs = sample_material_aware_designs(card, n=80, seed=11)
            self.assertEqual(len(designs), 80)
            self.assertTrue(all(d.topology in TOPOLOGIES for d in designs))
            rels = [estimate_relative_density(d) for d in designs]
            # the actual rel-density spread must be physically meaningful so
            # the pool can produce both feasible and infeasible designs
            spread = max(rels) - min(rels)
            self.assertGreater(spread, 0.06, f"too narrow rel-density spread for {preset}: {spread}")
            # both sides of the median should be populated
            median = sorted(rels)[len(rels) // 2]
            below = sum(1 for r in rels if r < median)
            above = sum(1 for r in rels if r > median)
            self.assertGreater(below, 5, f"too few low-density samples for {preset}")
            self.assertGreater(above, 5, f"too few high-density samples for {preset}")

    def test_seed_reproducible(self) -> None:
        cards = preset_material_cards()
        a = sample_material_aware_designs(cards["pa12"], n=12, seed=42)
        b = sample_material_aware_designs(cards["pa12"], n=12, seed=42)
        self.assertEqual([d.to_row() for d in a], [d.to_row() for d in b])


if __name__ == "__main__":
    unittest.main()
