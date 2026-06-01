from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.dimensionless import (
    MATERIAL_DIM_FIELDS,
    dimensionless_curve,
    material_dimensionless_features,
    material_feature_vector,
    physical_curve,
    scales_for_material,
)
from pinn_gym.core.materials import preset_material_cards


class DimensionlessTests(unittest.TestCase):
    def test_force_scale_matches_yield_times_area(self) -> None:
        cards = preset_material_cards()
        for name, card in cards.items():
            scales = scales_for_material(card)
            self.assertAlmostEqual(
                scales.force_scale_N,
                card.compressive_yield_strength_MPa * 50.0 ** 2,
                msg=f"force scale mismatch for {name}",
            )
            # energy scale is force_scale * envelope[m]
            self.assertAlmostEqual(scales.energy_scale_J, scales.force_scale_N * 0.050, places=6)

    def test_roundtrip_curve(self) -> None:
        card = preset_material_cards()["pa12"]
        scales = scales_for_material(card)
        disp = [0.0, 10.0, 25.0, 40.0, 50.0]
        force = [0.0, 1200.0, 2200.0, 2800.0, 3500.0]
        strain, f_hat = dimensionless_curve(disp, force, scales)
        d2, f2 = physical_curve(strain, f_hat, scales)
        for a, b in zip(disp, d2):
            self.assertAlmostEqual(a, b, places=8)
        for a, b in zip(force, f2):
            self.assertAlmostEqual(a, b, places=6)

    def test_dimensionless_features_have_expected_fields(self) -> None:
        card = preset_material_cards()["tpu"]
        feats = material_dimensionless_features(card)
        self.assertEqual(set(feats.keys()), set(MATERIAL_DIM_FIELDS))
        vec = material_feature_vector(card)
        self.assertEqual(len(vec), len(MATERIAL_DIM_FIELDS))
        # TPU's yield ratio should be much less than 1.0 (softer than PA12)
        self.assertLess(feats["yield_ratio"], 0.5)
        # PA-CF's stiffness ratio is greater than 1.0
        pa_cf = material_dimensionless_features(preset_material_cards()["pa_cf"])
        self.assertGreater(pa_cf["stiffness_ratio"], 1.0)
        # All values are finite
        for value in vec:
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
