from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_gym.core.materials import MaterialCard, load_material_card, preset_material_cards


class MaterialCardTests(unittest.TestCase):
    def test_presets_are_valid(self) -> None:
        presets = preset_material_cards()
        self.assertIn("pa12", presets)
        self.assertIn("pla", presets)
        for card in presets.values():
            self.assertEqual(card.validate(), [])
            self.assertGreater(card.density_g_per_mm3, 0.0)

    def test_load_user_json_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "card.json"
            payload = MaterialCard(material_name="test filament", density_g_cm3=1.17).to_dict()
            path.write_text(json.dumps(payload), encoding="utf-8")
            card = load_material_card(path)
            self.assertEqual(card.material_name, "test filament")
            self.assertAlmostEqual(card.density_g_cm3, 1.17)


if __name__ == "__main__":
    unittest.main()
