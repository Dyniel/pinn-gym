"""Material/filament cards for the POLMI numerical lattice gym.

Cards are deliberately explicit: they describe the assumptions used by the
numerical oracles, not certified material truth. User-supplied cards become
calibrated only when their fields are fitted to measured data for a specific
printer, filament batch, orientation, and process window.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MaterialCard:
    material_name: str = "PA12 numerical default"
    process_name: str = "SLS/MJF-like"
    density_g_cm3: float = 1.01
    elastic_modulus_MPa: float = 1600.0
    compressive_yield_strength_MPa: float = 46.0
    compressive_plateau_strength_MPa: float = 46.0
    failure_strain: float = 0.82
    strain_rate_sensitivity: float = 0.0
    poisson_ratio: float = 0.39
    minimum_printable_feature_mm: float = 0.50
    printer_tolerance_mm: float = 0.30
    layer_height_mm: float = 0.10
    nozzle_or_laser_spot_mm: float = 0.10
    anisotropy_z_factor: float = 1.00
    temperature_condition: str = "room_temperature"
    source_or_calibration_note: str = "Numerical default; not certified material data."

    @property
    def density_g_per_mm3(self) -> float:
        return self.density_g_cm3 / 1000.0

    def strain_rate_factor(self, strain_rate_s: float, reference_rate_s: float = 1.0) -> float:
        if self.strain_rate_sensitivity == 0.0 or strain_rate_s <= 0.0:
            return 1.0
        ratio = max(strain_rate_s / max(reference_rate_s, 1e-12), 1e-12)
        factor = 1.0 + self.strain_rate_sensitivity * math.log10(ratio)
        return max(0.25, min(3.0, factor))

    def validate(self) -> list[str]:
        errors: list[str] = []
        positive = {
            "density_g_cm3": self.density_g_cm3,
            "elastic_modulus_MPa": self.elastic_modulus_MPa,
            "compressive_yield_strength_MPa": self.compressive_yield_strength_MPa,
            "compressive_plateau_strength_MPa": self.compressive_plateau_strength_MPa,
            "failure_strain": self.failure_strain,
            "minimum_printable_feature_mm": self.minimum_printable_feature_mm,
            "printer_tolerance_mm": self.printer_tolerance_mm,
            "layer_height_mm": self.layer_height_mm,
            "nozzle_or_laser_spot_mm": self.nozzle_or_laser_spot_mm,
            "anisotropy_z_factor": self.anisotropy_z_factor,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"{name} must be positive")
        if not 0.0 < self.poisson_ratio < 0.5:
            errors.append("poisson_ratio must be in (0, 0.5)")
        if self.minimum_printable_feature_mm < 0.05:
            errors.append("minimum_printable_feature_mm is unrealistically small")
        if self.anisotropy_z_factor > 2.0:
            errors.append("anisotropy_z_factor should be <= 2 for this reduced oracle")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def scaled(
        self,
        *,
        name_suffix: str,
        density_scale: float = 1.0,
        modulus_scale: float = 1.0,
        yield_scale: float = 1.0,
        plateau_scale: float = 1.0,
        failure_scale: float = 1.0,
        tolerance_delta_mm: float = 0.0,
    ) -> "MaterialCard":
        return replace(
            self,
            material_name=f"{self.material_name} {name_suffix}".strip(),
            density_g_cm3=self.density_g_cm3 * density_scale,
            elastic_modulus_MPa=self.elastic_modulus_MPa * modulus_scale,
            compressive_yield_strength_MPa=self.compressive_yield_strength_MPa * yield_scale,
            compressive_plateau_strength_MPa=self.compressive_plateau_strength_MPa * plateau_scale,
            failure_strain=self.failure_strain * failure_scale,
            printer_tolerance_mm=max(1e-6, self.printer_tolerance_mm + tolerance_delta_mm),
        )


def default_pa12_card() -> MaterialCard:
    return MaterialCard()


def preset_material_cards() -> dict[str, MaterialCard]:
    return {
        "pa12": default_pa12_card(),
        "pla": MaterialCard(
            material_name="PLA numerical placeholder",
            process_name="FDM-like",
            density_g_cm3=1.24,
            elastic_modulus_MPa=2500.0,
            compressive_yield_strength_MPa=55.0,
            compressive_plateau_strength_MPa=38.0,
            failure_strain=0.18,
            strain_rate_sensitivity=0.015,
            poisson_ratio=0.36,
            minimum_printable_feature_mm=0.45,
            printer_tolerance_mm=0.20,
            layer_height_mm=0.20,
            nozzle_or_laser_spot_mm=0.40,
            anisotropy_z_factor=0.72,
            source_or_calibration_note="Synthetic placeholder for transfer tests; calibrate before claims.",
        ),
        "petg": MaterialCard(
            material_name="PETG numerical placeholder",
            process_name="FDM-like",
            density_g_cm3=1.27,
            elastic_modulus_MPa=1700.0,
            compressive_yield_strength_MPa=45.0,
            compressive_plateau_strength_MPa=33.0,
            failure_strain=0.35,
            strain_rate_sensitivity=0.010,
            poisson_ratio=0.39,
            minimum_printable_feature_mm=0.45,
            printer_tolerance_mm=0.22,
            layer_height_mm=0.20,
            nozzle_or_laser_spot_mm=0.40,
            anisotropy_z_factor=0.78,
            source_or_calibration_note="Synthetic placeholder for transfer tests; calibrate before claims.",
        ),
        "tpu": MaterialCard(
            material_name="TPU numerical placeholder",
            process_name="FDM-like",
            density_g_cm3=1.20,
            elastic_modulus_MPa=80.0,
            compressive_yield_strength_MPa=8.0,
            compressive_plateau_strength_MPa=6.0,
            failure_strain=1.20,
            strain_rate_sensitivity=0.020,
            poisson_ratio=0.47,
            minimum_printable_feature_mm=0.60,
            printer_tolerance_mm=0.30,
            layer_height_mm=0.20,
            nozzle_or_laser_spot_mm=0.40,
            anisotropy_z_factor=0.82,
            source_or_calibration_note="Synthetic placeholder for transfer tests; calibrate before claims.",
        ),
        "pa_cf": MaterialCard(
            material_name="PA-CF numerical placeholder",
            process_name="FDM-like",
            density_g_cm3=1.15,
            elastic_modulus_MPa=3500.0,
            compressive_yield_strength_MPa=75.0,
            compressive_plateau_strength_MPa=55.0,
            failure_strain=0.35,
            strain_rate_sensitivity=0.010,
            poisson_ratio=0.34,
            minimum_printable_feature_mm=0.55,
            printer_tolerance_mm=0.25,
            layer_height_mm=0.20,
            nozzle_or_laser_spot_mm=0.40,
            anisotropy_z_factor=0.65,
            source_or_calibration_note="Synthetic placeholder for transfer tests; calibrate before claims.",
        ),
    }


_ALIASES = {
    "poisson": "poisson_ratio",
    "elastic_modulus_mpa": "elastic_modulus_MPa",
    "compressive_yield_strength_mpa": "compressive_yield_strength_MPa",
    "compressive_plateau_strength_mpa": "compressive_plateau_strength_MPa",
}


def _coerce_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    fields = set(MaterialCard.__dataclass_fields__)
    out: dict[str, Any] = {}
    for key, value in payload.items():
        canonical = _ALIASES.get(str(key), _ALIASES.get(str(key).lower(), str(key)))
        if canonical in fields:
            out[canonical] = value
    return out


def material_card_from_mapping(payload: dict[str, Any]) -> MaterialCard:
    card = MaterialCard(**_coerce_mapping(payload))
    errors = card.validate()
    if errors:
        raise ValueError("; ".join(errors))
    return card


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        try:
            parsed: Any = float(value)
        except ValueError:
            parsed = value
        payload[key.strip()] = parsed
    return payload


def load_material_card(path_or_preset: str | Path | None) -> MaterialCard:
    if path_or_preset is None or str(path_or_preset).strip() == "":
        return default_pa12_card()
    key = str(path_or_preset).strip()
    presets = preset_material_cards()
    if key in presets:
        return presets[key]
    path = Path(key)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _parse_simple_yaml(text)
    if not isinstance(payload, dict):
        raise ValueError(f"material card must be a mapping: {path}")
    return material_card_from_mapping(payload)


def mass_g_from_relative_density(relative_density: float, material: MaterialCard, envelope_mm: float) -> float:
    return relative_density * envelope_mm**3 * material.density_g_per_mm3
