"""Dimensionless scales and material descriptors for the SR PINN.

This module implements the Buckingham-Pi nondimensionalization used by the
Scientific Reports material-card-conditioned PINN. The idea is that the
force-displacement response of a printable lattice under quasi-static
compression can be expressed in dimensionless form:

    f_hat(eps; geom*, mat*) = F / (sigma_y * A_env)

where ``eps = u / L_env`` is engineering strain, ``A_env = L_env**2`` is the
nominal cross-section of the cubic envelope and ``sigma_y`` is the
material-card compressive yield strength. Geometry features are made
dimensionless with the envelope, material features with a chosen reference
card so that scalars are O(1) across material families.

Once a network learns ``f_hat``, predictions for a new material card are
obtained by multiplying with ``sigma_y * A_env`` and stretching the strain
axis back to physical displacements. This is the property that the SR claim
calls "material-agnostic by construction": the architecture does not have to
re-train when ``sigma_y`` changes, only the I/O scales do.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .design_space import ENVELOPE_MM
from .materials import MaterialCard, default_pa12_card


REF_CARD = default_pa12_card()
REF_STRAIN_RATE_S = 100.0


@dataclass(frozen=True)
class DimensionlessScales:
    """Per-material I/O scales for the dimensionless PINN."""

    envelope_mm: float
    envelope_area_mm2: float
    yield_stress_MPa: float
    elastic_modulus_MPa: float
    force_scale_N: float
    energy_scale_J: float

    @property
    def envelope_m(self) -> float:
        return self.envelope_mm / 1000.0

    def force_to_dimensionless(self, force_n: float) -> float:
        return float(force_n) / max(self.force_scale_N, 1e-9)

    def dimensionless_to_force(self, f_hat: float) -> float:
        return float(f_hat) * self.force_scale_N

    def displacement_to_strain(self, displacement_mm: float) -> float:
        return float(displacement_mm) / max(self.envelope_mm, 1e-9)

    def strain_to_displacement(self, strain: float) -> float:
        return float(strain) * self.envelope_mm

    def energy_to_dimensionless(self, energy_j: float) -> float:
        return float(energy_j) / max(self.energy_scale_J, 1e-9)

    def dimensionless_to_energy(self, energy_hat: float) -> float:
        return float(energy_hat) * self.energy_scale_J

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def scales_for_material(material: MaterialCard, envelope_mm: float = ENVELOPE_MM) -> DimensionlessScales:
    area_mm2 = envelope_mm ** 2
    sigma_y_MPa = float(material.compressive_yield_strength_MPa)
    force_scale_N = sigma_y_MPa * area_mm2  # 1 MPa = 1 N/mm^2
    energy_scale_J = force_scale_N * (envelope_mm / 1000.0)
    return DimensionlessScales(
        envelope_mm=float(envelope_mm),
        envelope_area_mm2=area_mm2,
        yield_stress_MPa=sigma_y_MPa,
        elastic_modulus_MPa=float(material.elastic_modulus_MPa),
        force_scale_N=force_scale_N,
        energy_scale_J=energy_scale_J,
    )


MATERIAL_DIM_FIELDS: tuple[str, ...] = (
    "stiffness_ratio",
    "yield_ratio",
    "plateau_yield_ratio",
    "failure_strain_ratio",
    "density_ratio",
    "min_feature_ratio",
    "tolerance_ratio",
    "anisotropy_z",
    "strain_rate_factor",
    "yield_to_E_ratio",
)


def material_dimensionless_features(
    material: MaterialCard,
    *,
    reference: MaterialCard = REF_CARD,
    envelope_mm: float = ENVELOPE_MM,
) -> dict[str, float]:
    """Return the dimensionless material descriptors fed to the PINN.

    All ratios are nondimensional, taken against the reference card so that
    PA12-like cards land near 1.0 and unusual cards (TPU, PA-CF) reveal
    themselves as numerical outliers in feature space.
    """

    def safe(value: float) -> float:
        v = float(value)
        return v if math.isfinite(v) else 0.0

    yield_ratio = safe(material.compressive_yield_strength_MPa) / max(reference.compressive_yield_strength_MPa, 1e-9)
    plateau_yield_ratio = safe(material.compressive_plateau_strength_MPa) / max(material.compressive_yield_strength_MPa, 1e-9)
    stiffness_ratio = safe(material.elastic_modulus_MPa) / max(reference.elastic_modulus_MPa, 1e-9)
    failure_strain_ratio = safe(material.failure_strain) / max(reference.failure_strain, 1e-9)
    density_ratio = safe(material.density_g_cm3) / max(reference.density_g_cm3, 1e-9)
    min_feature_ratio = safe(material.minimum_printable_feature_mm) / max(envelope_mm, 1e-9)
    tolerance_ratio = safe(material.printer_tolerance_mm) / max(envelope_mm, 1e-9)
    strain_rate_factor = material.strain_rate_factor(REF_STRAIN_RATE_S)
    yield_to_E = safe(material.compressive_yield_strength_MPa) / max(material.elastic_modulus_MPa, 1e-9)
    return {
        "stiffness_ratio": stiffness_ratio,
        "yield_ratio": yield_ratio,
        "plateau_yield_ratio": plateau_yield_ratio,
        "failure_strain_ratio": failure_strain_ratio,
        "density_ratio": density_ratio,
        "min_feature_ratio": min_feature_ratio,
        "tolerance_ratio": tolerance_ratio,
        "anisotropy_z": safe(material.anisotropy_z_factor),
        "strain_rate_factor": float(strain_rate_factor),
        "yield_to_E_ratio": yield_to_E,
    }


def material_feature_vector(material: MaterialCard, envelope_mm: float = ENVELOPE_MM) -> list[float]:
    feats = material_dimensionless_features(material, envelope_mm=envelope_mm)
    return [feats[name] for name in MATERIAL_DIM_FIELDS]


def dimensionless_curve(
    displacement_mm: list[float],
    force_n: list[float],
    scales: DimensionlessScales,
) -> tuple[list[float], list[float]]:
    """Convert an oracle (displacement[mm], force[N]) curve to (strain, f_hat)."""

    strain = [scales.displacement_to_strain(d) for d in displacement_mm]
    f_hat = [scales.force_to_dimensionless(f) for f in force_n]
    return strain, f_hat


def physical_curve(
    strain: list[float],
    f_hat: list[float],
    scales: DimensionlessScales,
) -> tuple[list[float], list[float]]:
    """Inverse of :func:`dimensionless_curve`."""

    displacement = [scales.strain_to_displacement(e) for e in strain]
    force = [scales.dimensionless_to_force(f) for f in f_hat]
    return displacement, force
