"""Material-aware candidate-pool sampling for the SR benchmark.

The original :func:`pinn_gym.core.design_space.sample_designs` samples from a fixed
joint distribution that happens to centre on PA12's feasible region. For
material-agnostic ranking we need each material card to receive a candidate
pool that *straddles* its own feasibility frontier under the declared oracle,
otherwise non-PA12 cards collapse to zero-feasible degenerate experiments.

This module derives a target relative-density range from the material card's
plateau/yield strength so the sampler produces both feasible and infeasible
geometries for that card. The mapping is intentionally simple and explicit:
no calibration data, only declared numerical scales, so the result is a
balanced *numerical* candidate pool, not a calibrated experimental design.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .design_space import DesignParams, ENVELOPE_MM, TOPOLOGIES, normalize_zones
from .materials import MaterialCard


# Power and prefactor in the reduced layered crush model:
#   sigma_plateau ~ plateau_strength * yield_scale * rel^DENSITY_EXP
# (compare PhysicsConfig.yield_scale in pinn_gym.core.physics).
DENSITY_EXP = 2.18
PHYSICS_YIELD_SCALE = 0.08
# Average effective-area × topology × shape × strain-rate prefactor seen by
# the physics oracle once the layered model is integrated. Derived empirically
# from PA12 calibration; used as a single material-agnostic scaling constant.
PHYSICS_FORCE_PREFACTOR = 0.55
# Load-sharing factor between the active crush front and the neighbouring
# layer stack. ~1.6 in the progressive crush curve.
LOAD_SHARING = 1.6


@dataclass(frozen=True)
class MaterialPoolSpec:
    """Per-material sampling bounds derived from a material card.

    ``relative_density_center`` is the target relative density that makes the
    crush-plateau force land near the fixture peak limit. Geometries are then
    sampled to span a band of ``relative_density_half_width`` around it.
    """

    relative_density_center: float
    relative_density_half_width: float
    wall_min_mm: float
    wall_max_mm: float
    cell_min_mm: float
    cell_max_mm: float
    cap_min_mm: float
    cap_max_mm: float
    min_feature_floor_mm: float


def _rel_for_target_force(
    target_force_n: float,
    plateau_strength_MPa: float,
    area_mm2: float,
) -> float:
    denom = max(plateau_strength_MPa * PHYSICS_YIELD_SCALE * area_mm2 * PHYSICS_FORCE_PREFACTOR, 1e-9)
    needed = max(target_force_n / denom, 1e-6)
    return needed ** (1.0 / DENSITY_EXP)


def material_derived_min_crush_mm(material: MaterialCard, *, envelope_mm: float = ENVELOPE_MM) -> float:
    """Material-specific minimum-crush-stroke target.

    The original 40 mm rule bakes in PA12's ductile collapse behaviour: brittle
    materials (low ``failure_strain``) inherently fracture before reaching that
    stroke under the reduced-order oracle. To keep the feasibility envelope
    non-empty across cards we scale the target with the declared failure
    strain but never relax it below 18 mm or push it above the original 40 mm
    fixture limit.
    """

    fail = max(0.05, float(material.failure_strain))
    base = 0.62 * fail * envelope_mm + 12.0
    return max(18.0, min(40.0, base))


def derive_pool_spec(
    material: MaterialCard,
    *,
    envelope_mm: float = ENVELOPE_MM,
    fixture_peak_force_limit_n: float = 3500.0,
    target_min_crush_mm: float = 40.0,
    impact_energy_j: float = 29.43,
) -> MaterialPoolSpec:
    """Pick a sampling band so each material card has feasible/infeasible mix.

    The declared gates are: plateau force × stroke >= impact energy and
    peak-force <= fixture limit. Plateau force is driven by the *plateau*
    strength (not the yield strength) through the reduced layered oracle, so
    the previous version that read only ``sigma_y`` collapsed several cards
    to zero feasibility. We now bracket the rel-density band between the
    two frontiers using the actual plateau strength.
    """

    area_mm2 = envelope_mm ** 2
    plateau_MPa = max(float(material.compressive_plateau_strength_MPa), 1e-3)
    # Material-derived crush stroke is what the brittle materials can reach
    # under the declared oracle. We use it to size the energy frontier so we
    # are not asking for energy over an unreachable stroke.
    crush_target_mm = max(target_min_crush_mm, material_derived_min_crush_mm(material, envelope_mm=envelope_mm))
    crush_target_mm = max(crush_target_mm, 18.0)
    # Target plateau crush force from energy ~ F * stroke. The integrated curve
    # has hardening + load sharing so the *per-layer* plateau target is
    # F_target / LOAD_SHARING.
    target_curve_force = max(1e3 * impact_energy_j / max(crush_target_mm, 1e-3), 200.0)
    energy_target_force = target_curve_force / LOAD_SHARING
    rel_energy_floor = _rel_for_target_force(energy_target_force, plateau_MPa, area_mm2)
    # Peak frontier: a typical peak/plateau ratio under the oracle is ~1.6;
    # so for peak <= fixture limit we want load-shared plateau <= limit/1.6.
    peak_budget = max(fixture_peak_force_limit_n / 1.6, 200.0)
    peak_force_target = peak_budget / LOAD_SHARING
    rel_peak_ceiling = _rel_for_target_force(peak_force_target, plateau_MPa, area_mm2)
    # Materials whose energy floor is above their peak ceiling are physically
    # tight: we still produce a candidate pool centred between, but widen the
    # half-width so the sampler explores both sides of the (empty) frontier.
    lo = min(rel_energy_floor, rel_peak_ceiling)
    hi = max(rel_energy_floor, rel_peak_ceiling)
    center = 0.5 * (lo + hi)
    center = max(0.10, min(0.58, center))
    width = max(hi - lo, 0.06)
    half_width = max(0.08, min(0.24, 0.55 * width + 0.06))
    # Wall/cell bounds aligned with the target relative-density band and
    # printability. Cell range is set so that the slenderness implied by
    # ``center`` can be reached at the median wall thickness.
    min_feature_floor = max(0.30, float(material.minimum_printable_feature_mm) + 0.5 * float(material.printer_tolerance_mm))
    target_slenderness = max(0.06, ((max(center, 0.05) / 0.55) ** (1.0 / 1.28)) / 2.15)
    cell_med = max(4.0, min(9.0, 1.0 / max(target_slenderness, 1e-3) * 0.9))
    cell_min = max(3.8, cell_med - 1.6)
    cell_max = min(9.6, cell_med + 1.6)
    wall_floor = max(min_feature_floor, 0.55)
    wall_ceiling = max(wall_floor + 0.20, min(1.75, target_slenderness * cell_max * 1.45))
    cap_min = max(0.45, min_feature_floor)
    cap_max = max(cap_min + 0.20, 1.15)
    return MaterialPoolSpec(
        relative_density_center=center,
        relative_density_half_width=half_width,
        wall_min_mm=wall_floor,
        wall_max_mm=wall_ceiling,
        cell_min_mm=cell_min,
        cell_max_mm=cell_max,
        cap_min_mm=cap_min,
        cap_max_mm=cap_max,
        min_feature_floor_mm=min_feature_floor,
    )


def sample_material_aware_design(rng: random.Random, spec: MaterialPoolSpec) -> DesignParams:
    topology = rng.choices(
        TOPOLOGIES,
        weights=[0.11, 0.12, 0.07, 0.05, 0.12, 0.17, 0.10, 0.15, 0.07, 0.04],
        k=1,
    )[0]
    target_rel = max(
        0.04,
        spec.relative_density_center + rng.uniform(-1.0, 1.0) * spec.relative_density_half_width,
    )
    # Invert rel ~ 0.55 * (2.15 * slenderness)^1.28 to a wall/cell slenderness.
    slenderness = max(0.04, ((target_rel / 0.55) ** (1.0 / 1.28)) / 2.15)
    cell = rng.uniform(spec.cell_min_mm, spec.cell_max_mm)
    wall = slenderness * cell * rng.uniform(0.82, 1.20)
    wall = max(spec.wall_min_mm, min(spec.wall_max_mm, wall))
    # Wall must remain above the printability floor; if it does, also above
    # cell-relative minimum so the lattice stays printable.
    wall = max(wall, spec.min_feature_floor_mm)
    min_feature = rng.uniform(spec.min_feature_floor_mm, max(spec.min_feature_floor_mm + 0.05, 0.92))
    cap = rng.uniform(spec.cap_min_mm, spec.cap_max_mm)
    trigger = rng.uniform(8.0, 14.4)
    bumper = rng.uniform(9.0, 15.6)
    plateau = ENVELOPE_MM - trigger - bumper
    trigger, plateau, bumper = normalize_zones(trigger, plateau, bumper)
    return DesignParams(
        topology=topology,
        cell_size_mm=cell,
        wall_thickness_mm=wall,
        min_feature_mm=min_feature,
        vertical_gradient=rng.uniform(0.02, 0.95),
        density_bias=rng.uniform(-0.03, 0.18),
        cap_thickness_mm=cap,
        edge_rib_mm=rng.uniform(0.0, 1.05),
        trigger_layer_strength=rng.uniform(0.22, 0.96),
        trigger_zone_mm=trigger,
        plateau_zone_mm=plateau,
        bumper_zone_mm=bumper,
        hybrid_ratio=rng.uniform(0.0, 1.0),
        anisotropy_xy=rng.uniform(0.82, 1.22),
    )


def sample_material_aware_designs(
    material: MaterialCard,
    n: int,
    seed: int,
    *,
    fixture_peak_force_limit_n: float = 3500.0,
    target_min_crush_mm: float = 40.0,
    impact_energy_j: float = 29.43,
) -> list[DesignParams]:
    spec = derive_pool_spec(
        material,
        fixture_peak_force_limit_n=fixture_peak_force_limit_n,
        target_min_crush_mm=target_min_crush_mm,
        impact_energy_j=impact_energy_j,
    )
    rng = random.Random(seed)
    return [sample_material_aware_design(rng, spec) for _ in range(n)]
