"""Physics-based validation for POLMI candidates.

This module is intentionally heavier and more conservative than the bootstrap
surrogate in :mod:`pinn_gym.core.design_space`. It implements a reduced-order layered
crush FEM: the 50 mm specimen is represented as nonlinear elasto-plastic layers
in series under displacement control. It is not a replacement for Abaqus/LS-DYNA
or a calibrated material card, but it gives the pipeline a deterministic
mechanics gate with explicit PA12 assumptions, force-displacement curves, energy
integration, and failure margins.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

from .design_space import (
    ENVELOPE_MM,
    DesignParams,
    estimate_relative_density,
    integrate_curve_energy_j,
)
from .materials import MaterialCard, default_pa12_card, mass_g_from_relative_density
from .metrics import force_curve_metrics
from .mesh_quality import audit_stl_mesh
from .paths import ensure_dir


IMPACT_MASS_KG = 6.0
IMPACT_HEIGHT_M = 0.5
GRAVITY_M_S2 = 9.81
IMPACT_ENERGY_J = IMPACT_MASS_KG * GRAVITY_M_S2 * IMPACT_HEIGHT_M
IMPACT_VELOCITY_M_S = math.sqrt(2.0 * GRAVITY_M_S2 * IMPACT_HEIGHT_M)

PA12_ELASTIC_MODULUS_MPA = 1600.0
PA12_YIELD_STRESS_MPA = 46.0
PA12_POISSON = 0.39


@dataclass(frozen=True)
class PhysicsConfig:
    layers: int = 96
    steps: int = 360
    max_displacement_mm: float = ENVELOPE_MM
    strain_rate_s: float = IMPACT_VELOCITY_M_S / (ENVELOPE_MM / 1000.0)
    dynamic_amplification: float = 1.16
    yield_scale: float = 0.08
    min_effective_area_fraction: float = 0.16
    max_effective_area_fraction: float = 0.62
    fracture_strain_base: float = 0.82
    fixture_peak_force_limit_n: float = 3500.0
    target_min_crush_mm: float = 40.0
    material: MaterialCard = field(default_factory=default_pa12_card)


@dataclass(frozen=True)
class PhysicsResult:
    rank: int
    topology: str
    material_name: str
    mass_g: float
    relative_density: float
    physics_energy_50mm_J: float
    physics_energy_usable_J: float
    physics_sea_J_g: float
    physics_mean_crushing_force_N: float
    physics_cfe: float
    physics_plateau_cv: float
    physics_peak_force_N: float
    physics_plateau_force_N: float
    physics_collapse_mm: float
    physics_impact_stop_mm: float
    physics_failure_risk: float
    physics_energy_margin_J: float
    physics_energy_pass: bool
    physics_peak_force_pass: bool
    physics_crush_distance_pass: bool
    physics_survives_gate: bool
    physics_score: float
    stl_watertight_by_edges: bool | None = None
    stl_within_envelope: bool | None = None
    stl_open_edges: int | None = None
    stl_edge_count_not_two: int | None = None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ImpactConfig:
    dt_s: float = 2.0e-5
    max_time_s: float = 0.035
    damping_n_s_per_m: float = 18.0
    indenter_mass_kg: float = IMPACT_MASS_KG
    initial_velocity_m_s: float = IMPACT_VELOCITY_M_S
    gravity_m_s2: float = GRAVITY_M_S2
    stop_velocity_m_s: float = 0.02


@dataclass(frozen=True)
class ImpactResult:
    rank: int
    topology: str
    material_name: str
    mass_g: float
    impact_initial_ke_J: float
    impact_absorbed_J: float
    impact_energy_margin_J: float
    impact_sea_J_g: float
    impact_mean_crushing_force_N: float
    impact_cfe: float
    impact_peak_force_N: float
    impact_max_displacement_mm: float
    impact_stop_time_ms: float
    impact_residual_velocity_m_s: float
    impact_energy_balance_error_J: float
    impact_energy_pass: bool
    impact_peak_force_pass: bool
    impact_crush_distance_pass: bool
    impact_survives: bool
    impact_failure_risk: float
    impact_score: float

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Layer:
    height_mm: float
    rel_density: float
    stiffness_n_per_mm: float
    plateau_force_n: float
    dense_force_n: float
    yield_disp_mm: float
    densification_disp_mm: float
    hardening: float
    fracture_strain: float


def _topology_physics_factors(topology: str) -> tuple[float, float, float, float]:
    # plateau, stiffness, progressivity, fracture penalty
    table = {
        "gyroid": (1.00, 1.00, 1.08, 0.96),
        "diamond": (1.06, 1.08, 0.98, 0.98),
        "octet": (1.16, 1.24, 0.88, 1.08),
        "bcc": (0.92, 0.92, 0.84, 1.10),
        "hybrid": (1.04, 1.06, 1.14, 0.94),
        "schwarz_p": (0.98, 0.96, 1.22, 0.90),
        "diamond_graded": (1.03, 1.02, 1.16, 0.92),
        "gyroid_diamond_hybrid": (1.02, 0.98, 1.24, 0.90),
        "bccz_graded": (0.96, 0.94, 1.18, 0.96),
        "ot_like": (1.00, 1.00, 1.10, 0.98),
    }
    return table.get(topology, table["gyroid"])


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _layer_relative_density(params: DesignParams, z01: float, base_rel: float) -> float:
    # Positive gradient now means soft trigger at the impact side and denser
    # support-side bumper, matching the progressive-crush generator.
    trigger_start = 1.0 - params.trigger_zone_mm / ENVELOPE_MM
    bumper_end = params.bumper_zone_mm / ENVELOPE_MM
    gradient_multiplier = 1.0 + 0.62 * params.vertical_gradient * (0.5 - z01)
    trigger_softening = 1.0 - 0.28 * params.trigger_layer_strength * _clamp((z01 - trigger_start) / max(1e-6, 1.0 - trigger_start), 0.0, 1.0)
    bumper_boost = 1.0 + 0.18 * params.vertical_gradient * _clamp((bumper_end - z01) / max(1e-6, bumper_end), 0.0, 1.0)
    bottom_cap_bonus = 0.20 * (params.cap_thickness_mm / ENVELOPE_MM) if z01 < 0.04 else 0.0
    top_cap_bonus = 0.08 * (params.cap_thickness_mm / ENVELOPE_MM) * (1.0 - params.trigger_layer_strength) if z01 > 0.96 else 0.0
    return _clamp(base_rel * gradient_multiplier * trigger_softening * bumper_boost + bottom_cap_bonus + top_cap_bonus, 0.035, 0.75)


def build_layer_stack(params: DesignParams, config: PhysicsConfig) -> list[_Layer]:
    layers = max(8, int(config.layers))
    height = ENVELOPE_MM / layers
    base_rel = estimate_relative_density(params)
    plateau_factor, stiffness_factor, progressivity_factor, fracture_penalty = _topology_physics_factors(params.topology)
    material = config.material
    rate_factor = config.dynamic_amplification * material.strain_rate_factor(config.strain_rate_s)
    z_factor = _clamp(material.anisotropy_z_factor, 0.20, 2.0)
    fracture_base = _clamp(
        config.fracture_strain_base * material.failure_strain / max(default_pa12_card().failure_strain, 1e-12),
        0.18,
        1.60,
    )
    layers_out: list[_Layer] = []
    area_mm2 = ENVELOPE_MM * ENVELOPE_MM
    for i in range(layers):
        z01 = (i + 0.5) / layers
        rel = _layer_relative_density(params, z01, base_rel)
        slenderness = params.wall_thickness_mm / max(params.cell_size_mm, 1e-6)
        effective_area = _clamp(
            0.13 + 1.58 * rel + 0.16 * min(1.0, params.edge_rib_mm) + 0.45 * slenderness,
            config.min_effective_area_fraction,
            config.max_effective_area_fraction,
        )
        e_eff = material.elastic_modulus_MPa * z_factor * stiffness_factor * (0.22 * rel + 1.35 * rel * rel)
        stiffness = max(1.0, e_eff * area_mm2 * effective_area / height)
        sigma_plateau = (
            material.compressive_plateau_strength_MPa
            * config.yield_scale
            * plateau_factor
            * rate_factor
            * z_factor
            * (rel ** 2.18)
            * (0.78 + 0.42 * _clamp(slenderness / 0.15, 0.0, 1.4))
        )
        plateau_force = max(35.0, sigma_plateau * area_mm2 * effective_area)
        densification_strain = _clamp(
            (0.78 - 1.22 * rel) * progressivity_factor
            + 0.08 * params.trigger_layer_strength
            + 0.04 * _clamp(params.plateau_zone_mm / 28.0, 0.0, 1.4),
            0.32,
            0.82,
        )
        densification_disp = densification_strain * height
        yield_disp = min(max(plateau_force / stiffness, 0.002 * height), 0.22 * densification_disp)
        hardening = _clamp(0.10 + 0.42 * rel + 0.08 * max(0.0, -params.vertical_gradient), 0.08, 0.46)
        dense_force = plateau_force * (1.0 + hardening)
        fracture_strain = _clamp(
            fracture_base
            - 0.26 * max(0.0, rel - 0.32)
            - 0.16 * max(0.0, material.minimum_printable_feature_mm + 0.08 - params.min_feature_mm)
            - 0.06 * max(0.0, material.minimum_printable_feature_mm - params.wall_thickness_mm)
            - 0.08 * max(0.0, params.vertical_gradient - 0.75)
            - 0.06 * fracture_penalty,
            0.42,
            0.92,
        )
        layers_out.append(
            _Layer(
                height_mm=height,
                rel_density=rel,
                stiffness_n_per_mm=stiffness,
                plateau_force_n=plateau_force,
                dense_force_n=dense_force,
                yield_disp_mm=yield_disp,
                densification_disp_mm=densification_disp,
                hardening=hardening,
                fracture_strain=fracture_strain,
            )
        )
    return layers_out


def _material_mass_g(params: DesignParams, material: MaterialCard) -> float:
    return mass_g_from_relative_density(estimate_relative_density(params), material, ENVELOPE_MM)


def _layer_disp_for_force(force_n: float, layer: _Layer) -> float:
    if force_n <= layer.plateau_force_n:
        return force_n / layer.stiffness_n_per_mm
    crush_span = max(1e-9, layer.densification_disp_mm - layer.yield_disp_mm)
    if force_n <= layer.dense_force_n:
        x = _clamp((force_n / layer.plateau_force_n - 1.0) / max(layer.hardening, 1e-9), 0.0, 1.0)
        return layer.yield_disp_mm + math.sqrt(x) * crush_span
    dense_stiffness = layer.stiffness_n_per_mm * (2.8 + 9.0 * layer.rel_density)
    return min(0.97 * layer.height_mm, layer.densification_disp_mm + (force_n - layer.dense_force_n) / dense_stiffness)


def _stack_disp_for_force(force_n: float, layers: Iterable[_Layer]) -> float:
    return sum(_layer_disp_for_force(force_n, layer) for layer in layers)


def _force_for_stack_disp(displacement_mm: float, layers: list[_Layer]) -> float:
    if displacement_mm <= 0.0:
        return 0.0
    low = 0.0
    high = max(layer.dense_force_n for layer in layers) * 1.8
    while _stack_disp_for_force(high, layers) < displacement_mm and high < 250_000.0:
        high *= 1.6
    for _ in range(72):
        mid = 0.5 * (low + high)
        if _stack_disp_for_force(mid, layers) < displacement_mm:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _layer_strains(force_n: float, layers: list[_Layer]) -> list[float]:
    return [_layer_disp_for_force(force_n, layer) / layer.height_mm for layer in layers]


def _progressive_layer_capacity(layer: _Layer) -> float:
    strain_capacity = max(layer.densification_disp_mm / max(layer.height_mm, 1e-9), 1.08 * layer.fracture_strain)
    return layer.height_mm * _clamp(strain_capacity, 0.34, 0.92)


def _progressive_layer_force(params: DesignParams, layer: _Layer, local_progress: float, compacted_fraction: float) -> float:
    plateau_factor, _, progressivity_factor, _ = _topology_physics_factors(params.topology)
    slenderness = params.wall_thickness_mm / max(params.cell_size_mm, 1e-6)
    load_sharing = _clamp(
        1.55
        + 0.58 * _clamp(slenderness / 0.20, 0.0, 1.8)
        + 0.24 * params.edge_rib_mm
        + 0.16 * _clamp(progressivity_factor - 1.0, -0.4, 0.4)
        + 0.10 * _clamp(plateau_factor - 1.0, -0.3, 0.3),
        1.35,
        2.65,
    )
    ripple = 0.08 * math.sin(math.pi * _clamp(local_progress, 0.0, 1.0))
    hardening = 1.0 + 0.18 * local_progress + ripple + 0.08 * compacted_fraction
    return max(25.0, layer.plateau_force_n * load_sharing * hardening)


def _progressive_crush_curve(params: DesignParams, layers: list[_Layer], config: PhysicsConfig) -> tuple[list[float], list[float], float]:
    """Force law for a crush front propagating from impact side to base."""

    ordered = list(reversed(layers))  # z=1 is the trigger/impact side
    capacities = [_progressive_layer_capacity(layer) for layer in ordered]
    progressive_capacity = _clamp(sum(capacities), 0.18 * config.max_displacement_mm, 0.96 * config.max_displacement_mm)
    cumulative: list[float] = []
    total = 0.0
    for capacity in capacities:
        total += capacity
        cumulative.append(total)

    steps = max(8, int(config.steps))
    disp = [config.max_displacement_mm * i / (steps - 1) for i in range(steps)]
    force: list[float] = []
    first_force = _progressive_layer_force(params, ordered[0], 0.0, 0.0) if ordered else 0.0
    for d in disp:
        if d <= 0.0 or not ordered:
            force.append(0.0)
            continue
        if d <= progressive_capacity:
            idx = 0
            while idx < len(cumulative) - 1 and cumulative[idx] < d:
                idx += 1
            previous = 0.0 if idx == 0 else cumulative[idx - 1]
            local_progress = _clamp((d - previous) / max(capacities[idx], 1e-9), 0.0, 1.0)
            compacted_fraction = idx / max(1, len(ordered) - 1)
            f = _progressive_layer_force(params, ordered[idx], local_progress, compacted_fraction)
            if d < 1.25:
                f *= _clamp(d / 1.25, 0.0, 1.0)
            elif idx == 0:
                f = 0.68 * f + 0.32 * first_force
        else:
            over = d - progressive_capacity
            remaining = max(1.0, config.max_displacement_mm - progressive_capacity)
            compacted_forces = [_progressive_layer_force(params, layer, 1.0, 1.0) for layer in ordered]
            base_dense = max(compacted_forces) if compacted_forces else 0.0
            rel = estimate_relative_density(params)
            dense_stiffness = base_dense * (1.8 + 7.5 * rel + 0.8 * params.edge_rib_mm)
            f = base_dense * (1.0 + 0.12 * rel) + dense_stiffness * (over / remaining) ** 2.4
        force.append(max(0.0, f))
    return disp, force, progressive_capacity


def run_layered_crush_fem(params: DesignParams, config: PhysicsConfig | None = None) -> dict[str, object]:
    config = config or PhysicsConfig()
    layers = build_layer_stack(params, config)
    disp, force, collapse_mm = _progressive_crush_curve(params, layers, config)
    energy_50 = integrate_curve_energy_j(disp, force)
    usable_pairs = [(d, f) for d, f in zip(disp, force) if d <= collapse_mm]
    usable_disp = [x[0] for x in usable_pairs]
    usable_force = [x[1] for x in usable_pairs]
    energy_usable = integrate_curve_energy_j(usable_disp, usable_force) if len(usable_disp) >= 2 else 0.0
    impact_stop_mm = collapse_mm
    usable_cumulative = [0.0]
    for d0, d1, f0, f1 in zip(usable_disp[:-1], usable_disp[1:], usable_force[:-1], usable_force[1:]):
        usable_cumulative.append(usable_cumulative[-1] + 0.5 * (max(0.0, f0) + max(0.0, f1)) * max(0.0, d1 - d0) / 1000.0)
    for i in range(1, len(usable_cumulative)):
        if usable_cumulative[i] >= IMPACT_ENERGY_J:
            e0, e1 = usable_cumulative[i - 1], usable_cumulative[i]
            d0, d1 = usable_disp[i - 1], usable_disp[i]
            alpha = 0.0 if e1 <= e0 else (IMPACT_ENERGY_J - e0) / (e1 - e0)
            impact_stop_mm = d0 + alpha * (d1 - d0)
            break
    active = [f for d, f in zip(disp, force) if 3.0 <= d <= min(collapse_mm, 0.82 * config.max_displacement_mm)]
    plateau = sum(active) / len(active) if active else 0.0
    peak_limit = min(collapse_mm, impact_stop_mm if energy_usable >= IMPACT_ENERGY_J else collapse_mm)
    impact_forces = [f for d, f in zip(disp, force) if d <= peak_limit]
    peak = max(impact_forces) if impact_forces else (max(usable_force) if usable_force else 0.0)
    return {
        "displacement_mm": disp,
        "force_N": force,
        "energy_50mm_J": energy_50,
        "energy_usable_J": energy_usable,
        "peak_force_N": peak,
        "plateau_force_N": plateau,
        "collapse_mm": collapse_mm,
        "impact_stop_mm": impact_stop_mm,
        "progressive_crush_capacity_mm": collapse_mm,
        "layers": layers,
    }


def _interp_curve(x_mm: float, disp_mm: list[float], force_n: list[float]) -> float:
    if not disp_mm:
        return 0.0
    if x_mm <= disp_mm[0]:
        return force_n[0]
    if x_mm >= disp_mm[-1]:
        return force_n[-1]
    lo = 0
    hi = len(disp_mm) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if disp_mm[mid] <= x_mm:
            lo = mid
        else:
            hi = mid
    d0, d1 = disp_mm[lo], disp_mm[hi]
    f0, f1 = force_n[lo], force_n[hi]
    alpha = 0.0 if d1 <= d0 else (x_mm - d0) / (d1 - d0)
    return f0 + alpha * (f1 - f0)


def run_dynamic_impact(
    params: DesignParams,
    physics_config: PhysicsConfig | None = None,
    impact_config: ImpactConfig | None = None,
    rank: int = 0,
    mass_g: float | None = None,
) -> tuple[ImpactResult, dict[str, object]]:
    """Integrate a rigid indenter impact against the layered crush force law."""

    physics_config = physics_config or PhysicsConfig()
    impact_config = impact_config or ImpactConfig()
    sim = run_layered_crush_fem(params, physics_config)
    disp_mm = [float(x) for x in sim["displacement_mm"]]  # type: ignore[index]
    force_n = [float(x) for x in sim["force_N"]]  # type: ignore[index]
    collapse_mm = float(sim["collapse_mm"])
    initial_ke = 0.5 * impact_config.indenter_mass_kg * impact_config.initial_velocity_m_s**2
    mass = mass_g
    if mass is None or mass <= 0.0:
        mass = _material_mass_g(params, physics_config.material)

    t = 0.0
    x_m = 0.0
    v_m_s = impact_config.initial_velocity_m_s
    absorbed_j = 0.0
    damping_j = 0.0
    peak = 0.0
    history: list[dict[str, float]] = []
    max_steps = max(1, int(impact_config.max_time_s / max(impact_config.dt_s, 1e-9)))
    stopped = False
    failed = False
    for step in range(max_steps):
        x_mm = max(0.0, x_m * 1000.0)
        contact_force = _interp_curve(x_mm, disp_mm, force_n)
        damper = impact_config.damping_n_s_per_m * max(0.0, v_m_s)
        total_resistance = contact_force + damper
        acceleration = impact_config.gravity_m_s2 - total_resistance / max(impact_config.indenter_mass_kg, 1e-9)
        v_next = v_m_s + acceleration * impact_config.dt_s
        x_next = max(0.0, x_m + v_next * impact_config.dt_s)
        dx_m = max(0.0, x_next - x_m)
        absorbed_j += max(0.0, contact_force) * dx_m
        damping_j += damper * dx_m
        peak = max(peak, contact_force)
        t += impact_config.dt_s
        x_m, v_m_s = x_next, v_next
        if step % max(1, max_steps // 250) == 0:
            history.append(
                {
                    "time_ms": 1000.0 * t,
                    "disp_mm": x_m * 1000.0,
                    "velocity_m_s": v_m_s,
                    "force_N": contact_force,
                    "absorbed_J": absorbed_j,
                }
            )
        if x_m * 1000.0 >= collapse_mm:
            failed = True
            break
        if v_m_s <= impact_config.stop_velocity_m_s and absorbed_j >= 0.92 * initial_ke:
            stopped = True
            break

    gravity_work = impact_config.indenter_mass_kg * impact_config.gravity_m_s2 * x_m
    final_ke = 0.5 * impact_config.indenter_mass_kg * max(0.0, v_m_s) ** 2
    balance_error = (initial_ke + gravity_work) - (absorbed_j + damping_j + final_ke)
    max_disp_mm = x_m * 1000.0
    margin_j = absorbed_j - IMPACT_ENERGY_J
    mean_force = absorbed_j / max(max_disp_mm / 1000.0, 1e-12)
    cfe = mean_force / peak if peak > 1e-12 else 0.0
    risk = _clamp(
        0.06
        + 0.44 * _sigmoid(-margin_j / 3.0)
        + 0.26 * _sigmoid((max_disp_mm - 0.88 * collapse_mm) / 2.5)
        + 0.18 * _sigmoid((peak - physics_config.fixture_peak_force_limit_n) / 500.0)
        + (0.22 if failed else 0.0)
        + (0.10 if not stopped else 0.0),
        0.0,
        1.0,
    )
    energy_pass = absorbed_j >= 0.98 * IMPACT_ENERGY_J
    peak_pass = peak <= physics_config.fixture_peak_force_limit_n
    crush_pass = collapse_mm >= physics_config.target_min_crush_mm and max_disp_mm <= 0.96 * physics_config.max_displacement_mm
    survives = stopped and not failed and energy_pass and peak_pass and crush_pass and risk < 0.48
    score = (
        float(mass)
        + 70.0 * max(0.0, IMPACT_ENERGY_J - absorbed_j)
        + 42.0 * risk
        + 0.030 * max(0.0, peak - physics_config.fixture_peak_force_limit_n)
        + 16.0 * (0.0 if peak_pass else 1.0)
        + 10.0 * (0.0 if crush_pass else 1.0)
        + 0.22 * max(0.0, max_disp_mm - 42.0)
    )
    result = ImpactResult(
        rank=rank,
        topology=params.topology,
        material_name=physics_config.material.material_name,
        mass_g=float(mass),
        impact_initial_ke_J=initial_ke,
        impact_absorbed_J=absorbed_j,
        impact_energy_margin_J=margin_j,
        impact_sea_J_g=absorbed_j / max(float(mass), 1e-12),
        impact_mean_crushing_force_N=mean_force,
        impact_cfe=cfe,
        impact_peak_force_N=peak,
        impact_max_displacement_mm=max_disp_mm,
        impact_stop_time_ms=1000.0 * t,
        impact_residual_velocity_m_s=max(0.0, v_m_s),
        impact_energy_balance_error_J=balance_error,
        impact_energy_pass=energy_pass,
        impact_peak_force_pass=peak_pass,
        impact_crush_distance_pass=crush_pass,
        impact_survives=survives,
        impact_failure_risk=risk,
        impact_score=score,
    )
    return result, {"history": history, "force_law": sim}


def evaluate_candidate_physics(
    row: dict[str, str | float],
    config: PhysicsConfig | None = None,
    stl_path: Path | None = None,
    audit_stl: bool = False,
) -> tuple[PhysicsResult, dict[str, object]]:
    config = config or PhysicsConfig()
    params = DesignParams.from_row(row)
    rank = int(float(row.get("rank", 0))) if "rank" in row else 0
    rel = estimate_relative_density(params)
    mass = _material_mass_g(params, config.material)
    sim = run_layered_crush_fem(params, config)
    energy_usable = float(sim["energy_usable_J"])
    energy_50 = float(sim["energy_50mm_J"])
    peak = float(sim["peak_force_N"])
    plateau = float(sim["plateau_force_N"])
    collapse = float(sim["collapse_mm"])
    impact_stop = float(sim["impact_stop_mm"])
    energy_margin = energy_usable - IMPACT_ENERGY_J
    curve_metrics = force_curve_metrics(
        [float(x) for x in sim["displacement_mm"]],  # type: ignore[index]
        [float(x) for x in sim["force_N"]],  # type: ignore[index]
        mass,
        target_energy_j=IMPACT_ENERGY_J,
        target_stroke_mm=config.target_min_crush_mm,
        usable_stroke_mm=collapse,
    )
    mesh_quality = audit_stl_mesh(stl_path) if audit_stl and stl_path is not None and stl_path.exists() else None
    mesh_penalty = 0.0
    if mesh_quality is not None:
        if mesh_quality.watertight_by_edges is False:
            mesh_penalty += 0.25
        if mesh_quality.within_envelope is False:
            mesh_penalty += 0.25
    risk = _clamp(
        0.04
        + 0.42 * _sigmoid(-energy_margin / 4.0)
        + 0.22 * _sigmoid((peak - config.fixture_peak_force_limit_n) / 500.0)
        + 0.22 * _sigmoid((32.0 - collapse) / 3.0)
        + 0.14 * _sigmoid((config.material.minimum_printable_feature_mm + 0.08 - params.min_feature_mm) / 0.05)
        + mesh_penalty,
        0.0,
        1.0,
    )
    energy_pass = energy_usable >= IMPACT_ENERGY_J
    peak_pass = peak <= config.fixture_peak_force_limit_n
    crush_pass = collapse >= config.target_min_crush_mm
    survives = energy_pass and peak_pass and crush_pass and risk < 0.42
    score = (
        mass
        + 55.0 * max(0.0, IMPACT_ENERGY_J - energy_usable)
        + 38.0 * risk
        + 0.025 * max(0.0, peak - config.fixture_peak_force_limit_n)
        + 18.0 * (0.0 if peak_pass else 1.0)
        + 8.0 * (0.0 if crush_pass else 1.0)
        + 0.4 * max(0.0, 30.0 - collapse)
    )
    result = PhysicsResult(
        rank=rank,
        topology=params.topology,
        material_name=config.material.material_name,
        mass_g=mass,
        relative_density=rel,
        physics_energy_50mm_J=energy_50,
        physics_energy_usable_J=energy_usable,
        physics_sea_J_g=energy_usable / max(mass, 1e-12),
        physics_mean_crushing_force_N=float(curve_metrics["mean_crushing_force_N"]),
        physics_cfe=float(curve_metrics["crush_force_efficiency"]),
        physics_plateau_cv=float(curve_metrics["plateau_force_cv"]),
        physics_peak_force_N=peak,
        physics_plateau_force_N=plateau,
        physics_collapse_mm=collapse,
        physics_impact_stop_mm=impact_stop,
        physics_failure_risk=risk,
        physics_energy_margin_J=energy_margin,
        physics_energy_pass=energy_pass,
        physics_peak_force_pass=peak_pass,
        physics_crush_distance_pass=crush_pass,
        physics_survives_gate=survives,
        physics_score=score,
        stl_watertight_by_edges=mesh_quality.watertight_by_edges if mesh_quality else None,
        stl_within_envelope=mesh_quality.within_envelope if mesh_quality else None,
        stl_open_edges=mesh_quality.open_edges if mesh_quality else None,
        stl_edge_count_not_two=mesh_quality.edge_count_not_two if mesh_quality else None,
    )
    return result, sim


def _write_curve_csv(path: Path, displacement_mm: list[float], force_n: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["displacement_mm", "force_N"])
        writer.writerows(zip(displacement_mm, force_n))


def run_physics_gate(
    top_csv: Path,
    out_dir: Path,
    top_n: int = 50,
    config: PhysicsConfig | None = None,
    stl_dir: Path | None = None,
    audit_stl: bool = False,
    write_curves: int = 20,
) -> dict[str, object]:
    config = config or PhysicsConfig()
    out_dir = ensure_dir(out_dir)
    curve_dir = ensure_dir(out_dir / "curves")
    with Path(top_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:top_n]
    results: list[PhysicsResult] = []
    for row in rows:
        stl_path = None
        if stl_dir is not None and row.get("rank") and row.get("topology"):
            stl_path = Path(stl_dir) / f"rank_{int(float(row['rank'])):03d}_{row['topology']}.stl"
        result, sim = evaluate_candidate_physics(row, config=config, stl_path=stl_path, audit_stl=audit_stl)
        results.append(result)
        if len(results) <= write_curves:
            _write_curve_csv(
                curve_dir / f"rank_{result.rank:03d}_{result.topology}_physics_curve.csv",
                sim["displacement_mm"],  # type: ignore[arg-type]
                sim["force_N"],  # type: ignore[arg-type]
            )
    ranked = sorted(results, key=lambda item: item.physics_score)
    fields = list(ranked[0].to_row().keys()) if ranked else list(PhysicsResult.__dataclass_fields__.keys())
    out_csv = out_dir / "physics_candidates.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in ranked:
            writer.writerow(result.to_row())
    summary = {
        "top_csv": str(top_csv),
        "out_csv": str(out_csv),
        "curve_dir": str(curve_dir),
        "evaluated": len(results),
        "survivors": sum(1 for item in results if item.physics_survives_gate),
        "config": asdict(config),
        "best": ranked[0].to_row() if ranked else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# POLMI Physics Gate",
        "",
        f"Evaluated: `{summary['evaluated']}`",
        f"Survivors: `{summary['survivors']}`",
        f"Input: `{top_csv}`",
        "",
        "| Physics Rank | Original Rank | Topology | Mass g | E usable J | Peak N | Plateau N | Stop mm | Collapse mm | Risk | Score |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, result in enumerate(ranked[:20], start=1):
        lines.append(
            f"| {i} | {result.rank} | {result.topology} | {result.mass_g:.3f} | "
            f"{result.physics_energy_usable_J:.2f} | {result.physics_peak_force_N:.0f} | "
            f"{result.physics_plateau_force_N:.0f} | {result.physics_impact_stop_mm:.1f} | "
            f"{result.physics_collapse_mm:.1f} | "
            f"{result.physics_failure_risk:.3f} | {result.physics_score:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_dynamic_impact_gate(
    top_csv: Path,
    out_dir: Path,
    top_n: int = 120,
    physics_config: PhysicsConfig | None = None,
    impact_config: ImpactConfig | None = None,
    write_histories: int = 20,
) -> dict[str, object]:
    physics_config = physics_config or PhysicsConfig()
    impact_config = impact_config or ImpactConfig()
    out_dir = ensure_dir(out_dir)
    history_dir = ensure_dir(out_dir / "histories")
    with Path(top_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:top_n]
    results: list[ImpactResult] = []
    for row in rows:
        params = DesignParams.from_row(row)
        rank = int(float(row.get("rank", len(results) + 1)))
        mass = _material_mass_g(params, physics_config.material)
        result, payload = run_dynamic_impact(
            params,
            physics_config=physics_config,
            impact_config=impact_config,
            rank=rank,
            mass_g=mass,
        )
        results.append(result)
        if len(results) <= write_histories:
            with (history_dir / f"rank_{rank:03d}_{params.topology}_impact_history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                fieldnames = ["time_ms", "disp_mm", "velocity_m_s", "force_N", "absorbed_J"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(payload["history"])  # type: ignore[arg-type]

    ranked = sorted(results, key=lambda item: item.impact_score)
    fields = list(ranked[0].to_row().keys()) if ranked else list(ImpactResult.__dataclass_fields__.keys())
    out_csv = out_dir / "dynamic_impact_candidates.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in ranked:
            writer.writerow(result.to_row())
    summary = {
        "top_csv": str(top_csv),
        "out_csv": str(out_csv),
        "history_dir": str(history_dir),
        "evaluated": len(results),
        "survivors": sum(1 for item in results if item.impact_survives),
        "physics_config": asdict(physics_config),
        "impact_config": asdict(impact_config),
        "best": ranked[0].to_row() if ranked else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# POLMI Dynamic Impact Gate",
        "",
        f"Evaluated: `{summary['evaluated']}`",
        f"Survivors: `{summary['survivors']}`",
        f"Input: `{top_csv}`",
        "",
        "| Impact Rank | Original Rank | Topology | Mass g | Absorbed J | Peak N | Max disp mm | Stop ms | Risk | Score |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, result in enumerate(ranked[:20], start=1):
        lines.append(
            f"| {i} | {result.rank} | {result.topology} | {result.mass_g:.3f} | "
            f"{result.impact_absorbed_J:.2f} | {result.impact_peak_force_N:.0f} | "
            f"{result.impact_max_displacement_mm:.1f} | {result.impact_stop_time_ms:.2f} | "
            f"{result.impact_failure_risk:.3f} | {result.impact_score:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _sensitivity_variants(base_config: PhysicsConfig) -> list[tuple[str, PhysicsConfig, float]]:
    variants: list[tuple[str, PhysicsConfig, float]] = []
    base_material = base_config.material
    for scale in (0.06, 0.08, 0.10, 0.12):
        variants.append((f"yield_scale_{scale:.2f}", replace(base_config, yield_scale=scale), 0.0))
    for factor in (1.00, 1.16, 1.32, 1.50):
        variants.append((f"strain_rate_factor_{factor:.2f}", replace(base_config, dynamic_amplification=factor), 0.0))
    for fracture in (0.66, 0.74, 0.82, 0.90):
        variants.append((f"fracture_strain_{fracture:.2f}", replace(base_config, fracture_strain_base=fracture), 0.0))
    for peak_limit in (2600.0, 3200.0, 3800.0, 4600.0):
        variants.append((f"contact_peak_limit_{peak_limit:.0f}N", replace(base_config, fixture_peak_force_limit_n=peak_limit), 0.0))
    for min_feature_delta in (-0.08, -0.04, 0.0, 0.06):
        variants.append((f"min_feature_delta_{min_feature_delta:+.2f}mm", base_config, min_feature_delta))
    for scale in (0.90, 1.00, 1.10):
        material = base_material.scaled(name_suffix=f"density_x{scale:.2f}", density_scale=scale)
        variants.append((f"material_density_x{scale:.2f}", replace(base_config, material=material), 0.0))
    for scale in (0.85, 1.00, 1.15):
        material = base_material.scaled(name_suffix=f"plateau_x{scale:.2f}", plateau_scale=scale, yield_scale=scale)
        variants.append((f"material_strength_x{scale:.2f}", replace(base_config, material=material), 0.0))
    for scale in (0.80, 1.00, 1.20):
        material = base_material.scaled(name_suffix=f"failure_x{scale:.2f}", failure_scale=scale)
        variants.append((f"material_failure_x{scale:.2f}", replace(base_config, material=material), 0.0))
    for delta in (-0.05, 0.0, 0.05):
        material = base_material.scaled(name_suffix=f"tolerance_{delta:+.2f}mm", tolerance_delta_mm=delta)
        variants.append((f"material_tolerance_{delta:+.2f}mm", replace(base_config, material=material), 0.0))
    return variants


def run_sensitivity_sweep(
    top_csv: Path,
    out_dir: Path,
    top_n: int = 12,
    config: PhysicsConfig | None = None,
) -> dict[str, object]:
    config = config or PhysicsConfig()
    out_dir = ensure_dir(out_dir)
    with Path(top_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:top_n]
    out_rows: list[dict[str, object]] = []
    for row in rows:
        base_params = DesignParams.from_row(row)
        rank = int(float(row.get("rank", len(out_rows) + 1)))
        for name, variant_config, min_feature_delta in _sensitivity_variants(config):
            params = base_params
            if abs(min_feature_delta) > 1e-12:
                adjusted_min_feature = _clamp(
                    base_params.min_feature_mm + min_feature_delta,
                    max(0.05, variant_config.material.minimum_printable_feature_mm - 0.08),
                    1.20,
                )
                params = replace(
                    base_params,
                    min_feature_mm=adjusted_min_feature,
                    wall_thickness_mm=max(adjusted_min_feature, base_params.wall_thickness_mm + min_feature_delta),
                )
            variant_row = row.copy()
            variant_row.update(params.to_row())
            scenario_mass = _material_mass_g(params, variant_config.material)
            result, _ = evaluate_candidate_physics(variant_row, config=variant_config)
            impact, _ = run_dynamic_impact(
                params,
                physics_config=variant_config,
                rank=rank,
                mass_g=scenario_mass,
            )
            out = {
                "rank": rank,
                "topology": params.topology,
                "scenario": name,
                "material_name": variant_config.material.material_name,
                "material_density_g_cm3": variant_config.material.density_g_cm3,
                "material_plateau_strength_MPa": variant_config.material.compressive_plateau_strength_MPa,
                "material_failure_strain": variant_config.material.failure_strain,
                "material_min_feature_mm": variant_config.material.minimum_printable_feature_mm,
                "mass_g": result.mass_g,
                "energy_usable_J": result.physics_energy_usable_J,
                "sea_J_g": result.physics_sea_J_g,
                "cfe": result.physics_cfe,
                "energy_margin_J": result.physics_energy_margin_J,
                "collapse_mm": result.physics_collapse_mm,
                "peak_force_N": result.physics_peak_force_N,
                "failure_risk": result.physics_failure_risk,
                "survives_physics": result.physics_survives_gate,
                "impact_absorbed_J": impact.impact_absorbed_J,
                "impact_max_displacement_mm": impact.impact_max_displacement_mm,
                "impact_risk": impact.impact_failure_risk,
                "survives_impact": impact.impact_survives,
            }
            out_rows.append(out)
    out_csv = out_dir / "sensitivity_sweep.csv"
    fields = list(out_rows[0].keys()) if out_rows else [
        "rank",
        "topology",
        "scenario",
        "material_name",
        "material_density_g_cm3",
        "material_plateau_strength_MPa",
        "material_failure_strain",
        "material_min_feature_mm",
        "mass_g",
        "energy_usable_J",
        "sea_J_g",
        "cfe",
        "energy_margin_J",
        "collapse_mm",
        "peak_force_N",
        "failure_risk",
        "survives_physics",
        "impact_absorbed_J",
        "impact_max_displacement_mm",
        "impact_risk",
        "survives_impact",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    by_rank: dict[int, list[dict[str, object]]] = {}
    for item in out_rows:
        by_rank.setdefault(int(item["rank"]), []).append(item)
    robust = []
    for rank, items in sorted(by_rank.items()):
        robust.append(
            {
                "rank": rank,
                "scenarios": len(items),
                "physics_survival_rate": sum(1 for item in items if item["survives_physics"]) / max(1, len(items)),
                "impact_survival_rate": sum(1 for item in items if item["survives_impact"]) / max(1, len(items)),
                "worst_energy_margin_J": min(float(item["energy_margin_J"]) for item in items),
                "worst_impact_absorbed_J": min(float(item["impact_absorbed_J"]) for item in items),
                "max_failure_risk": max(float(item["failure_risk"]) for item in items),
            }
        )
    summary = {
        "top_csv": str(top_csv),
        "out_csv": str(out_csv),
        "evaluated_designs": len(rows),
        "scenarios_per_design": len(_sensitivity_variants(config)),
        "robustness": robust,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
