"""Parametric lattice design space and bootstrap response model.

This module intentionally stays dependency-light. It is used by tests, audits,
candidate generation, and the Slurm job before any GPU libraries are imported.
The pseudo-response is not a FEM substitute; it gives the neural models a
physically shaped bootstrap target that is then used for ranking and inverse
search with explicit uncertainty.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ENVELOPE_MM = 50.0
PA12_DENSITY_G_PER_MM3 = 1.01e-3
TARGET_ENERGY_J = 29.4
SAFE_ENERGY_J = 35.0
CURVE_POINTS = 96

TOPOLOGIES = [
    "gyroid",
    "diamond",
    "octet",
    "bcc",
    "hybrid",
    "schwarz_p",
    "diamond_graded",
    "gyroid_diamond_hybrid",
    "bccz_graded",
    "ot_like",
]

PARAM_FIELDS = [
    "topology",
    "cell_size_mm",
    "wall_thickness_mm",
    "min_feature_mm",
    "vertical_gradient",
    "density_bias",
    "cap_thickness_mm",
    "edge_rib_mm",
    "trigger_layer_strength",
    "trigger_zone_mm",
    "plateau_zone_mm",
    "bumper_zone_mm",
    "hybrid_ratio",
    "anisotropy_xy",
]

SCALAR_TARGET_FIELDS = [
    "mass_g",
    "relative_density",
    "energy_abs_J",
    "force_peak_N",
    "force_plateau_N",
    "early_energy_20mm_J",
    "peak_plateau_ratio",
    "progressive_crush_score",
    "collapse_displacement_mm",
    "failure_probability",
]


@dataclass(frozen=True)
class DesignParams:
    topology: str
    cell_size_mm: float
    wall_thickness_mm: float
    min_feature_mm: float
    vertical_gradient: float
    density_bias: float
    cap_thickness_mm: float
    edge_rib_mm: float
    trigger_layer_strength: float
    trigger_zone_mm: float
    plateau_zone_mm: float
    bumper_zone_mm: float
    hybrid_ratio: float
    anisotropy_xy: float

    def to_row(self) -> dict[str, str]:
        row = asdict(self)
        return {k: _format_value(v) for k, v in row.items()}

    @classmethod
    def from_row(cls, row: dict[str, str | float]) -> "DesignParams":
        trigger = _row_float(row, "trigger_zone_mm", 11.0)
        plateau = _row_float(row, "plateau_zone_mm", 27.0)
        bumper = _row_float(row, "bumper_zone_mm", 12.0)
        trigger, plateau, bumper = normalize_zones(trigger, plateau, bumper)
        return cls(
            topology=str(row["topology"]),
            cell_size_mm=float(row["cell_size_mm"]),
            wall_thickness_mm=float(row["wall_thickness_mm"]),
            min_feature_mm=float(row["min_feature_mm"]),
            vertical_gradient=float(row["vertical_gradient"]),
            density_bias=float(row["density_bias"]),
            cap_thickness_mm=float(row["cap_thickness_mm"]),
            edge_rib_mm=float(row["edge_rib_mm"]),
            trigger_layer_strength=float(row["trigger_layer_strength"]),
            trigger_zone_mm=trigger,
            plateau_zone_mm=plateau,
            bumper_zone_mm=bumper,
            hybrid_ratio=float(row["hybrid_ratio"]),
            anisotropy_xy=float(row["anisotropy_xy"]),
        )


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _row_float(row: dict[str, str | float], name: str, default: float) -> float:
    try:
        return float(row.get(name, default))
    except (TypeError, ValueError):
        return default


def normalize_zones(trigger_mm: float, plateau_mm: float, bumper_mm: float) -> tuple[float, float, float]:
    trigger = clamp(trigger_mm, 6.0, 16.0)
    bumper = clamp(bumper_mm, 7.0, 17.0)
    plateau = clamp(plateau_mm, 18.0, 36.0)
    total = trigger + plateau + bumper
    if total <= 1e-9:
        return 11.0, 27.0, 12.0
    scale = ENVELOPE_MM / total
    trigger = clamp(trigger * scale, 6.0, 16.0)
    bumper = clamp(bumper * scale, 7.0, 17.0)
    plateau = ENVELOPE_MM - trigger - bumper
    if plateau < 18.0:
        deficit = 18.0 - plateau
        trigger = max(6.0, trigger - 0.45 * deficit)
        bumper = max(7.0, bumper - 0.55 * deficit)
        plateau = ENVELOPE_MM - trigger - bumper
    return trigger, plateau, bumper


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def sample_design(rng: random.Random) -> DesignParams:
    topology = rng.choices(
        TOPOLOGIES,
        weights=[0.11, 0.12, 0.07, 0.05, 0.12, 0.17, 0.10, 0.15, 0.07, 0.04],
        k=1,
    )[0]
    cell_size = rng.uniform(4.4, 8.8)
    if topology in {"bccz_graded", "ot_like", "octet", "bcc"}:
        cell_size = rng.uniform(4.2, 7.8)
    min_feature = rng.uniform(0.50, 0.84)
    wall = rng.uniform(max(min_feature, 0.56), 1.32)
    if topology in {"octet", "bcc", "bccz_graded", "ot_like"}:
        wall = rng.uniform(max(min_feature, 0.60), 1.45)
    trigger = rng.uniform(8.0, 13.8)
    bumper = rng.uniform(9.0, 15.2)
    plateau = ENVELOPE_MM - trigger - bumper
    trigger, plateau, bumper = normalize_zones(trigger, plateau, bumper)
    return DesignParams(
        topology=topology,
        cell_size_mm=cell_size,
        wall_thickness_mm=wall,
        min_feature_mm=min_feature,
        vertical_gradient=rng.uniform(0.02, 0.95),
        density_bias=rng.uniform(-0.02, 0.14),
        cap_thickness_mm=rng.uniform(0.45, 1.00),
        edge_rib_mm=rng.uniform(0.0, 0.95),
        trigger_layer_strength=rng.uniform(0.25, 0.95),
        trigger_zone_mm=trigger,
        plateau_zone_mm=plateau,
        bumper_zone_mm=bumper,
        hybrid_ratio=rng.uniform(0.0, 1.0),
        anisotropy_xy=rng.uniform(0.85, 1.20),
    )


def sample_designs(n: int, seed: int) -> list[DesignParams]:
    rng = random.Random(seed)
    return [sample_design(rng) for _ in range(n)]


def _topology_factor(params: DesignParams) -> tuple[float, float, float]:
    density_factor = {
        "gyroid": 0.52,
        "diamond": 0.57,
        "octet": 0.48,
        "bcc": 0.40,
        "hybrid": 0.54,
        "schwarz_p": 0.50,
        "diamond_graded": 0.56,
        "gyroid_diamond_hybrid": 0.54,
        "bccz_graded": 0.43,
        "ot_like": 0.46,
    }[params.topology]
    sea_factor = {
        "gyroid": 1.52,
        "diamond": 1.44,
        "octet": 1.32,
        "bcc": 1.10,
        "hybrid": 1.66,
        "schwarz_p": 1.78,
        "diamond_graded": 1.62,
        "gyroid_diamond_hybrid": 1.74,
        "bccz_graded": 1.58,
        "ot_like": 1.50,
    }[params.topology]
    peak_factor = {
        "gyroid": 1.10,
        "diamond": 1.16,
        "octet": 1.32,
        "bcc": 1.24,
        "hybrid": 1.08,
        "schwarz_p": 0.94,
        "diamond_graded": 0.98,
        "gyroid_diamond_hybrid": 0.92,
        "bccz_graded": 1.02,
        "ot_like": 1.00,
    }[params.topology]
    return density_factor, sea_factor, peak_factor


def estimate_relative_density(params: DesignParams) -> float:
    density_factor, _, _ = _topology_factor(params)
    slenderness = params.wall_thickness_mm / params.cell_size_mm
    shell_like = density_factor * (2.15 * slenderness) ** 1.28
    gradient_bonus = 1.0 + 0.08 * abs(params.vertical_gradient)
    cap_bonus = 0.030 * (params.cap_thickness_mm / 0.8)
    rib_bonus = 0.025 * (params.edge_rib_mm / 0.8)
    hybrid_bonus = 0.030 * params.hybrid_ratio if params.topology in {"hybrid", "gyroid_diamond_hybrid"} else 0.0
    rel = params.density_bias + shell_like * gradient_bonus + cap_bonus + rib_bonus + hybrid_bonus
    return clamp(rel, 0.055, 0.72)


def geometry_feature_row(params: DesignParams) -> dict[str, float]:
    rel = estimate_relative_density(params)
    mass = rel * ENVELOPE_MM**3 * PA12_DENSITY_G_PER_MM3
    z_gradient_abs = abs(params.vertical_gradient)
    printability_margin = params.min_feature_mm - 0.5
    slenderness = params.wall_thickness_mm / params.cell_size_mm
    trigger_fraction = params.trigger_zone_mm / ENVELOPE_MM
    plateau_fraction = params.plateau_zone_mm / ENVELOPE_MM
    bumper_fraction = params.bumper_zone_mm / ENVELOPE_MM
    zone_balance = clamp(1.0 - abs(params.trigger_zone_mm - 11.0) / 13.0 - abs(params.bumper_zone_mm - 12.0) / 15.0, 0.0, 1.0)
    topology_progressivity = {
        "gyroid": 0.03,
        "diamond": 0.02,
        "octet": -0.04,
        "bcc": -0.02,
        "hybrid": 0.07,
        "schwarz_p": 0.12,
        "diamond_graded": 0.10,
        "gyroid_diamond_hybrid": 0.13,
        "bccz_graded": 0.08,
        "ot_like": 0.06,
    }[params.topology]
    progressivity = clamp(
        0.35
        + 0.28 * max(0.0, params.vertical_gradient)
        + 0.20 * params.trigger_layer_strength
        + 0.10 * plateau_fraction
        + 0.08 * zone_balance
        + 0.08 * params.hybrid_ratio
        + topology_progressivity
        - 0.12 * max(0.0, rel - 0.42),
        0.05,
        1.0,
    )
    return {
        "relative_density_est": rel,
        "mass_g_est": mass,
        "slenderness": slenderness,
        "printability_margin_mm": printability_margin,
        "gradient_abs": z_gradient_abs,
        "progressivity_est": progressivity,
        "cap_fraction": params.cap_thickness_mm / ENVELOPE_MM,
        "rib_fraction": params.edge_rib_mm / ENVELOPE_MM,
        "trigger_fraction": trigger_fraction,
        "plateau_fraction": plateau_fraction,
        "bumper_fraction": bumper_fraction,
        "zone_balance": zone_balance,
    }


def feature_names() -> list[str]:
    topo = [f"topology_{name}" for name in TOPOLOGIES]
    numeric = [
        "cell_size_mm",
        "wall_thickness_mm",
        "min_feature_mm",
        "vertical_gradient",
        "density_bias",
        "cap_thickness_mm",
        "edge_rib_mm",
        "trigger_layer_strength",
        "trigger_zone_mm",
        "plateau_zone_mm",
        "bumper_zone_mm",
        "hybrid_ratio",
        "anisotropy_xy",
    ]
    engineered = list(geometry_feature_row(sample_design(random.Random(0))).keys())
    return topo + numeric + engineered


def row_to_feature_map(row: dict[str, str | float]) -> dict[str, float]:
    params = DesignParams.from_row(row)
    out = {f"topology_{name}": (1.0 if params.topology == name else 0.0) for name in TOPOLOGIES}
    out.update(
        {
            "cell_size_mm": params.cell_size_mm,
            "wall_thickness_mm": params.wall_thickness_mm,
            "min_feature_mm": params.min_feature_mm,
            "vertical_gradient": params.vertical_gradient,
            "density_bias": params.density_bias,
            "cap_thickness_mm": params.cap_thickness_mm,
            "edge_rib_mm": params.edge_rib_mm,
            "trigger_layer_strength": params.trigger_layer_strength,
            "trigger_zone_mm": params.trigger_zone_mm,
            "plateau_zone_mm": params.plateau_zone_mm,
            "bumper_zone_mm": params.bumper_zone_mm,
            "hybrid_ratio": params.hybrid_ratio,
            "anisotropy_xy": params.anisotropy_xy,
        }
    )
    out.update(geometry_feature_row(params))
    return out


def row_to_named_features(row: dict[str, str | float], names: list[str]) -> list[float]:
    feature_map = row_to_feature_map(row)
    return [feature_map.get(name, 0.0) for name in names]


def row_to_features(row: dict[str, str | float]) -> list[float]:
    return row_to_named_features(row, feature_names())


def displacement_axis(n: int = CURVE_POINTS) -> list[float]:
    if n < 2:
        return [0.0]
    return [ENVELOPE_MM * i / (n - 1) for i in range(n)]


def integrate_curve_energy_j(displacement_mm: list[float], force_n: list[float]) -> float:
    total = 0.0
    for d0, d1, f0, f1 in zip(displacement_mm[:-1], displacement_mm[1:], force_n[:-1], force_n[1:]):
        dx_m = max(0.0, d1 - d0) / 1000.0
        total += 0.5 * (max(0.0, f0) + max(0.0, f1)) * dx_m
    return total


def pseudo_response(params: DesignParams, curve_points: int = CURVE_POINTS) -> dict[str, object]:
    rel = estimate_relative_density(params)
    mass_g = rel * ENVELOPE_MM**3 * PA12_DENSITY_G_PER_MM3
    _, sea_factor, peak_factor = _topology_factor(params)
    features = geometry_feature_row(params)
    progressivity = features["progressivity_est"]

    low_feature_risk = sigmoid((0.535 - params.min_feature_mm) * 24.0)
    low_density_risk = sigmoid((0.115 - rel) * 20.0)
    brittle_risk = sigmoid((params.vertical_gradient - 0.82) * 5.0)
    thin_wall_risk = sigmoid((0.58 - params.wall_thickness_mm) * 9.0)
    cap_risk = sigmoid((0.50 - params.cap_thickness_mm) * 8.0)
    zone_risk = sigmoid((8.0 - params.trigger_zone_mm) * 1.2) + sigmoid((9.0 - params.bumper_zone_mm) * 1.0)
    failure_probability = clamp(
        0.06
        + 0.26 * low_feature_risk
        + 0.24 * low_density_risk
        + 0.14 * brittle_risk
        + 0.18 * thin_wall_risk
        + 0.08 * cap_risk
        + 0.04 * zone_risk,
        0.02,
        0.92,
    )

    density_optimum = math.exp(-((rel - 0.23) / 0.15) ** 2)
    sea_j_per_g = sea_factor * (0.72 + 0.46 * progressivity + 0.28 * density_optimum)
    sea_j_per_g *= 1.0 - 0.28 * failure_probability
    collapse_mm = clamp(
        25.0
        + 23.0 * progressivity
        + 2.5 * (params.plateau_zone_mm / ENVELOPE_MM)
        + 1.8 * params.trigger_layer_strength
        - 17.0 * failure_probability
        - 9.0 * max(0.0, rel - 0.46)
        + 1.2 * min(1.0, params.edge_rib_mm),
        16.0,
        49.0,
    )
    energy_abs_j = clamp(mass_g * sea_j_per_g * (collapse_mm / 42.0), 1.0, 95.0)
    force_plateau_n = energy_abs_j / max(collapse_mm / 1000.0, 0.008)
    peak_relief = 0.24 * params.trigger_layer_strength + 0.10 * features["zone_balance"] + 0.08 * progressivity
    force_peak_n = force_plateau_n * (
        peak_factor
        + 0.38 * max(0.0, rel - 0.30)
        - peak_relief
    )
    force_peak_n = max(force_plateau_n * 0.82, force_peak_n)

    disp = displacement_axis(curve_points)
    raw_curve: list[float] = []
    trigger_end = min(0.42, params.trigger_zone_mm / max(collapse_mm, 1e-6))
    bumper_start = max(trigger_end + 0.18, 1.0 - params.bumper_zone_mm / max(collapse_mm, 1e-6))
    for d in disp:
        x = d / max(collapse_mm, 1e-6)
        if x <= trigger_end:
            ramp = x / max(trigger_end, 1e-6)
            force = force_peak_n * (0.18 + 0.82 * ramp**1.35)
            force *= 1.0 - 0.18 * params.trigger_layer_strength * (1.0 - ramp)
        elif x <= bumper_start:
            local = (x - trigger_end) / max(bumper_start - trigger_end, 1e-6)
            wave = 1.0 + 0.035 * math.sin(10.0 * local + params.hybrid_ratio * 2.0)
            softening = 1.0 - 0.07 * params.trigger_layer_strength * local
            force = force_plateau_n * wave * softening
            force = max(0.18 * force_plateau_n, min(force_peak_n * 1.03, force))
        elif x <= 1.0:
            tail = (x - bumper_start) / max(1.0 - bumper_start, 1e-6)
            hardening = 1.0 + (0.20 + 0.42 * max(0.0, rel - 0.22)) * tail
            force = force_plateau_n * hardening
            force = min(force_peak_n * (1.05 + 0.22 * tail), force)
        else:
            tail = min(1.0, (x - 1.0) / 0.35)
            force = force_plateau_n * (1.0 + 1.6 * tail * max(0.0, rel - 0.18))
            if failure_probability > 0.42:
                force *= max(0.22, 1.0 - 1.2 * tail * (failure_probability - 0.42))
        raw_curve.append(max(0.0, force))

    raw_energy = integrate_curve_energy_j(disp, raw_curve)
    scale = energy_abs_j / raw_energy if raw_energy > 1e-9 else 1.0
    curve = [f * scale for f in raw_curve]
    force_peak_n = max(curve)
    active = [f for d, f in zip(disp, curve) if d <= collapse_mm]
    force_plateau_n = sum(active) / len(active) if active else force_plateau_n
    early_pairs = [(d, f) for d, f in zip(disp, curve) if d <= 20.0]
    early_energy_20mm_j = (
        integrate_curve_energy_j([d for d, _ in early_pairs], [f for _, f in early_pairs]) if len(early_pairs) >= 2 else 0.0
    )
    peak_plateau_ratio = force_peak_n / max(force_plateau_n, 1e-9)
    progressive_crush_score = clamp(
        0.40 * progressivity
        + 0.22 * (collapse_mm / ENVELOPE_MM)
        + 0.18 * clamp(1.55 - peak_plateau_ratio, 0.0, 1.0)
        + 0.12 * clamp(1.0 - early_energy_20mm_j / max(energy_abs_j, 1e-9), 0.0, 1.0)
        + 0.08 * features["zone_balance"],
        0.0,
        1.0,
    )

    return {
        "mass_g": mass_g,
        "relative_density": rel,
        "energy_abs_J": integrate_curve_energy_j(disp, curve),
        "force_peak_N": force_peak_n,
        "force_plateau_N": force_plateau_n,
        "early_energy_20mm_J": early_energy_20mm_j,
        "peak_plateau_ratio": peak_plateau_ratio,
        "progressive_crush_score": progressive_crush_score,
        "collapse_displacement_mm": collapse_mm,
        "failure_probability": failure_probability,
        "curve_force_N": curve,
        "curve_displacement_mm": disp,
    }


def candidate_header(curve_points: int = CURVE_POINTS) -> list[str]:
    curve_fields = [f"curve_{i:03d}" for i in range(curve_points)]
    return PARAM_FIELDS + list(geometry_feature_row(sample_design(random.Random(1))).keys()) + SCALAR_TARGET_FIELDS + curve_fields


def design_to_candidate_row(params: DesignParams, curve_points: int = CURVE_POINTS) -> dict[str, str]:
    row = params.to_row()
    row.update({k: _format_value(v) for k, v in geometry_feature_row(params).items()})
    response = pseudo_response(params, curve_points=curve_points)
    for key in SCALAR_TARGET_FIELDS:
        row[key] = _format_value(response[key])
    for i, force in enumerate(response["curve_force_N"]):
        row[f"curve_{i:03d}"] = _format_value(force)
    return row


def write_candidates_csv(path: Path, designs: Iterable[DesignParams], curve_points: int = CURVE_POINTS) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = candidate_header(curve_points)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for params in designs:
            writer.writerow(design_to_candidate_row(params, curve_points=curve_points))
            count += 1
    meta = {
        "rows": count,
        "curve_points": curve_points,
        "target_energy_J": TARGET_ENERGY_J,
        "safe_energy_J": SAFE_ENERGY_J,
        "param_fields": PARAM_FIELDS,
        "scalar_target_fields": SCALAR_TARGET_FIELDS,
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return count


def read_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
