"""Mechanics and benchmark metrics for force-displacement lattice responses."""

from __future__ import annotations

import math
from statistics import pstdev


def integrate_energy_j(displacement_mm: list[float], force_n: list[float], limit_mm: float | None = None) -> float:
    if len(displacement_mm) < 2 or len(force_n) < 2:
        return 0.0
    total = 0.0
    pairs = list(zip(displacement_mm, force_n))
    for (d0, f0), (d1, f1) in zip(pairs[:-1], pairs[1:]):
        if limit_mm is not None and d0 >= limit_mm:
            break
        local_d1 = d1
        local_f1 = f1
        if limit_mm is not None and d1 > limit_mm:
            alpha = 0.0 if d1 <= d0 else (limit_mm - d0) / (d1 - d0)
            local_d1 = limit_mm
            local_f1 = f0 + alpha * (f1 - f0)
        dx_m = max(0.0, local_d1 - d0) / 1000.0
        total += 0.5 * (max(0.0, f0) + max(0.0, local_f1)) * dx_m
        if limit_mm is not None and d1 >= limit_mm:
            break
    return total


def force_curve_metrics(
    displacement_mm: list[float],
    force_n: list[float],
    mass_g: float,
    *,
    target_energy_j: float = 29.43,
    target_stroke_mm: float = 40.0,
    usable_stroke_mm: float | None = None,
) -> dict[str, float]:
    if not displacement_mm or not force_n:
        return {
            "absorbed_energy_J": 0.0,
            "specific_energy_absorption_J_g": 0.0,
            "peak_force_N": 0.0,
            "mean_crushing_force_N": 0.0,
            "crush_force_efficiency": 0.0,
            "plateau_force_mean_N": 0.0,
            "plateau_force_std_N": 0.0,
            "plateau_force_cv": 0.0,
            "energy_margin_J": -target_energy_j,
            "stroke_mm": 0.0,
        }
    stroke = max(0.0, min(float(usable_stroke_mm or target_stroke_mm), max(displacement_mm)))
    active_force = [max(0.0, f) for d, f in zip(displacement_mm, force_n) if d <= stroke]
    energy = integrate_energy_j(displacement_mm, force_n, stroke)
    peak = max(active_force) if active_force else 0.0
    mean_force = energy / max(stroke / 1000.0, 1e-12)
    cfe = mean_force / peak if peak > 1e-12 else 0.0
    lo = 0.20 * stroke
    hi = 0.80 * stroke
    plateau = [max(0.0, f) for d, f in zip(displacement_mm, force_n) if lo <= d <= hi]
    if not plateau:
        plateau = [max(0.0, f) for d, f in zip(displacement_mm, force_n) if d <= stroke]
    plateau_mean = sum(plateau) / len(plateau) if plateau else 0.0
    plateau_std = pstdev(plateau) if len(plateau) > 1 else 0.0
    plateau_cv = plateau_std / plateau_mean if plateau_mean > 1e-12 else 0.0
    sea = energy / mass_g if mass_g > 1e-12 else 0.0
    return {
        "absorbed_energy_J": energy,
        "specific_energy_absorption_J_g": sea,
        "peak_force_N": peak,
        "mean_crushing_force_N": mean_force,
        "crush_force_efficiency": cfe,
        "plateau_force_mean_N": plateau_mean,
        "plateau_force_std_N": plateau_std,
        "plateau_force_cv": plateau_cv,
        "energy_margin_J": energy - target_energy_j,
        "stroke_mm": stroke,
    }


def nrmse(pred: list[float], target: list[float]) -> float:
    if len(pred) != len(target) or not pred:
        return math.nan
    mse = sum((a - b) ** 2 for a, b in zip(pred, target)) / len(pred)
    scale = max(target) - min(target)
    if scale <= 1e-12:
        scale = max(abs(x) for x in target) if target else 1.0
    return math.sqrt(mse) / max(scale, 1e-12)


def rmse(pred: list[float], target: list[float]) -> float:
    if len(pred) != len(target) or not pred:
        return math.nan
    mse = sum((a - b) ** 2 for a, b in zip(pred, target)) / len(pred)
    return math.sqrt(mse)


def interp_curve(displacement_mm: list[float], force_n: list[float], x_mm: float) -> float:
    if not displacement_mm or not force_n:
        return 0.0
    if x_mm <= displacement_mm[0]:
        return float(force_n[0])
    if x_mm >= displacement_mm[-1]:
        return float(force_n[-1])
    lo = 0
    hi = min(len(displacement_mm), len(force_n)) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if displacement_mm[mid] <= x_mm:
            lo = mid
        else:
            hi = mid
    d0, d1 = displacement_mm[lo], displacement_mm[hi]
    f0, f1 = force_n[lo], force_n[hi]
    alpha = 0.0 if d1 <= d0 else (x_mm - d0) / (d1 - d0)
    return float(f0 + alpha * (f1 - f0))


def resample_force_curve(
    displacement_mm: list[float],
    force_n: list[float],
    target_displacement_mm: list[float],
) -> list[float]:
    return [interp_curve(displacement_mm, force_n, x) for x in target_displacement_mm]


def force_curve_error_metrics(
    pred_displacement_mm: list[float],
    pred_force_n: list[float],
    target_displacement_mm: list[float],
    target_force_n: list[float],
    *,
    limit_mm: float | None = None,
) -> dict[str, float]:
    if not pred_displacement_mm or not target_displacement_mm:
        return {
            "curve_rmse_N": math.nan,
            "curve_nrmse": math.nan,
            "energy_integral_error_J": math.nan,
            "energy_integral_abs_error_J": math.nan,
            "energy_integral_rel_error": math.nan,
        }
    axis = [x for x in target_displacement_mm if limit_mm is None or x <= limit_mm]
    if len(axis) < 2:
        axis = target_displacement_mm[:]
    pred = resample_force_curve(pred_displacement_mm, pred_force_n, axis)
    target = resample_force_curve(target_displacement_mm, target_force_n, axis)
    pred_energy = integrate_energy_j(axis, pred, limit_mm=limit_mm)
    target_energy = integrate_energy_j(axis, target, limit_mm=limit_mm)
    energy_error = pred_energy - target_energy
    return {
        "curve_rmse_N": rmse(pred, target),
        "curve_nrmse": nrmse(pred, target),
        "energy_integral_error_J": energy_error,
        "energy_integral_abs_error_J": abs(energy_error),
        "energy_integral_rel_error": abs(energy_error) / max(abs(target_energy), 1e-12),
    }


def physical_violation_rate(
    rows: list[dict[str, object]],
    *,
    energy_key: str,
    peak_key: str,
    crush_key: str,
    risk_key: str | None = None,
    target_energy_j: float = 29.43,
    peak_limit_n: float = 3500.0,
    min_crush_mm: float = 40.0,
    max_risk: float | None = None,
) -> float:
    if not rows:
        return math.nan
    violations = 0
    for row in rows:
        energy = _float(row.get(energy_key))
        peak = _float(row.get(peak_key))
        crush = _float(row.get(crush_key))
        violates = (
            not math.isfinite(energy)
            or not math.isfinite(peak)
            or not math.isfinite(crush)
            or energy < target_energy_j
            or peak > peak_limit_n
            or crush < min_crush_mm
        )
        if risk_key is not None and max_risk is not None:
            risk = _float(row.get(risk_key))
            violates = violates or not math.isfinite(risk) or risk > max_risk
        if violates:
            violations += 1
    return violations / len(rows)


def precision_at_k(feasible: list[bool], k: int) -> float:
    if k <= 0 or not feasible:
        return math.nan
    selected = feasible[: min(k, len(feasible))]
    return sum(1 for item in selected if item) / len(selected)


def best_feasible_regret(selected_masses_g: list[float], selected_feasible: list[bool], oracle_masses_g: list[float]) -> float:
    oracle = [mass for mass in oracle_masses_g if math.isfinite(mass)]
    if not oracle:
        return math.nan
    selected = [mass for mass, feasible in zip(selected_masses_g, selected_feasible) if feasible and math.isfinite(mass)]
    if not selected:
        return math.inf
    return min(selected) - min(oracle)


def best_selected_feasible_mass(selected_masses_g: list[float], selected_feasible: list[bool]) -> float:
    selected = [mass for mass, feasible in zip(selected_masses_g, selected_feasible) if feasible and math.isfinite(mass)]
    return min(selected) if selected else math.inf


def relative_best_feasible_regret(
    selected_masses_g: list[float],
    selected_feasible: list[bool],
    oracle_masses_g: list[float],
) -> float:
    oracle = [mass for mass in oracle_masses_g if math.isfinite(mass)]
    if not oracle:
        return math.nan
    best_oracle = min(oracle)
    regret = best_feasible_regret(selected_masses_g, selected_feasible, oracle)
    if not math.isfinite(regret):
        return regret
    return regret / max(best_oracle, 1e-12)


def _float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default
