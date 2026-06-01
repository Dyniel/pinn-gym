"""Compare several reduced physical models on the same POLMI candidates."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from .design_space import (
    ENVELOPE_MM,
    PA12_DENSITY_G_PER_MM3,
    DesignParams,
    estimate_relative_density,
    integrate_curve_energy_j,
)
from .paths import ensure_dir
from .physics import (
    IMPACT_ENERGY_J,
    PA12_ELASTIC_MODULUS_MPA,
    PA12_YIELD_STRESS_MPA,
    PhysicsConfig,
    _clamp,
    _topology_physics_factors,
    build_layer_stack,
    evaluate_candidate_physics,
    run_layered_crush_fem,
)


BAKEOFF_MODELS = ("layered_v1", "layered_soft", "homogenized_ga", "buckling_stack")


@dataclass(frozen=True)
class PhysicsBakeoffConfig:
    layers: int = 96
    steps: int = 360
    max_displacement_mm: float = ENVELOPE_MM
    target_energy_j: float = IMPACT_ENERGY_J
    peak_limit_n: float = 3500.0
    min_crush_mm: float = 40.0
    max_failure_risk: float = 0.58
    yield_scale: float = 0.08
    dynamic_amplification: float = 1.16
    write_curves: int = 24
    models: tuple[str, ...] = BAKEOFF_MODELS


@dataclass(frozen=True)
class BakeoffModelResult:
    model: str
    rank: int
    topology: str
    mass_g: float
    energy_usable_J: float
    peak_force_N: float
    crush_mm: float
    impact_stop_mm: float
    failure_risk: float
    energy_pass: bool
    peak_pass: bool
    crush_pass: bool
    model_pass: bool
    model_score: float
    displacement_mm: list[float]
    force_N: list[float]

    def metric_row(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_energy_usable_J": self.energy_usable_J,
            f"{prefix}_peak_force_N": self.peak_force_N,
            f"{prefix}_crush_mm": self.crush_mm,
            f"{prefix}_impact_stop_mm": self.impact_stop_mm,
            f"{prefix}_failure_risk": self.failure_risk,
            f"{prefix}_energy_pass": self.energy_pass,
            f"{prefix}_peak_pass": self.peak_pass,
            f"{prefix}_crush_pass": self.crush_pass,
            f"{prefix}_pass": self.model_pass,
            f"{prefix}_score": self.model_score,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else ""
    if value is None:
        return ""
    return str(value)


def _mass_from_row(row: dict[str, str], params: DesignParams) -> float:
    for key in ("mass_g_mean", "mass_g", "coarse_mass_g"):
        try:
            value = float(row.get(key, "") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0 and math.isfinite(value):
            return value
    return estimate_relative_density(params) * ENVELOPE_MM**3 * PA12_DENSITY_G_PER_MM3


def _linspace(max_value: float, count: int) -> list[float]:
    count = max(8, int(count))
    return [max_value * i / (count - 1) for i in range(count)]


def _energy_until(disp: list[float], force: list[float], limit_mm: float) -> float:
    if len(disp) < 2:
        return 0.0
    d_out: list[float] = []
    f_out: list[float] = []
    for d, f in zip(disp, force):
        if d <= limit_mm:
            d_out.append(d)
            f_out.append(f)
        else:
            if d_out:
                d0, f0 = d_out[-1], f_out[-1]
                alpha = 0.0 if d <= d0 else (limit_mm - d0) / (d - d0)
                d_out.append(limit_mm)
                f_out.append(f0 + alpha * (f - f0))
            break
    return integrate_curve_energy_j(d_out, f_out) if len(d_out) >= 2 else 0.0


def _impact_stop_mm(disp: list[float], force: list[float], target_energy_j: float) -> float:
    energy = 0.0
    for d0, d1, f0, f1 in zip(disp[:-1], disp[1:], force[:-1], force[1:]):
        de = 0.5 * (max(0.0, f0) + max(0.0, f1)) * max(0.0, d1 - d0) / 1000.0
        if energy + de >= target_energy_j:
            alpha = 0.0 if de <= 1e-12 else (target_energy_j - energy) / de
            return d0 + alpha * (d1 - d0)
        energy += de
    return disp[-1] if disp else 0.0


def _risk(energy_j: float, peak_n: float, crush_mm: float, params: DesignParams, config: PhysicsBakeoffConfig) -> float:
    energy_margin = energy_j - config.target_energy_j
    return _clamp(
        0.05
        + 0.34 / (1.0 + math.exp(energy_margin / 3.5))
        + 0.28 / (1.0 + math.exp(-(peak_n - config.peak_limit_n) / 550.0))
        + 0.24 / (1.0 + math.exp((crush_mm - config.min_crush_mm) / 2.2))
        + 0.08 / (1.0 + math.exp((params.min_feature_mm - 0.58) / 0.04)),
        0.0,
        1.0,
    )


def _score(mass_g: float, energy_j: float, peak_n: float, crush_mm: float, risk: float, config: PhysicsBakeoffConfig) -> float:
    return (
        mass_g
        + 65.0 * max(0.0, config.target_energy_j - energy_j)
        + 0.040 * max(0.0, peak_n - config.peak_limit_n)
        + 24.0 * max(0.0, config.min_crush_mm - crush_mm)
        + 58.0 * risk
    )


def _result_from_curve(
    model: str,
    rank: int,
    params: DesignParams,
    mass_g: float,
    disp: list[float],
    force: list[float],
    collapse_mm: float,
    risk_override: float | None,
    config: PhysicsBakeoffConfig,
) -> BakeoffModelResult:
    collapse = _clamp(collapse_mm, 0.0, config.max_displacement_mm)
    energy = _energy_until(disp, force, collapse)
    stop = _impact_stop_mm([d for d in disp if d <= collapse], [f for d, f in zip(disp, force) if d <= collapse], config.target_energy_j)
    if energy < config.target_energy_j:
        peak_limit = collapse
    else:
        peak_limit = min(collapse, stop)
    active_peak = [f for d, f in zip(disp, force) if d <= peak_limit]
    peak = max(active_peak) if active_peak else 0.0
    risk = _risk(energy, peak, collapse, params, config) if risk_override is None else risk_override
    energy_pass = energy >= config.target_energy_j
    peak_pass = peak <= config.peak_limit_n
    crush_pass = collapse >= config.min_crush_mm
    model_pass = energy_pass and peak_pass and crush_pass and risk <= config.max_failure_risk
    score = _score(mass_g, energy, peak, collapse, risk, config)
    return BakeoffModelResult(
        model=model,
        rank=rank,
        topology=params.topology,
        mass_g=mass_g,
        energy_usable_J=energy,
        peak_force_N=peak,
        crush_mm=collapse,
        impact_stop_mm=stop,
        failure_risk=risk,
        energy_pass=energy_pass,
        peak_pass=peak_pass,
        crush_pass=crush_pass,
        model_pass=model_pass,
        model_score=score,
        displacement_mm=disp,
        force_N=force,
    )


def _physics_config(config: PhysicsBakeoffConfig) -> PhysicsConfig:
    return PhysicsConfig(
        layers=config.layers,
        steps=config.steps,
        max_displacement_mm=config.max_displacement_mm,
        dynamic_amplification=config.dynamic_amplification,
        yield_scale=config.yield_scale,
        fixture_peak_force_limit_n=config.peak_limit_n,
        target_min_crush_mm=config.min_crush_mm,
    )


def _layered_v1(row: dict[str, str], params: DesignParams, rank: int, mass_g: float, config: PhysicsBakeoffConfig) -> BakeoffModelResult:
    physics_config = _physics_config(config)
    result, sim = evaluate_candidate_physics(row, config=physics_config)
    return BakeoffModelResult(
        model="layered_v1",
        rank=rank,
        topology=params.topology,
        mass_g=mass_g,
        energy_usable_J=float(result.physics_energy_usable_J),
        peak_force_N=float(result.physics_peak_force_N),
        crush_mm=float(result.physics_collapse_mm),
        impact_stop_mm=float(result.physics_impact_stop_mm),
        failure_risk=float(result.physics_failure_risk),
        energy_pass=bool(result.physics_energy_pass),
        peak_pass=bool(result.physics_peak_force_pass),
        crush_pass=bool(result.physics_crush_distance_pass),
        model_pass=bool(result.physics_survives_gate),
        model_score=float(result.physics_score),
        displacement_mm=[float(x) for x in sim["displacement_mm"]],  # type: ignore[index]
        force_N=[float(x) for x in sim["force_N"]],  # type: ignore[index]
    )


def _layered_soft(row: dict[str, str], params: DesignParams, rank: int, mass_g: float, config: PhysicsBakeoffConfig) -> BakeoffModelResult:
    base = _physics_config(config)
    variant = replace(
        base,
        dynamic_amplification=max(0.92, 0.88 * base.dynamic_amplification),
        yield_scale=0.62 * base.yield_scale,
        min_effective_area_fraction=max(0.08, 0.68 * base.min_effective_area_fraction),
        max_effective_area_fraction=max(0.32, 0.78 * base.max_effective_area_fraction),
        fracture_strain_base=min(1.04, base.fracture_strain_base + 0.12),
    )
    sim = run_layered_crush_fem(params, variant)
    return _result_from_curve(
        "layered_soft",
        rank,
        params,
        mass_g,
        [float(x) for x in sim["displacement_mm"]],  # type: ignore[index]
        [float(x) for x in sim["force_N"]],  # type: ignore[index]
        float(sim["collapse_mm"]),
        None,
        config,
    )


def _homogenized_ga(row: dict[str, str], params: DesignParams, rank: int, mass_g: float, config: PhysicsBakeoffConfig) -> BakeoffModelResult:
    rel = estimate_relative_density(params)
    plateau_factor, stiffness_factor, progressivity_factor, fracture_penalty = _topology_physics_factors(params.topology)
    slenderness = params.wall_thickness_mm / max(params.cell_size_mm, 1e-6)
    area_mm2 = ENVELOPE_MM * ENVELOPE_MM
    cell_factor = _clamp(6.4 / max(params.cell_size_mm, 1e-6), 0.72, 1.35)
    anisotropy = _clamp(params.anisotropy_xy, 0.82, 1.22)
    sigma_plateau = (
        PA12_YIELD_STRESS_MPA
        * config.yield_scale
        * config.dynamic_amplification
        * plateau_factor
        * (0.82 + 0.38 * _clamp(slenderness / 0.16, 0.0, 1.5))
        * (rel ** 1.72)
        * cell_factor
        * anisotropy
    )
    area_eff = _clamp(0.18 + 1.25 * rel + 0.10 * params.edge_rib_mm, 0.12, 0.56)
    plateau_force = max(25.0, sigma_plateau * area_mm2 * area_eff)
    densification_strain = _clamp(
        (0.86 - 1.08 * rel) * progressivity_factor
        + 0.09 * params.trigger_layer_strength
        + 0.05 * _clamp(params.plateau_zone_mm / 30.0, 0.0, 1.25)
        - 0.03 * max(0.0, fracture_penalty - 1.0),
        0.34,
        0.88,
    )
    crush_mm = _clamp(densification_strain * ENVELOPE_MM, 5.0, config.max_displacement_mm)
    yield_mm = _clamp((0.0025 + 0.016 * rel) * ENVELOPE_MM, 0.25, 2.3)
    dense_start = 0.92 * crush_mm
    dense_stiffness = PA12_ELASTIC_MODULUS_MPA * stiffness_factor * area_eff * area_mm2 / ENVELOPE_MM * (0.012 + 0.08 * rel)
    disp = _linspace(config.max_displacement_mm, config.steps)
    force: list[float] = []
    for d in disp:
        if d <= yield_mm:
            f = plateau_force * d / max(yield_mm, 1e-9)
        elif d <= dense_start:
            x = (d - yield_mm) / max(1e-9, dense_start - yield_mm)
            f = plateau_force * (0.94 + 0.12 * x + 0.04 * math.sin(math.pi * x))
        else:
            over = d - dense_start
            f = plateau_force * 1.08 + dense_stiffness * (over / max(1.0, ENVELOPE_MM - dense_start)) ** 2
        force.append(max(0.0, f))
    return _result_from_curve("homogenized_ga", rank, params, mass_g, disp, force, crush_mm, None, config)


def _buckling_stack(row: dict[str, str], params: DesignParams, rank: int, mass_g: float, config: PhysicsBakeoffConfig) -> BakeoffModelResult:
    base = _physics_config(config)
    variant = replace(
        base,
        yield_scale=0.82 * base.yield_scale,
        dynamic_amplification=max(1.0, 0.96 * base.dynamic_amplification),
        fracture_strain_base=min(1.0, base.fracture_strain_base + 0.08),
    )
    layers = list(reversed(build_layer_stack(params, variant)))
    plateau_factor, _, progressivity_factor, fracture_penalty = _topology_physics_factors(params.topology)
    layer_caps: list[float] = []
    layer_forces: list[float] = []
    for i, layer in enumerate(layers):
        z_progress = i / max(1, len(layers) - 1)
        cap = layer.height_mm * _clamp(
            0.54
            + 0.30 * layer.fracture_strain
            + 0.10 * progressivity_factor
            + 0.08 * params.trigger_layer_strength
            - 0.16 * layer.rel_density,
            0.28,
            0.90,
        )
        layer_caps.append(cap)
        euler_like = PA12_ELASTIC_MODULUS_MPA * (params.wall_thickness_mm / max(params.cell_size_mm, 1e-6)) ** 3
        geometric_force = euler_like * ENVELOPE_MM * ENVELOPE_MM * (0.55 + 0.75 * layer.rel_density)
        force = (
            0.58 * layer.plateau_force_n
            + 0.012 * geometric_force
            + 24.0
            + 18.0 * z_progress * params.vertical_gradient
        ) * _clamp(plateau_factor, 0.82, 1.18) * _clamp(1.08 - 0.08 * fracture_penalty, 0.88, 1.08)
        layer_forces.append(max(18.0, force))

    progressive_capacity = min(config.max_displacement_mm, sum(layer_caps))
    disp = _linspace(config.max_displacement_mm, config.steps)
    force: list[float] = []
    cumulative = []
    total = 0.0
    for cap in layer_caps:
        total += cap
        cumulative.append(total)

    for d in disp:
        if d <= progressive_capacity:
            idx = 0
            while idx < len(cumulative) - 1 and cumulative[idx] < d:
                idx += 1
            prev = 0.0 if idx == 0 else cumulative[idx - 1]
            local = _clamp((d - prev) / max(layer_caps[idx], 1e-9), 0.0, 1.0)
            f = layer_forces[idx] * (0.82 + 0.26 * local) + 0.06 * max(layer_forces[: idx + 1])
        else:
            over = d - progressive_capacity
            dense_force = max(layer_forces) * (1.08 + 0.11 * estimate_relative_density(params))
            dense_stiffness = max(350.0, dense_force * (0.35 + 3.2 * estimate_relative_density(params)))
            f = dense_force + dense_stiffness * (over / max(1.0, config.max_displacement_mm - progressive_capacity)) ** 2
        force.append(max(0.0, f))

    return _result_from_curve("buckling_stack", rank, params, mass_g, disp, force, progressive_capacity, None, config)


ModelRunner = Callable[[dict[str, str], DesignParams, int, float, PhysicsBakeoffConfig], BakeoffModelResult]

MODEL_RUNNERS: dict[str, ModelRunner] = {
    "layered_v1": _layered_v1,
    "layered_soft": _layered_soft,
    "homogenized_ga": _homogenized_ga,
    "buckling_stack": _buckling_stack,
}


def _aggregate(row: dict[str, str], model_results: list[BakeoffModelResult], config: PhysicsBakeoffConfig) -> dict[str, object]:
    energies = [result.energy_usable_J for result in model_results]
    peaks = [result.peak_force_N for result in model_results]
    crushes = [result.crush_mm for result in model_results]
    risks = [result.failure_risk for result in model_results]
    energy_passes = sum(1 for result in model_results if result.energy_pass)
    peak_passes = sum(1 for result in model_results if result.peak_pass)
    crush_passes = sum(1 for result in model_results if result.crush_pass)
    model_passes = sum(1 for result in model_results if result.model_pass)
    model_count = max(1, len(model_results))
    energy_min = min(energies) if energies else 0.0
    peak_max = max(peaks) if peaks else 0.0
    crush_min = min(crushes) if crushes else 0.0
    risk_max = max(risks) if risks else 1.0
    robust_pass = (
        energy_passes >= math.ceil(0.5 * model_count)
        and peak_passes >= math.ceil(0.5 * model_count)
        and crush_passes >= math.ceil(0.5 * model_count)
        and model_passes >= max(1, math.ceil(0.34 * model_count))
    )
    consensus_score = (
        float(row.get("score", "0") or 0.0)
        + 80.0 * max(0.0, config.target_energy_j - energy_min)
        + 0.050 * max(0.0, peak_max - config.peak_limit_n)
        + 28.0 * max(0.0, config.min_crush_mm - crush_min)
        + 65.0 * risk_max
        - 6.0 * model_passes
        - 1.8 * (energy_passes + peak_passes + crush_passes)
    )
    return {
        "model_count": model_count,
        "models_energy_pass": energy_passes,
        "models_peak_pass": peak_passes,
        "models_crush_pass": crush_passes,
        "models_full_pass": model_passes,
        "consensus_energy_min_J": energy_min,
        "consensus_peak_max_N": peak_max,
        "consensus_crush_min_mm": crush_min,
        "consensus_failure_risk_max": risk_max,
        "robust_pass": robust_pass,
        "bakeoff_score": consensus_score,
    }


def _write_curve(path: Path, results: list[BakeoffModelResult]) -> None:
    ensure_dir(path.parent)
    max_len = max((len(result.displacement_mm) for result in results), default=0)
    fieldnames = ["index", "displacement_mm"] + [f"{result.model}_force_N" for result in results]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(max_len):
            disp = next((result.displacement_mm[i] for result in results if i < len(result.displacement_mm)), 0.0)
            row: dict[str, object] = {"index": i, "displacement_mm": disp}
            for result in results:
                row[f"{result.model}_force_N"] = result.force_N[i] if i < len(result.force_N) else ""
            writer.writerow({key: _format(value) for key, value in row.items()})


def run_physics_bakeoff(
    top_csv: Path,
    out_dir: Path,
    top_n: int = 500,
    config: PhysicsBakeoffConfig | None = None,
) -> dict[str, object]:
    config = config or PhysicsBakeoffConfig()
    out_dir = ensure_dir(out_dir)
    curve_dir = ensure_dir(out_dir / "curves")
    requested_models = tuple(config.models)
    unknown = [name for name in requested_models if name not in MODEL_RUNNERS]
    if unknown:
        raise ValueError(f"unknown bakeoff model(s): {', '.join(unknown)}")

    input_rows = _read_csv(Path(top_csv))[:top_n]
    output_rows: list[dict[str, object]] = []
    model_counts = {name: {"energy": 0, "peak": 0, "crush": 0, "full": 0} for name in requested_models}
    failure_reasons: dict[str, int] = {}
    for row_i, source_row in enumerate(input_rows, start=1):
        params = DesignParams.from_row(source_row)
        rank = int(float(source_row.get("rank", row_i) or row_i))
        mass_g = _mass_from_row(source_row, params)
        model_results = [MODEL_RUNNERS[name](source_row, params, rank, mass_g, config) for name in requested_models]
        out_row: dict[str, object] = {
            "rank": rank,
            "source_order": row_i,
            "topology": params.topology,
            "mass_g": mass_g,
            "source_score": source_row.get("score", ""),
        }
        out_row.update(_aggregate(source_row, model_results, config))
        for result in model_results:
            prefix = result.model
            out_row.update(result.metric_row(prefix))
            model_counts[result.model]["energy"] += int(result.energy_pass)
            model_counts[result.model]["peak"] += int(result.peak_pass)
            model_counts[result.model]["crush"] += int(result.crush_pass)
            model_counts[result.model]["full"] += int(result.model_pass)
        reasons = []
        if int(out_row["models_energy_pass"]) == 0:
            reasons.append("energy")
        if int(out_row["models_peak_pass"]) == 0:
            reasons.append("peak")
        if int(out_row["models_crush_pass"]) == 0:
            reasons.append("crush")
        if not reasons:
            reasons.append("mixed")
        reason_key = "+".join(reasons)
        out_row["failure_reason_summary"] = reason_key
        failure_reasons[reason_key] = failure_reasons.get(reason_key, 0) + 1
        output_rows.append(out_row)
        if row_i <= config.write_curves:
            _write_curve(curve_dir / f"rank_{rank:03d}_{params.topology}_bakeoff_curves.csv", model_results)

    ranked = sorted(output_rows, key=lambda item: float(item["bakeoff_score"]))
    fieldnames = list(ranked[0].keys()) if ranked else ["rank", "bakeoff_score"]
    out_csv = out_dir / "physics_bakeoff_candidates.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: _format(row.get(key)) for key in fieldnames})

    robust_rows = [row for row in ranked if str(row.get("robust_pass")) == "True" or row.get("robust_pass") is True]
    robust_csv = out_dir / "robust_candidates.csv"
    with robust_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in robust_rows:
            writer.writerow({key: _format(row.get(key)) for key in fieldnames})

    summary = {
        "top_csv": str(top_csv),
        "out_csv": str(out_csv),
        "robust_csv": str(robust_csv),
        "curve_dir": str(curve_dir),
        "evaluated": len(output_rows),
        "robust_pass": len(robust_rows),
        "models": requested_models,
        "model_pass_counts": model_counts,
        "failure_reasons": failure_reasons,
        "config": asdict(config),
        "best": ranked[0] if ranked else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# POLMI Physics Bakeoff",
        "",
        f"Input: `{top_csv}`",
        f"Evaluated: `{len(output_rows)}`",
        f"Robust pass: `{len(robust_rows)}`",
        "",
        "| Rank | Topology | Score | Model Passes | E min J | Peak max N | Crush min mm | Reason |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked[:30]:
        lines.append(
            f"| {row['rank']} | {row['topology']} | {float(row['bakeoff_score']):.2f} | "
            f"{row['models_full_pass']} | {float(row['consensus_energy_min_J']):.2f} | "
            f"{float(row['consensus_peak_max_N']):.0f} | {float(row['consensus_crush_min_mm']):.1f} | "
            f"{row['failure_reason_summary']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
