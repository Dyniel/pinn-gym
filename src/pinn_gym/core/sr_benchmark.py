"""Scientific Reports benchmark orchestration.

This module ties the material-aware sampler, the declared physics oracle and
the dimensionless material-PINN together into a small, scriptable pipeline.
It writes per-material datasets, trains the dimensionless PINN, evaluates
per-material curve/energy/violation/ranking metrics, and produces a
cross-material transfer matrix.

The pipeline is intentionally split into Slurm-friendly stages:

* :func:`build_sr_dataset` builds material-specific candidate pools and runs
  the declared oracle (CPU only).
* :func:`train_sr_models` trains one or more PINN variants on a single
  material's pool (per-material job in a Slurm array). Optionally trains a
  pooled multi-material model.
* :func:`evaluate_sr_run` runs baselines + checkpoints and writes ranking
  tables and the cross-material transfer matrix.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .design_space import (
    DesignParams,
    design_to_candidate_row,
    displacement_axis,
    pseudo_response,
)
from .dimensionless import (
    DimensionlessScales,
    material_dimensionless_features,
    scales_for_material,
)
from .material_aware_sampler import material_derived_min_crush_mm, sample_material_aware_designs
from .material_pinn import (
    MaterialPINNConfig,
    predict_force_curves,
    train_material_pinn,
)
from .materials import MaterialCard, load_material_card
from .metrics import (
    best_feasible_regret,
    best_selected_feasible_mass,
    force_curve_error_metrics,
    integrate_energy_j,
    precision_at_k,
    relative_best_feasible_regret,
)
from .paths import ensure_dir
from .physics import IMPACT_ENERGY_J, PhysicsConfig, evaluate_candidate_physics, run_dynamic_impact


@dataclass(frozen=True)
class SRBuildConfig:
    presets: tuple[str, ...] = ("pa12", "pla", "petg", "tpu", "pa_cf")
    train_n: int = 4000
    eval_n: int = 1200
    seed: int = 20260519
    layers: int = 96
    steps: int = 320
    max_displacement_mm: float = 50.0
    dynamic_amplification: float = 1.16
    yield_scale: float = 0.08
    fixture_peak_force_limit_n: float = 3500.0
    target_min_crush_mm: float = 40.0
    # When True, the build derives a per-material ``target_min_crush_mm`` from
    # the material card's failure_strain so brittle cards (PLA/PETG/PA-CF) do
    # not collapse to zero feasibility purely because PA12's 40 mm rule is
    # unreachable under the reduced-order oracle for them.
    material_aware_crush_target: bool = True
    # 0 = single-process; positive = number of worker processes for the oracle.
    # The oracle is CPU-bound (NumPy-free pure Python), so workers scale nearly
    # linearly. Defaults to ``min(os.cpu_count(), 8)`` when 0 and ``os`` is
    # available, but stays single-threaded when explicitly set to 1.
    oracle_workers: int = 0


@dataclass(frozen=True)
class SREvalConfig:
    precision_ks: tuple[int, ...] = (1, 3, 5, 10, 25, 50)
    target_energy_j: float = IMPACT_ENERGY_J
    peak_limit_n: float = 3500.0
    min_crush_mm: float = 40.0
    curve_limit_mm: float = 40.0
    random_seed: int = 20260519


def build_sr_dataset(
    out_dir: Path,
    *,
    config: SRBuildConfig | None = None,
) -> dict[str, object]:
    """Generate material-aware candidate pools and oracle labels per material.

    Unlike :func:`pinn_gym.core.material_gym.build_material_gym_datasets`, the sampler
    is *re-seeded per material* and tuned to that material's yield strength,
    so every preset receives a candidate pool that brackets its declared
    feasibility frontier.
    """

    config = config or SRBuildConfig()
    out_dir = ensure_dir(out_dir)
    summaries: list[dict[str, object]] = []
    for offset, preset in enumerate(config.presets):
        material = load_material_card(preset)
        material_crush_target = (
            material_derived_min_crush_mm(material, envelope_mm=config.max_displacement_mm)
            if config.material_aware_crush_target
            else config.target_min_crush_mm
        )
        # Never relax above what the user asked for: the PA12 default of 40 mm
        # remains an upper bound, but brittle cards get a physically reachable
        # target so their pool can contain feasible designs.
        effective_target_min_crush_mm = min(config.target_min_crush_mm, material_crush_target)
        physics_config = PhysicsConfig(
            layers=config.layers,
            steps=config.steps,
            max_displacement_mm=config.max_displacement_mm,
            dynamic_amplification=config.dynamic_amplification,
            yield_scale=config.yield_scale,
            fixture_peak_force_limit_n=config.fixture_peak_force_limit_n,
            target_min_crush_mm=effective_target_min_crush_mm,
            material=material,
        )
        train_designs = sample_material_aware_designs(
            material,
            config.train_n,
            seed=config.seed + offset * 1009,
            fixture_peak_force_limit_n=config.fixture_peak_force_limit_n,
            target_min_crush_mm=effective_target_min_crush_mm,
        )
        eval_designs = sample_material_aware_designs(
            material,
            config.eval_n,
            seed=config.seed + offset * 1009 + 7,
            fixture_peak_force_limit_n=config.fixture_peak_force_limit_n,
            target_min_crush_mm=effective_target_min_crush_mm,
        )
        preset_dir = ensure_dir(out_dir / preset)
        material.to_json(preset_dir / "material_card.json")
        scales = scales_for_material(material)
        (preset_dir / "scales.json").write_text(json.dumps(scales.to_dict(), indent=2), encoding="utf-8")
        (preset_dir / "material_features.json").write_text(
            json.dumps(material_dimensionless_features(material), indent=2), encoding="utf-8"
        )
        train_rows = _evaluate_oracle_batch(train_designs, physics_config, config.oracle_workers)
        eval_rows = _evaluate_oracle_batch(eval_designs, physics_config, config.oracle_workers)
        _write_rows(preset_dir / "train.csv", train_rows)
        _write_rows(preset_dir / "eval.csv", eval_rows)
        feasible_train = sum(1 for row in train_rows if _truthy(row["oracle_feasible"]))
        feasible_eval = sum(1 for row in eval_rows if _truthy(row["oracle_feasible"]))
        summary = {
            "preset": preset,
            "material_name": material.material_name,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_feasible": feasible_train,
            "eval_feasible": feasible_eval,
            "train_feasible_rate": feasible_train / max(1, len(train_rows)),
            "eval_feasible_rate": feasible_eval / max(1, len(eval_rows)),
            "target_min_crush_mm_effective": effective_target_min_crush_mm,
        }
        (preset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
    payload = {"out_dir": str(out_dir), "config": asdict(config), "datasets": summaries}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_dataset_readme(payload), encoding="utf-8")
    return payload


def train_sr_models(
    dataset_dir: Path,
    checkpoint_root: Path,
    *,
    presets: list[str],
    pooled: bool = True,
    methods: tuple[str, ...] = ("mlp_softplus", "pinn_energy", "pinn_full"),
    config: MaterialPINNConfig | None = None,
) -> dict[str, object]:
    """Train PINN variants. If ``pooled`` is True, also trains a multi-material
    model on the union of all presets (this is the SR main run); otherwise
    trains per-material models for the single-preset transfer baseline.
    """

    config = config or MaterialPINNConfig()
    dataset_dir = Path(dataset_dir)
    checkpoint_root = ensure_dir(Path(checkpoint_root))
    trained: list[dict[str, object]] = []
    if pooled:
        out = ensure_dir(checkpoint_root / "pooled")
        for method in methods:
            metrics = train_material_pinn(
                dataset_dir,
                out,
                presets=list(presets),
                method=method,
                config=config,
            )
            trained.append({"scope": "pooled", "presets": list(presets), "method": method, **metrics})
    else:
        for preset in presets:
            out = ensure_dir(checkpoint_root / preset)
            for method in methods:
                metrics = train_material_pinn(
                    dataset_dir,
                    out,
                    presets=[preset],
                    method=method,
                    config=config,
                )
                trained.append({"scope": "per_material", "presets": [preset], "method": method, **metrics})
    summary_path = checkpoint_root / ("train_summary_pooled.json" if pooled else "train_summary_per_material.json")
    summary_path.write_text(
        json.dumps([{k: v for k, v in entry.items() if k != "history"} for entry in trained], indent=2),
        encoding="utf-8",
    )
    return {"trained": trained, "summary": str(summary_path)}


def evaluate_sr_run(
    dataset_dir: Path,
    checkpoint_root: Path,
    out_dir: Path,
    *,
    presets: list[str],
    config: SREvalConfig | None = None,
    include_transfer: bool = True,
) -> dict[str, object]:
    """Evaluate baselines + trained checkpoints, including cross-material transfer."""

    config = config or SREvalConfig()
    dataset_dir = Path(dataset_dir)
    out_dir = ensure_dir(out_dir)
    checkpoint_root = Path(checkpoint_root) if checkpoint_root else None

    method_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for eval_preset in presets:
        preset_dir = dataset_dir / eval_preset
        eval_csv = preset_dir / "eval.csv"
        if not eval_csv.exists():
            continue
        eval_rows = _read_rows(eval_csv)
        material = load_material_card(preset_dir / "material_card.json" if (preset_dir / "material_card.json").exists() else eval_preset)
        scales = scales_for_material(material)
        baselines = _baseline_predictions(eval_rows, scales, config, eval_preset)
        for method, preds in baselines.items():
            method_rows.append(_aggregate_metrics(eval_preset, "self", method, preds, eval_rows, config))
            curve_rows.extend(_curve_records(eval_preset, "self", method, preds))
        if checkpoint_root is not None:
            # pooled and per_material checkpoints
            for scope_dir in ["pooled", eval_preset]:
                model_dir = checkpoint_root / scope_dir
                if not model_dir.exists():
                    continue
                for checkpoint in sorted(model_dir.glob("material_pinn_*.pt")):
                    method = checkpoint.stem.replace("material_pinn_", "") + (
                        "" if scope_dir == "pooled" else "_self"
                    )
                    scope = "pooled" if scope_dir == "pooled" else "per_material"
                    preds = _checkpoint_predictions(eval_rows, checkpoint, material, scales, config)
                    method_rows.append(_aggregate_metrics(eval_preset, scope, method, preds, eval_rows, config))
                    curve_rows.extend(_curve_records(eval_preset, scope, method, preds))

    # Cross-material transfer: pooled model is evaluated using each material's
    # scales (already covered above as 'pooled'); per-material checkpoints are
    # evaluated across other materials.
    if include_transfer and checkpoint_root is not None:
        for train_preset in presets:
            train_dir = checkpoint_root / train_preset
            if not train_dir.exists() or train_preset == "pooled":
                continue
            for checkpoint in sorted(train_dir.glob("material_pinn_*.pt")):
                method_base = checkpoint.stem.replace("material_pinn_", "")
                for eval_preset in presets:
                    if eval_preset == train_preset:
                        continue
                    preset_dir = dataset_dir / eval_preset
                    eval_csv = preset_dir / "eval.csv"
                    if not eval_csv.exists():
                        continue
                    eval_rows = _read_rows(eval_csv)
                    material = load_material_card(
                        preset_dir / "material_card.json"
                        if (preset_dir / "material_card.json").exists()
                        else eval_preset
                    )
                    scales = scales_for_material(material)
                    preds = _checkpoint_predictions(eval_rows, checkpoint, material, scales, config)
                    metrics = _aggregate_metrics(
                        eval_preset, "transfer", f"{method_base}_from_{train_preset}", preds, eval_rows, config
                    )
                    transfer_rows.append(
                        {**metrics, "train_preset": train_preset, "eval_preset": eval_preset, "method_base": method_base}
                    )

    metrics_csv = out_dir / "method_metrics.csv"
    transfer_csv = out_dir / "transfer_metrics.csv"
    curves_csv = out_dir / "per_candidate_curve_metrics.csv"
    _write_rows(metrics_csv, [{key: _format(value) for key, value in row.items()} for row in method_rows])
    _write_rows(transfer_csv, [{key: _format(value) for key, value in row.items()} for row in transfer_rows])
    _write_rows(curves_csv, [{key: _format(value) for key, value in row.items()} for row in curve_rows])
    payload = {
        "dataset_dir": str(dataset_dir),
        "checkpoint_root": str(checkpoint_root) if checkpoint_root else None,
        "out_dir": str(out_dir),
        "presets": presets,
        "config": asdict(config),
        "method_metrics_csv": str(metrics_csv),
        "transfer_metrics_csv": str(transfer_csv),
        "per_candidate_curve_metrics_csv": str(curves_csv),
        "rows": method_rows,
        "transfer_rows": transfer_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_evaluation_markdown(payload), encoding="utf-8")
    return payload


def _resolve_oracle_workers(requested: int) -> int:
    if requested >= 1:
        return int(requested)
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 8))


def _evaluate_oracle_batch(
    designs: list[DesignParams],
    physics_config: PhysicsConfig,
    workers_request: int,
) -> list[dict[str, object]]:
    if not designs:
        return []
    workers = _resolve_oracle_workers(workers_request)
    if workers <= 1 or len(designs) < 32:
        return [_oracle_row(design, physics_config) for design in designs]
    # Spread the work across processes. The oracle is fully pure-Python so the
    # GIL prevents threading from helping; processes are necessary.
    chunk = max(1, (len(designs) + workers - 1) // workers)
    args = [(designs[i : i + chunk], physics_config) for i in range(0, len(designs), chunk)]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_rows in pool.map(_oracle_chunk, args):
            rows.extend(chunk_rows)
    return rows


def _oracle_chunk(payload: tuple[list[DesignParams], PhysicsConfig]) -> list[dict[str, object]]:
    designs, physics_config = payload
    return [_oracle_row(design, physics_config) for design in designs]


def _oracle_row(design: DesignParams, config: PhysicsConfig) -> dict[str, object]:
    from .design_space import geometry_feature_row

    rank_field = abs(hash((design.topology, round(design.cell_size_mm, 4), round(design.wall_thickness_mm, 4)))) % 10_000_000
    row = {"rank": rank_field, **design.to_row()}
    row.update({k: _format(v) for k, v in geometry_feature_row(design).items()})
    physics, sim = evaluate_candidate_physics(row, config=config)
    impact, _ = run_dynamic_impact(design, physics_config=config, rank=rank_field, mass_g=physics.mass_g)
    disp = [float(x) for x in sim["displacement_mm"]]  # type: ignore[index]
    force = [float(x) for x in sim["force_N"]]  # type: ignore[index]
    out: dict[str, object] = {
        "rank": rank_field,
        **design.to_row(),
    }
    out.update({k: _format(v) for k, v in geometry_feature_row(design).items()})
    out.update(
        {
            "mass_g": physics.mass_g,
            "relative_density": physics.relative_density,
            "energy_abs_J": physics.physics_energy_usable_J,
            "force_peak_N": physics.physics_peak_force_N,
            "force_plateau_N": physics.physics_plateau_force_N,
            "early_energy_20mm_J": integrate_energy_j(disp, force, limit_mm=20.0),
            "peak_plateau_ratio": physics.physics_peak_force_N / max(physics.physics_plateau_force_N, 1e-12),
            "progressive_crush_score": max(0.0, 1.0 - physics.physics_plateau_cv),
            "collapse_displacement_mm": physics.physics_collapse_mm,
            "failure_probability": physics.physics_failure_risk,
        }
    )
    for i, value in enumerate(force):
        out[f"curve_{i:03d}"] = value
    out.update({f"physics_{key}": value for key, value in physics.to_row().items() if key not in out})
    out.update({f"impact_{key}": value for key, value in impact.to_row().items() if f"impact_{key}" not in out})
    out["oracle_feasible"] = physics.physics_survives_gate and impact.impact_survives
    return out


def _baseline_predictions(
    eval_rows: list[dict[str, str]],
    scales: DimensionlessScales,
    config: SREvalConfig,
    preset: str,
) -> dict[str, list[dict[str, object]]]:
    rng = random.Random(config.random_seed + sum(ord(ch) for ch in preset))
    out = {"random": [], "lightest": [], "pseudo_bootstrap": [], "oracle_upper_bound": []}
    for idx, row in enumerate(eval_rows):
        oracle_disp, oracle_force = _curve_from_row(row)
        pseudo = pseudo_response(DesignParams.from_row(row), curve_points=len(oracle_disp))
        pseudo_disp = [float(x) for x in pseudo["curve_displacement_mm"]]  # type: ignore[index]
        pseudo_force = [float(x) for x in pseudo["curve_force_N"]]  # type: ignore[index]
        pseudo_energy = integrate_energy_j(pseudo_disp, pseudo_force, limit_mm=config.curve_limit_mm)
        pseudo_peak = max(pseudo_force) if pseudo_force else 0.0
        base = _prediction_row(row, pseudo_disp, pseudo_force, config)
        base.update(
            {
                "pred_energy_J": pseudo_energy,
                "pred_peak_N": pseudo_peak,
                "pred_feasible": pseudo_energy >= config.target_energy_j and pseudo_peak <= config.peak_limit_n,
            }
        )
        out["pseudo_bootstrap"].append(
            {
                **base,
                "pred_score": _score(_f(row, "mass_g"), pseudo_energy, pseudo_peak, config),
            }
        )
        out["lightest"].append({**base, "pred_score": _f(row, "mass_g"), "pred_feasible": True})
        out["random"].append({**base, "pred_score": rng.random() + idx * 1e-9, "pred_feasible": True})
        oracle_energy = _f(row, "energy_abs_J")
        oracle_peak = _f(row, "force_peak_N")
        out["oracle_upper_bound"].append(
            {
                **_prediction_row(row, oracle_disp, oracle_force, config),
                "pred_score": _f(row, "mass_g") if _truthy(row.get("oracle_feasible")) else math.inf,
                "pred_energy_J": oracle_energy,
                "pred_peak_N": oracle_peak,
                "pred_feasible": _truthy(row.get("oracle_feasible")),
            }
        )
    return out


def _checkpoint_predictions(
    eval_rows: list[dict[str, str]],
    checkpoint: Path,
    material: MaterialCard,
    scales: DimensionlessScales,
    config: SREvalConfig,
) -> list[dict[str, object]]:
    raw = predict_force_curves(checkpoint, eval_rows, material, device="cpu")
    out: list[dict[str, object]] = []
    for row, prediction in zip(eval_rows, raw):
        pred_disp = list(prediction["pred_displacement_mm"])  # type: ignore[arg-type]
        pred_force = list(prediction["pred_force_N"])  # type: ignore[arg-type]
        pred_energy = integrate_energy_j(pred_disp, pred_force, limit_mm=config.curve_limit_mm)
        pred_peak = max(pred_force) if pred_force else 0.0
        base = _prediction_row(row, pred_disp, pred_force, config)
        base.update(
            {
                "pred_energy_J": pred_energy,
                "pred_peak_N": pred_peak,
                "pred_feasible": pred_energy >= config.target_energy_j and pred_peak <= config.peak_limit_n,
                "pred_score": _score(_f(row, "mass_g"), pred_energy, pred_peak, config),
            }
        )
        out.append(base)
    return out


def _aggregate_metrics(
    preset: str,
    scope: str,
    method: str,
    preds: list[dict[str, object]],
    eval_rows: list[dict[str, str]],
    config: SREvalConfig,
) -> dict[str, object]:
    ranked = sorted(preds, key=lambda row: _f(row, "pred_score", math.inf))
    feasible = [_truthy(row.get("oracle_feasible")) for row in ranked]
    masses = [_f(row, "mass_g") for row in ranked]
    oracle_masses = [_f(row, "mass_g") for row in eval_rows if _truthy(row.get("oracle_feasible"))]
    out: dict[str, object] = {
        "preset": preset,
        "scope": scope,
        "method": method,
        "evaluated": len(ranked),
        "oracle_feasible": len(oracle_masses),
        "best_oracle_feasible_mass_g": min(oracle_masses, default=math.nan),
        "predicted_feasible_rate": sum(1 for row in ranked if _truthy(row.get("pred_feasible"))) / max(1, len(ranked)),
        "physical_violation_rate": _predicted_violation_rate(ranked),
        "mean_curve_rmse_N": _mean(_f(row, "curve_rmse_N") for row in ranked),
        "mean_curve_nrmse": _mean(_f(row, "curve_nrmse") for row in ranked),
        "mean_energy_integral_abs_error_J": _mean(_f(row, "energy_integral_abs_error_J") for row in ranked),
    }
    for k in config.precision_ks:
        out[f"precision_at_{k}"] = precision_at_k(feasible, k)
        out[f"best_selected_feasible_mass_at_{k}_g"] = best_selected_feasible_mass(masses[:k], feasible[:k])
        out[f"regret_at_{k}_g"] = best_feasible_regret(masses[:k], feasible[:k], oracle_masses)
        out[f"relative_regret_at_{k}"] = relative_best_feasible_regret(masses[:k], feasible[:k], oracle_masses)
    return out


def _curve_records(preset: str, scope: str, method: str, preds: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "preset": preset,
            "scope": scope,
            "method": method,
            "rank": row.get("rank", ""),
            "topology": row.get("topology", ""),
            "mass_g": row.get("mass_g", ""),
            "oracle_feasible": row.get("oracle_feasible", ""),
            "pred_feasible": row.get("pred_feasible", ""),
            "pred_score": row.get("pred_score", ""),
            "curve_rmse_N": row.get("curve_rmse_N", ""),
            "curve_nrmse": row.get("curve_nrmse", ""),
            "energy_integral_abs_error_J": row.get("energy_integral_abs_error_J", ""),
        }
        for row in preds
    ]


def _prediction_row(
    row: dict[str, str],
    pred_disp: list[float],
    pred_force: list[float],
    config: SREvalConfig,
) -> dict[str, object]:
    oracle_disp, oracle_force = _curve_from_row(row)
    errors = force_curve_error_metrics(pred_disp, pred_force, oracle_disp, oracle_force, limit_mm=config.curve_limit_mm)
    return {
        "rank": row.get("rank", ""),
        "topology": row.get("topology", ""),
        "mass_g": _f(row, "mass_g"),
        "oracle_feasible": row.get("oracle_feasible", False),
        **errors,
    }


def _curve_from_row(row: dict[str, str]) -> tuple[list[float], list[float]]:
    fields = sorted(key for key in row if key.startswith("curve_"))
    force = [_f(row, key, 0.0) for key in fields]
    return displacement_axis(len(force) or 96), force


def _score(mass_g: float, energy_j: float, peak_n: float, config: SREvalConfig) -> float:
    energy_gap = max(0.0, config.target_energy_j - energy_j)
    peak_excess = max(0.0, peak_n - config.peak_limit_n)
    if energy_gap <= 1e-12 and peak_excess <= 1e-12:
        return mass_g
    return mass_g + 10_000.0 + 80.0 * energy_gap + 0.030 * peak_excess


def _predicted_violation_rate(rows: list[dict[str, object]]) -> float:
    if not rows:
        return math.nan
    predicted = [row for row in rows if _truthy(row.get("pred_feasible"))]
    if not predicted:
        return 0.0
    violations = sum(1 for row in predicted if not _truthy(row.get("oracle_feasible")))
    return violations / len(predicted)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else str(value)
    return value


def _mean(values: Any) -> float:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    return sum(cleaned) / len(cleaned) if cleaned else math.nan


def _dataset_readme(payload: dict[str, object]) -> str:
    lines = ["# POLMI SR material-aware pools", ""]
    for item in payload["datasets"]:  # type: ignore[index]
        lines.append(
            "- {preset}: train={train_rows} (feasible {train_feasible}/{train_feasible_rate:.2%}), "
            "eval={eval_rows} (feasible {eval_feasible}/{eval_feasible_rate:.2%})".format(**item)
        )
    return "\n".join(lines) + "\n"


def _evaluation_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# POLMI SR Benchmark Evaluation",
        "",
        f"Dataset: `{payload['dataset_dir']}`",
        f"Checkpoints: `{payload['checkpoint_root']}`",
        "",
        "## Per-material metrics",
        "",
        "| Material | Scope | Method | P@1 | P@5 | P@10 | Regret@10 g | Curve NRMSE | Energy err J | Pred violation |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:  # type: ignore[index]
        lines.append(
            f"| {row['preset']} | {row['scope']} | {row['method']} | "
            f"{float(row.get('precision_at_1', math.nan)):.3f} | "
            f"{float(row.get('precision_at_5', math.nan)):.3f} | "
            f"{float(row.get('precision_at_10', math.nan)):.3f} | "
            f"{float(row.get('regret_at_10_g', math.nan)):.3f} | "
            f"{float(row.get('mean_curve_nrmse', math.nan)):.3f} | "
            f"{float(row.get('mean_energy_integral_abs_error_J', math.nan)):.3f} | "
            f"{float(row.get('physical_violation_rate', math.nan)):.3f} |"
        )
    transfer = payload.get("transfer_rows", [])
    if transfer:
        lines.extend(["", "## Cross-material transfer", "",
                       "| Train | Eval | Method | P@5 | Curve NRMSE | Pred violation |",
                       "|---|---|---|---:|---:|---:|"])
        for row in transfer:
            lines.append(
                f"| {row.get('train_preset', '')} | {row.get('eval_preset', '')} | {row.get('method_base', '')} | "
                f"{float(row.get('precision_at_5', math.nan)):.3f} | "
                f"{float(row.get('mean_curve_nrmse', math.nan)):.3f} | "
                f"{float(row.get('physical_violation_rate', math.nan)):.3f} |"
            )
    return "\n".join(lines) + "\n"
