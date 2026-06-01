"""Material-specific POLMI PINN gym datasets and comparisons."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .design_space import (
    CURVE_POINTS,
    PARAM_FIELDS,
    SCALAR_TARGET_FIELDS,
    DesignParams,
    design_to_candidate_row,
    displacement_axis,
    geometry_feature_row,
    pseudo_response,
    row_to_features,
    sample_designs,
)
from .materials import load_material_card
from .metrics import best_feasible_regret, force_curve_error_metrics, integrate_energy_j, precision_at_k
from .paths import ensure_dir
from .physics import IMPACT_ENERGY_J, PhysicsConfig, evaluate_candidate_physics, run_dynamic_impact
from .pinn import CrushPINN


@dataclass(frozen=True)
class MaterialGymBuildConfig:
    presets: tuple[str, ...] = ("pa12", "pla", "petg", "tpu", "pa_cf")
    train_n: int = 5000
    eval_n: int = 1000
    seed: int = 20260518
    layers: int = 96
    steps: int = 360
    max_displacement_mm: float = 50.0
    dynamic_amplification: float = 1.16
    yield_scale: float = 0.08
    fixture_peak_force_limit_n: float = 3500.0
    target_min_crush_mm: float = 40.0


@dataclass(frozen=True)
class MaterialGymCompareConfig:
    precision_ks: tuple[int, ...] = (1, 3, 5, 10, 25, 50)
    target_energy_j: float = IMPACT_ENERGY_J
    peak_limit_n: float = 3500.0
    min_crush_mm: float = 40.0
    curve_limit_mm: float = 40.0
    random_seed: int = 20260518


def build_material_gym_datasets(
    out_dir: Path,
    *,
    candidates_csv: Path | None = None,
    config: MaterialGymBuildConfig | None = None,
) -> dict[str, object]:
    config = config or MaterialGymBuildConfig()
    out_dir = ensure_dir(out_dir)
    base_rows = _candidate_rows(candidates_csv, config.train_n + config.eval_n, config.seed)
    if len(base_rows) < config.train_n + config.eval_n:
        raise ValueError(f"need at least {config.train_n + config.eval_n} candidate rows, got {len(base_rows)}")
    train_base = base_rows[: config.train_n]
    eval_base = base_rows[config.train_n : config.train_n + config.eval_n]

    datasets: list[dict[str, object]] = []
    for preset in config.presets:
        material = load_material_card(preset)
        physics_config = PhysicsConfig(
            layers=config.layers,
            steps=config.steps,
            max_displacement_mm=config.max_displacement_mm,
            dynamic_amplification=config.dynamic_amplification,
            yield_scale=config.yield_scale,
            fixture_peak_force_limit_n=config.fixture_peak_force_limit_n,
            target_min_crush_mm=config.target_min_crush_mm,
            material=material,
        )
        preset_dir = ensure_dir(out_dir / preset)
        material.to_json(preset_dir / "material_card.json")
        train_rows = [_oracle_row(row, physics_config) for row in train_base]
        eval_rows = [_oracle_row(row, physics_config) for row in eval_base]
        train_csv = preset_dir / "train.csv"
        eval_csv = preset_dir / "eval.csv"
        _write_rows(train_csv, train_rows)
        _write_rows(eval_csv, eval_rows)
        feasible_train = sum(1 for row in train_rows if _truthy(row["oracle_feasible"]))
        feasible_eval = sum(1 for row in eval_rows if _truthy(row["oracle_feasible"]))
        summary = {
            "preset": preset,
            "material_name": material.material_name,
            "train_csv": str(train_csv),
            "eval_csv": str(eval_csv),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_feasible": feasible_train,
            "eval_feasible": feasible_eval,
            "train_feasible_rate": feasible_train / max(1, len(train_rows)),
            "eval_feasible_rate": feasible_eval / max(1, len(eval_rows)),
        }
        (preset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        datasets.append(summary)

    payload = {"out_dir": str(out_dir), "config": asdict(config), "datasets": datasets}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_dataset_readme(payload), encoding="utf-8")
    return payload


def compare_material_gym(
    dataset_dir: Path,
    out_dir: Path,
    *,
    checkpoint_root: Path | None = None,
    config: MaterialGymCompareConfig | None = None,
) -> dict[str, object]:
    config = config or MaterialGymCompareConfig()
    dataset_dir = Path(dataset_dir)
    out_dir = ensure_dir(out_dir)
    preset_dirs = [path for path in sorted(dataset_dir.iterdir()) if (path / "eval.csv").exists()]
    if not preset_dirs:
        raise ValueError(f"no material gym eval CSVs found in {dataset_dir}")

    all_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for preset_dir in preset_dirs:
        preset = preset_dir.name
        eval_rows = _read_rows(preset_dir / "eval.csv")
        methods = _baseline_predictions(eval_rows, config, preset)
        if checkpoint_root is not None:
            model_root = Path(checkpoint_root) / preset
            methods.update(_checkpoint_predictions(eval_rows, model_root, config))
        oracle_masses = [_f(row, "mass_g") for row in eval_rows if _truthy(row.get("oracle_feasible"))]
        for method, preds in methods.items():
            ranked = sorted(preds, key=lambda row: _f(row, "pred_score", math.inf))
            feasible = [_truthy(row.get("oracle_feasible")) for row in ranked]
            masses = [_f(row, "mass_g") for row in ranked]
            method_row: dict[str, object] = {
                "preset": preset,
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
                method_row[f"precision_at_{k}"] = precision_at_k(feasible, k)
                method_row[f"regret_at_{k}_g"] = best_feasible_regret(masses[:k], feasible[:k], oracle_masses)
            all_rows.append(method_row)
            for row in ranked:
                curve_rows.append(
                    {
                        "preset": preset,
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
                )

    metrics_csv = out_dir / "method_metrics.csv"
    curves_csv = out_dir / "per_candidate_curve_metrics.csv"
    _write_rows(metrics_csv, [{key: _format(value) for key, value in row.items()} for row in all_rows])
    _write_rows(curves_csv, [{key: _format(value) for key, value in row.items()} for row in curve_rows])
    payload = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "checkpoint_root": str(checkpoint_root) if checkpoint_root else None,
        "method_metrics_csv": str(metrics_csv),
        "per_candidate_curve_metrics_csv": str(curves_csv),
        "config": asdict(config),
        "rows": all_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_comparison_markdown(payload), encoding="utf-8")
    return payload


def _candidate_rows(candidates_csv: Path | None, n: int, seed: int) -> list[dict[str, str]]:
    if candidates_csv is None:
        return [design_to_candidate_row(params) for params in sample_designs(n, seed=seed)]
    rows = _read_rows(Path(candidates_csv))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < n:
        needed = n - len(rows)
        rows.extend(design_to_candidate_row(params) for params in sample_designs(needed, seed=seed + 17))
    return rows[:n]


def _oracle_row(row: dict[str, str], config: PhysicsConfig) -> dict[str, object]:
    params = DesignParams.from_row(row)
    physics, sim = evaluate_candidate_physics(row, config=config)
    impact, _ = run_dynamic_impact(params, physics_config=config, rank=_rank(row), mass_g=physics.mass_g)
    disp = [float(x) for x in sim["displacement_mm"]]  # type: ignore[index]
    force = [float(x) for x in sim["force_N"]]  # type: ignore[index]
    out: dict[str, object] = {"rank": _rank(row), **params.to_row()}
    out.update({key: _format(value) for key, value in geometry_feature_row(params).items()})
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
    config: MaterialGymCompareConfig,
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
        out["lightest"].append(
            {
                **base,
                "pred_score": _f(row, "mass_g"),
                "pred_feasible": True,
            }
        )
        out["random"].append(
            {
                **base,
                "pred_score": rng.random() + idx * 1e-9,
                "pred_feasible": True,
            }
        )
        oracle_energy = _f(row, "energy_abs_J")
        oracle_peak = _f(row, "force_peak_N")
        out["oracle_upper_bound"].append(
            {
                **_prediction_row(row, oracle_disp, oracle_force, config),
                "pred_score": _f(row, "mass_g") if _truthy(row.get("oracle_feasible")) else math.inf,
                "pred_energy_J": oracle_energy,
                "pred_peak_N": oracle_peak,
                "pred_feasible": row.get("oracle_feasible", False),
            }
        )
    return out


def _checkpoint_predictions(
    eval_rows: list[dict[str, str]],
    model_root: Path,
    config: MaterialGymCompareConfig,
) -> dict[str, list[dict[str, object]]]:
    if not model_root.exists():
        return {}
    out: dict[str, list[dict[str, object]]] = {}
    for checkpoint in sorted(model_root.glob("*/crush_pinn.pt")):
        method = checkpoint.parent.name
        try:
            out[method] = _predict_checkpoint(eval_rows, checkpoint, config)
        except Exception as exc:
            out[f"{method}_ERROR"] = [{"rank": "", "topology": "", "mass_g": math.nan, "pred_score": math.inf, "error": str(exc)}]
    return out


def _predict_checkpoint(
    eval_rows: list[dict[str, str]],
    checkpoint: Path,
    config: MaterialGymCompareConfig,
) -> list[dict[str, object]]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    curve_fields = list(payload.get("curve_fields") or [])
    n_points = len(curve_fields) or len([key for key in eval_rows[0] if key.startswith("curve_")]) or CURVE_POINTS
    disp = displacement_axis(n_points)
    model_config = payload.get("config", {})
    model = CrushPINN(
        input_dim=len(payload.get("feature_names") or []) + 1,
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        blocks=int(model_config.get("blocks", 5)),
    )
    model.model.load_state_dict(payload["model_state"])
    model.eval()
    x_mean = payload["x_mean"]
    x_std = payload["x_std"]
    force_scale = float(payload["force_scale"])
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        d_tensor = torch.tensor(disp, dtype=torch.float32) / 50.0
        for row in eval_rows:
            features = torch.tensor(row_to_features(row), dtype=torch.float32)
            features = (features - x_mean) / x_std
            full_features = features.repeat(n_points, 1)
            pred = model(torch.cat([full_features, d_tensor.unsqueeze(1)], dim=1)).cpu().tolist()
            pred_force = [float(value) * force_scale for value in pred]
            energy = integrate_energy_j(disp, pred_force, limit_mm=config.curve_limit_mm)
            peak = max(pred_force) if pred_force else 0.0
            pred_row = _prediction_row(row, disp, pred_force, config)
            pred_row.update(
                {
                    "pred_energy_J": energy,
                    "pred_peak_N": peak,
                    "pred_feasible": energy >= config.target_energy_j and peak <= config.peak_limit_n,
                    "pred_score": _score(_f(row, "mass_g"), energy, peak, config),
                }
            )
            rows.append(pred_row)
    return rows


def _prediction_row(
    row: dict[str, str],
    pred_disp: list[float],
    pred_force: list[float],
    config: MaterialGymCompareConfig,
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
    return displacement_axis(len(force) or CURVE_POINTS), force


def _score(mass_g: float, energy_j: float, peak_n: float, config: MaterialGymCompareConfig) -> float:
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


def _rank(row: dict[str, Any]) -> int:
    try:
        return int(float(row.get("rank", 0)))
    except (TypeError, ValueError):
        return 0


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
    lines = ["# POLMI Material PINN Gym", ""]
    for item in payload["datasets"]:  # type: ignore[index]
        lines.append(
            "- {preset}: train={train_rows}, eval={eval_rows}, eval_feasible={eval_feasible} ({eval_feasible_rate:.3f})".format(
                **item
            )
        )
    return "\n".join(lines) + "\n"


def _comparison_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# POLMI Material Gym Comparison",
        "",
        f"Dataset: `{payload['dataset_dir']}`",
        f"Checkpoints: `{payload['checkpoint_root']}`",
        "",
        "| Material | Method | P@1 | P@5 | P@10 | Regret@10 g | Curve NRMSE | Energy abs err J | Pred violation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:  # type: ignore[index]
        lines.append(
            f"| {row['preset']} | {row['method']} | "
            f"{float(row.get('precision_at_1', math.nan)):.3f} | "
            f"{float(row.get('precision_at_5', math.nan)):.3f} | "
            f"{float(row.get('precision_at_10', math.nan)):.3f} | "
            f"{float(row.get('regret_at_10_g', math.nan)):.3f} | "
            f"{float(row.get('mean_curve_nrmse', math.nan)):.3f} | "
            f"{float(row.get('mean_energy_integral_abs_error_J', math.nan)):.3f} | "
            f"{float(row.get('physical_violation_rate', math.nan)):.3f} |"
        )
    return "\n".join(lines) + "\n"
