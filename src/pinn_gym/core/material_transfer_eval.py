"""Evaluation metrics for POLMI material-card transfer runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .design_space import CURVE_POINTS, DesignParams, displacement_axis, pseudo_response
from .materials import load_material_card
from .metrics import (
    best_feasible_regret,
    force_curve_error_metrics,
    physical_violation_rate,
    precision_at_k,
)
from .paths import ensure_dir
from .physics import IMPACT_ENERGY_J, PhysicsConfig, run_layered_crush_fem


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    fields = fieldnames or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rank(row: dict[str, Any]) -> int | None:
    try:
        return int(float(row["rank"]))
    except (KeyError, TypeError, ValueError):
        return None


def _index_by_rank(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        rank = _rank(row)
        if rank is not None:
            out[rank] = row
    return out


def _float(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    if row is None:
        return default
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


def _discover_presets(result_dir: Path) -> list[str]:
    presets = []
    for path in sorted(result_dir.glob("*/physics_gate/physics_candidates.csv")):
        if path.is_file():
            presets.append(path.parents[1].name)
    return presets


def _top_csv_from_result_dir(result_dir: Path, fallback: Path | None = None) -> Path:
    if fallback is not None:
        return fallback
    summaries = sorted(result_dir.glob("*/physics_gate/summary.json"))
    for path in summaries:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        top_csv = payload.get("top_csv")
        if top_csv:
            return Path(str(top_csv))
    raise ValueError("could not infer top CSV from result dir; pass --top-csv")


def _physics_config(result_dir: Path, preset: str) -> PhysicsConfig:
    summary_path = result_dir / preset / "physics_gate" / "summary.json"
    card_path = result_dir / "cards" / f"{preset}.json"
    config_payload: dict[str, Any] = {}
    if summary_path.exists():
        try:
            config_payload = json.loads(summary_path.read_text(encoding="utf-8")).get("config", {})
        except json.JSONDecodeError:
            config_payload = {}
    material = load_material_card(card_path if card_path.exists() else preset)
    fields = set(PhysicsConfig.__dataclass_fields__) - {"material"}
    kwargs = {key: config_payload[key] for key in fields if key in config_payload}
    return PhysicsConfig(**kwargs, material=material)


def _predicted_curve(row: dict[str, str]) -> tuple[list[float], list[float]]:
    curve_fields = sorted(key for key in row if key.startswith("curve_"))
    if curve_fields:
        force = [_float(row, key, 0.0) for key in curve_fields]
        return displacement_axis(len(force)), force
    response = pseudo_response(DesignParams.from_row(row), curve_points=CURVE_POINTS)
    return response["curve_displacement_mm"], response["curve_force_N"]  # type: ignore[return-value]


def _candidate_score(physics: dict[str, str] | None, dynamic: dict[str, str] | None) -> float:
    score = _float(physics, "physics_score", 0.0) + _float(dynamic, "impact_score", 0.0)
    if not math.isfinite(score):
        return math.inf
    return score


def _oracle_feasible(physics: dict[str, str] | None, dynamic: dict[str, str] | None) -> bool:
    return bool(
        physics
        and dynamic
        and _truthy(physics.get("physics_survives_gate"))
        and _truthy(dynamic.get("impact_survives"))
    )


def _combined_rows(physics: dict[int, dict[str, str]], dynamic: dict[int, dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank in sorted(set(physics) | set(dynamic)):
        p_row = physics.get(rank)
        d_row = dynamic.get(rank)
        rows.append(
            {
                "rank": rank,
                "physics_energy_usable_J": _float(p_row, "physics_energy_usable_J", 0.0),
                "physics_peak_force_N": _float(p_row, "physics_peak_force_N", math.inf),
                "physics_collapse_mm": _float(p_row, "physics_collapse_mm", 0.0),
                "physics_failure_risk": _float(p_row, "physics_failure_risk", 1.0),
                "physics_survives_gate": _truthy(p_row.get("physics_survives_gate")) if p_row else False,
                "impact_absorbed_J": _float(d_row, "impact_absorbed_J", 0.0),
                "impact_peak_force_N": _float(d_row, "impact_peak_force_N", math.inf),
                "impact_max_displacement_mm": _float(d_row, "impact_max_displacement_mm", 0.0),
                "impact_failure_risk": _float(d_row, "impact_failure_risk", 1.0),
                "impact_survives": _truthy(d_row.get("impact_survives")) if d_row else False,
                "oracle_feasible": _oracle_feasible(p_row, d_row),
                "mass_g": _float(p_row, "mass_g", _float(d_row, "mass_g", math.nan)),
            }
        )
    return rows


def evaluate_material_transfer_run(
    result_dir: Path,
    out_dir: Path,
    *,
    top_csv: Path | None = None,
    source_presets: list[str] | None = None,
    target_presets: list[str] | None = None,
    precision_ks: list[int] | None = None,
    target_energy_j: float = IMPACT_ENERGY_J,
    peak_limit_n: float = 3500.0,
    min_crush_mm: float = 40.0,
    max_failure_risk: float = 0.48,
    curve_limit_mm: float = 40.0,
) -> dict[str, object]:
    result_dir = Path(result_dir)
    out_dir = ensure_dir(out_dir)
    top_csv = _top_csv_from_result_dir(result_dir, top_csv)
    precision_ks = precision_ks or [1, 3, 5, 10]
    presets = _discover_presets(result_dir)
    if not presets:
        raise ValueError(f"no material-card outputs found in {result_dir}")
    source_presets = source_presets or presets
    target_presets = target_presets or presets

    top_rows = _read_csv(top_csv)
    top_by_rank = _index_by_rank(top_rows)
    physics_by_preset: dict[str, dict[int, dict[str, str]]] = {}
    dynamic_by_preset: dict[str, dict[int, dict[str, str]]] = {}
    for preset in presets:
        physics_by_preset[preset] = _index_by_rank(_read_csv(result_dir / preset / "physics_gate" / "physics_candidates.csv"))
        dynamic_by_preset[preset] = _index_by_rank(_read_csv(result_dir / preset / "dynamic_impact" / "dynamic_impact_candidates.csv"))

    curve_rows: list[dict[str, object]] = []
    for preset in target_presets:
        if preset not in physics_by_preset:
            continue
        config = _physics_config(result_dir, preset)
        for rank, physics_row in sorted(physics_by_preset[preset].items()):
            top_row = top_by_rank.get(rank)
            if top_row is None:
                continue
            params = DesignParams.from_row(top_row)
            pred_disp, pred_force = _predicted_curve(top_row)
            oracle = run_layered_crush_fem(params, config)
            target_disp = [float(x) for x in oracle["displacement_mm"]]  # type: ignore[index]
            target_force = [float(x) for x in oracle["force_N"]]  # type: ignore[index]
            errors = force_curve_error_metrics(
                [float(x) for x in pred_disp],
                [float(x) for x in pred_force],
                target_disp,
                target_force,
                limit_mm=curve_limit_mm,
            )
            curve_rows.append(
                {
                    "preset": preset,
                    "rank": rank,
                    "topology": physics_row.get("topology", top_row.get("topology", "")),
                    "mass_g": _float(physics_row, "mass_g"),
                    **errors,
                }
            )

    preset_rows: list[dict[str, object]] = []
    for preset in target_presets:
        physics = physics_by_preset.get(preset, {})
        dynamic = dynamic_by_preset.get(preset, {})
        rows = _combined_rows(physics, dynamic)
        feasible_count = sum(1 for row in rows if row["oracle_feasible"])
        preset_rows.append(
            {
                "preset": preset,
                "evaluated": len(rows),
                "oracle_feasible": feasible_count,
                "oracle_feasible_rate": feasible_count / max(1, len(rows)),
                "physics_violation_rate": physical_violation_rate(
                    rows,
                    energy_key="physics_energy_usable_J",
                    peak_key="physics_peak_force_N",
                    crush_key="physics_collapse_mm",
                    risk_key="physics_failure_risk",
                    target_energy_j=target_energy_j,
                    peak_limit_n=peak_limit_n,
                    min_crush_mm=min_crush_mm,
                    max_risk=0.42,
                ),
                "impact_violation_rate": physical_violation_rate(
                    rows,
                    energy_key="impact_absorbed_J",
                    peak_key="impact_peak_force_N",
                    crush_key="impact_max_displacement_mm",
                    risk_key="impact_failure_risk",
                    target_energy_j=0.98 * target_energy_j,
                    peak_limit_n=peak_limit_n,
                    min_crush_mm=0.0,
                    max_risk=max_failure_risk,
                ),
                "combined_physical_violation_rate": 1.0 - feasible_count / max(1, len(rows)),
                "best_oracle_feasible_mass_g": min(
                    [float(row["mass_g"]) for row in rows if row["oracle_feasible"] and math.isfinite(float(row["mass_g"]))],
                    default=math.nan,
                ),
            }
        )

    transfer_rows: list[dict[str, object]] = []
    for source in source_presets:
        if source not in physics_by_preset:
            continue
        source_order = sorted(
            set(physics_by_preset[source]) | set(dynamic_by_preset.get(source, {})),
            key=lambda rank: _candidate_score(physics_by_preset[source].get(rank), dynamic_by_preset[source].get(rank)),
        )
        for target in target_presets:
            if target == source or target not in physics_by_preset:
                continue
            target_physics = physics_by_preset[target]
            target_dynamic = dynamic_by_preset[target]
            ranked = [rank for rank in source_order if rank in target_physics and rank in target_dynamic]
            feasible = [_oracle_feasible(target_physics.get(rank), target_dynamic.get(rank)) for rank in ranked]
            masses = [_float(target_physics.get(rank), "mass_g", _float(target_dynamic.get(rank), "mass_g")) for rank in ranked]
            oracle_masses = [
                _float(target_physics.get(rank), "mass_g")
                for rank in set(target_physics) & set(target_dynamic)
                if _oracle_feasible(target_physics.get(rank), target_dynamic.get(rank))
            ]
            row: dict[str, object] = {
                "source_preset": source,
                "target_preset": target,
                "evaluated": len(ranked),
                "oracle_feasible": sum(1 for item in feasible if item),
                "oracle_best_feasible_mass_g": min(oracle_masses, default=math.nan),
            }
            for k in precision_ks:
                row[f"precision_at_{k}"] = precision_at_k(feasible, k)
                row[f"regret_at_{k}_g"] = best_feasible_regret(masses[:k], feasible[:k], oracle_masses)
            transfer_rows.append(row)

    aggregate_curve = _aggregate_curve_rows(curve_rows)
    curve_csv = out_dir / "curve_metrics.csv"
    preset_csv = out_dir / "physical_metrics.csv"
    transfer_csv = out_dir / "transfer_metrics.csv"
    _write_csv(curve_csv, [{key: _format(value) for key, value in row.items()} for row in curve_rows])
    _write_csv(preset_csv, [{key: _format(value) for key, value in row.items()} for row in preset_rows])
    _write_csv(transfer_csv, [{key: _format(value) for key, value in row.items()} for row in transfer_rows])

    summary = {
        "result_dir": str(result_dir),
        "top_csv": str(top_csv),
        "out_dir": str(out_dir),
        "curve_metrics_csv": str(curve_csv),
        "physical_metrics_csv": str(preset_csv),
        "transfer_metrics_csv": str(transfer_csv),
        "presets": presets,
        "curve_aggregate": aggregate_curve,
        "physical_metrics": preset_rows,
        "transfer_pairs": len(transfer_rows),
        "config": {
            "precision_ks": precision_ks,
            "target_energy_j": target_energy_j,
            "peak_limit_n": peak_limit_n,
            "min_crush_mm": min_crush_mm,
            "max_failure_risk": max_failure_risk,
            "curve_limit_mm": curve_limit_mm,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _aggregate_curve_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_preset: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_preset.setdefault(str(row["preset"]), []).append(row)
    out: dict[str, dict[str, float]] = {}
    for preset, items in sorted(by_preset.items()):
        out[preset] = {
            "curve_rmse_N_mean": _mean(_float(item, "curve_rmse_N") for item in items),
            "curve_nrmse_mean": _mean(_float(item, "curve_nrmse") for item in items),
            "energy_integral_abs_error_J_mean": _mean(_float(item, "energy_integral_abs_error_J") for item in items),
            "energy_integral_rel_error_mean": _mean(_float(item, "energy_integral_rel_error") for item in items),
        }
    return out


def _mean(values: Any) -> float:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    return sum(cleaned) / len(cleaned) if cleaned else math.nan


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# POLMI Material Transfer Metrics",
        "",
        f"Input: `{summary['result_dir']}`",
        f"Top CSV: `{summary['top_csv']}`",
        "",
        "## Curve Metrics",
        "",
        "| Preset | RMSE N | NRMSE | Energy abs error J | Energy rel error |",
        "|---|---:|---:|---:|---:|",
    ]
    curve_aggregate = summary.get("curve_aggregate", {})
    if isinstance(curve_aggregate, dict):
        for preset, metrics in curve_aggregate.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                f"| {preset} | {float(metrics['curve_rmse_N_mean']):.2f} | "
                f"{float(metrics['curve_nrmse_mean']):.3f} | "
                f"{float(metrics['energy_integral_abs_error_J_mean']):.2f} | "
                f"{float(metrics['energy_integral_rel_error_mean']):.3f} |"
            )
    lines += [
        "",
        "## Physical Metrics",
        "",
        "| Preset | Evaluated | Feasible | Feasible rate | Violation rate | Best feasible mass g |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("physical_metrics", []):  # type: ignore[union-attr]
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row['preset']} | {int(row['evaluated'])} | {int(row['oracle_feasible'])} | "
            f"{float(row['oracle_feasible_rate']):.3f} | "
            f"{float(row['combined_physical_violation_rate']):.3f} | "
            f"{float(row['best_oracle_feasible_mass_g']):.3f} |"
        )
    lines += [
        "",
        f"Transfer pairs: `{summary['transfer_pairs']}`",
        "",
    ]
    return "\n".join(lines)
