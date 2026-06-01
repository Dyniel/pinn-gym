"""Reviewer-response experiment orchestration for the SR PINN benchmark.

The helpers in this module deliberately wrap existing pipeline functions
instead of adding a second training stack. They produce CSV/Markdown artefacts
for the common reviewer requests: repeated seeds, physics-loss weight
ablations, and pooled-model tuning.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .material_pinn import MaterialPINNConfig
from .paths import ensure_dir
from .sr_benchmark import SRBuildConfig, SREvalConfig, build_sr_dataset, evaluate_sr_run, train_sr_models


@dataclass(frozen=True)
class LossWeightSpec:
    name: str
    boundary_weight: float = 0.05
    energy_weight: float = 0.20
    peak_weight: float = 0.10
    monotonicity_weight: float = 0.05
    smoothness_weight: float = 0.02


@dataclass(frozen=True)
class PooledTuneSpec:
    name: str
    epochs: int = 300
    batch_size: int = 8192
    rows_per_material: int = 6000
    hidden_dim: int = 512
    blocks: int = 7
    lr: float = 2e-3
    weight_decay: float = 1e-5
    boundary_weight: float = 0.05
    energy_weight: float = 0.20
    peak_weight: float = 0.10
    monotonicity_weight: float = 0.05
    smoothness_weight: float = 0.02
    peak_soft_bound: float = 1.08
    monotonic_strain_after: float = 0.78
    methods: tuple[str, ...] = ("pinn_full",)


DEFAULT_LOSS_WEIGHT_GRID: tuple[LossWeightSpec, ...] = (
    LossWeightSpec("data_boundary", energy_weight=0.0, peak_weight=0.0, monotonicity_weight=0.0, smoothness_weight=0.0),
    LossWeightSpec("energy_only", peak_weight=0.0, monotonicity_weight=0.0, smoothness_weight=0.0),
    LossWeightSpec("full_default"),
    LossWeightSpec("full_energy_strong", energy_weight=0.50),
    LossWeightSpec("full_peak_strong", peak_weight=0.30),
)

DEFAULT_POOLED_TUNE_GRID: tuple[PooledTuneSpec, ...] = (
    PooledTuneSpec("pooled_default"),
    PooledTuneSpec("pooled_compact", epochs=260, hidden_dim=384, blocks=6, lr=2e-3),
    PooledTuneSpec("pooled_wide_slow", epochs=340, hidden_dim=640, blocks=8, lr=1.2e-3, weight_decay=3e-5),
    PooledTuneSpec("pooled_energy_heavy", energy_weight=0.50, peak_weight=0.10),
)


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("at least one seed is required")
    return out


def load_loss_weight_grid(value: str | None) -> list[LossWeightSpec]:
    """Load loss weights from JSON/path or an inline compact spec.

    Inline format:
    ``name:boundary,energy,peak,monotonicity,smoothness;name2:...``.
    """

    if not value:
        return list(DEFAULT_LOSS_WEIGHT_GRID)
    path = _maybe_path(value)
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [_loss_spec_from_dict(item) for item in payload]
    if value.lstrip().startswith("["):
        return [_loss_spec_from_dict(item) for item in json.loads(value)]
    specs: list[LossWeightSpec] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"bad loss-weight spec {chunk!r}; expected name:b,e,p,m,s")
        name, raw_values = chunk.split(":", 1)
        values = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
        if len(values) != 5:
            raise ValueError(f"bad loss-weight spec {chunk!r}; expected five numeric weights")
        specs.append(
            LossWeightSpec(
                name=_slug(name),
                boundary_weight=values[0],
                energy_weight=values[1],
                peak_weight=values[2],
                monotonicity_weight=values[3],
                smoothness_weight=values[4],
            )
        )
    if not specs:
        raise ValueError("loss-weight grid is empty")
    return specs


def load_pooled_tune_grid(value: str | None) -> list[PooledTuneSpec]:
    if not value:
        return list(DEFAULT_POOLED_TUNE_GRID)
    path = _maybe_path(value)
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    return [_tune_spec_from_dict(item) for item in payload]


def run_sr_seed_repeats(
    out_dir: Path,
    *,
    presets: list[str],
    seeds: list[int],
    build_config: SRBuildConfig,
    train_config: MaterialPINNConfig,
    eval_config: SREvalConfig,
    methods: tuple[str, ...],
    train_pooled: bool = True,
    train_per_material: bool = False,
    skip_training: bool = False,
    include_transfer: bool = True,
) -> dict[str, object]:
    out_dir = ensure_dir(out_dir)
    all_method_rows: list[dict[str, object]] = []
    all_transfer_rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []

    for seed in seeds:
        run_dir = ensure_dir(out_dir / f"seed_{seed}")
        dataset_dir = run_dir / "datasets"
        checkpoint_root = run_dir / "checkpoints"
        eval_dir = run_dir / "eval"

        build_payload = build_sr_dataset(dataset_dir, config=replace(build_config, seed=seed, presets=tuple(presets)))
        if not skip_training:
            seeded_train_config = replace(train_config, seed=seed)
            if train_pooled:
                train_sr_models(
                    dataset_dir,
                    checkpoint_root,
                    presets=presets,
                    pooled=True,
                    methods=methods,
                    config=seeded_train_config,
                )
            if train_per_material:
                train_sr_models(
                    dataset_dir,
                    checkpoint_root,
                    presets=presets,
                    pooled=False,
                    methods=methods,
                    config=seeded_train_config,
                )

        eval_payload = evaluate_sr_run(
            dataset_dir,
            checkpoint_root if not skip_training else None,
            eval_dir,
            presets=presets,
            config=replace(eval_config, random_seed=seed),
            include_transfer=include_transfer and not skip_training,
        )
        method_rows = _tag_rows(_read_rows(Path(eval_payload["method_metrics_csv"])), seed=seed)
        transfer_rows = _tag_rows(_read_rows(Path(eval_payload["transfer_metrics_csv"])), seed=seed)
        all_method_rows.extend(method_rows)
        all_transfer_rows.extend(transfer_rows)
        runs.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "dataset_summary": build_payload["out_dir"],
                "method_metrics_csv": eval_payload["method_metrics_csv"],
                "transfer_metrics_csv": eval_payload["transfer_metrics_csv"],
            }
        )

    method_csv = out_dir / "method_metrics_by_seed.csv"
    transfer_csv = out_dir / "transfer_metrics_by_seed.csv"
    summary_csv = out_dir / "method_metrics_seed_summary.csv"
    _write_rows(method_csv, all_method_rows)
    _write_rows(transfer_csv, all_transfer_rows)
    summary_rows = summarize_metric_rows(all_method_rows, group_keys=("preset", "scope", "method"))
    _write_rows(summary_csv, summary_rows)

    payload = {
        "out_dir": str(out_dir),
        "presets": presets,
        "seeds": seeds,
        "methods": list(methods),
        "skip_training": skip_training,
        "train_pooled": train_pooled,
        "train_per_material": train_per_material,
        "method_metrics_by_seed_csv": str(method_csv),
        "transfer_metrics_by_seed_csv": str(transfer_csv),
        "method_metrics_seed_summary_csv": str(summary_csv),
        "runs": runs,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_repeat_summary_md(payload, summary_rows), encoding="utf-8")
    return payload


def run_loss_weight_ablation(
    dataset_dir: Path,
    out_dir: Path,
    *,
    presets: list[str],
    weights: list[LossWeightSpec],
    train_config: MaterialPINNConfig,
    eval_config: SREvalConfig,
    methods: tuple[str, ...] = ("pinn_full",),
    pooled: bool = True,
    include_transfer: bool = False,
) -> dict[str, object]:
    out_dir = ensure_dir(out_dir)
    all_rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []

    for spec in weights:
        label = _slug(spec.name)
        run_dir = ensure_dir(out_dir / label)
        checkpoint_root = run_dir / "checkpoints"
        eval_dir = run_dir / "eval"
        config = replace(
            train_config,
            boundary_weight=spec.boundary_weight,
            energy_weight=spec.energy_weight,
            peak_weight=spec.peak_weight,
            monotonicity_weight=spec.monotonicity_weight,
            smoothness_weight=spec.smoothness_weight,
        )
        train_sr_models(
            dataset_dir,
            checkpoint_root,
            presets=presets,
            pooled=pooled,
            methods=methods,
            config=config,
        )
        eval_payload = evaluate_sr_run(
            dataset_dir,
            checkpoint_root,
            eval_dir,
            presets=presets,
            config=eval_config,
            include_transfer=include_transfer,
        )
        rows = _read_rows(Path(eval_payload["method_metrics_csv"]))
        for row in rows:
            row.update(_loss_spec_row(spec))
            row["ablation"] = label
        all_rows.extend(rows)
        runs.append({"ablation": label, "weights": _loss_spec_row(spec), "eval_dir": str(eval_dir)})

    metrics_csv = out_dir / "loss_weight_ablation_metrics.csv"
    summary_csv = out_dir / "loss_weight_ablation_summary.csv"
    _write_rows(metrics_csv, all_rows)
    summary_rows = summarize_metric_rows(all_rows, group_keys=("ablation", "preset", "scope", "method"))
    _write_rows(summary_csv, summary_rows)
    payload = {
        "out_dir": str(out_dir),
        "dataset_dir": str(dataset_dir),
        "presets": presets,
        "methods": list(methods),
        "pooled": pooled,
        "metrics_csv": str(metrics_csv),
        "summary_csv": str(summary_csv),
        "runs": runs,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_loss_ablation_summary_md(payload, summary_rows), encoding="utf-8")
    return payload


def run_pooled_tuning(
    dataset_dir: Path,
    out_dir: Path,
    *,
    presets: list[str],
    candidates: list[PooledTuneSpec],
    base_config: MaterialPINNConfig,
    eval_config: SREvalConfig,
) -> dict[str, object]:
    out_dir = ensure_dir(out_dir)
    all_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for spec in candidates:
        label = _slug(spec.name)
        run_dir = ensure_dir(out_dir / label)
        checkpoint_root = run_dir / "checkpoints"
        eval_dir = run_dir / "eval"
        train_config = _config_from_tune_spec(base_config, spec)
        train_sr_models(
            dataset_dir,
            checkpoint_root,
            presets=presets,
            pooled=True,
            methods=spec.methods,
            config=train_config,
        )
        eval_payload = evaluate_sr_run(
            dataset_dir,
            checkpoint_root,
            eval_dir,
            presets=presets,
            config=eval_config,
            include_transfer=False,
        )
        rows = _read_rows(Path(eval_payload["method_metrics_csv"]))
        for row in rows:
            row["tune_candidate"] = label
            row.update(_tune_spec_row(spec))
        all_rows.extend(rows)
        pooled_rows = [row for row in rows if row.get("scope") == "pooled"]
        candidate_rows.append(_pooled_candidate_summary(label, spec, pooled_rows))

    metrics_csv = out_dir / "pooled_tuning_metrics.csv"
    results_csv = out_dir / "pooled_tuning_results.csv"
    candidate_rows = sorted(candidate_rows, key=lambda row: _finite_float(row.get("objective"), math.inf))
    _write_rows(metrics_csv, all_rows)
    _write_rows(results_csv, candidate_rows)
    payload = {
        "out_dir": str(out_dir),
        "dataset_dir": str(dataset_dir),
        "presets": presets,
        "metrics_csv": str(metrics_csv),
        "results_csv": str(results_csv),
        "best": candidate_rows[0] if candidate_rows else None,
        "objective": "mean_curve_nrmse_mean + physical_violation_rate_mean + 0.01 * mean_energy_integral_abs_error_J_mean + feasibility_collapse_penalty",
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_pooled_tuning_summary_md(payload, candidate_rows), encoding="utf-8")
    return payload


def summarize_metric_rows(rows: list[dict[str, object]], *, group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)

    metric_names = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in group_keys and key not in {"seed"} and _looks_numeric(row.get(key))
        }
    )
    out: list[dict[str, object]] = []
    for key_values, group in sorted(groups.items()):
        item: dict[str, object] = {key: value for key, value in zip(group_keys, key_values)}
        item["repeats"] = len(group)
        seeds = sorted({str(row.get("seed", "")) for row in group if row.get("seed", "") != ""})
        if seeds:
            item["seeds"] = ",".join(seeds)
        for metric in metric_names:
            values = [_to_float(row.get(metric)) for row in group]
            finite = [value for value in values if math.isfinite(value)]
            item[f"{metric}_finite_n"] = len(finite)
            item[f"{metric}_nonfinite_n"] = len(values) - len(finite)
            if finite:
                mean = sum(finite) / len(finite)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = _std(finite, mean)
                item[f"{metric}_min"] = min(finite)
                item[f"{metric}_max"] = max(finite)
        out.append(item)
    return out


def _loss_spec_from_dict(item: dict[str, object]) -> LossWeightSpec:
    return LossWeightSpec(
        name=_slug(str(item["name"])),
        boundary_weight=float(item.get("boundary_weight", 0.05)),
        energy_weight=float(item.get("energy_weight", 0.20)),
        peak_weight=float(item.get("peak_weight", 0.10)),
        monotonicity_weight=float(item.get("monotonicity_weight", 0.05)),
        smoothness_weight=float(item.get("smoothness_weight", 0.02)),
    )


def _tune_spec_from_dict(item: dict[str, object]) -> PooledTuneSpec:
    methods = item.get("methods", ("pinn_full",))
    if isinstance(methods, str):
        method_tuple = tuple(part.strip() for part in methods.split(",") if part.strip())
    else:
        method_tuple = tuple(str(part) for part in methods)  # type: ignore[arg-type]
    return PooledTuneSpec(
        name=_slug(str(item["name"])),
        epochs=int(item.get("epochs", 300)),
        batch_size=int(item.get("batch_size", 8192)),
        rows_per_material=int(item.get("rows_per_material", 6000)),
        hidden_dim=int(item.get("hidden_dim", 512)),
        blocks=int(item.get("blocks", 7)),
        lr=float(item.get("lr", 2e-3)),
        weight_decay=float(item.get("weight_decay", 1e-5)),
        boundary_weight=float(item.get("boundary_weight", 0.05)),
        energy_weight=float(item.get("energy_weight", 0.20)),
        peak_weight=float(item.get("peak_weight", 0.10)),
        monotonicity_weight=float(item.get("monotonicity_weight", 0.05)),
        smoothness_weight=float(item.get("smoothness_weight", 0.02)),
        peak_soft_bound=float(item.get("peak_soft_bound", 1.08)),
        monotonic_strain_after=float(item.get("monotonic_strain_after", 0.78)),
        methods=method_tuple or ("pinn_full",),
    )


def _config_from_tune_spec(base: MaterialPINNConfig, spec: PooledTuneSpec) -> MaterialPINNConfig:
    return replace(
        base,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        rows_per_material=spec.rows_per_material,
        hidden_dim=spec.hidden_dim,
        blocks=spec.blocks,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        boundary_weight=spec.boundary_weight,
        energy_weight=spec.energy_weight,
        peak_weight=spec.peak_weight,
        monotonicity_weight=spec.monotonicity_weight,
        smoothness_weight=spec.smoothness_weight,
        peak_soft_bound=spec.peak_soft_bound,
        monotonic_strain_after=spec.monotonic_strain_after,
    )


def _pooled_candidate_summary(label: str, spec: PooledTuneSpec, rows: list[dict[str, object]]) -> dict[str, object]:
    item = {"tune_candidate": label, **_tune_spec_row(spec), "rows": len(rows)}
    for metric in (
        "mean_curve_nrmse",
        "mean_energy_integral_abs_error_J",
        "physical_violation_rate",
        "predicted_feasible_rate",
        "regret_at_10_g",
        "relative_regret_at_10",
    ):
        values = [_to_float(row.get(metric)) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        item[f"{metric}_finite_n"] = len(finite)
        item[f"{metric}_nonfinite_n"] = len(values) - len(finite)
        item[f"{metric}_mean"] = sum(finite) / len(finite) if finite else math.nan
    nrmse = _finite_float(item.get("mean_curve_nrmse_mean"), math.inf)
    violation = _finite_float(item.get("physical_violation_rate_mean"), math.inf)
    energy = _finite_float(item.get("mean_energy_integral_abs_error_J_mean"), math.inf)
    # Collapse guard: a model that predicts *no* candidate feasible trivially
    # scores violation=0, which the raw objective would reward. Penalize that
    # only on pools that actually contain oracle-feasible designs, so genuinely
    # zero-feasible material cards (e.g. pla/petg/pa_cf) are not penalized.
    collapse_flags = [
        1.0 if _finite_float(row.get("predicted_feasible_rate"), 0.0) <= 1e-9 else 0.0
        for row in rows
        if _finite_float(row.get("oracle_feasible"), 0.0) > 0.0
    ]
    collapse_penalty = (sum(collapse_flags) / len(collapse_flags)) if collapse_flags else 0.0
    item["feasibility_collapse_penalty"] = collapse_penalty
    item["objective"] = nrmse + violation + 0.01 * energy + collapse_penalty
    return item


def _loss_spec_row(spec: LossWeightSpec) -> dict[str, object]:
    return {key: value for key, value in asdict(spec).items() if key != "name"}


def _tune_spec_row(spec: PooledTuneSpec) -> dict[str, object]:
    row = asdict(spec)
    row["methods"] = ",".join(spec.methods)
    row.pop("name", None)
    return row


def _tag_rows(rows: list[dict[str, object]], *, seed: int) -> list[dict[str, object]]:
    return [{"seed": seed, **row} for row in rows]


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
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
        writer.writerows([{key: _format_value(row.get(key, "")) for key in fields} for row in rows])


def _format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else str(value)
    return value


def _maybe_path(value: str) -> Path | None:
    if value.startswith("@"):
        return Path(value[1:])
    path = Path(value)
    return path if path.suffix.lower() == ".json" or "/" in value else None


def _slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    out = "_".join(part for part in out.split("_") if part)
    if not out:
        raise ValueError("empty name")
    return out


def _looks_numeric(value: object) -> bool:
    return math.isfinite(_to_float(value)) or str(value).strip().lower() in {"inf", "-inf", "nan"}


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _finite_float(value: object, default: float) -> float:
    out = _to_float(value)
    return out if math.isfinite(out) else default


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _repeat_summary_md(payload: dict[str, object], rows: list[dict[str, object]]) -> str:
    lines = [
        "# SR repeated-seed summary",
        "",
        f"Seeds: `{','.join(str(seed) for seed in payload['seeds'])}`",
        f"Method metrics by seed: `{payload['method_metrics_by_seed_csv']}`",
        f"Seed summary: `{payload['method_metrics_seed_summary_csv']}`",
        "",
        "## Top pooled rows",
        "",
    ]
    for row in rows[:20]:
        if row.get("scope") == "pooled":
            lines.append(
                "- {preset} / {method}: nRMSE={nrmse}, violation={violation}, regret@10={regret}".format(
                    preset=row.get("preset"),
                    method=row.get("method"),
                    nrmse=row.get("mean_curve_nrmse_mean", ""),
                    violation=row.get("physical_violation_rate_mean", ""),
                    regret=row.get("regret_at_10_g_mean", ""),
                )
            )
    return "\n".join(lines) + "\n"


def _loss_ablation_summary_md(payload: dict[str, object], rows: list[dict[str, object]]) -> str:
    lines = [
        "# SR physics-loss weight ablation",
        "",
        f"Metrics: `{payload['metrics_csv']}`",
        f"Summary: `{payload['summary_csv']}`",
        "",
        "## Rows",
        "",
    ]
    for row in rows[:40]:
        lines.append(
            "- {ablation} / {preset} / {method}: nRMSE={nrmse}, energy_err={energy}, violation={violation}, regret@10={regret}".format(
                ablation=row.get("ablation"),
                preset=row.get("preset"),
                method=row.get("method"),
                nrmse=row.get("mean_curve_nrmse_mean", ""),
                energy=row.get("mean_energy_integral_abs_error_J_mean", ""),
                violation=row.get("physical_violation_rate_mean", ""),
                regret=row.get("regret_at_10_g_mean", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _pooled_tuning_summary_md(payload: dict[str, object], rows: list[dict[str, object]]) -> str:
    lines = [
        "# SR pooled-model tuning",
        "",
        f"Objective: `{payload['objective']}`",
        f"Metrics: `{payload['metrics_csv']}`",
        f"Results: `{payload['results_csv']}`",
        "",
        "## Candidate ranking",
        "",
    ]
    for row in rows:
        lines.append(
            "- {name}: objective={objective}, nRMSE={nrmse}, violation={violation}, energy_err={energy}".format(
                name=row.get("tune_candidate"),
                objective=row.get("objective"),
                nrmse=row.get("mean_curve_nrmse_mean"),
                violation=row.get("physical_violation_rate_mean"),
                energy=row.get("mean_energy_integral_abs_error_J_mean"),
            )
        )
    return "\n".join(lines) + "\n"
