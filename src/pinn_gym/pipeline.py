"""Pipeline orchestration — the kombajn.

This module is a thin dispatcher. It maps the validated :class:`GymConfig`
onto the scientific core (build / train / evaluate / plots / audit) and
writes a single run directory under ``runs/<name>_<timestamp>/``.

Stage outputs land under predictable subpaths so that re-running a stage with
``--stage`` overrides picks up the previous outputs without ambiguity.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import GymConfig

log = logging.getLogger("pinn_gym.pipeline")

# Methods that actually run training. The rest are decision-rule baselines
# that need no checkpoint and are evaluated directly inside sr_benchmark.
TRAINABLE_METHODS = frozenset({"mlp_softplus", "pinn_energy", "pinn_full"})


@dataclass(frozen=True)
class RunPaths:
    root: Path
    datasets: Path
    checkpoints: Path
    figures: Path
    tables: Path
    logs: Path

    @classmethod
    def create(cls, parent: Path, name: str) -> "RunPaths":
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = parent / f"{name}_{stamp}"
        root.mkdir(parents=True, exist_ok=False)
        sub = {
            "datasets": root / "datasets",
            "checkpoints": root / "checkpoints",
            "figures": root / "figures",
            "tables": root / "tables",
            "logs": root / "logs",
        }
        for p in sub.values():
            p.mkdir(parents=True, exist_ok=True)
        return cls(root=root, **sub)

    @classmethod
    def open(cls, root: Path) -> "RunPaths":
        return cls(
            root=root,
            datasets=root / "datasets",
            checkpoints=root / "checkpoints",
            figures=root / "figures",
            tables=root / "tables",
            logs=root / "logs",
        )


def _default_runs_root() -> Path:
    return Path(__file__).resolve().parents[2] / "runs"


def prepare_run_dir(cfg: GymConfig) -> RunPaths:
    parent = Path(cfg.run.output_root) if cfg.run.output_root else _default_runs_root()
    parent.mkdir(parents=True, exist_ok=True)
    return RunPaths.create(parent, cfg.run.name)


def write_manifest(paths: RunPaths, cfg: GymConfig) -> Path:
    manifest = paths.root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config": cfg.model_dump(mode="python"),
                "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "pinn_gym_version": _read_version(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _read_version() -> str:
    from . import __version__

    return __version__


def run(cfg: GymConfig, *, dry_run: bool = False) -> RunPaths:
    """Execute the configured stages.

    With ``dry_run=True``, prepare the run directory and write the manifest but
    perform no scientific work. Useful for CI and config sanity checks.
    """
    paths = prepare_run_dir(cfg)
    write_manifest(paths, cfg)
    _attach_file_log(paths)

    if dry_run:
        log.info("dry-run: stages skipped, manifest written to %s", paths.root / "manifest.json")
        return paths

    for stage in cfg.run.stages:
        log.info("=== stage: %s ===", stage)
        _dispatch(stage, cfg, paths)
    return paths


def _attach_file_log(paths: RunPaths) -> None:
    handler = logging.FileHandler(paths.logs / "pipeline.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.setLevel(logging.INFO)
    log.addHandler(handler)


def _dispatch(stage: str, cfg: GymConfig, paths: RunPaths) -> None:
    handlers = {
        "build": _stage_build,
        "train": _stage_train,
        "evaluate": _stage_evaluate,
        "plots": _stage_plots,
        "audit": _stage_audit,
    }
    handlers[stage](cfg, paths)


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------


def _stage_build(cfg: GymConfig, paths: RunPaths) -> None:
    from .core import SRBuildConfig, build_sr_dataset

    build_cfg = SRBuildConfig(
        presets=tuple(cfg.materials.presets),
        train_n=cfg.candidate_pool.train_n,
        eval_n=cfg.candidate_pool.eval_n,
        seed=cfg.run.seed,
        layers=cfg.candidate_pool.layers,
        steps=cfg.candidate_pool.steps,
        max_displacement_mm=cfg.candidate_pool.max_displacement_mm,
        dynamic_amplification=cfg.candidate_pool.dynamic_amplification,
        yield_scale=cfg.candidate_pool.yield_scale,
        fixture_peak_force_limit_n=cfg.candidate_pool.fixture_peak_force_limit_n,
        target_min_crush_mm=cfg.candidate_pool.target_min_crush_mm,
        material_aware_crush_target=cfg.materials.material_aware_crush_target,
        oracle_workers=cfg.candidate_pool.oracle_workers,
    )
    payload = build_sr_dataset(paths.datasets, config=build_cfg)
    (paths.tables / "build_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    log.info("build: %d material(s), datasets at %s", len(cfg.materials.presets), paths.datasets)


def _stage_train(cfg: GymConfig, paths: RunPaths) -> None:
    from .core import MaterialPINNConfig, train_sr_models

    methods = tuple(m for m in cfg.train.methods if m in TRAINABLE_METHODS)
    if not methods:
        log.info("train: only baseline methods requested; skipping GPU training")
        return

    pinn_cfg = MaterialPINNConfig(
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        rows_per_material=cfg.train.rows_per_material,
        hidden_dim=cfg.train.hidden_dim,
        blocks=cfg.train.blocks,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        device=cfg.run.device,
        seed=cfg.run.seed,
        boundary_weight=cfg.train.loss_weights.boundary,
        energy_weight=cfg.train.loss_weights.energy,
        peak_weight=cfg.train.loss_weights.peak,
        monotonicity_weight=cfg.train.loss_weights.monotonicity,
        smoothness_weight=cfg.train.loss_weights.smoothness,
        peak_soft_bound=cfg.train.peak_soft_bound,
        monotonic_strain_after=cfg.train.monotonic_strain_after,
    )
    payload = train_sr_models(
        paths.datasets,
        paths.checkpoints,
        presets=list(cfg.materials.presets),
        pooled=cfg.train.pooled,
        methods=methods,
        config=pinn_cfg,
    )
    log.info("train: %d model(s) written to %s", len(payload["trained"]), paths.checkpoints)


def _stage_evaluate(cfg: GymConfig, paths: RunPaths) -> None:
    from .core import IMPACT_ENERGY_J, SREvalConfig, evaluate_sr_run

    target_energy = (
        IMPACT_ENERGY_J
        if cfg.evaluate.target_energy_j == "auto"
        else float(cfg.evaluate.target_energy_j)
    )
    eval_cfg = SREvalConfig(
        precision_ks=tuple(cfg.evaluate.precision_ks),
        target_energy_j=target_energy,
        peak_limit_n=cfg.evaluate.peak_limit_n,
        min_crush_mm=cfg.evaluate.min_crush_mm,
        curve_limit_mm=cfg.evaluate.curve_limit_mm,
        random_seed=cfg.run.seed,
    )
    payload = evaluate_sr_run(
        paths.datasets,
        paths.checkpoints,
        paths.tables,
        presets=list(cfg.materials.presets),
        config=eval_cfg,
        include_transfer=cfg.evaluate.include_transfer_matrix,
    )
    log.info(
        "evaluate: %s + transfer=%s",
        payload.get("method_metrics_csv", "?"),
        cfg.evaluate.include_transfer_matrix,
    )


def _stage_plots(cfg: GymConfig, paths: RunPaths) -> None:
    from .core import render_sr_figures

    payload = render_sr_figures(
        paths.tables,
        paths.datasets,
        paths.figures,
        presets=list(cfg.materials.presets),
    )
    log.info("plots: %d figure(s) at %s", len(payload), paths.figures)


def _stage_audit(cfg: GymConfig, paths: RunPaths) -> None:
    from .core import audit_stl_directory, export_design_stl, write_mesh_quality_report
    from .core.design_space import DesignParams

    if cfg.audit.stl_export_count <= 0:
        log.info("audit: stl_export_count=0, skipping STL audit")
        return

    stl_root = paths.root / "audit"
    stl_root.mkdir(parents=True, exist_ok=True)

    audited_results = []
    for preset in cfg.materials.presets:
        eval_csv = paths.datasets / preset / "eval.csv"
        if not eval_csv.exists():
            log.warning("audit: missing eval.csv for %s, skipping", preset)
            continue
        rows = _top_feasible_rows(eval_csv, cfg.audit.stl_export_count)
        if not rows:
            log.warning("audit: no feasible rows for %s, skipping", preset)
            continue
        out_dir = stl_root / preset
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(rows, start=1):
            params = DesignParams.from_row(row)
            stl_path = out_dir / f"rank_{rank:03d}_{params.topology}.stl"
            export_design_stl(
                params,
                stl_path,
                backend=cfg.audit.stl_backend,
                resolution=cfg.audit.stl_resolution,
                stl_format=cfg.audit.stl_format,
            )
        audited_results.extend(audit_stl_directory(out_dir))

    if audited_results:
        report_path = paths.tables / "mesh_quality.json"
        write_mesh_quality_report(audited_results, report_path)
        log.info("audit: %d STL(s) audited, report at %s", len(audited_results), report_path)
        bad = [r for r in audited_results if _is_audit_failure(r)]
        if bad and not cfg.audit.warn_only:
            raise RuntimeError(f"audit failed: {len(bad)}/{len(audited_results)} STL(s) failed checks")
    else:
        log.info("audit: nothing to audit")


def _top_feasible_rows(eval_csv: Path, top_k: int) -> list[dict[str, str]]:
    with eval_csv.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if _row_is_feasible(row)]
    rows.sort(key=lambda r: float(r.get("mass_g", "inf") or "inf"))
    return rows[:top_k]


def _row_is_feasible(row: dict[str, str]) -> bool:
    raw = row.get("feasible")
    if raw is None:
        return False
    s = str(raw).strip().lower()
    return s in {"1", "true", "yes"} or (s.replace(".", "", 1).isdigit() and float(s) > 0.5)


def _is_audit_failure(quality: object) -> bool:
    return bool(
        getattr(quality, "error", None)
        or getattr(quality, "watertight_by_edges", True) is False
        or getattr(quality, "within_envelope", True) is False
    )
