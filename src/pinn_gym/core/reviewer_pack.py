#!/usr/bin/env python3
"""Assemble and verify a publication-ready reviewer pack.

Reads the four reviewer stages (seed repeats, loss-weight ablation, canonical
regret evaluation, pooled tuning) and writes curated CSV tables plus an
auto-filled SUMMARY.md into ``<run-dir>/publication_pack`` (override with
``--out``). All headline numbers in SUMMARY.md are read from the run, so the
pack stays correct for whatever configuration produced the run.

The public CLI wraps :func:`build_pack` through ``pinn-gym reviewer-pack``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _f(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: object, nd: int = 3) -> str:
    x = _f(value)
    if math.isnan(x):
        return str(value) if value not in (None, "") else "n/a"
    if math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


REQUIRED_TABLES: tuple[str, ...] = (
    "pooled_regret_per_card.csv",
    "baselines_per_card.csv",
    "seed_repeats_ci.csv",
    "loss_weight_ablation.csv",
    "pooled_tuning.csv",
)


def build_pack(run_dir: Path, out_dir: Path) -> dict[str, object]:
    tables = out_dir / "tables"
    raw = out_dir / "raw_per_stage"
    tables.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    # --- Stage 4: canonical regret evaluation -> per-card pooled table -------
    canon = _read(run_dir / "sr_eval_with_regret" / "method_metrics.csv")
    keep = [
        "preset", "scope", "method", "oracle_feasible", "mean_curve_nrmse",
        "predicted_feasible_rate", "physical_violation_rate",
        "mean_energy_integral_abs_error_J", "precision_at_5", "precision_at_10",
        "regret_at_5_g", "regret_at_10_g", "relative_regret_at_10",
    ]
    pooled_rows = [r for r in canon if r.get("scope") == "pooled"]
    # also keep per-card random + oracle baselines for context
    context = [r for r in canon if r.get("scope") == "self"
               and r.get("method") in {"random", "oracle_upper_bound"}]
    _write(tables / "pooled_regret_per_card.csv", pooled_rows, keep)
    _write(tables / "baselines_per_card.csv", context, keep)

    # --- Stage 1: seed repeats -> mean/SD CI table ---------------------------
    seed_sum = _read(run_dir / "sr_repeats" / "method_metrics_seed_summary.csv")
    seed_keep_metrics = ["mean_curve_nrmse", "predicted_feasible_rate",
                         "physical_violation_rate", "regret_at_10_g"]
    seed_rows = []
    for r in seed_sum:
        if r.get("scope") != "pooled":
            continue
        out = {"preset": r.get("preset"), "method": r.get("method"),
               "repeats": r.get("repeats", "")}
        for m in seed_keep_metrics:
            out[f"{m}_mean"] = r.get(f"{m}_mean", "")
            out[f"{m}_std"] = r.get(f"{m}_std", "")
        seed_rows.append(out)
    _write(tables / "seed_repeats_ci.csv", seed_rows)

    # --- Stage 2: loss-weight ablation ---------------------------------------
    loss = _read(run_dir / "sr_loss_weight_ablation" / "loss_weight_ablation_summary.csv")
    loss_rows = [r for r in loss if r.get("scope") == "pooled"]
    _write(tables / "loss_weight_ablation.csv", loss_rows)

    # --- Stage 5: pooled tuning ----------------------------------------------
    tuning = _read(run_dir / "sr_pooled_tuning" / "pooled_tuning_results.csv")
    _write(tables / "pooled_tuning.csv", tuning)

    # raw copies for traceability
    for rel in [
        "sr_eval_with_regret/method_metrics.csv",
        "sr_repeats/method_metrics_by_seed.csv",
        "sr_loss_weight_ablation/loss_weight_ablation_metrics.csv",
        "sr_pooled_tuning/pooled_tuning_metrics.csv",
    ]:
        src = run_dir / rel
        if src.exists():
            (raw / rel.replace("/", "__")).write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8")

    _write_summary(out_dir, run_dir, pooled_rows, context, seed_rows, loss_rows, tuning)
    sha_path = write_sha256sums(out_dir)
    return {
        "out_dir": str(out_dir),
        "tables": [str(tables / name) for name in REQUIRED_TABLES],
        "sha256sums": str(sha_path),
    }


def verify_pack(pack_dir: Path) -> dict[str, object]:
    """Validate the reviewer pack shape and SHA256 manifest."""

    missing: list[str] = []
    tables = pack_dir / "tables"
    if not tables.is_dir():
        missing.append("tables/")
    for name in REQUIRED_TABLES:
        path = tables / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(f"tables/{name}")

    if not ((pack_dir / "SUMMARY.md").is_file() or (pack_dir / "INTERPRETATION.md").is_file()):
        missing.append("SUMMARY.md or INTERPRETATION.md")

    sha_path = pack_dir / "SHA256SUMS"
    if not sha_path.is_file() or sha_path.stat().st_size == 0:
        missing.append("SHA256SUMS")

    sha_errors: list[str] = []
    if sha_path.is_file():
        for lineno, line in enumerate(sha_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                expected, rel = line.split("  ", 1)
            except ValueError:
                sha_errors.append(f"SHA256SUMS:{lineno}: malformed line")
                continue
            path = pack_dir / rel
            if not path.is_file():
                sha_errors.append(f"{rel}: missing")
                continue
            actual = _sha256(path)
            if actual != expected:
                sha_errors.append(f"{rel}: sha256 mismatch")

    ok = not missing and not sha_errors
    return {
        "pack_dir": str(pack_dir),
        "ok": ok,
        "missing": missing,
        "sha_errors": sha_errors,
    }


def write_sha256sums(root: Path) -> Path:
    """Write a deterministic SHA256 manifest for every file in ``root``."""

    sha_path = root / "SHA256SUMS"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != sha_path.name
    )
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    sha_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return sha_path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _by(rows, **kw):
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()):
            return r
    return None


def _write_summary(out_dir, run_dir, pooled, context, seed_rows, loss_rows, tuning) -> None:
    presets = []
    for r in pooled:
        if r.get("preset") not in presets:
            presets.append(r.get("preset"))

    pooled_nrmse = [_f(r.get("mean_curve_nrmse")) for r in pooled
                    if r.get("method") == "pinn_full"]
    pooled_nrmse = [x for x in pooled_nrmse if math.isfinite(x)]
    macro = statistics.mean(pooled_nrmse) if pooled_nrmse else math.nan

    best = None
    if tuning:
        best = min(tuning, key=lambda r: _f(r.get("objective")) if math.isfinite(_f(r.get("objective"))) else math.inf)

    lines = [
        "# Reviewer experiments — publication pack",
        "",
        f"Auto-generated from `{run_dir}`.",
        "All metrics per material card, scoped to the declared numerical oracle.",
        "",
        "## Headline numbers (read from this run)",
        "",
        f"- Pooled `pinn_full` macro curve nRMSE: **{_fmt(macro)}** "
        f"(over {len(pooled_nrmse)} cards).",
    ]
    if best:
        lines.append(
            f"- Best tuned pooled candidate: **{best.get('tune_candidate')}** "
            f"(hidden={best.get('hidden_dim')}, blocks={best.get('blocks')}, "
            f"epochs={best.get('epochs')}) — macro nRMSE "
            f"{_fmt(best.get('mean_curve_nrmse_mean'))}, "
            f"regret@10 {_fmt(best.get('regret_at_10_g_mean'))}, "
            f"collapse_penalty {_fmt(best.get('feasibility_collapse_penalty'))}."
        )
    lines += ["", "## Per-card pooled model (canonical regret eval)", "",
              "| card | nRMSE | pred-feasible | viol | P@10 | regret@10 [g] |",
              "|---|---:|---:|---:|---:|---:|"]
    for p in presets:
        r = _by(pooled, preset=p, method="pinn_full")
        b = _by(context, preset=p, method="random")
        if not r:
            continue
        rg = r.get("regret_at_10_g", "")
        rg_disp = "N/A" if str(rg).strip().lower() in {"nan", ""} else _fmt(rg)
        lines.append(
            f"| {p} | {_fmt(r.get('mean_curve_nrmse'))} | "
            f"{_fmt(r.get('predicted_feasible_rate'))} | "
            f"{_fmt(r.get('physical_violation_rate'))} | "
            f"{_fmt(r.get('precision_at_10'))} | {rg_disp} |"
        )
        if b:
            lines[-1] += f"  <!-- random nRMSE {_fmt(b.get('mean_curve_nrmse'))} -->"

    lines += ["", "## Seed CI (pooled pinn_full, mean ± SD)", "",
              "| card | nRMSE mean | nRMSE SD | pred-feasible mean |",
              "|---|---:|---:|---:|"]
    for r in seed_rows:
        if r.get("method") != "pinn_full":
            continue
        lines.append(
            f"| {r.get('preset')} | {_fmt(r.get('mean_curve_nrmse_mean'))} | "
            f"{_fmt(r.get('mean_curve_nrmse_std'))} | "
            f"{_fmt(r.get('predicted_feasible_rate_mean'))} |"
        )

    lines += [
        "",
        "## Files",
        "- `tables/pooled_regret_per_card.csv` — main pooled results + regret.",
        "- `tables/baselines_per_card.csv` — random/oracle baselines for context.",
        "- `tables/seed_repeats_ci.csv` — 5-seed mean/SD.",
        "- `tables/loss_weight_ablation.csv` — physics-loss-weight ablation.",
        "- `tables/pooled_tuning.csv` — architecture/optimiser grid.",
        "- `raw_per_stage/` — unfiltered per-stage CSVs.",
        "",
        "## Notes",
        "- `regret = inf`: feasible pool exists but none selected in top-k.",
        "- `regret = N/A`: card has no oracle-feasible design (ranking undefined).",
        "- Report the network size and physics-loss weights with every number; "
        "feasibility prediction is sensitive to the energy-loss weight.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="reviewer_experiments_* run directory")
    ap.add_argument("--out", default=None, help="output dir (default: <run-dir>/publication_pack)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "publication_pack"
    payload = build_pack(run_dir, out_dir)
    print(f"publication pack written to: {payload['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
