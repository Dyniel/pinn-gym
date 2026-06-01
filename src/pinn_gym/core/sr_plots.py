"""Plotting helpers for the Scientific Reports benchmark.

Figures are stratified by material card, matching the SR scoping rule that
metrics must be per-material before any macro-average. Plotting uses
matplotlib with the default backend so it can run on a Slurm node without
display. All functions write PNG files and return their paths.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .design_space import displacement_axis
from .paths import ensure_dir


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def plot_pool_feasibility(dataset_summary: Path, out_path: Path) -> Path:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(Path(dataset_summary).read_text(encoding="utf-8"))
    presets = [item["preset"] for item in payload["datasets"]]
    train_rates = [float(item["train_feasible_rate"]) for item in payload["datasets"]]
    eval_rates = [float(item["eval_feasible_rate"]) for item in payload["datasets"]]
    x = range(len(presets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], train_rates, width=width, label="train")
    ax.bar([i + width / 2 for i in x], eval_rates, width=width, label="eval")
    ax.set_xticks(list(x))
    ax.set_xticklabels(presets)
    ax.set_ylabel("oracle feasible rate")
    ax.set_title("Material-aware pool feasibility by card")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return Path(out_path)


def plot_nrmse_distribution(method_metrics_csv: Path, out_path: Path, *, scope: str = "pooled") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in _read_rows(method_metrics_csv) if row.get("scope") == scope]
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(_f(row, "mean_curve_nrmse"))
    methods = list(by_method.keys())
    values = [by_method[m] for m in methods]
    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.0), 4))
    ax.boxplot(values, labels=methods, showfliers=False)
    ax.set_ylabel("mean curve NRMSE")
    ax.set_title(f"Curve NRMSE by method ({scope}) across materials")
    ax.grid(True, axis="y", alpha=0.3)
    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return Path(out_path)


def plot_violation_rates(method_metrics_csv: Path, out_path: Path, *, scope: str = "pooled") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in _read_rows(method_metrics_csv) if row.get("scope") == scope]
    if not rows:
        return _empty_plot(out_path, f"no rows for scope={scope}")
    methods = sorted({row["method"] for row in rows})
    presets = sorted({row["preset"] for row in rows})
    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.4), 4))
    width = 0.8 / max(1, len(presets))
    for idx, preset in enumerate(presets):
        values = []
        for method in methods:
            match = [row for row in rows if row["preset"] == preset and row["method"] == method]
            values.append(_f(match[0], "physical_violation_rate") if match else math.nan)
        positions = [i + width * (idx - (len(presets) - 1) / 2) for i in range(len(methods))]
        ax.bar(positions, values, width=width, label=preset)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("physical violation rate")
    ax.set_title(f"Predicted-feasible violation rate ({scope})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="material", fontsize=8)
    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return Path(out_path)


def plot_precision_at_k(method_metrics_csv: Path, out_dir: Path, *, scope: str = "pooled") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in _read_rows(method_metrics_csv) if row.get("scope") == scope]
    if not rows:
        return []
    out_dir = ensure_dir(out_dir)
    paths: list[Path] = []
    presets = sorted({row["preset"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    ks = sorted({int(k.replace("precision_at_", ""))
                  for row in rows for k in row.keys()
                  if k.startswith("precision_at_") and row[k] not in {"", "nan"}})
    for preset in presets:
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in methods:
            match = [row for row in rows if row["preset"] == preset and row["method"] == method]
            if not match:
                continue
            ys = [_f(match[0], f"precision_at_{k}") for k in ks]
            ax.plot(ks, ys, marker="o", label=method)
        ax.set_xlabel("k")
        ax.set_ylabel("precision@k")
        ax.set_title(f"Precision@k for material {preset} ({scope})")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        out_path = out_dir / f"precision_at_k_{preset}_{scope}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_transfer_heatmap(transfer_csv: Path, out_path: Path, *, metric: str = "mean_curve_nrmse") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_rows(transfer_csv)
    if not rows:
        return _empty_plot(out_path, "no transfer rows")
    train_set = sorted({row["train_preset"] for row in rows})
    eval_set = sorted({row["eval_preset"] for row in rows})
    if not train_set or not eval_set:
        return _empty_plot(out_path, "no transfer pairs")
    grid = [[math.nan for _ in eval_set] for _ in train_set]
    for row in rows:
        if row.get("method_base") != "pinn_full":
            continue
        try:
            i = train_set.index(row["train_preset"])
            j = eval_set.index(row["eval_preset"])
        except ValueError:
            continue
        grid[i][j] = _f(row, metric)
    fig, ax = plt.subplots(figsize=(1.2 * len(eval_set) + 2, 1.2 * len(train_set) + 2))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(eval_set)))
    ax.set_xticklabels(eval_set, rotation=20, ha="right")
    ax.set_yticks(range(len(train_set)))
    ax.set_yticklabels(train_set)
    ax.set_xlabel("eval material")
    ax.set_ylabel("train material")
    ax.set_title(f"Cross-material transfer ({metric}, pinn_full)")
    for i in range(len(train_set)):
        for j in range(len(eval_set)):
            value = grid[i][j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return Path(out_path)


def plot_force_curve_overlay(eval_csv: Path, out_path: Path, *, n_examples: int = 8) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_rows(eval_csv)
    if not rows:
        return _empty_plot(out_path, "no eval rows")
    curve_fields = sorted([name for name in rows[0] if name.startswith("curve_")])
    if not curve_fields:
        return _empty_plot(out_path, "no curve fields")
    n_points = len(curve_fields)
    disp = displacement_axis(n_points)
    feasible = [row for row in rows if str(row.get("oracle_feasible", "")).lower() in {"true", "1"}][:n_examples]
    infeasible = [row for row in rows if str(row.get("oracle_feasible", "")).lower() not in {"true", "1"}][:n_examples]
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in feasible:
        forces = [_f(row, field) for field in curve_fields]
        ax.plot(disp, forces, color="tab:green", alpha=0.55, linewidth=1.0)
    for row in infeasible:
        forces = [_f(row, field) for field in curve_fields]
        ax.plot(disp, forces, color="tab:red", alpha=0.45, linewidth=0.9, linestyle="--")
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("oracle force [N]")
    ax.set_title("Oracle force-displacement responses (green=feasible, red=infeasible)")
    ax.grid(True, alpha=0.3)
    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return Path(out_path)


def _empty_plot(out_path: Path, note: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, note, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return Path(out_path)


def render_sr_figures(eval_dir: Path, dataset_dir: Path, out_dir: Path, *, presets: list[str]) -> dict[str, str]:
    """Produce the standard set of SR figures from a benchmark evaluation."""

    eval_dir = Path(eval_dir)
    dataset_dir = Path(dataset_dir)
    out_dir = ensure_dir(out_dir)
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        (out_dir / "matplotlib_missing.txt").write_text(
            f"matplotlib not available in this environment ({exc}). "
            "Install it (see pyproject.toml) and re-run sr-plots.\n",
            encoding="utf-8",
        )
        return {"error": "matplotlib not installed"}
    figures: dict[str, str] = {}
    dataset_summary = dataset_dir / "summary.json"
    if dataset_summary.exists():
        figures["pool_feasibility"] = str(
            plot_pool_feasibility(dataset_summary, out_dir / "pool_feasibility.png")
        )
    method_csv = eval_dir / "method_metrics.csv"
    if method_csv.exists():
        figures["nrmse_pooled"] = str(plot_nrmse_distribution(method_csv, out_dir / "nrmse_pooled.png", scope="pooled"))
        figures["violations_pooled"] = str(plot_violation_rates(method_csv, out_dir / "violations_pooled.png", scope="pooled"))
        figures["nrmse_self"] = str(plot_nrmse_distribution(method_csv, out_dir / "nrmse_self.png", scope="self"))
        figures["precision_at_k"] = ";".join(
            str(p) for p in plot_precision_at_k(method_csv, out_dir / "precision_at_k", scope="pooled")
        )
    transfer_csv = eval_dir / "transfer_metrics.csv"
    if transfer_csv.exists():
        figures["transfer_nrmse"] = str(plot_transfer_heatmap(transfer_csv, out_dir / "transfer_nrmse.png", metric="mean_curve_nrmse"))
        figures["transfer_violation"] = str(plot_transfer_heatmap(transfer_csv, out_dir / "transfer_violation.png", metric="physical_violation_rate"))
    for preset in presets:
        eval_csv = dataset_dir / preset / "eval.csv"
        if eval_csv.exists():
            figures[f"curves_{preset}"] = str(plot_force_curve_overlay(eval_csv, out_dir / f"curves_{preset}.png"))
    return figures
