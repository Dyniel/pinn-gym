"""ROM-guided candidate search using feedback from existing physics gates."""

from __future__ import annotations

import csv
import heapq
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .design_space import (
    PARAM_FIELDS,
    DesignParams,
    feature_names,
    geometry_feature_row,
    pseudo_response,
    row_to_features,
    sample_designs,
)
from .paths import ensure_dir
from .stl import export_design_stl


ROM_TARGET_FIELDS = [
    "physics_energy_usable_J",
    "physics_peak_force_N",
    "physics_collapse_mm",
    "physics_failure_risk",
    "physics_score",
    "impact_absorbed_J",
    "impact_peak_force_N",
    "impact_max_displacement_mm",
    "impact_failure_risk",
    "impact_score",
]

SURROGATE_CONTEXT_FIELDS = [
    "score",
    "mass_g_mean",
    "energy_abs_J_lcb",
    "force_peak_N_mean",
    "force_plateau_N_mean",
    "collapse_displacement_mm_mean",
    "failure_probability_mean",
    "peak_plateau_ratio_mean",
    "progressive_crush_score_mean",
]


@dataclass(frozen=True)
class RomSearchConfig:
    seed: int = 20260517
    search_n: int = 1_000_000
    batch_size: int = 8192
    top_k: int = 1000
    target_energy_j: float = 29.43
    peak_limit_n: float = 3500.0
    min_crush_mm: float = 40.0
    target_mass_low_g: float = 26.0
    target_mass_high_g: float = 36.0
    export_stl_count: int = 32
    stl_backend: str = "auto"
    stl_resolution: int = 128
    stl_format: str = "binary"
    device: str = "cuda"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _index_by_rank(rows: Iterable[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            rank = int(float(row["rank"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[rank] = row
    return out


def _f(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
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


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else ""
    if value is None:
        return ""
    return str(value)


def _dynamic_defaults(physics: dict[str, str] | None, dynamic: dict[str, str] | None) -> dict[str, float]:
    physics_energy = _f(physics, "physics_energy_usable_J", 0.0)
    physics_peak = _f(physics, "physics_peak_force_N", 1e6)
    physics_collapse = _f(physics, "physics_collapse_mm", 0.0)
    physics_risk = _f(physics, "physics_failure_risk", 1.0)
    physics_score = _f(physics, "physics_score", 1e4)
    return {
        "impact_absorbed_J": _f(dynamic, "impact_absorbed_J", physics_energy),
        "impact_peak_force_N": _f(dynamic, "impact_peak_force_N", physics_peak),
        "impact_max_displacement_mm": _f(dynamic, "impact_max_displacement_mm", physics_collapse),
        "impact_failure_risk": _f(dynamic, "impact_failure_risk", physics_risk),
        "impact_score": _f(dynamic, "impact_score", physics_score),
    }


def build_rom_feedback_dataset(
    run_dirs: list[Path],
    out_csv: Path,
    top_n: int = 1000,
) -> dict[str, object]:
    """Join top-candidate parameters with available ROM/dynamic feedback."""

    rows_out: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        top = _index_by_rank(_read_csv(run_dir / "postanalysis" / "top_candidates.csv")[:top_n])
        physics = _index_by_rank(_read_csv(run_dir / "physics_gate" / "physics_candidates.csv"))
        dynamic = _index_by_rank(_read_csv(run_dir / "dynamic_impact" / "dynamic_impact_candidates.csv"))
        ranks = sorted(set(top) & (set(physics) | set(dynamic)))
        for rank in ranks:
            top_row = top[rank]
            physics_row = physics.get(rank)
            dynamic_row = dynamic.get(rank)
            params = DesignParams.from_row(top_row)
            row: dict[str, object] = {
                "source_run": run_dir.name,
                "source_rank": rank,
                "rank": len(rows_out) + 1,
            }
            row.update(params.to_row())
            row.update({key: _format(value) for key, value in geometry_feature_row(params).items()})
            for field in SURROGATE_CONTEXT_FIELDS:
                row[f"surrogate_{field}"] = top_row.get(field, "")
            row.update(
                {
                    "physics_energy_usable_J": _f(physics_row, "physics_energy_usable_J", _f(dynamic_row, "impact_absorbed_J", 0.0)),
                    "physics_peak_force_N": _f(physics_row, "physics_peak_force_N", _f(dynamic_row, "impact_peak_force_N", 1e6)),
                    "physics_collapse_mm": _f(physics_row, "physics_collapse_mm", _f(dynamic_row, "impact_max_displacement_mm", 0.0)),
                    "physics_failure_risk": _f(physics_row, "physics_failure_risk", _f(dynamic_row, "impact_failure_risk", 1.0)),
                    "physics_score": _f(physics_row, "physics_score", _f(dynamic_row, "impact_score", 1e4)),
                }
            )
            row.update(_dynamic_defaults(physics_row, dynamic_row))
            if all(math.isfinite(float(row[field])) for field in ROM_TARGET_FIELDS):
                rows_out.append(row)

    if not rows_out:
        raise ValueError("no ROM feedback rows could be built from the supplied runs")

    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    fieldnames = ["source_run", "source_rank", "rank"] + PARAM_FIELDS + list(geometry_feature_row(DesignParams.from_row(rows_out[0])).keys())
    fieldnames += [f"surrogate_{field}" for field in SURROGATE_CONTEXT_FIELDS] + ROM_TARGET_FIELDS
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow({key: _format(row.get(key)) for key in fieldnames})

    by_run: dict[str, int] = {}
    for row in rows_out:
        name = str(row["source_run"])
        by_run[name] = by_run.get(name, 0) + 1
    summary = {"out_csv": str(out_csv), "rows": len(rows_out), "run_dirs": [str(x) for x in run_dirs], "rows_by_run": by_run}
    out_csv.with_suffix(out_csv.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_rom_tensors(feedback_csv: Path, max_rows: int | None = None):
    import torch

    xs: list[list[float]] = []
    ys: list[list[float]] = []
    rows = _read_csv(feedback_csv)
    for row_i, row in enumerate(rows):
        if max_rows is not None and row_i >= max_rows:
            break
        try:
            x = row_to_features(row)
            y = [float(row[field]) for field in ROM_TARGET_FIELDS]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in x + y):
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"no usable ROM feedback rows in {feedback_csv}")
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def _standardize(tensor):
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (tensor - mean) / std, mean, std


def _r2_score(pred, target) -> float:
    ss_res = (pred - target).pow(2).sum()
    ss_tot = (target - target.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-9)
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def train_rom_reranker(
    feedback_csv: Path,
    out_dir: Path,
    epochs: int = 260,
    batch_size: int = 512,
    hidden_dim: int = 384,
    blocks: int = 6,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.18,
    max_rows: int | None = None,
    device: str = "cuda",
    seed: int = 20260517,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .torch_models import PolmiSurrogate

    out_dir = ensure_dir(out_dir)
    x, y = _load_rom_tensors(Path(feedback_csv), max_rows=max_rows)
    if x.shape[0] < 50:
        raise ValueError("need at least 50 ROM feedback rows for reranker training")
    x_z, x_mean, x_std = _standardize(x)
    y_z, y_mean, y_std = _standardize(y)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    n = x.shape[0]
    perm = torch.randperm(n)
    val_n = max(16, int(n * val_fraction))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    loader = DataLoader(
        TensorDataset(x_z[train_idx], y_z[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch_device.type == "cuda",
    )
    model = PolmiSurrogate(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dim=hidden_dim,
        blocks=blocks,
        dropout=0.04,
    ).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=torch_device.type == "cuda")
    val_x = x_z[val_idx].to(torch_device)
    val_y = y_z[val_idx].to(torch_device)
    val_raw = y[val_idx].to(torch_device)
    best = {"val_loss": math.inf, "epoch": -1, "r2": -math.inf}
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for bx, by in loader:
            bx = bx.to(torch_device, non_blocking=True)
            by = by.to(torch_device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch_device.type == "cuda"):
                pred = model(bx)
                loss = (pred - by).pow(2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        model.eval()
        with torch.no_grad():
            pred_z = model(val_x)
            val_loss = float((pred_z - val_y).pow(2).mean().detach().cpu())
            pred_raw = pred_z * y_std.to(torch_device) + y_mean.to(torch_device)
            r2 = _r2_score(pred_raw, val_raw)
        train_loss = total / max(1, batches)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "r2": r2})
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "r2": r2}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "feature_names": feature_names(),
                    "target_names": ROM_TARGET_FIELDS,
                    "config": {
                        "feedback_csv": str(feedback_csv),
                        "rows": n,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "hidden_dim": hidden_dim,
                        "blocks": blocks,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "device": str(torch_device),
                        "seed": seed,
                    },
                },
                out_dir / "rom_reranker.pt",
            )
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(f"[rom-reranker] epoch={epoch:04d} train={train_loss:.5f} val={val_loss:.5f} r2={r2:.4f}", flush=True)

    payload = {
        "feedback_csv": str(feedback_csv),
        "out_dir": str(out_dir),
        "checkpoint": str(out_dir / "rom_reranker.pt"),
        "rows": n,
        "best": best,
        "history_tail": history[-10:],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_rom_checkpoint(model_dir: Path, device: str):
    import torch

    from .torch_models import PolmiSurrogate

    ckpt_path = Path(model_dir) / "rom_reranker.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing ROM reranker checkpoint: {ckpt_path}")
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
    model = PolmiSurrogate(
        input_dim=len(ckpt["feature_names"]),
        output_dim=len(ckpt["target_names"]),
        hidden_dim=int(ckpt["config"]["hidden_dim"]),
        blocks=int(ckpt["config"]["blocks"]),
        dropout=0.0,
    ).to(torch_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, torch_device


def _pred_map(target_names: list[str], values: list[float]) -> dict[str, float]:
    out = {}
    for name, value in zip(target_names, values):
        if name.endswith("_risk"):
            value = max(0.0, min(1.0, value))
        elif name.endswith("_score"):
            value = max(0.0, value)
        else:
            value = max(0.0, value)
        out[f"rom_pred_{name}"] = value
    return out


def _rom_score(pred: dict[str, float], mass_g: float, config: RomSearchConfig) -> float:
    physics_energy = pred.get("rom_pred_physics_energy_usable_J", 0.0)
    impact_energy = pred.get("rom_pred_impact_absorbed_J", physics_energy)
    physics_peak = pred.get("rom_pred_physics_peak_force_N", 1e6)
    impact_peak = pred.get("rom_pred_impact_peak_force_N", physics_peak)
    physics_crush = pred.get("rom_pred_physics_collapse_mm", 0.0)
    impact_crush = pred.get("rom_pred_impact_max_displacement_mm", physics_crush)
    physics_risk = pred.get("rom_pred_physics_failure_risk", 1.0)
    impact_risk = pred.get("rom_pred_impact_failure_risk", physics_risk)
    conservative_energy = min(physics_energy, impact_energy)
    conservative_peak = max(physics_peak, impact_peak)
    conservative_crush = min(physics_crush, impact_crush)
    conservative_risk = max(physics_risk, impact_risk)
    return (
        mass_g
        + 95.0 * max(0.0, config.target_energy_j - conservative_energy)
        + 0.050 * max(0.0, conservative_peak - config.peak_limit_n)
        + 18.0 * max(0.0, config.min_crush_mm - conservative_crush)
        + 90.0 * conservative_risk
        + 8.0 * max(0.0, config.target_mass_low_g - mass_g)
        + 0.8 * max(0.0, mass_g - config.target_mass_high_g)
    )


def run_rom_guided_search(
    model_dir: Path,
    out_dir: Path,
    config: RomSearchConfig | None = None,
) -> dict[str, object]:
    import torch

    config = config or RomSearchConfig()
    out_dir = ensure_dir(out_dir)
    stl_dir = ensure_dir(out_dir / "stl")
    model, ckpt, torch_device = _load_rom_checkpoint(model_dir, device=config.device)
    target_names = list(ckpt["target_names"])
    rng = random.Random(config.seed)
    heap: list[tuple[float, int, dict[str, str]]] = []
    counter = 0
    feature_name_list = feature_names()
    mass_idx = feature_name_list.index("mass_g_est")
    for start in range(0, config.search_n, config.batch_size):
        n = min(config.batch_size, config.search_n - start)
        designs = sample_designs(n, seed=rng.randint(0, 2_000_000_000))
        rows = [design.to_row() for design in designs]
        x = torch.tensor([row_to_features(row) for row in rows], dtype=torch.float32, device=torch_device)
        with torch.no_grad():
            xz = (x - ckpt["x_mean"].to(torch_device)) / ckpt["x_std"].to(torch_device)
            pred = model(xz) * ckpt["y_std"].to(torch_device) + ckpt["y_mean"].to(torch_device)
        pred_cpu = pred.detach().cpu()
        masses = x[:, mass_idx].detach().cpu().tolist()
        batch_scores: list[float] = []
        batch_pred_maps: list[dict[str, float]] = []
        for i in range(n):
            pred_dict = _pred_map(target_names, pred_cpu[i].tolist())
            batch_pred_maps.append(pred_dict)
            batch_scores.append(_rom_score(pred_dict, float(masses[i]), config))
        candidate_count = min(config.top_k, n)
        best_idx = sorted(range(n), key=lambda idx: batch_scores[idx])[:candidate_count]
        for idx in best_idx:
            params = designs[idx]
            response = pseudo_response(params)
            row = params.to_row()
            row.update({key: _format(value) for key, value in geometry_feature_row(params).items()})
            pred_dict = batch_pred_maps[idx]
            score = batch_scores[idx]
            row.update(
                {
                    "score": f"{score:.8g}",
                    "rom_score": f"{score:.8g}",
                    "rank_source": "rom_reranker",
                    "mass_g_mean": f"{float(response['mass_g']):.8g}",
                    "energy_abs_J_mean": f"{float(response['energy_abs_J']):.8g}",
                    "energy_abs_J_lcb": f"{float(response['energy_abs_J']):.8g}",
                    "force_peak_N_mean": f"{float(response['force_peak_N']):.8g}",
                    "force_plateau_N_mean": f"{float(response['force_plateau_N']):.8g}",
                    "collapse_displacement_mm_mean": f"{float(response['collapse_displacement_mm']):.8g}",
                    "failure_probability_mean": f"{float(response['failure_probability']):.8g}",
                    "early_energy_20mm_J_mean": f"{float(response['early_energy_20mm_J']):.8g}",
                    "peak_plateau_ratio_mean": f"{float(response['peak_plateau_ratio']):.8g}",
                    "progressive_crush_score_mean": f"{float(response['progressive_crush_score']):.8g}",
                }
            )
            row.update({key: f"{value:.8g}" for key, value in pred_dict.items()})
            item = (-score, counter, row)
            if len(heap) < config.top_k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            counter += 1
        if start == 0 or (start + n) % max(config.batch_size, config.search_n // 10) == 0:
            print(f"[rom-guided-search] searched {start+n}/{config.search_n}", flush=True)

    selected = [item[2] for item in sorted(heap, key=lambda item: -item[0])]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = str(rank)
    preferred = [
        "rank",
        "score",
        "rom_score",
        "rank_source",
        *PARAM_FIELDS,
        "mass_g_mean",
        "energy_abs_J_lcb",
        "force_peak_N_mean",
        "force_plateau_N_mean",
        "collapse_displacement_mm_mean",
        "failure_probability_mean",
        "early_energy_20mm_J_mean",
        "peak_plateau_ratio_mean",
        "progressive_crush_score_mean",
    ]
    pred_fields = [f"rom_pred_{field}" for field in ROM_TARGET_FIELDS]
    extras = [key for key in selected[0].keys() if key not in set(preferred + pred_fields)] if selected else []
    fieldnames = preferred + pred_fields + extras
    top_csv = out_dir / "top_candidates.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    exported = []
    for row in selected[: config.export_stl_count]:
        params = DesignParams.from_row(row)
        stl_path = stl_dir / f"rank_{int(row['rank']):03d}_{params.topology}.stl"
        export_design_stl(params, stl_path, backend=config.stl_backend, resolution=config.stl_resolution, stl_format=config.stl_format)
        exported.append(str(stl_path))

    lines = [
        "# POLMI ROM-Guided Search",
        "",
        f"Search candidates: `{config.search_n}`",
        f"Top K: `{config.top_k}`",
        f"Exported STL: `{len(exported)}`",
        "",
        "| Rank | Score | Topology | Mass g | ROM E J | ROM Peak N | ROM Crush mm | ROM Risk |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected[:20]:
        lines.append(
            f"| {row['rank']} | {row['score']} | {row['topology']} | {row['mass_g_mean']} | "
            f"{row['rom_pred_impact_absorbed_J']} | {row['rom_pred_impact_peak_force_N']} | "
            f"{row['rom_pred_impact_max_displacement_mm']} | {row['rom_pred_impact_failure_risk']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"top_csv": str(top_csv), "exported_stl": exported, "search_n": config.search_n, "top_k": config.top_k, "config": asdict(config)}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def select_candidates_by_physics(
    top_csv: Path,
    physics_csv: Path,
    out_csv: Path,
    top_n: int = 500,
) -> dict[str, object]:
    top_rows = _index_by_rank(_read_csv(top_csv))
    physics_rows = _read_csv(physics_csv)
    joined: list[dict[str, str]] = []
    for ph in sorted(physics_rows, key=lambda row: _f(row, "physics_score", math.inf)):
        try:
            rank = int(float(ph["rank"]))
        except (KeyError, ValueError):
            continue
        top = top_rows.get(rank)
        if top is None:
            continue
        row = dict(top)
        row.update({f"coarse_{key}": value for key, value in ph.items() if key != "rank"})
        joined.append(row)
        if len(joined) >= top_n:
            break
    if not joined:
        raise ValueError("no candidates could be selected by physics score")
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    preferred = list(joined[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=preferred, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(joined)
    summary = {"top_csv": str(top_csv), "physics_csv": str(physics_csv), "out_csv": str(out_csv), "rows": len(joined)}
    out_csv.with_suffix(out_csv.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _shortlist_score(
    mass_g: float,
    physics: dict[str, str],
    dynamic: dict[str, str],
    target_energy_j: float,
    peak_limit_n: float,
) -> float:
    physics_energy = _f(physics, "physics_energy_usable_J", 0.0)
    dynamic_energy = _f(dynamic, "impact_absorbed_J", 0.0)
    physics_peak = _f(physics, "physics_peak_force_N", peak_limit_n)
    dynamic_peak = _f(dynamic, "impact_peak_force_N", peak_limit_n)
    physics_risk = _f(physics, "physics_failure_risk", 1.0)
    dynamic_risk = _f(dynamic, "impact_failure_risk", 1.0)
    return (
        mass_g
        + 38.0 * max(0.0, target_energy_j - physics_energy)
        + 48.0 * max(0.0, 0.98 * target_energy_j - dynamic_energy)
        + 0.0015 * max(0.0, physics_peak - peak_limit_n)
        + 0.0025 * max(0.0, dynamic_peak - peak_limit_n)
        + 4.0 * physics_risk
        + 7.0 * dynamic_risk
        - 0.08 * max(0.0, physics_energy - target_energy_j)
    )


def select_finalist_shortlist(
    top_csv: Path,
    physics_csv: Path,
    dynamic_csv: Path,
    out_csv: Path,
    top_n: int = 30,
    target_energy_j: float = 29.43,
    peak_limit_n: float = 3500.0,
    min_crush_mm: float = 40.0,
    max_failure_risk: float = 0.48,
) -> dict[str, object]:
    """Select original top-table rows that pass both fixed physics and impact gates."""

    top_rows = _index_by_rank(_read_csv(top_csv))
    physics_rows = _index_by_rank(_read_csv(physics_csv))
    dynamic_rows = _index_by_rank(_read_csv(dynamic_csv))
    joined: list[dict[str, str]] = []
    for rank in sorted(set(top_rows) & set(physics_rows) & set(dynamic_rows)):
        top = top_rows[rank]
        ph = physics_rows[rank]
        dy = dynamic_rows[rank]
        physics_pass = _truthy(ph.get("physics_survives_gate")) or (
            _f(ph, "physics_energy_usable_J", 0.0) >= target_energy_j
            and _f(ph, "physics_peak_force_N", math.inf) <= peak_limit_n
            and _f(ph, "physics_collapse_mm", 0.0) >= min_crush_mm
            and _f(ph, "physics_failure_risk", 1.0) <= max_failure_risk
        )
        dynamic_crush_pass = (
            _truthy(dy.get("impact_crush_distance_pass"))
            if "impact_crush_distance_pass" in dy
            else _f(ph, "physics_collapse_mm", 0.0) >= min_crush_mm
        )
        dynamic_pass = _truthy(dy.get("impact_survives")) or (
            _f(dy, "impact_absorbed_J", 0.0) >= 0.98 * target_energy_j
            and _f(dy, "impact_peak_force_N", math.inf) <= peak_limit_n
            and dynamic_crush_pass
            and _f(dy, "impact_failure_risk", 1.0) <= max_failure_risk
        )
        if not (physics_pass and dynamic_pass):
            continue
        mass_g = _f(top, "mass_g_mean", _f(ph, "mass_g", _f(dy, "mass_g", math.inf)))
        row = dict(top)
        row.update(
            {
                "shortlist_score": _format(_shortlist_score(mass_g, ph, dy, target_energy_j, peak_limit_n)),
                "shortlist_physics_energy_usable_J": ph.get("physics_energy_usable_J", ""),
                "shortlist_physics_peak_force_N": ph.get("physics_peak_force_N", ""),
                "shortlist_physics_collapse_mm": ph.get("physics_collapse_mm", ""),
                "shortlist_physics_failure_risk": ph.get("physics_failure_risk", ""),
                "shortlist_dynamic_absorbed_J": dy.get("impact_absorbed_J", ""),
                "shortlist_dynamic_peak_force_N": dy.get("impact_peak_force_N", ""),
                "shortlist_dynamic_max_displacement_mm": dy.get("impact_max_displacement_mm", ""),
                "shortlist_dynamic_crush_distance_pass": str(dynamic_crush_pass),
                "shortlist_dynamic_failure_risk": dy.get("impact_failure_risk", ""),
            }
        )
        joined.append(row)
    joined.sort(key=lambda row: (float(row["shortlist_score"]), _f(row, "mass_g_mean", math.inf)))
    joined = joined[:top_n]
    if not joined:
        raise ValueError("no candidates passed both physics and dynamic gates")

    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    fieldnames: list[str] = []
    for row in joined:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(joined)

    summary = {
        "top_csv": str(top_csv),
        "physics_csv": str(physics_csv),
        "dynamic_csv": str(dynamic_csv),
        "out_csv": str(out_csv),
        "rows": len(joined),
        "top_n": top_n,
        "best": {key: joined[0].get(key, "") for key in ("rank", "topology", "mass_g_mean", "shortlist_score")},
    }
    out_csv.with_suffix(out_csv.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
