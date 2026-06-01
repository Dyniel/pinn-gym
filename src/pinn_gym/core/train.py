"""Train the POLMI neural surrogate."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from .design_space import CURVE_POINTS, SCALAR_TARGET_FIELDS, DesignParams, feature_names, pseudo_response, row_to_features
from .paths import ensure_dir, project_root


def _curve_fields(fieldnames: list[str]) -> list[str]:
    return sorted([x for x in fieldnames if x.startswith("curve_")])


def load_candidate_tensors(path: Path, max_rows: int | None = None):
    import torch

    xs: list[list[float]] = []
    ys: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        curves = _curve_fields(reader.fieldnames)
        target_names = list(SCALAR_TARGET_FIELDS) + curves
        for row_i, row in enumerate(reader):
            if max_rows is not None and row_i >= max_rows:
                break
            try:
                xs.append(row_to_features(row))
                bootstrap: dict[str, object] | None = None
                scalar_values: list[float] = []
                for name in SCALAR_TARGET_FIELDS:
                    value = row.get(name, "")
                    if value == "":
                        if bootstrap is None:
                            bootstrap = pseudo_response(DesignParams.from_row(row), curve_points=len(curves) or CURVE_POINTS)
                        scalar_values.append(float(bootstrap[name]))
                    else:
                        scalar_values.append(float(value))
                curve_values = [float(row[name]) for name in curves]
                ys.append(scalar_values + curve_values)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"bad row {row_i} in {path}: {exc}") from exc
    if not xs:
        raise ValueError(f"no candidates loaded from {path}")
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y, target_names


def _standardize(x):
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (x - mean) / std, mean, std


def _weighted_loss(pred, target, scalar_count: int):
    import torch

    weights = torch.ones(target.shape[1], device=target.device)
    if target.shape[1] > scalar_count:
        weights[scalar_count:] = 0.35
    return ((pred - target).pow(2) * weights).mean()


def _r2_score(pred, target) -> float:
    ss_res = (pred - target).pow(2).sum()
    ss_tot = (target - target.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-9)
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def train_surrogate(
    candidates_csv: Path,
    run_dir: Path,
    epochs: int = 240,
    batch_size: int = 2048,
    ensemble_size: int = 5,
    hidden_dim: int = 512,
    blocks: int = 8,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    max_rows: int | None = None,
    val_fraction: float = 0.12,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .torch_models import PolmiSurrogate

    run_dir = ensure_dir(run_dir)
    x, y, target_names = load_candidate_tensors(candidates_csv, max_rows=max_rows)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("non-finite values in training tensors")

    x_z, x_mean, x_std = _standardize(x)
    y_z, y_mean, y_std = _standardize(y)
    n = x_z.shape[0]
    if n < 100:
        raise ValueError("need at least 100 candidate rows for training")

    base_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    feature_names_list = feature_names()
    scalar_count = len(SCALAR_TARGET_FIELDS)
    models_metrics: list[dict[str, Any]] = []
    config = {
        "candidates_csv": str(candidates_csv),
        "rows": n,
        "feature_names": feature_names_list,
        "target_names": target_names,
        "epochs": epochs,
        "batch_size": batch_size,
        "ensemble_size": ensemble_size,
        "hidden_dim": hidden_dim,
        "blocks": blocks,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": str(base_device),
        "val_fraction": val_fraction,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for model_idx in range(ensemble_size):
        seed = 10_000 + model_idx * 997
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        perm = torch.randperm(n)
        val_n = max(64, int(n * val_fraction))
        val_idx = perm[:val_n]
        train_idx = perm[val_n:]
        train_ds = TensorDataset(x_z[train_idx], y_z[train_idx])
        val_x = x_z[val_idx].to(base_device)
        val_y = y_z[val_idx].to(base_device)
        val_y_raw = y[val_idx].to(base_device)

        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=base_device.type == "cuda")
        model = PolmiSurrogate(
            input_dim=x_z.shape[1],
            output_dim=y_z.shape[1],
            hidden_dim=hidden_dim,
            blocks=blocks,
            dropout=0.05,
        ).to(base_device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        scaler = torch.amp.GradScaler("cuda", enabled=base_device.type == "cuda")
        best = {"val_loss": math.inf, "epoch": -1, "r2": -math.inf}
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_total = 0.0
            batches = 0
            for bx, by in loader:
                bx = bx.to(base_device, non_blocking=True)
                by = by.to(base_device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=base_device.type == "cuda"):
                    pred = model(bx)
                    loss = _weighted_loss(pred, by, scalar_count)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch}, model {model_idx}")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                train_loss_total += float(loss.detach().cpu())
                batches += 1
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = float(_weighted_loss(val_pred, val_y, scalar_count).detach().cpu())
                val_pred_raw = val_pred * y_std.to(base_device) + y_mean.to(base_device)
                r2 = _r2_score(val_pred_raw[:, :scalar_count], val_y_raw[:, :scalar_count])
            train_loss = train_loss_total / max(1, batches)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "scalar_r2": r2})
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "epoch": epoch, "r2": r2}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "x_mean": x_mean,
                        "x_std": x_std,
                        "y_mean": y_mean,
                        "y_std": y_std,
                        "feature_names": feature_names_list,
                        "target_names": target_names,
                        "config": config,
                    },
                    run_dir / f"model_seed{model_idx}.pt",
                )
            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                print(
                    f"[model {model_idx+1}/{ensemble_size}] epoch={epoch:04d} "
                    f"train={train_loss:.5f} val={val_loss:.5f} r2={r2:.4f}",
                    flush=True,
                )

        model_metric = {
            "model_index": model_idx,
            "seed": seed,
            "best_val_loss": best["val_loss"],
            "best_epoch": best["epoch"],
            "best_scalar_r2": best["r2"],
            "history_tail": history[-10:],
            "checkpoint": str(run_dir / f"model_seed{model_idx}.pt"),
        }
        models_metrics.append(model_metric)
        (run_dir / f"history_seed{model_idx}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    metrics = {"config": config, "models": models_metrics}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def default_run_dir(run_name: str | None = None) -> Path:
    root = project_root()
    name = run_name or "pinn_gym_surrogate"
    return root / "simulations" / "runs" / name
