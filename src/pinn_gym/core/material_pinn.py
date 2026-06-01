"""Dimensionless material-card-conditioned PINN for the SR benchmark.

The network learns a dimensionless force ratio ``f_hat(eps, geom*, mat*)``
where ``eps = u / L_env`` and ``f_hat = F / (sigma_y * A_env)``. Physics-
informed losses are written in the dimensionless space so they are the same
for every material card:

* boundary residual ``f_hat(eps=0) = 0`` (no pre-load),
* energy residual ``int f_hat deps`` matches the oracle energy / energy-scale,
* peak-bound residual ``f_hat <= 1 + densification_slack`` (soft yielding),
* densification monotonicity ``df_hat/deps >= 0`` after a learnable plateau,
* curve supervision in dimensionless units.

The network ingests a small bank of dimensionless material descriptors so a
single trained model can be evaluated on unseen material cards by passing
their descriptors at inference time. This is the SR claim: the model is
material-agnostic by *construction* (Buckingham-Pi scaling), and the learned
representation is shared across all cards in the training pool.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .design_space import (
    CURVE_POINTS,
    DesignParams,
    displacement_axis,
    feature_names,
    row_to_features,
)
from .dimensionless import (
    MATERIAL_DIM_FIELDS,
    DimensionlessScales,
    material_dimensionless_features,
    material_feature_vector,
    scales_for_material,
)
from .materials import MaterialCard, load_material_card
from .paths import ensure_dir


@dataclass(frozen=True)
class MaterialPINNConfig:
    epochs: int = 240
    batch_size: int = 8192
    rows_per_material: int = 6000
    # Gradient steps per epoch. 0 = auto: scale with the number of material
    # cards in the pool so a pooled (multi-material) run gets the same per-row
    # gradient coverage as a single-material run. Without this, a pooled run
    # over 5x the data is trained with the same number of steps and collapses
    # (predicts 0% feasible, curve nRMSE worse than the random baseline).
    steps_per_epoch: int = 0
    hidden_dim: int = 384
    blocks: int = 6
    lr: float = 2e-3
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 20260519
    # physics-informed weights, all dimensionless
    boundary_weight: float = 0.05
    energy_weight: float = 0.20
    peak_weight: float = 0.10
    monotonicity_weight: float = 0.05
    smoothness_weight: float = 0.02
    peak_soft_bound: float = 1.08
    monotonic_strain_after: float = 0.78
    train_methods: tuple[str, ...] = ("mlp_softplus", "pinn_energy", "pinn_full")


@dataclass
class _MultiMaterialDataset:
    geom_features: Any  # tensor (N_rows_total, F_geom)
    mat_features: Any  # tensor (N_rows_total, F_mat)
    strain_axis: Any  # tensor (n_points,)
    f_hat: Any  # tensor (N_rows_total, n_points)
    energy_hat: Any  # tensor (N_rows_total,)
    geom_mean: Any
    geom_std: Any
    mat_mean: Any
    mat_std: Any
    presets: list[str] = field(default_factory=list)
    preset_index: Any = None  # tensor of int per row


def _load_preset_rows(preset_dir: Path, rows_per_material: int, seed: int) -> tuple[list[dict[str, str]], list[str]]:
    rng = random.Random(seed)
    train_csv = preset_dir / "train.csv"
    if not train_csv.exists():
        return [], []
    with train_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        curve_fields = sorted([name for name in (reader.fieldnames or []) if name.startswith("curve_")])
        rows: list[dict[str, str]] = []
        seen = 0
        limit = max(0, int(rows_per_material))
        for row in reader:
            if not curve_fields or not all(field in row for field in curve_fields):
                continue
            seen += 1
            if len(rows) < limit:
                rows.append(row)
                continue
            if limit > 0:
                j = rng.randrange(seen)
                if j < limit:
                    rows[j] = row
    return rows, curve_fields


def _build_multi_material_dataset(
    dataset_dir: Path,
    presets: list[str],
    *,
    rows_per_material: int,
    seed: int,
    device: str,
) -> _MultiMaterialDataset:
    import torch

    all_geom: list[list[float]] = []
    all_mat: list[list[float]] = []
    all_curves: list[list[float]] = []
    all_energies: list[float] = []
    preset_idx: list[int] = []
    curve_fields_ref: list[str] | None = None
    used_presets: list[str] = []
    for idx, preset in enumerate(presets):
        preset_dir = Path(dataset_dir) / preset
        rows, curve_fields = _load_preset_rows(preset_dir, rows_per_material, seed + idx * 13)
        if not rows:
            continue
        if curve_fields_ref is None:
            curve_fields_ref = curve_fields
        elif curve_fields != curve_fields_ref:
            raise ValueError(f"curve fields differ for preset {preset}")
        material_path = preset_dir / "material_card.json"
        material = load_material_card(material_path if material_path.exists() else preset)
        scales = scales_for_material(material)
        mat_vec = material_feature_vector(material)
        for row in rows:
            all_geom.append(row_to_features(row))
            all_mat.append(mat_vec)
            curve = [float(row[field]) for field in curve_fields]
            f_hat = [scales.force_to_dimensionless(value) for value in curve]
            all_curves.append(f_hat)
            energy_j = _safe_float(row.get("energy_abs_J"))
            all_energies.append(scales.energy_to_dimensionless(energy_j))
            preset_idx.append(idx)
        used_presets.append(preset)
    if not all_geom or curve_fields_ref is None:
        raise ValueError(f"no training rows found under {dataset_dir} for presets {presets}")
    geom = torch.tensor(all_geom, dtype=torch.float32)
    mat = torch.tensor(all_mat, dtype=torch.float32)
    f_hat = torch.tensor(all_curves, dtype=torch.float32)
    energies = torch.tensor(all_energies, dtype=torch.float32)
    geom_mean = geom.mean(dim=0)
    geom_std = geom.std(dim=0).clamp_min(1e-6)
    mat_mean = mat.mean(dim=0)
    mat_std = mat.std(dim=0).clamp_min(1e-6)
    strain = torch.tensor(displacement_axis(len(curve_fields_ref)), dtype=torch.float32) / 50.0
    geom_norm = (geom - geom_mean) / geom_std
    mat_norm = (mat - mat_mean) / mat_std
    preset_t = torch.tensor(preset_idx, dtype=torch.long)
    return _MultiMaterialDataset(
        geom_features=geom_norm.to(device),
        mat_features=mat_norm.to(device),
        strain_axis=strain.to(device),
        f_hat=f_hat.to(device),
        energy_hat=energies.to(device),
        geom_mean=geom_mean,
        geom_std=geom_std,
        mat_mean=mat_mean,
        mat_std=mat_std,
        presets=used_presets,
        preset_index=preset_t.to(device),
    )


class MaterialPINN:
    """Wrapper around a small fully-connected dimensionless PINN."""

    def __init__(self, geom_dim: int, mat_dim: int, hidden_dim: int, blocks: int):
        import torch
        from torch import nn

        in_dim = geom_dim + mat_dim + 1
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(blocks):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
        self.torch = torch

    def to(self, device: Any) -> "MaterialPINN":
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, x):
        return self.softplus(self.model(x)).squeeze(-1)


def _integrate_strain(f_hat, strain):
    """Trapezoid integral of f_hat over engineering strain (dimensionless)."""

    dx = strain[1:] - strain[:-1]
    f0 = f_hat[:, :-1]
    f1 = f_hat[:, 1:]
    return (0.5 * (f0 + f1) * dx).sum(dim=1)


def train_material_pinn(
    dataset_dir: Path,
    out_dir: Path,
    *,
    presets: list[str],
    method: str = "pinn_full",
    config: MaterialPINNConfig | None = None,
) -> dict[str, object]:
    """Train the dimensionless PINN on a pooled multi-material dataset.

    ``method`` selects the loss family. ``mlp_softplus`` keeps only data and
    boundary; ``pinn_energy`` adds the dimensionless energy residual;
    ``pinn_full`` enables every physics-informed term.
    """

    import torch
    from torch import nn

    config = config or MaterialPINNConfig()
    torch.manual_seed(config.seed)
    out_dir = ensure_dir(out_dir)
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = _build_multi_material_dataset(
        Path(dataset_dir),
        presets,
        rows_per_material=config.rows_per_material,
        seed=config.seed,
        device=str(device),
    )
    geom = data.geom_features
    mat = data.mat_features
    strain = data.strain_axis
    f_hat = data.f_hat
    energy_hat = data.energy_hat
    n_rows = geom.shape[0]
    n_points = strain.shape[0]
    model = MaterialPINN(geom.shape[1], mat.shape[1], hidden_dim=config.hidden_dim, blocks=config.blocks).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse = nn.MSELoss()

    use_energy = method in {"pinn_energy", "pinn_full"}
    use_peak = method == "pinn_full"
    use_monotonic = method == "pinn_full"
    use_smooth = method == "pinn_full"
    monotonic_mask = (strain >= config.monotonic_strain_after).to(torch.float32)

    # Scale gradient steps with the pool size so pooled multi-material runs get
    # the same per-row coverage as a single-material run (see config docstring).
    # Use the number of material cards actually loaded (robust when the dataset
    # has fewer rows than rows_per_material, e.g. train_n < rows_per_material).
    materials_in_pool = max(1, len(data.presets))
    steps_per_epoch = config.steps_per_epoch if config.steps_per_epoch > 0 else materials_in_pool
    total_steps = config.epochs * steps_per_epoch
    print(
        "[material-pinn:{method}] rows={rows} points={pts} materials_in_pool={mats} "
        "steps_per_epoch={spe} total_steps={tot}".format(
            method=method, rows=n_rows, pts=n_points,
            mats=materials_in_pool, spe=steps_per_epoch, tot=total_steps,
        ),
        flush=True,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        for _ in range(steps_per_epoch):
            row_idx = torch.randint(0, n_rows, (config.batch_size,), device=device)
            point_idx = torch.randint(0, n_points, (config.batch_size,), device=device)
            g = geom[row_idx]
            m = mat[row_idx]
            eps = strain[point_idx].unsqueeze(1)
            pred = model(torch.cat([g, m, eps], dim=1))
            target = f_hat[row_idx, point_idx]
            supervised = mse(pred, target)

            zero_pred = model(torch.cat([g[:1024], m[:1024], torch.zeros_like(eps[:1024])], dim=1))
            boundary = (zero_pred ** 2).mean()

            loss = supervised + config.boundary_weight * boundary
            energy_loss = torch.tensor(0.0, device=device)
            peak_loss = torch.tensor(0.0, device=device)
            monotonic_loss = torch.tensor(0.0, device=device)
            smooth_loss = torch.tensor(0.0, device=device)

            if use_energy or use_peak or use_monotonic or use_smooth:
                energy_rows = torch.randint(0, n_rows, (min(256, n_rows),), device=device)
                full_g = geom[energy_rows].repeat_interleave(n_points, dim=0)
                full_m = mat[energy_rows].repeat_interleave(n_points, dim=0)
                full_eps = strain.repeat(len(energy_rows)).unsqueeze(1)
                full_pred = model(torch.cat([full_g, full_m, full_eps], dim=1)).reshape(len(energy_rows), n_points)
                if use_energy:
                    pred_energy = _integrate_strain(full_pred, strain)
                    energy_loss = mse(pred_energy, energy_hat[energy_rows])
                    loss = loss + config.energy_weight * energy_loss
                if use_peak:
                    overshoot = torch.relu(full_pred - config.peak_soft_bound)
                    peak_loss = (overshoot ** 2).mean()
                    loss = loss + config.peak_weight * peak_loss
                if use_monotonic:
                    slope = full_pred[:, 1:] - full_pred[:, :-1]
                    mask = monotonic_mask[1:]
                    negative = torch.relu(-slope) * mask
                    monotonic_loss = (negative ** 2).mean()
                    loss = loss + config.monotonicity_weight * monotonic_loss
                if use_smooth:
                    second = full_pred[:, 2:] - 2.0 * full_pred[:, 1:-1] + full_pred[:, :-2]
                    smooth_loss = (second ** 2).mean()
                    loss = loss + config.smoothness_weight * smooth_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        if epoch == 1 or epoch == config.epochs or epoch % max(1, config.epochs // 10) == 0:
            item = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "supervised": float(supervised.detach().cpu()),
                "boundary": float(boundary.detach().cpu()),
                "energy": float(energy_loss.detach().cpu() if torch.is_tensor(energy_loss) else energy_loss),
                "peak": float(peak_loss.detach().cpu() if torch.is_tensor(peak_loss) else peak_loss),
                "monotonic": float(monotonic_loss.detach().cpu() if torch.is_tensor(monotonic_loss) else monotonic_loss),
                "smoothness": float(smooth_loss.detach().cpu() if torch.is_tensor(smooth_loss) else smooth_loss),
            }
            history.append(item)
            print(
                "[material-pinn:{method}] epoch={epoch:04.0f} loss={loss:.5f} data={supervised:.5f} "
                "energy={energy:.5f} peak={peak:.5f}".format(method=method, **item),
                flush=True,
            )

    checkpoint = out_dir / f"material_pinn_{method}.pt"
    torch.save(
        {
            "model_state": model.model.state_dict(),
            "config": asdict(config),
            "method": method,
            "geom_feature_names": feature_names(),
            "material_feature_names": list(MATERIAL_DIM_FIELDS),
            "geom_mean": data.geom_mean,
            "geom_std": data.geom_std,
            "mat_mean": data.mat_mean,
            "mat_std": data.mat_std,
            "strain_axis": strain.detach().cpu(),
            "presets": data.presets,
        },
        checkpoint,
    )
    metrics = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "checkpoint": str(checkpoint),
        "method": method,
        "presets": data.presets,
        "rows": int(n_rows),
        "materials_in_pool": int(materials_in_pool),
        "steps_per_epoch_effective": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "history": history,
        "config": asdict(config),
    }
    (out_dir / f"metrics_{method}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def predict_force_curves(
    checkpoint_path: Path,
    eval_rows: list[dict[str, str]],
    material: MaterialCard,
    *,
    device: str = "cpu",
) -> list[dict[str, object]]:
    """Predict force[N] curves on an eval pool using a trained PINN.

    Batched GPU/CPU implementation. The previous implementation did one model
    forward per candidate, which is too slow for full SR transfer evaluation.
    """

    import os
    import time
    import torch

    t0 = time.time()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scales = scales_for_material(material)
    mat_features = material_feature_vector(material)

    geom_mean = payload["geom_mean"]
    geom_std = payload["geom_std"]
    mat_mean = payload["mat_mean"]
    mat_std = payload["mat_std"]
    strain_axis = payload["strain_axis"]

    geom_feature_names = list(payload["geom_feature_names"])
    config = payload.get("config", {})

    model = MaterialPINN(
        geom_dim=len(geom_feature_names),
        mat_dim=len(payload["material_feature_names"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
        blocks=int(config.get("blocks", 5)),
    )
    model.model.load_state_dict(payload["model_state"])

    requested = device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"

    target_device = torch.device(requested)
    model.to(target_device)
    model.eval()

    n_rows = len(eval_rows)
    n_points = int(strain_axis.shape[0])
    batch_size = int(os.environ.get("POLMI_PRED_BATCH", "8192"))

    if batch_size < 1:
        batch_size = 8192

    print(
        f"[predict_force_curves] checkpoint={checkpoint_path} rows={n_rows} "
        f"points={n_points} batch={batch_size} device={target_device}",
        flush=True,
    )

    geom_mean = geom_mean.to(target_device).float() if hasattr(geom_mean, "to") else torch.tensor(geom_mean, dtype=torch.float32, device=target_device)
    geom_std = geom_std.to(target_device).float() if hasattr(geom_std, "to") else torch.tensor(geom_std, dtype=torch.float32, device=target_device)
    mat_mean = mat_mean.to(target_device).float() if hasattr(mat_mean, "to") else torch.tensor(mat_mean, dtype=torch.float32, device=target_device)
    mat_std = mat_std.to(target_device).float() if hasattr(mat_std, "to") else torch.tensor(mat_std, dtype=torch.float32, device=target_device)

    strain_axis_d = strain_axis.to(target_device).float()
    strain_col_template = strain_axis_d.view(1, n_points, 1)

    mat_tensor = torch.tensor(mat_features, dtype=torch.float32, device=target_device)
    mat_tensor = (mat_tensor - mat_mean) / mat_std

    displacement = [scales.strain_to_displacement(float(e)) for e in strain_axis.tolist()]

    out: list[dict[str, object]] = []

    with torch.inference_mode():
        for start_i in range(0, n_rows, batch_size):
            end_i = min(start_i + batch_size, n_rows)
            batch_rows = eval_rows[start_i:end_i]
            bsz = len(batch_rows)

            geom = torch.tensor(
                [row_to_features(row) for row in batch_rows],
                dtype=torch.float32,
                device=target_device,
            )
            geom = (geom - geom_mean) / geom_std

            full_g = geom[:, None, :].expand(bsz, n_points, geom.shape[1])
            full_m = mat_tensor.view(1, 1, -1).expand(bsz, n_points, mat_tensor.numel())
            full_eps = strain_col_template.expand(bsz, n_points, 1)

            x = torch.cat([full_g, full_m, full_eps], dim=2).reshape(bsz * n_points, -1)
            pred_hat = model(x).reshape(bsz, n_points).detach().cpu()

            for row, pred_vec in zip(batch_rows, pred_hat):
                force = [scales.dimensionless_to_force(float(value)) for value in pred_vec.tolist()]
                out.append(
                    {
                        "rank": row.get("rank", ""),
                        "topology": row.get("topology", ""),
                        "mass_g": _safe_float(row.get("mass_g")),
                        "oracle_feasible": row.get("oracle_feasible", False),
                        "pred_displacement_mm": displacement,
                        "pred_force_N": force,
                    }
                )

            elapsed = time.time() - t0
            rate = end_i / max(elapsed, 1e-9)
            print(
                f"[predict_force_curves] {end_i}/{n_rows} rows "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} rows/s",
                flush=True,
            )

    print(
        f"[predict_force_curves] done rows={n_rows} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return out

def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default
