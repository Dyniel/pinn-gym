"""Physics-informed neural network utilities for POLMI crush curves.

The first PINN target is deliberately practical: learn force-displacement
responses while enforcing boundary, energy, and smoothness residuals. The same
network shape can later be reused for vascular stent work by swapping the scalar
compression displacement for radial crimp/expansion coordinates and replacing
the energy residual with hoop/radial equilibrium terms.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .design_space import DesignParams, displacement_axis, feature_names, row_to_features
from .paths import ensure_dir
from .physics import PA12_ELASTIC_MODULUS_MPA, PA12_POISSON


@dataclass(frozen=True)
class PINNConfig:
    epochs: int = 120
    batch_size: int = 8192
    max_rows: int = 50_000
    hidden_dim: int = 256
    blocks: int = 5
    lr: float = 2e-3
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 20260501
    boundary_weight: float = 0.05
    energy_weight: float = 0.15
    smoothness_weight: float = 0.02


@dataclass(frozen=True)
class NeuralFEMConfig:
    epochs: int = 300
    resolution: int = 28
    samples: int = 4096
    hidden_dim: int = 128
    blocks: int = 4
    lr: float = 1e-3
    device: str = "cuda"
    seed: int = 20260514
    compression_mm: float = 8.0
    equilibrium_weight: float = 1.0
    boundary_weight: float = 8.0
    contact_weight: float = 2.0
    energy_weight: float = 0.20
    damage_weight: float = 0.08


class CrushPINN:
    """Small wrapper to avoid importing torch at module import time."""

    def __init__(self, input_dim: int, hidden_dim: int, blocks: int):
        import torch
        from torch import nn

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(blocks):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
        self.torch = torch

    def to(self, device: Any) -> "CrushPINN":
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


class NeuralFEMNet:
    """Coordinate network for displacement and damage fields over voxel/SDF input."""

    def __init__(self, input_dim: int, hidden_dim: int, blocks: int):
        import torch
        from torch import nn

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(blocks):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.model = nn.Sequential(*layers)
        self.torch = torch

    def to(self, device: Any) -> "NeuralFEMNet":
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, x):
        raw = self.model(x)
        u = raw[:, :3]
        damage = self.torch.sigmoid(raw[:, 3:4])
        return u, damage


def _load_rows(candidates_csv: Path, max_rows: int, seed: int) -> tuple[list[dict[str, str]], list[str]]:
    rng = random.Random(seed)
    with Path(candidates_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        curve_fields = sorted([name for name in (reader.fieldnames or []) if name.startswith("curve_")])
        rows: list[dict[str, str]] = []
        seen = 0
        limit = max(0, int(max_rows))
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


def _make_training_tensors(rows: list[dict[str, str]], curve_fields: list[str], device: str):
    import torch

    features = torch.tensor([row_to_features(row) for row in rows], dtype=torch.float32)
    disp = torch.tensor(displacement_axis(len(curve_fields)), dtype=torch.float32) / 50.0
    curves = torch.tensor([[float(row[field]) for field in curve_fields] for row in rows], dtype=torch.float32)
    energies = torch.tensor([float(row["energy_abs_J"]) for row in rows], dtype=torch.float32)
    x_mean = features.mean(dim=0)
    x_std = features.std(dim=0).clamp_min(1e-6)
    flat_curves = curves.reshape(-1)
    if flat_curves.numel() > 1_000_000:
        stride = max(1, flat_curves.numel() // 1_000_000)
        flat_curves = flat_curves[::stride]
    force_scale = torch.quantile(flat_curves, 0.95).clamp_min(1.0)
    features = (features - x_mean) / x_std
    return {
        "features": features.to(device),
        "disp": disp.to(device),
        "curves": (curves / force_scale).to(device),
        "energies": energies.to(device),
        "x_mean": x_mean,
        "x_std": x_std,
        "force_scale": force_scale,
    }


def _batch_inputs(features, disp, row_idx, point_idx):
    f = features[row_idx]
    d = disp[point_idx].unsqueeze(1)
    return f, d, f.new_zeros((f.shape[0], 0))


def _integrate_energy_torch(force_n, disp_norm, force_scale):
    # displacement is normalized to 50 mm; dx_m = dx_norm * 0.05 m.
    dx = (disp_norm[1:] - disp_norm[:-1]) * 0.05
    f0 = force_n[:, :-1] * force_scale
    f1 = force_n[:, 1:] * force_scale
    return (0.5 * (f0 + f1) * dx).sum(dim=1)


def train_crush_pinn(candidates_csv: Path, out_dir: Path, config: PINNConfig | None = None) -> dict[str, object]:
    import torch
    from torch import nn

    config = config or PINNConfig()
    torch.manual_seed(config.seed)
    rows, curve_fields = _load_rows(candidates_csv, config.max_rows, config.seed)
    if not rows:
        raise ValueError(f"no curve rows found in {candidates_csv}")
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = _make_training_tensors(rows, curve_fields, str(device))
    features = data["features"]
    disp = data["disp"]
    curves = data["curves"]
    energies = data["energies"]
    force_scale = data["force_scale"].to(device)
    model = CrushPINN(input_dim=len(feature_names()) + 1, hidden_dim=config.hidden_dim, blocks=config.blocks).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    history: list[dict[str, float]] = []
    n_rows = features.shape[0]
    n_points = disp.shape[0]
    for epoch in range(1, config.epochs + 1):
        model.train()
        row_idx = torch.randint(0, n_rows, (config.batch_size,), device=device)
        point_idx = torch.randint(0, n_points, (config.batch_size,), device=device)
        batch_features = features[row_idx]
        batch_disp = disp[point_idx].unsqueeze(1)
        pred = model(torch.cat([batch_features, batch_disp], dim=1))
        supervised = mse(pred, curves[row_idx, point_idx])

        zero_pred = model(torch.cat([batch_features[:1024], torch.zeros_like(batch_disp[:1024])], dim=1))
        boundary = (zero_pred**2).mean()

        energy_rows = torch.randint(0, n_rows, (min(256, n_rows),), device=device)
        full_features = features[energy_rows].repeat_interleave(n_points, dim=0)
        full_disp = disp.repeat(len(energy_rows)).unsqueeze(1)
        full_pred = model(torch.cat([full_features, full_disp], dim=1)).reshape(len(energy_rows), n_points)
        energy_pred = _integrate_energy_torch(full_pred, disp, force_scale)
        energy = mse(energy_pred / 50.0, energies[energy_rows] / 50.0)
        second = full_pred[:, 2:] - 2.0 * full_pred[:, 1:-1] + full_pred[:, :-2]
        smoothness = (second**2).mean()

        loss = supervised + config.boundary_weight * boundary + config.energy_weight * energy + config.smoothness_weight * smoothness
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if epoch == 1 or epoch == config.epochs or epoch % max(1, config.epochs // 10) == 0:
            item = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "supervised": float(supervised.detach().cpu()),
                "boundary": float(boundary.detach().cpu()),
                "energy": float(energy.detach().cpu()),
                "smoothness": float(smoothness.detach().cpu()),
            }
            history.append(item)
            print(
                "[pinn] epoch={epoch:04.0f} loss={loss:.5f} data={supervised:.5f} energy={energy:.5f}".format(**item),
                flush=True,
            )
    out_dir = ensure_dir(out_dir)
    checkpoint = out_dir / "crush_pinn.pt"
    torch.save(
        {
            "model_state": model.model.state_dict(),
            "config": asdict(config),
            "feature_names": feature_names(),
            "x_mean": data["x_mean"],
            "x_std": data["x_std"],
            "force_scale": data["force_scale"],
            "curve_fields": curve_fields,
        },
        checkpoint,
    )
    metrics = {
        "candidates_csv": str(candidates_csv),
        "out_dir": str(out_dir),
        "checkpoint": str(checkpoint),
        "rows": len(rows),
        "curve_points": len(curve_fields),
        "history": history,
        "config": asdict(config),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _solid_sdf_proxy(solid) -> Any:
    import numpy as np

    try:
        from scipy.ndimage import distance_transform_edt

        inside = distance_transform_edt(solid)
        outside = distance_transform_edt(~solid)
        sdf = inside - outside
        scale = max(1.0, float(np.max(np.abs(sdf))))
        return (sdf / scale).astype(np.float32)
    except Exception:
        return (solid.astype(np.float32) * 2.0 - 1.0).astype(np.float32)


def _sample_voxel_points(params: DesignParams, config: NeuralFEMConfig):
    import numpy as np

    from .voxel_fem import voxel_solid

    solid = voxel_solid(params, config.resolution)
    sdf = _solid_sdf_proxy(solid)
    coords_idx = np.argwhere(solid)
    if len(coords_idx) == 0:
        raise ValueError("voxelized design has no solid voxels")
    rng = np.random.default_rng(config.seed)
    take = min(config.samples, len(coords_idx))
    chosen = coords_idx[rng.choice(len(coords_idx), size=take, replace=len(coords_idx) < take)]
    denom = max(1, config.resolution - 1)
    coords = (chosen.astype(np.float32) / denom) * 2.0 - 1.0
    sdf_values = sdf[chosen[:, 0], chosen[:, 1], chosen[:, 2]].reshape(-1, 1).astype(np.float32)
    return coords, sdf_values


def _linear_elastic_stress(strain, damage):
    import torch

    e = PA12_ELASTIC_MODULUS_MPA
    nu = PA12_POISSON
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    trace = strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2]
    eye = torch.eye(3, device=strain.device, dtype=strain.dtype).unsqueeze(0)
    degradation = (1.0 - damage).clamp_min(0.04).view(-1, 1, 1) ** 2
    return degradation * (lam * trace.view(-1, 1, 1) * eye + 2.0 * mu * strain)


def train_neural_fem_quasistatic(
    row: dict[str, str | float],
    out_dir: Path,
    config: NeuralFEMConfig | None = None,
) -> dict[str, object]:
    """Train a small physics-informed neural FEM field model for one candidate.

    Inputs are normalized voxel coordinates plus an SDF proxy and imposed
    compression. Outputs are displacement `u=(ux,uy,uz)` and scalar damage.
    Loss terms cover quasi-static equilibrium, fixed base, rigid top indenter
    displacement/contact, elastic energy, and damage/dissipation regularity.
    """

    import torch

    config = config or NeuralFEMConfig()
    torch.manual_seed(config.seed)
    params = DesignParams.from_row(row)
    coords_np, sdf_np = _sample_voxel_points(params, config)
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    coords_base = torch.tensor(coords_np, dtype=torch.float32, device=device)
    sdf = torch.tensor(sdf_np, dtype=torch.float32, device=device)
    compression_norm = torch.full((coords_base.shape[0], 1), config.compression_mm / 50.0, dtype=torch.float32, device=device)
    net = NeuralFEMNet(input_dim=5, hidden_dim=config.hidden_dim, blocks=config.blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=1e-6)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        net.train()
        coords = coords_base.detach().clone().requires_grad_(True)
        x = torch.cat([coords, sdf, compression_norm], dim=1)
        u, damage = net(x)
        grads = []
        for comp in range(3):
            grad = torch.autograd.grad(u[:, comp].sum(), coords, create_graph=True, retain_graph=True)[0]
            grads.append(grad)
        grad_u = torch.stack(grads, dim=1)
        strain = 0.5 * (grad_u + grad_u.transpose(1, 2))
        stress = _linear_elastic_stress(strain, damage)

        div_terms = []
        for i in range(3):
            div_i = 0.0
            for j in range(3):
                grad_sigma = torch.autograd.grad(stress[:, i, j].sum(), coords, create_graph=True, retain_graph=True)[0][:, j]
                div_i = div_i + grad_sigma
            div_terms.append(div_i)
        equilibrium = torch.stack(div_terms, dim=1).pow(2).mean()

        bottom = coords[:, 2] < -0.86
        top = coords[:, 2] > 0.86
        if bottom.any():
            bottom_loss = u[bottom].pow(2).mean()
        else:
            bottom_loss = u.pow(2).mean() * 0.0
        if top.any():
            target_uz = -config.compression_mm / 50.0
            top_loss = (u[top, 2] - target_uz).pow(2).mean() + 0.15 * u[top, :2].pow(2).mean()
            contact = torch.relu(u[top, 2] - target_uz).pow(2).mean()
        else:
            top_loss = u.pow(2).mean() * 0.0
            contact = top_loss
        boundary = bottom_loss + top_loss

        strain_energy = 0.5 * (stress * strain).sum(dim=(1, 2)).clamp_min(0.0).mean()
        damage_drive = torch.linalg.norm(strain.reshape(strain.shape[0], -1), dim=1, keepdim=True)
        dissipation = (damage * damage_drive).mean()
        energy = torch.relu(dissipation - strain_energy / max(PA12_ELASTIC_MODULUS_MPA, 1e-9)).pow(2)
        damage_reg = damage.mean() + torch.relu(damage - 0.92).pow(2).mean()

        loss = (
            config.equilibrium_weight * equilibrium
            + config.boundary_weight * boundary
            + config.contact_weight * contact
            + config.energy_weight * energy
            + config.damage_weight * damage_reg
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.model.parameters(), 10.0)
        opt.step()
        if epoch == 1 or epoch == config.epochs or epoch % max(1, config.epochs // 10) == 0:
            item = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "equilibrium": float(equilibrium.detach().cpu()),
                "boundary": float(boundary.detach().cpu()),
                "contact": float(contact.detach().cpu()),
                "energy": float(energy.detach().cpu()),
                "damage_mean": float(damage.mean().detach().cpu()),
                "strain_energy_proxy": float(strain_energy.detach().cpu()),
            }
            history.append(item)
            print(
                "[neural-fem] epoch={epoch:04.0f} loss={loss:.5f} eq={equilibrium:.5f} bc={boundary:.5f}".format(**item),
                flush=True,
            )

    out_dir = ensure_dir(out_dir)
    checkpoint = out_dir / "neural_fem_quasistatic.pt"
    torch.save(
        {
            "model_state": net.model.state_dict(),
            "config": asdict(config),
            "design": params.to_row(),
            "input_fields": ["x", "y", "z", "sdf_proxy", "compression_norm"],
            "outputs": ["ux", "uy", "uz", "damage"],
        },
        checkpoint,
    )
    metrics = {
        "out_dir": str(out_dir),
        "checkpoint": str(checkpoint),
        "topology": params.topology,
        "samples": int(coords_base.shape[0]),
        "config": asdict(config),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
