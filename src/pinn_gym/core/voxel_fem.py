"""Sparse 3D voxel FEM checks for POLMI candidates.

This is a scalar displacement FEM on a voxelized implicit lattice. It solves a
3D spring/continuum graph for vertical compression stiffness, giving an
independent load-path sanity check from the actual topology rather than only the
parametric features. It is deliberately conservative and cheap enough to run on
many candidates before a full external explicit FEM workflow exists.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .design_space import ENVELOPE_MM, DesignParams
from .paths import ensure_dir
from .physics import PA12_ELASTIC_MODULUS_MPA


@dataclass(frozen=True)
class VoxelFEMConfig:
    resolution: int = 36
    displacement_mm: float = 1.0
    lateral_coupling: float = 0.22
    diagonal_coupling: float = 0.08
    stiffness_scale: float = 0.075
    cg_rtol: float = 1e-7
    cg_maxiter: int = 4000


@dataclass(frozen=True)
class VoxelFEMResult:
    rank: int
    topology: str
    resolution: int
    solid_voxels: int
    voxel_relative_density: float
    connected_unknowns: int
    top_contact_voxels: int
    bottom_contact_voxels: int
    voxel_stiffness_N_per_mm: float
    reaction_force_N: float
    elastic_energy_J: float
    solver_info: int
    mean_strain_zz: float = 0.0
    max_stress_MPa: float = 0.0
    damage_proxy: float = 0.0
    equilibrium_residual: float = 0.0
    contact_residual: float = 0.0
    energy_residual: float = 0.0
    dissipation_proxy: float = 0.0

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def voxel_solid(params: DesignParams, resolution: int) -> np.ndarray:
    import numpy as np

    resolution = int(max(12, min(128, resolution)))
    h = ENVELOPE_MM / resolution
    half = ENVELOPE_MM / 2.0
    axis = -half + h / 2 + np.arange(resolution, dtype=np.float32) * h
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    k = 2.0 * np.pi / max(3.5, params.cell_size_mm)
    gx = k * x * params.anisotropy_xy
    gy = k * y / max(0.75, params.anisotropy_xy)
    gz = k * z
    gyroid = np.sin(gx) * np.cos(gy) + np.sin(gy) * np.cos(gz) + np.sin(gz) * np.cos(gx)
    diamond = (
        np.sin(gx) * np.sin(gy) * np.sin(gz)
        + np.sin(gx) * np.cos(gy) * np.cos(gz)
        + np.cos(gx) * np.sin(gy) * np.cos(gz)
        + np.cos(gx) * np.cos(gy) * np.sin(gz)
    )
    schwarz_p = np.cos(gx) + np.cos(gy) + np.cos(gz)
    bccz = 0.42 * gyroid + 0.35 * diamond + 0.23 * np.sin(gz) * (np.cos(gx) + np.cos(gy))
    if params.topology in {"diamond", "diamond_graded"}:
        field = diamond
    elif params.topology == "schwarz_p":
        field = schwarz_p / 1.5
    elif params.topology == "gyroid_diamond_hybrid":
        z_blend = np.clip(params.hybrid_ratio + 0.25 * (0.5 - ((z + half) / ENVELOPE_MM)), 0.0, 1.0)
        field = (1.0 - z_blend) * gyroid + z_blend * diamond
    elif params.topology in {"bccz_graded", "ot_like"}:
        mix = 0.72 if params.topology == "bccz_graded" else 0.48
        field = mix * bccz + (1.0 - mix) * (0.55 * gyroid + 0.45 * diamond)
    elif params.topology in {"octet", "bcc"}:
        field = 0.55 * gyroid + 0.45 * diamond
    else:
        blend = params.hybrid_ratio if params.topology == "hybrid" else 0.0
        field = (1.0 - blend) * gyroid + blend * diamond
    z01 = (z + half) / ENVELOPE_MM
    threshold = (params.wall_thickness_mm / max(3.5, params.cell_size_mm)) * 1.7
    trigger_start = 1.0 - params.trigger_zone_mm / ENVELOPE_MM
    bumper_end = params.bumper_zone_mm / ENVELOPE_MM
    trigger_softening = 1.0 - 0.30 * params.trigger_layer_strength * np.clip((z01 - trigger_start) / max(1e-6, 1.0 - trigger_start), 0.0, 1.0)
    bumper_boost = 1.0 + 0.24 * params.vertical_gradient * np.clip((bumper_end - z01) / max(1e-6, bumper_end), 0.0, 1.0)
    threshold = threshold * (1.0 + params.vertical_gradient * (0.5 - z01)) * trigger_softening * bumper_boost
    threshold = np.maximum(threshold, params.min_feature_mm / max(3.5, params.cell_size_mm))
    solid = np.abs(field) <= threshold
    solid |= z <= (-half + params.cap_thickness_mm)
    top_cap = max(params.min_feature_mm, params.cap_thickness_mm * (0.45 + 0.35 * (1.0 - params.trigger_layer_strength)))
    solid |= z >= (half - top_cap)
    if params.edge_rib_mm > 0.05:
        rib = params.edge_rib_mm
        rib_top = half - 0.55 * params.trigger_zone_mm
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                cx = sx * (half - rib / 2)
                cy = sy * (half - rib / 2)
                solid |= (np.abs(x - cx) <= rib / 2) & (np.abs(y - cy) <= rib / 2) & (z <= rib_top)
    return solid


def _cg_solve(matrix, rhs, config: VoxelFEMConfig):
    from scipy.sparse.linalg import cg

    try:
        return cg(matrix, rhs, rtol=config.cg_rtol, atol=0.0, maxiter=config.cg_maxiter)
    except TypeError:
        return cg(matrix, rhs, tol=config.cg_rtol, maxiter=config.cg_maxiter)


def _neural_fem_precheck_metrics(
    solid: np.ndarray,
    reaction_force_n: float,
    stiffness_n_per_mm: float,
    config: VoxelFEMConfig,
) -> dict[str, float]:
    import numpy as np

    """Voxel/SDF field proxies for the neural-FEM gate.

    The full neural solver in :mod:`pinn_gym.core.pinn` optimizes these residual families
    with autograd. This cheap precheck computes the same report columns from the
    voxel field so hundreds of candidates can be screened before dynamic impact.
    """

    n = solid.shape[0]
    area_mm2 = ENVELOPE_MM * ENVELOPE_MM
    strain_zz = abs(config.displacement_mm) / ENVELOPE_MM
    top_contact = float(solid[:, :, -1].sum())
    bottom_contact = float(solid[:, :, 0].sum())
    contact_area_fraction = max(top_contact, bottom_contact) / max(1.0, float(n * n))
    effective_area_mm2 = max(1.0, area_mm2 * max(0.02, contact_area_fraction))
    stress_mpa = abs(reaction_force_n) / effective_area_mm2

    z_density = solid.mean(axis=(0, 1)).astype(np.float64)
    density_jump = float(np.abs(np.diff(z_density)).mean()) if len(z_density) > 1 else 0.0
    disconnected_penalty = float((z_density < 1.0 / max(1, n)).mean())
    equilibrium_residual = density_jump + disconnected_penalty
    contact_residual = abs(top_contact - bottom_contact) / max(1.0, top_contact + bottom_contact)
    continuum_energy_j = 0.5 * stiffness_n_per_mm * config.displacement_mm**2 / 1000.0
    stress_energy_j = 0.5 * stress_mpa * strain_zz * area_mm2 * ENVELOPE_MM / 1000.0
    energy_residual = abs(continuum_energy_j - stress_energy_j) / max(1e-6, continuum_energy_j + stress_energy_j)
    damage_proxy = 1.0 / (1.0 + math.exp(-(stress_mpa - 0.55 * PA12_ELASTIC_MODULUS_MPA * strain_zz) / 6.0))
    dissipation_proxy = damage_proxy * strain_zz * max(0.0, stress_mpa - 0.35 * PA12_ELASTIC_MODULUS_MPA * strain_zz)
    return {
        "mean_strain_zz": strain_zz,
        "max_stress_MPa": stress_mpa,
        "damage_proxy": damage_proxy,
        "equilibrium_residual": equilibrium_residual,
        "contact_residual": contact_residual,
        "energy_residual": energy_residual,
        "dissipation_proxy": dissipation_proxy,
    }


def run_scalar_voxel_fem(params: DesignParams, config: VoxelFEMConfig | None = None, rank: int = 0) -> VoxelFEMResult:
    import numpy as np
    from scipy.sparse import coo_matrix

    config = config or VoxelFEMConfig()
    solid = voxel_solid(params, config.resolution)
    n = solid.shape[0]
    h = ENVELOPE_MM / n
    ids = -np.ones_like(solid, dtype=np.int64)
    coords = np.argwhere(solid)
    for idx, (i, j, k) in enumerate(coords):
        ids[i, j, k] = idx
    total = len(coords)
    if total == 0:
        return VoxelFEMResult(rank, params.topology, n, 0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, -1)

    top = coords[:, 2] == n - 1
    bottom = coords[:, 2] == 0
    prescribed = top | bottom
    unknown_global = np.where(~prescribed)[0]
    unknown_map = -np.ones(total, dtype=np.int64)
    unknown_map[unknown_global] = np.arange(len(unknown_global))
    u_prescribed = np.zeros(total, dtype=np.float64)
    u_prescribed[top] = -config.displacement_mm

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(len(unknown_global), dtype=np.float64)
    reaction = 0.0

    base_k = PA12_ELASTIC_MODULUS_MPA * h * config.stiffness_scale
    directions = [
        (1, 0, 0, config.lateral_coupling),
        (0, 1, 0, config.lateral_coupling),
        (0, 0, 1, 1.0),
        (1, 1, 0, config.diagonal_coupling),
        (1, -1, 0, config.diagonal_coupling),
        (1, 0, 1, 0.5),
        (0, 1, 1, 0.5),
    ]

    def add_diag(a: int, value: float) -> None:
        rows.append(a)
        cols.append(a)
        data.append(value)

    def add_offdiag(a: int, b: int, value: float) -> None:
        rows.append(a)
        cols.append(b)
        data.append(value)

    for global_a, (i, j, k) in enumerate(coords):
        for di, dj, dk, factor in directions:
            ni, nj, nk = int(i + di), int(j + dj), int(k + dk)
            if not (0 <= ni < n and 0 <= nj < n and 0 <= nk < n):
                continue
            global_b = int(ids[ni, nj, nk])
            if global_b < 0:
                continue
            length_factor = math.sqrt(di * di + dj * dj + dk * dk)
            spring_k = base_k * factor / max(length_factor, 1e-9)
            ua = unknown_map[global_a]
            ub = unknown_map[global_b]
            if ua >= 0 and ub >= 0:
                add_diag(int(ua), spring_k)
                add_diag(int(ub), spring_k)
                add_offdiag(int(ua), int(ub), -spring_k)
                add_offdiag(int(ub), int(ua), -spring_k)
            elif ua >= 0:
                add_diag(int(ua), spring_k)
                rhs[int(ua)] += spring_k * u_prescribed[global_b]
            elif ub >= 0:
                add_diag(int(ub), spring_k)
                rhs[int(ub)] += spring_k * u_prescribed[global_a]
            if top[global_a] and not top[global_b]:
                reaction += spring_k * (u_prescribed[global_a] - (u_prescribed[global_b] if prescribed[global_b] else 0.0))
            elif top[global_b] and not top[global_a]:
                reaction += spring_k * (u_prescribed[global_b] - (u_prescribed[global_a] if prescribed[global_a] else 0.0))

    if len(unknown_global) and rows:
        matrix = coo_matrix((data, (rows, cols)), shape=(len(unknown_global), len(unknown_global))).tocsr()
        sol, info = _cg_solve(matrix, -rhs, config)
        # Recompute top reaction with solved unknown displacements.
        u_all = u_prescribed.copy()
        u_all[unknown_global] = sol
        reaction = 0.0
        for global_a, (i, j, k) in enumerate(coords):
            for di, dj, dk, factor in directions:
                ni, nj, nk = int(i + di), int(j + dj), int(k + dk)
                if not (0 <= ni < n and 0 <= nj < n and 0 <= nk < n):
                    continue
                global_b = int(ids[ni, nj, nk])
                if global_b < 0 or top[global_a] == top[global_b]:
                    continue
                length_factor = math.sqrt(di * di + dj * dj + dk * dk)
                spring_k = base_k * factor / max(length_factor, 1e-9)
                if top[global_a]:
                    reaction += spring_k * (u_all[global_a] - u_all[global_b])
                else:
                    reaction += spring_k * (u_all[global_b] - u_all[global_a])
    else:
        info = -2

    reaction = abs(float(reaction))
    stiffness = reaction / max(config.displacement_mm, 1e-9)
    energy_j = 0.5 * stiffness * config.displacement_mm * config.displacement_mm / 1000.0
    residuals = _neural_fem_precheck_metrics(solid, reaction, stiffness, config)
    return VoxelFEMResult(
        rank=rank,
        topology=params.topology,
        resolution=n,
        solid_voxels=total,
        voxel_relative_density=total / float(n**3),
        connected_unknowns=len(unknown_global),
        top_contact_voxels=int(top.sum()),
        bottom_contact_voxels=int(bottom.sum()),
        voxel_stiffness_N_per_mm=stiffness,
        reaction_force_N=reaction,
        elastic_energy_J=energy_j,
        solver_info=int(info),
        **residuals,
    )


def run_voxel_fem_gate(top_csv: Path, out_dir: Path, top_n: int = 20, config: VoxelFEMConfig | None = None) -> dict[str, object]:
    config = config or VoxelFEMConfig()
    with Path(top_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:top_n]
    results = []
    for row in rows:
        rank = int(float(row.get("rank", len(results) + 1)))
        results.append(run_scalar_voxel_fem(DesignParams.from_row(row), config=config, rank=rank))
    out_dir = ensure_dir(out_dir)
    out_csv = out_dir / "voxel_fem_candidates.csv"
    fieldnames = list(results[0].to_row().keys()) if results else list(VoxelFEMResult.__dataclass_fields__.keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())
    summary = {
        "top_csv": str(top_csv),
        "out_csv": str(out_csv),
        "evaluated": len(results),
        "config": asdict(config),
        "best_stiffness": max((result.voxel_stiffness_N_per_mm for result in results), default=0.0),
        "best_equilibrium_residual": min((result.equilibrium_residual for result in results), default=0.0),
        "worst_damage_proxy": max((result.damage_proxy for result in results), default=0.0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
