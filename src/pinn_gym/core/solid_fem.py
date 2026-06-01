"""Vector 3D voxel solid-FEM precheck for POLMI candidates.

This module is the next fidelity step after :mod:`pinn_gym.core.voxel_fem`: each solid
voxel carries a 3D displacement vector and neighboring voxels are coupled by
axial vector springs. The model is still an in-house voxel-FEM approximation,
not a certified large-deformation explicit solver, but it resolves full
displacement, strain, stress, reaction force, strain energy, and a conservative
damage proxy over the actual TPMS/SDF geometry.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .design_space import ENVELOPE_MM, PA12_DENSITY_G_PER_MM3, DesignParams
from .paths import ensure_dir
from .physics import (
    GRAVITY_M_S2,
    IMPACT_MASS_KG,
    IMPACT_VELOCITY_M_S,
    PA12_ELASTIC_MODULUS_MPA,
    PA12_POISSON,
    PA12_YIELD_STRESS_MPA,
)
from .voxel_fem import voxel_solid


@dataclass(frozen=True)
class VectorFEMConfig:
    resolution: int = 24
    displacement_mm: float = 4.0
    stiffness_scale: float = 0.055
    diagonal_coupling: float = 0.45
    body_diagonal_coupling: float = 0.22
    stabilization: float = 1e-7
    cg_rtol: float = 1e-7
    cg_maxiter: int = 8000
    damage_softening_mpa: float = 10.0


@dataclass(frozen=True)
class VectorFEMResult:
    rank: int
    topology: str
    resolution: int
    solid_voxels: int
    vector_dofs: int
    unknown_dofs: int
    top_contact_voxels: int
    bottom_contact_voxels: int
    reaction_force_N: float
    stiffness_N_per_mm: float
    strain_energy_J: float
    mean_von_mises_MPa: float
    max_von_mises_MPa: float
    p95_von_mises_MPa: float
    mean_strain_energy_density: float
    max_principal_strain: float
    damage_proxy: float
    equilibrium_residual: float
    contact_residual: float
    solver_info: int

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExplicitImpactFEMConfig:
    resolution: int = 18
    dt_s: float = 2.0e-6
    max_time_s: float = 0.010
    stiffness_scale: float = 0.040
    diagonal_coupling: float = 0.40
    body_diagonal_coupling: float = 0.18
    stabilization: float = 1e-7
    mass_scale: float = 4.0
    nodal_damping_s: float = 120.0
    contact_penalty_N_per_mm: float = 150.0
    contact_damping_N_s_per_mm: float = 0.010
    indenter_mass_kg: float = IMPACT_MASS_KG
    initial_velocity_m_s: float = IMPACT_VELOCITY_M_S
    initial_clearance_mm: float = 0.05
    stop_velocity_m_s: float = 0.03
    damage_softening_mpa: float = 10.0
    failure_damage: float = 0.88
    enable_plasticity: bool = True
    yield_strain: float = PA12_YIELD_STRESS_MPA / PA12_ELASTIC_MODULUS_MPA
    hardening_ratio: float = 0.08
    fracture_plastic_strain: float = 0.16
    damage_growth_strain: float = 0.10
    failed_spring_damage: float = 0.92
    enable_self_contact: bool = True
    self_contact_radius_fraction: float = 0.82
    self_contact_exclusion_fraction: float = 1.05
    self_contact_penalty_N_per_mm: float = 55.0
    self_contact_damping_N_s_per_mm: float = 0.004
    self_contact_stride: int = 5


@dataclass(frozen=True)
class ExplicitImpactFEMResult:
    rank: int
    topology: str
    resolution: int
    solid_voxels: int
    vector_dofs: int
    simulated_time_ms: float
    impact_initial_ke_J: float
    impact_absorbed_J: float
    peak_contact_force_N: float
    max_indenter_displacement_mm: float
    residual_velocity_m_s: float
    nodal_kinetic_energy_J: float
    strain_energy_J: float
    plastic_dissipation_J: float
    energy_balance_error_J: float
    peak_self_contact_force_N: float
    max_self_contact_pairs: int
    max_plastic_strain: float
    failed_springs_fraction: float
    p95_von_mises_MPa: float
    max_von_mises_MPa: float
    max_principal_strain: float
    damage_proxy: float
    survives_explicit_fem: bool
    explicit_fem_score: float

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def _cg_solve(matrix, rhs, config: VectorFEMConfig):
    from scipy.sparse.linalg import cg

    try:
        return cg(matrix, rhs, rtol=config.cg_rtol, atol=0.0, maxiter=config.cg_maxiter)
    except TypeError:
        return cg(matrix, rhs, tol=config.cg_rtol, maxiter=config.cg_maxiter)


def _neighbor_directions() -> list[tuple[int, int, int, float]]:
    out: list[tuple[int, int, int, float]] = []
    for di in (0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj < 0:
                    continue
                if di == 0 and dj == 0 and dk <= 0:
                    continue
                if di == dj == dk == 0:
                    continue
                nonzero = sum(1 for item in (di, dj, dk) if item != 0)
                if nonzero == 1:
                    factor = 1.0
                elif nonzero == 2:
                    factor = 0.45
                else:
                    factor = 0.22
                out.append((di, dj, dk, factor))
    return out


def _assemble_vector_stiffness(solid, config: VectorFEMConfig):
    import numpy as np
    from scipy.sparse import coo_matrix

    n = int(solid.shape[0])
    h = ENVELOPE_MM / n
    coords = np.argwhere(solid)
    total = len(coords)
    ids = -np.ones_like(solid, dtype=np.int64)
    for idx, (i, j, k) in enumerate(coords):
        ids[i, j, k] = idx

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    base_k = PA12_ELASTIC_MODULUS_MPA * h * config.stiffness_scale

    def add(a: int, b: int, value: float) -> None:
        rows.append(a)
        cols.append(b)
        data.append(value)

    for global_a, (i, j, k) in enumerate(coords):
        for di, dj, dk, default_factor in _neighbor_directions():
            ni, nj, nk = int(i + di), int(j + dj), int(k + dk)
            if not (0 <= ni < n and 0 <= nj < n and 0 <= nk < n):
                continue
            global_b = int(ids[ni, nj, nk])
            if global_b < 0:
                continue
            direction = np.array([di, dj, dk], dtype=np.float64)
            length = float(np.linalg.norm(direction))
            unit = direction / max(length, 1e-12)
            nonzero = int(np.count_nonzero(direction))
            if nonzero == 2:
                factor = config.diagonal_coupling
            elif nonzero == 3:
                factor = config.body_diagonal_coupling
            else:
                factor = default_factor
            local = base_k * factor * np.outer(unit, unit) / max(length, 1e-12)
            for p in range(3):
                for q in range(3):
                    value = float(local[p, q])
                    a_p = 3 * global_a + p
                    a_q = 3 * global_a + q
                    b_p = 3 * global_b + p
                    b_q = 3 * global_b + q
                    add(a_p, a_q, value)
                    add(b_p, b_q, value)
                    add(a_p, b_q, -value)
                    add(b_p, a_q, -value)

    dofs = 3 * total
    if dofs and config.stabilization > 0.0:
        diag = max(base_k, 1.0) * config.stabilization
        for dof in range(dofs):
            add(dof, dof, diag)
    return coo_matrix((data, (rows, cols)), shape=(dofs, dofs)).tocsr(), coords, ids


def _build_explicit_springs(solid, coords, ids, config: ExplicitImpactFEMConfig):
    import numpy as np

    n = int(solid.shape[0])
    h = ENVELOPE_MM / n
    base_k = PA12_ELASTIC_MODULUS_MPA * h * config.stiffness_scale
    a_idx: list[int] = []
    b_idx: list[int] = []
    units: list[list[float]] = []
    rest_lengths: list[float] = []
    stiffness: list[float] = []
    bonded_pairs: set[tuple[int, int]] = set()
    for global_a, (i, j, k) in enumerate(coords):
        for di, dj, dk, default_factor in _neighbor_directions():
            ni, nj, nk = int(i + di), int(j + dj), int(k + dk)
            if not (0 <= ni < n and 0 <= nj < n and 0 <= nk < n):
                continue
            global_b = int(ids[ni, nj, nk])
            if global_b < 0:
                continue
            direction = np.array([di, dj, dk], dtype=np.float64)
            length_factor = float(np.linalg.norm(direction))
            unit = direction / max(length_factor, 1e-12)
            nonzero = int(np.count_nonzero(direction))
            if nonzero == 2:
                factor = config.diagonal_coupling
            elif nonzero == 3:
                factor = config.body_diagonal_coupling
            else:
                factor = default_factor
            a_idx.append(global_a)
            b_idx.append(global_b)
            units.append(unit.tolist())
            rest_lengths.append(length_factor * h)
            stiffness.append(base_k * factor / max(length_factor, 1e-12))
            bonded_pairs.add((min(global_a, global_b), max(global_a, global_b)))
    return {
        "a": np.asarray(a_idx, dtype=np.int64),
        "b": np.asarray(b_idx, dtype=np.int64),
        "unit": np.asarray(units, dtype=np.float64),
        "rest": np.asarray(rest_lengths, dtype=np.float64),
        "k": np.asarray(stiffness, dtype=np.float64),
        "bonded_pairs": bonded_pairs,
    }


def _explicit_internal_forces(u_vox, springs, plastic_strain, damage, config: ExplicitImpactFEMConfig, update_state: bool = True):
    import numpy as np

    internal = np.zeros_like(u_vox)
    a = springs["a"]
    b = springs["b"]
    unit = springs["unit"]
    rest = springs["rest"]
    k_spring = springs["k"]
    extension = np.einsum("ij,ij->i", u_vox[b] - u_vox[a], unit)
    elastic_extension = extension - plastic_strain * rest
    trial_force = k_spring * elastic_extension
    plastic_dissipation_j = 0.0
    if update_state and config.enable_plasticity and len(trial_force):
        yield_extension = config.yield_strain * rest * (1.0 + config.hardening_ratio * np.abs(plastic_strain))
        yield_force = k_spring * yield_extension
        over = np.abs(trial_force) > yield_force
        if np.any(over):
            denom = k_spring[over] * rest[over] * (1.0 + config.hardening_ratio)
            delta_plastic = (np.abs(trial_force[over]) - yield_force[over]) / np.maximum(denom, 1e-12)
            plastic_strain[over] += np.sign(trial_force[over]) * delta_plastic
            elastic_extension[over] = np.sign(trial_force[over]) * yield_extension[over]
            trial_force[over] = k_spring[over] * elastic_extension[over]
            plastic_dissipation_j = float(np.sum(yield_force[over] * delta_plastic * rest[over]) / 1000.0)
        plastic_abs = np.abs(plastic_strain)
        damage_drive = np.maximum(0.0, plastic_abs - config.fracture_plastic_strain) / max(config.damage_growth_strain, 1e-12)
        damage[:] = np.maximum(damage, 1.0 - np.exp(-damage_drive))
    degraded_force = trial_force * np.maximum(0.03, 1.0 - damage) ** 2
    force_vec = degraded_force[:, None] * unit
    np.add.at(internal, a, -force_vec)
    np.add.at(internal, b, force_vec)
    strain_energy_j = float(0.5 * np.sum(np.maximum(0.0, 1.0 - damage) ** 2 * k_spring * elastic_extension**2) / 1000.0)
    return internal, strain_energy_j, plastic_dissipation_j


def _self_contact_forces(base_pos, u_vox, v_vox, springs, config: ExplicitImpactFEMConfig, h: float):
    import numpy as np
    from scipy.spatial import cKDTree

    external = np.zeros_like(u_vox)
    if not config.enable_self_contact or len(base_pos) < 2:
        return external, 0.0, 0
    radius = max(1e-6, config.self_contact_radius_fraction * h)
    rest_exclusion = max(radius, config.self_contact_exclusion_fraction * h)
    pos = base_pos + u_vox
    tree = cKDTree(pos)
    pairs = tree.query_pairs(radius)
    if not pairs:
        return external, 0.0, 0
    bonded = springs["bonded_pairs"]
    total_force = 0.0
    active_pairs = 0
    for a, b in pairs:
        key = (a, b) if a < b else (b, a)
        if key in bonded:
            continue
        ref_dist = float(np.linalg.norm(base_pos[b] - base_pos[a]))
        if ref_dist <= rest_exclusion:
            continue
        delta = pos[b] - pos[a]
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-12 or dist >= radius:
            continue
        normal = delta / dist
        rel_v = float(np.dot(v_vox[b] - v_vox[a], normal))
        magnitude = config.self_contact_penalty_N_per_mm * (radius - dist) + config.self_contact_damping_N_s_per_mm * max(0.0, -rel_v)
        force = magnitude * normal
        external[a] -= force
        external[b] += force
        total_force += abs(magnitude)
        active_pairs += 1
    return external, total_force, active_pairs


def _vector_config_from_explicit(config: ExplicitImpactFEMConfig) -> VectorFEMConfig:
    return VectorFEMConfig(
        resolution=config.resolution,
        displacement_mm=4.0,
        stiffness_scale=config.stiffness_scale,
        diagonal_coupling=config.diagonal_coupling,
        body_diagonal_coupling=config.body_diagonal_coupling,
        stabilization=config.stabilization,
        damage_softening_mpa=config.damage_softening_mpa,
    )


def _prescribed_dofs(coords, resolution: int, displacement_mm: float):
    import numpy as np

    total = len(coords)
    prescribed = np.zeros(3 * total, dtype=bool)
    values = np.zeros(3 * total, dtype=np.float64)
    bottom = coords[:, 2] == 0
    top = coords[:, 2] == resolution - 1
    for global_idx, is_bottom in enumerate(bottom):
        if is_bottom:
            prescribed[3 * global_idx : 3 * global_idx + 3] = True
    for global_idx, is_top in enumerate(top):
        if is_top:
            prescribed[3 * global_idx + 2] = True
            values[3 * global_idx + 2] = -abs(displacement_mm)
    return prescribed, values, top, bottom


def _strain_stress_metrics(solid, coords, ids, u_vox, config: VectorFEMConfig) -> dict[str, float]:
    import numpy as np

    n = int(solid.shape[0])
    h = ENVELOPE_MM / n
    e = PA12_ELASTIC_MODULUS_MPA
    nu = PA12_POISSON
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    von: list[float] = []
    energy_density: list[float] = []
    principal_abs: list[float] = []

    def displacement_at(i: int, j: int, k: int, fallback: int):
        idx = int(ids[i, j, k])
        if idx >= 0:
            return u_vox[idx]
        return u_vox[fallback]

    for global_idx, (i, j, k) in enumerate(coords):
        grad = np.zeros((3, 3), dtype=np.float64)
        for axis, (di, dj, dk) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            minus = (int(i - di), int(j - dj), int(k - dk))
            plus = (int(i + di), int(j + dj), int(k + dk))
            has_minus = (
                0 <= minus[0] < n
                and 0 <= minus[1] < n
                and 0 <= minus[2] < n
                and ids[minus[0], minus[1], minus[2]] >= 0
            )
            has_plus = (
                0 <= plus[0] < n
                and 0 <= plus[1] < n
                and 0 <= plus[2] < n
                and ids[plus[0], plus[1], plus[2]] >= 0
            )
            if has_minus and has_plus:
                up = displacement_at(*plus, fallback=global_idx)
                um = displacement_at(*minus, fallback=global_idx)
                grad[:, axis] = (up - um) / (2.0 * h)
            elif has_plus:
                up = displacement_at(*plus, fallback=global_idx)
                grad[:, axis] = (up - u_vox[global_idx]) / h
            elif has_minus:
                um = displacement_at(*minus, fallback=global_idx)
                grad[:, axis] = (u_vox[global_idx] - um) / h
        strain = 0.5 * (grad + grad.T)
        trace = float(np.trace(strain))
        stress = lam * trace * np.eye(3) + 2.0 * mu * strain
        dev = stress - np.eye(3) * float(np.trace(stress)) / 3.0
        vm = math.sqrt(max(0.0, 1.5 * float((dev * dev).sum())))
        von.append(vm)
        energy_density.append(max(0.0, 0.5 * float((stress * strain).sum())))
        try:
            eig = np.linalg.eigvalsh(strain)
            principal_abs.append(float(np.max(np.abs(eig))))
        except Exception:
            principal_abs.append(float(np.linalg.norm(strain)))

    if not von:
        return {
            "mean_von_mises_MPa": 0.0,
            "max_von_mises_MPa": 0.0,
            "p95_von_mises_MPa": 0.0,
            "mean_strain_energy_density": 0.0,
            "max_principal_strain": 0.0,
            "damage_proxy": 1.0,
        }
    von_arr = np.asarray(von, dtype=np.float64)
    damage = 1.0 / (1.0 + math.exp(-(float(np.percentile(von_arr, 95)) - PA12_YIELD_STRESS_MPA) / config.damage_softening_mpa))
    return {
        "mean_von_mises_MPa": float(von_arr.mean()),
        "max_von_mises_MPa": float(von_arr.max()),
        "p95_von_mises_MPa": float(np.percentile(von_arr, 95)),
        "mean_strain_energy_density": float(np.mean(energy_density)),
        "max_principal_strain": float(max(principal_abs, default=0.0)),
        "damage_proxy": damage,
    }


def run_vector_voxel_fem(params: DesignParams, config: VectorFEMConfig | None = None, rank: int = 0) -> VectorFEMResult:
    import numpy as np

    config = config or VectorFEMConfig()
    solid = voxel_solid(params, config.resolution)
    resolution = int(solid.shape[0])
    matrix, coords, ids = _assemble_vector_stiffness(solid, config)
    total = len(coords)
    if total == 0:
        return VectorFEMResult(rank, params.topology, resolution, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1)

    prescribed, values, top, bottom = _prescribed_dofs(coords, resolution, config.displacement_mm)
    unknown = ~prescribed
    if not np.any(top) or not np.any(bottom):
        return VectorFEMResult(
            rank,
            params.topology,
            resolution,
            total,
            3 * total,
            int(unknown.sum()),
            int(top.sum()),
            int(bottom.sum()),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            -3,
        )

    u = values.copy()
    if int(unknown.sum()) > 0:
        k_uu = matrix[unknown][:, unknown]
        rhs = -matrix[unknown][:, prescribed] @ values[prescribed]
        sol, info = _cg_solve(k_uu, rhs, config)
        u[unknown] = sol
    else:
        info = -2

    residual = matrix @ u
    top_z_dofs = np.array([3 * idx + 2 for idx, is_top in enumerate(top) if is_top], dtype=np.int64)
    reaction = abs(float(residual[top_z_dofs].sum())) if len(top_z_dofs) else 0.0
    stiffness = reaction / max(abs(config.displacement_mm), 1e-9)
    energy_j = max(0.0, 0.5 * float(u @ (matrix @ u)) / 1000.0)
    unknown_residual = float(np.linalg.norm(residual[unknown])) / max(1.0, abs(reaction))
    top_uz = np.array([u[3 * idx + 2] for idx, is_top in enumerate(top) if is_top], dtype=np.float64)
    contact_residual = float(np.std(top_uz)) / max(1e-9, abs(config.displacement_mm)) if len(top_uz) else 1.0
    metrics = _strain_stress_metrics(solid, coords, ids, u.reshape(total, 3), config)

    return VectorFEMResult(
        rank=rank,
        topology=params.topology,
        resolution=resolution,
        solid_voxels=total,
        vector_dofs=3 * total,
        unknown_dofs=int(unknown.sum()),
        top_contact_voxels=int(top.sum()),
        bottom_contact_voxels=int(bottom.sum()),
        reaction_force_N=reaction,
        stiffness_N_per_mm=stiffness,
        strain_energy_J=energy_j,
        equilibrium_residual=unknown_residual,
        contact_residual=contact_residual,
        solver_info=int(info),
        **metrics,
    )


def run_vector_fem_gate(top_csv: Path, out_dir: Path, top_n: int = 50, config: VectorFEMConfig | None = None) -> dict[str, object]:
    config = config or VectorFEMConfig()
    with Path(top_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:top_n]
    results: list[VectorFEMResult] = []
    for row in rows:
        rank = int(float(row.get("rank", len(results) + 1)))
        results.append(run_vector_voxel_fem(DesignParams.from_row(row), config=config, rank=rank))

    out_dir = ensure_dir(out_dir)
    out_csv = out_dir / "vector_fem_candidates.csv"
    fieldnames = list(results[0].to_row().keys()) if results else list(VectorFEMResult.__dataclass_fields__.keys())
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
        "best_stiffness": max((item.stiffness_N_per_mm for item in results), default=0.0),
        "worst_damage_proxy": max((item.damage_proxy for item in results), default=0.0),
        "max_von_mises_MPa": max((item.max_von_mises_MPa for item in results), default=0.0),
        "max_equilibrium_residual": max((item.equilibrium_residual for item in results), default=0.0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# POLMI Vector Voxel FEM",
        "",
        f"Evaluated: `{len(results)}`",
        f"Input: `{top_csv}`",
        "",
        "| Rank | Topology | Voxels | Reaction N | Stiffness N/mm | p95 VM MPa | Damage | Eq residual |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results[:20]:
        lines.append(
            f"| {item.rank} | {item.topology} | {item.solid_voxels} | {item.reaction_force_N:.1f} | "
            f"{item.stiffness_N_per_mm:.1f} | {item.p95_von_mises_MPa:.2f} | {item.damage_proxy:.3f} | "
            f"{item.equilibrium_residual:.3e} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_explicit_vector_impact(
    params: DesignParams,
    config: ExplicitImpactFEMConfig | None = None,
    rank: int = 0,
) -> tuple[ExplicitImpactFEMResult, dict[str, object]]:
    import numpy as np

    config = config or ExplicitImpactFEMConfig()
    vector_config = _vector_config_from_explicit(config)
    solid = voxel_solid(params, config.resolution)
    resolution = int(solid.shape[0])
    matrix, coords, ids = _assemble_vector_stiffness(solid, vector_config)
    total = len(coords)
    if total == 0:
        result = ExplicitImpactFEMResult(
            rank=rank,
            topology=params.topology,
            resolution=resolution,
            solid_voxels=0,
            vector_dofs=0,
            simulated_time_ms=0.0,
            impact_initial_ke_J=0.0,
            impact_absorbed_J=0.0,
            peak_contact_force_N=0.0,
            max_indenter_displacement_mm=0.0,
            residual_velocity_m_s=0.0,
            nodal_kinetic_energy_J=0.0,
            strain_energy_J=0.0,
            plastic_dissipation_J=0.0,
            energy_balance_error_J=0.0,
            peak_self_contact_force_N=0.0,
            max_self_contact_pairs=0,
            max_plastic_strain=0.0,
            failed_springs_fraction=0.0,
            p95_von_mises_MPa=0.0,
            max_von_mises_MPa=0.0,
            max_principal_strain=0.0,
            damage_proxy=1.0,
            survives_explicit_fem=False,
            explicit_fem_score=1e6,
        )
        return result, {"history": []}

    h = ENVELOPE_MM / resolution
    dofs = 3 * total
    springs = _build_explicit_springs(solid, coords, ids, config)
    plastic_strain = np.zeros(len(springs["a"]), dtype=np.float64)
    spring_damage = np.zeros(len(springs["a"]), dtype=np.float64)
    base_pos = (coords.astype(np.float64) + 0.5) * h
    top = coords[:, 2] == resolution - 1
    bottom = coords[:, 2] == 0
    fixed = np.zeros(dofs, dtype=bool)
    for global_idx, is_bottom in enumerate(bottom):
        if is_bottom:
            fixed[3 * global_idx : 3 * global_idx + 3] = True
    free = ~fixed
    top_indices = np.array([idx for idx, is_top in enumerate(top) if is_top], dtype=np.int64)
    top_z_dofs = 3 * top_indices + 2

    density_kg_per_mm3 = PA12_DENSITY_G_PER_MM3 * 1e-3
    voxel_mass_kg = max(1e-12, density_kg_per_mm3 * h**3 * config.mass_scale)
    mass = np.full(dofs, voxel_mass_kg, dtype=np.float64)

    u = np.zeros(dofs, dtype=np.float64)
    v = np.zeros(dofs, dtype=np.float64)
    indenter_z = ENVELOPE_MM + config.initial_clearance_mm
    indenter_v = -abs(config.initial_velocity_m_s) * 1000.0
    initial_ke = 0.5 * config.indenter_mass_kg * (abs(config.initial_velocity_m_s) ** 2)
    peak_contact = 0.0
    peak_self_contact = 0.0
    max_self_contact_pairs = 0
    contact_work_j = 0.0
    plastic_dissipation_j = 0.0
    max_indenter_disp = 0.0
    history: list[dict[str, float]] = []
    max_steps = max(1, int(config.max_time_s / max(config.dt_s, 1e-12)))
    history_stride = max(1, max_steps // 240)
    failed = False
    stopped = False

    for step in range(max_steps):
        u_vox = u.reshape(total, 3)
        v_vox = v.reshape(total, 3)
        internal_vox, strain_energy_j, plastic_step_j = _explicit_internal_forces(
            u_vox, springs, plastic_strain, spring_damage, config
        )
        plastic_dissipation_j += plastic_step_j
        internal = internal_vox.reshape(dofs)
        external_vox = np.zeros((total, 3), dtype=np.float64)
        contact_force_up = 0.0
        if len(top_indices):
            surface_z = (coords[top_indices, 2].astype(np.float64) + 1.0) * h + u_vox[top_indices, 2]
            node_vz = v_vox[top_indices, 2]
            penetration = np.maximum(0.0, surface_z - indenter_z)
            relative_v = np.maximum(0.0, node_vz - indenter_v)
            contact_node_force = -(config.contact_penalty_N_per_mm * penetration + config.contact_damping_N_s_per_mm * relative_v)
            external_vox[top_indices, 2] += contact_node_force
            contact_force_up = -float(contact_node_force.sum())
            peak_contact = max(peak_contact, contact_force_up)

        if config.enable_self_contact and step % max(1, config.self_contact_stride) == 0:
            self_contact, self_force, self_pairs = _self_contact_forces(base_pos, u_vox, v_vox, springs, config, h)
            external_vox += self_contact
            peak_self_contact = max(peak_self_contact, self_force)
            max_self_contact_pairs = max(max_self_contact_pairs, self_pairs)

        external = external_vox.reshape(dofs)

        damping = config.nodal_damping_s * mass * v
        acceleration = np.zeros(dofs, dtype=np.float64)
        acceleration[free] = 1000.0 * (external[free] - internal[free] - damping[free]) / mass[free]
        v[free] += acceleration[free] * config.dt_s
        u[free] += v[free] * config.dt_s
        u[fixed] = 0.0
        v[fixed] = 0.0

        indenter_a = -GRAVITY_M_S2 * 1000.0 + 1000.0 * contact_force_up / max(config.indenter_mass_kg, 1e-9)
        previous_z = indenter_z
        indenter_v += indenter_a * config.dt_s
        indenter_z += indenter_v * config.dt_s
        indenter_drop = max(0.0, ENVELOPE_MM + config.initial_clearance_mm - indenter_z)
        max_indenter_disp = max(max_indenter_disp, indenter_drop)
        contact_work_j += max(0.0, contact_force_up) * max(0.0, previous_z - indenter_z) / 1000.0

        if step % history_stride == 0 or step == max_steps - 1:
            nodal_ke_j = 0.5 * float(np.sum(mass * (v / 1000.0) ** 2))
            history.append(
                {
                    "time_ms": 1000.0 * step * config.dt_s,
                    "indenter_disp_mm": indenter_drop,
                    "indenter_velocity_m_s": indenter_v / 1000.0,
                    "contact_force_N": contact_force_up,
                    "strain_energy_J": strain_energy_j,
                    "plastic_dissipation_J": plastic_dissipation_j,
                    "nodal_kinetic_J": nodal_ke_j,
                    "contact_work_J": contact_work_j,
                    "self_contact_force_N": peak_self_contact,
                    "self_contact_pairs": float(max_self_contact_pairs),
                }
            )

        if max_indenter_disp >= 0.96 * ENVELOPE_MM:
            failed = True
            break
        if len(spring_damage) and float(np.mean(spring_damage >= config.failed_spring_damage)) > 0.10:
            failed = True
            break
        if contact_work_j >= 0.95 * initial_ke and abs(indenter_v) / 1000.0 <= config.stop_velocity_m_s:
            stopped = True
            break

    internal_vox, strain_energy_j, _ = _explicit_internal_forces(
        u.reshape(total, 3), springs, plastic_strain, spring_damage, config, update_state=False
    )
    nodal_ke_j = 0.5 * float(np.sum(mass * (v / 1000.0) ** 2))
    indenter_ke_j = 0.5 * config.indenter_mass_kg * (indenter_v / 1000.0) ** 2
    gravity_work_j = config.indenter_mass_kg * GRAVITY_M_S2 * max_indenter_disp / 1000.0
    balance_error = initial_ke + gravity_work_j - (
        strain_energy_j + plastic_dissipation_j + nodal_ke_j + indenter_ke_j + contact_work_j
    )
    metrics = _strain_stress_metrics(solid, coords, ids, u.reshape(total, 3), vector_config)
    max_plastic = float(np.max(np.abs(plastic_strain))) if len(plastic_strain) else 0.0
    failed_fraction = float(np.mean(spring_damage >= config.failed_spring_damage)) if len(spring_damage) else 0.0
    material_damage = max(metrics["damage_proxy"], float(np.max(spring_damage)) if len(spring_damage) else 0.0)
    failed = failed or material_damage >= config.failure_damage or failed_fraction > 0.10
    survives = stopped and not failed and contact_work_j >= 0.90 * initial_ke and max_indenter_disp < 0.88 * ENVELOPE_MM
    score = (
        20.0 * max(0.0, initial_ke - contact_work_j)
        + 30.0 * material_damage
        + 18.0 * failed_fraction
        + 0.003 * max(0.0, peak_contact - 4500.0)
        + 0.001 * max(0.0, peak_self_contact - 3000.0)
        + 0.45 * max(0.0, max_indenter_disp - 38.0)
        + (20.0 if failed else 0.0)
    )
    result = ExplicitImpactFEMResult(
        rank=rank,
        topology=params.topology,
        resolution=resolution,
        solid_voxels=total,
        vector_dofs=dofs,
        simulated_time_ms=1000.0 * min(max_steps, step + 1) * config.dt_s,
        impact_initial_ke_J=initial_ke,
        impact_absorbed_J=contact_work_j,
        peak_contact_force_N=peak_contact,
        max_indenter_displacement_mm=max_indenter_disp,
        residual_velocity_m_s=indenter_v / 1000.0,
        nodal_kinetic_energy_J=nodal_ke_j,
        strain_energy_J=strain_energy_j,
        plastic_dissipation_J=plastic_dissipation_j,
        energy_balance_error_J=balance_error,
        peak_self_contact_force_N=peak_self_contact,
        max_self_contact_pairs=max_self_contact_pairs,
        max_plastic_strain=max_plastic,
        failed_springs_fraction=failed_fraction,
        p95_von_mises_MPa=metrics["p95_von_mises_MPa"],
        max_von_mises_MPa=metrics["max_von_mises_MPa"],
        max_principal_strain=metrics["max_principal_strain"],
        damage_proxy=material_damage,
        survives_explicit_fem=survives,
        explicit_fem_score=score,
    )
    return result, {"history": history}


def run_explicit_vector_impact_gate(
    top_csv: Path,
    out_dir: Path,
    top_n: int = 20,
    config: ExplicitImpactFEMConfig | None = None,
    write_histories: int = 8,
) -> dict[str, object]:
    config = config or ExplicitImpactFEMConfig()
    with Path(top_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:top_n]
    out_dir = ensure_dir(out_dir)
    history_dir = ensure_dir(out_dir / "histories")
    results: list[ExplicitImpactFEMResult] = []
    for row in rows:
        rank = int(float(row.get("rank", len(results) + 1)))
        result, payload = run_explicit_vector_impact(DesignParams.from_row(row), config=config, rank=rank)
        results.append(result)
        if len(results) <= write_histories:
            with (history_dir / f"rank_{rank:03d}_{result.topology}_explicit_vector_history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                fieldnames = [
                    "time_ms",
                    "indenter_disp_mm",
                    "indenter_velocity_m_s",
                    "contact_force_N",
                    "strain_energy_J",
                    "plastic_dissipation_J",
                    "nodal_kinetic_J",
                    "contact_work_J",
                    "self_contact_force_N",
                    "self_contact_pairs",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(payload["history"])  # type: ignore[arg-type]

    ranked = sorted(results, key=lambda item: item.explicit_fem_score)
    fields = list(ranked[0].to_row().keys()) if ranked else list(ExplicitImpactFEMResult.__dataclass_fields__.keys())
    out_csv = out_dir / "explicit_vector_impact_candidates.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in ranked:
            writer.writerow(result.to_row())
    summary = {
        "top_csv": str(top_csv),
        "out_csv": str(out_csv),
        "history_dir": str(history_dir),
        "evaluated": len(results),
        "survivors": sum(1 for item in results if item.survives_explicit_fem),
        "config": asdict(config),
        "best": ranked[0].to_row() if ranked else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# POLMI Explicit Vector Impact FEM",
        "",
        f"Evaluated: `{len(results)}`",
        f"Survivors: `{summary['survivors']}`",
        f"Input: `{top_csv}`",
        "",
        "| Rank | Original | Topology | Absorbed J | Peak N | Disp mm | p95 VM MPa | Damage | Score |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, item in enumerate(ranked[:20], start=1):
        lines.append(
            f"| {i} | {item.rank} | {item.topology} | {item.impact_absorbed_J:.2f} | "
            f"{item.peak_contact_force_N:.0f} | {item.max_indenter_displacement_mm:.1f} | "
            f"{item.p95_von_mises_MPa:.2f} | {item.damage_proxy:.3f} | {item.explicit_fem_score:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
