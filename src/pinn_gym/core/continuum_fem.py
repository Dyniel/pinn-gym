"""Continuum hexahedral FEM for POLMI voxel/SDF candidates.

This module is the closest in-house solver to a conventional FEM stack in this
project. It builds 8-node hexahedral elements from a voxelized candidate, uses
2x2x2 Gauss integration points, evaluates large-deformation kinematics through
the deformation gradient, carries per-Gauss-point plastic/damage state, and
reports convergence/comparison artifacts.

It is still a compact research solver, not a replacement for a validated
commercial explicit FEM package. The point is to make the numerical assumptions
visible and testable inside the pipeline.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .design_space import ENVELOPE_MM, DesignParams
from .materials import load_material_card
from .paths import ensure_dir
from .physics import PA12_ELASTIC_MODULUS_MPA, PA12_POISSON, PA12_YIELD_STRESS_MPA
from .voxel_fem import voxel_solid


@dataclass(frozen=True)
class PA12MaterialCard:
    elastic_modulus_MPa: float = PA12_ELASTIC_MODULUS_MPA
    poisson: float = PA12_POISSON
    yield_stress_MPa: float = PA12_YIELD_STRESS_MPA
    tangent_modulus_MPa: float = 110.0
    density_g_cm3: float = 1.01
    fracture_plastic_strain: float = 0.18
    damage_growth_strain: float = 0.12
    strain_rate_factor: float = 1.16
    contact_friction: float = 0.20
    source: str = "default PA12 placeholder; calibrate with print/process data before final claims"

    @classmethod
    def from_json(cls, path: Path | None) -> "PA12MaterialCard":
        if path is None:
            return cls()
        generic = load_material_card(path)
        return cls(
            elastic_modulus_MPa=generic.elastic_modulus_MPa,
            poisson=generic.poisson_ratio,
            yield_stress_MPa=generic.compressive_yield_strength_MPa,
            density_g_cm3=generic.density_g_cm3,
            fracture_plastic_strain=generic.failure_strain,
            strain_rate_factor=generic.strain_rate_factor(60.0),
            source=generic.source_or_calibration_note,
        )

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class HexFEMConfig:
    resolution: int = 10
    displacement_mm: float = 4.0
    load_steps: int = 8
    relax_iterations: int = 140
    pseudo_dt: float = 4e-4
    damping: float = 0.72
    stiffness_scale: float = 0.65
    enable_plasticity: bool = True
    enable_self_contact: bool = True
    self_contact_radius_fraction: float = 0.72
    self_contact_penalty_N_per_mm: float = 18.0
    self_contact_stride: int = 8


@dataclass(frozen=True)
class HexFEMResult:
    rank: int
    topology: str
    resolution: int
    nodes: int
    elements: int
    gauss_points: int
    displacement_mm: float
    reaction_force_N: float
    stiffness_N_per_mm: float
    strain_energy_J: float
    plastic_dissipation_J: float
    max_von_mises_MPa: float
    p95_von_mises_MPa: float
    max_principal_green_strain: float
    max_plastic_strain: float
    damage_proxy: float
    failed_gauss_fraction: float
    max_self_contact_pairs: int
    residual_norm_N: float
    convergence_error: float

    def to_row(self) -> dict[str, object]:
        return asdict(self)


_HEX_LOCAL = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
]


def _gauss_points():
    a = 1.0 / math.sqrt(3.0)
    for xi in (-a, a):
        for eta in (-a, a):
            for zeta in (-a, a):
                yield xi, eta, zeta, 1.0


def _shape_grad_parent(xi: float, eta: float, zeta: float):
    import numpy as np

    grads = np.zeros((8, 3), dtype=np.float64)
    for i, (sx, sy, sz) in enumerate(_HEX_LOCAL):
        grads[i, 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
        grads[i, 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
        grads[i, 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
    return grads


def _build_hex_mesh(params: DesignParams, resolution: int):
    import numpy as np

    solid = voxel_solid(params, resolution)
    n = int(solid.shape[0])
    h = ENVELOPE_MM / n
    node_index: dict[tuple[int, int, int], int] = {}
    nodes: list[tuple[float, float, float]] = []
    elements: list[list[int]] = []
    for i, j, k in np.argwhere(solid):
        corners = [
            (int(i), int(j), int(k)),
            (int(i + 1), int(j), int(k)),
            (int(i + 1), int(j + 1), int(k)),
            (int(i), int(j + 1), int(k)),
            (int(i), int(j), int(k + 1)),
            (int(i + 1), int(j), int(k + 1)),
            (int(i + 1), int(j + 1), int(k + 1)),
            (int(i), int(j + 1), int(k + 1)),
        ]
        element = []
        for corner in corners:
            if corner not in node_index:
                node_index[corner] = len(nodes)
                nodes.append((corner[0] * h, corner[1] * h, corner[2] * h))
            element.append(node_index[corner])
        elements.append(element)
    return np.asarray(nodes, dtype=np.float64), np.asarray(elements, dtype=np.int64), h


def _von_mises(stress):
    import numpy as np

    dev = stress - np.eye(3) * float(np.trace(stress)) / 3.0
    return math.sqrt(max(0.0, 1.5 * float((dev * dev).sum())))


def _element_forces(
    nodes_ref,
    nodes_cur,
    elements,
    material: PA12MaterialCard,
    config: HexFEMConfig,
    plastic_state,
    damage_state,
    update_state: bool,
):
    import numpy as np

    forces = np.zeros_like(nodes_cur)
    h_est = ENVELOPE_MM / max(1, config.resolution)
    det_j = (h_est / 2.0) ** 3
    grad_scale = 2.0 / h_est
    e = material.elastic_modulus_MPa * config.stiffness_scale * material.strain_rate_factor
    nu = material.poisson
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    hardening = material.tangent_modulus_MPa
    vm_values: list[float] = []
    principal_values: list[float] = []
    strain_energy = 0.0
    plastic_dissipation = 0.0
    gp_index = 0

    for element in elements:
        x = nodes_cur[element]
        for xi, eta, zeta, weight in _gauss_points():
            grad_parent = _shape_grad_parent(xi, eta, zeta)
            grad_ref = grad_parent * grad_scale
            f_def = x.T @ grad_ref
            c = f_def.T @ f_def
            green = 0.5 * (c - np.eye(3))
            second = lam * float(np.trace(green)) * np.eye(3) + 2.0 * mu * green
            vm = _von_mises(second)
            if config.enable_plasticity:
                yield_now = material.yield_stress_MPa + hardening * plastic_state[gp_index]
                if vm > yield_now:
                    scale = yield_now / max(vm, 1e-12)
                    hydro = np.eye(3) * float(np.trace(second)) / 3.0
                    dev = second - hydro
                    second = hydro + scale * dev
                    delta_plastic = (vm - yield_now) / max(e + hardening, 1e-12)
                    if update_state:
                        plastic_state[gp_index] += delta_plastic
                        drive = max(0.0, plastic_state[gp_index] - material.fracture_plastic_strain)
                        damage_state[gp_index] = max(
                            damage_state[gp_index],
                            1.0 - math.exp(-drive / max(material.damage_growth_strain, 1e-12)),
                        )
                    plastic_dissipation += yield_now * delta_plastic * det_j * weight / 1000.0
                    vm = _von_mises(second)
            degradation = max(0.03, 1.0 - damage_state[gp_index]) ** 2
            second *= degradation
            first = f_def @ second
            for local_i, node_i in enumerate(element):
                forces[node_i] -= first @ grad_ref[local_i] * det_j * weight
            strain_energy += max(0.0, 0.5 * float((second * green).sum()) * det_j * weight / 1000.0)
            vm_values.append(vm)
            try:
                principal_values.append(float(np.max(np.abs(np.linalg.eigvalsh(green)))))
            except Exception:
                principal_values.append(float(np.linalg.norm(green)))
            gp_index += 1
    return forces, strain_energy, plastic_dissipation, vm_values, principal_values


def _surface_self_contact(nodes_ref, nodes_cur, velocity, config: HexFEMConfig, h: float):
    import numpy as np
    from scipy.spatial import cKDTree

    if not config.enable_self_contact or len(nodes_cur) < 2:
        return np.zeros_like(nodes_cur), 0
    z = nodes_ref[:, 2]
    surface = (z < 1e-9) | (z > ENVELOPE_MM - 1e-9)
    surface |= nodes_ref[:, 0] < 1e-9
    surface |= nodes_ref[:, 0] > ENVELOPE_MM - 1e-9
    surface |= nodes_ref[:, 1] < 1e-9
    surface |= nodes_ref[:, 1] > ENVELOPE_MM - 1e-9
    ids = np.where(surface)[0]
    if len(ids) < 2:
        return np.zeros_like(nodes_cur), 0
    pos = nodes_cur[ids]
    radius = max(1e-6, config.self_contact_radius_fraction * h)
    tree = cKDTree(pos)
    forces = np.zeros_like(nodes_cur)
    active = 0
    for a_local, b_local in tree.query_pairs(radius):
        a = int(ids[a_local])
        b = int(ids[b_local])
        if np.linalg.norm(nodes_ref[b] - nodes_ref[a]) < 1.05 * h:
            continue
        delta = nodes_cur[b] - nodes_cur[a]
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-12 or dist >= radius:
            continue
        normal = delta / dist
        rel_v = float(np.dot(velocity[b] - velocity[a], normal))
        mag = config.self_contact_penalty_N_per_mm * (radius - dist) + 0.01 * max(0.0, -rel_v)
        forces[a] -= mag * normal
        forces[b] += mag * normal
        active += 1
    return forces, active


def run_hex_fem_compression(
    params: DesignParams,
    config: HexFEMConfig | None = None,
    material: PA12MaterialCard | None = None,
    rank: int = 0,
) -> tuple[HexFEMResult, dict[str, object]]:
    import numpy as np

    config = config or HexFEMConfig()
    material = material or PA12MaterialCard()
    nodes_ref, elements, h = _build_hex_mesh(params, config.resolution)
    if len(nodes_ref) == 0 or len(elements) == 0:
        empty = HexFEMResult(
            rank=rank,
            topology=params.topology,
            resolution=config.resolution,
            nodes=0,
            elements=0,
            gauss_points=0,
            displacement_mm=config.displacement_mm,
            reaction_force_N=0.0,
            stiffness_N_per_mm=0.0,
            strain_energy_J=0.0,
            plastic_dissipation_J=0.0,
            max_von_mises_MPa=0.0,
            p95_von_mises_MPa=0.0,
            max_principal_green_strain=0.0,
            max_plastic_strain=0.0,
            damage_proxy=1.0,
            failed_gauss_fraction=1.0,
            max_self_contact_pairs=0,
            residual_norm_N=1e9,
            convergence_error=1e9,
        )
        return empty, {"curve": []}

    u = np.zeros_like(nodes_ref)
    velocity = np.zeros_like(nodes_ref)
    bottom = nodes_ref[:, 2] < 1e-9
    top = nodes_ref[:, 2] > ENVELOPE_MM - 1e-9
    fixed = bottom.copy()
    gp_count = len(elements) * 8
    plastic = np.zeros(gp_count, dtype=np.float64)
    damage = np.zeros(gp_count, dtype=np.float64)
    curve: list[dict[str, float]] = []
    max_self_pairs = 0
    residual_norm = 0.0

    for step in range(1, max(1, config.load_steps) + 1):
        target = -config.displacement_mm * step / max(1, config.load_steps)
        for _ in range(max(1, config.relax_iterations)):
            nodes_cur = nodes_ref + u
            force, _, _, _, _ = _element_forces(nodes_ref, nodes_cur, elements, material, config, plastic, damage, update_state=False)
            if config.enable_self_contact:
                contact, pairs = _surface_self_contact(nodes_ref, nodes_cur, velocity, config, h)
                force += contact
                max_self_pairs = max(max_self_pairs, pairs)
            free = ~(fixed | top)
            residual_norm = float(np.linalg.norm(force[free]))
            nodal_stiffness = max(1.0, material.elastic_modulus_MPa * config.stiffness_scale * h)
            velocity[free] = config.damping * velocity[free] + config.pseudo_dt * force[free] / nodal_stiffness
            step_norm = np.linalg.norm(velocity[free], axis=1)
            over = step_norm > 0.04 * h
            if np.any(over):
                velocity[free][over] *= ((0.04 * h) / np.maximum(step_norm[over], 1e-12))[:, None]
            u[free] += config.pseudo_dt * velocity[free]
            u[fixed] = 0.0
            velocity[fixed] = 0.0
            u[top, 2] = target
            velocity[top] = 0.0
        nodes_cur = nodes_ref + u
        force, energy, plastic_step, vm, principal = _element_forces(nodes_ref, nodes_cur, elements, material, config, plastic, damage, update_state=True)
        reaction = abs(float(force[top, 2].sum())) if top.any() else 0.0
        curve.append({"displacement_mm": abs(target), "reaction_force_N": reaction, "strain_energy_J": energy})

    nodes_cur = nodes_ref + u
    force, energy, plastic_diss, vm, principal = _element_forces(nodes_ref, nodes_cur, elements, material, config, plastic, damage, update_state=False)
    reaction = abs(float(force[top, 2].sum())) if top.any() else 0.0
    vm_sorted = sorted(vm)
    p95 = vm_sorted[int(0.95 * (len(vm_sorted) - 1))] if vm_sorted else 0.0
    failed_fraction = float((damage > 0.90).mean()) if len(damage) else 0.0
    max_damage = float(damage.max()) if len(damage) else 0.0
    result = HexFEMResult(
        rank=rank,
        topology=params.topology,
        resolution=config.resolution,
        nodes=len(nodes_ref),
        elements=len(elements),
        gauss_points=gp_count,
        displacement_mm=config.displacement_mm,
        reaction_force_N=reaction,
        stiffness_N_per_mm=reaction / max(config.displacement_mm, 1e-9),
        strain_energy_J=energy,
        plastic_dissipation_J=plastic_diss,
        max_von_mises_MPa=max(vm, default=0.0),
        p95_von_mises_MPa=p95,
        max_principal_green_strain=max(principal, default=0.0),
        max_plastic_strain=float(plastic.max()) if len(plastic) else 0.0,
        damage_proxy=max_damage,
        failed_gauss_fraction=failed_fraction,
        max_self_contact_pairs=max_self_pairs,
        residual_norm_N=residual_norm,
        convergence_error=residual_norm / max(1.0, abs(reaction)),
    )
    return result, {"curve": curve, "material": asdict(material), "config": asdict(config)}


def run_hex_fem_gate(
    top_csv: Path,
    out_dir: Path,
    top_n: int = 10,
    config: HexFEMConfig | None = None,
    material: PA12MaterialCard | None = None,
    write_curves: int = 5,
) -> dict[str, object]:
    config = config or HexFEMConfig()
    material = material or PA12MaterialCard()
    out_dir = ensure_dir(out_dir)
    curve_dir = ensure_dir(out_dir / "curves")
    with Path(top_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:top_n]
    results: list[HexFEMResult] = []
    for row in rows:
        rank = int(float(row.get("rank", len(results) + 1)))
        result, payload = run_hex_fem_compression(DesignParams.from_row(row), config=config, material=material, rank=rank)
        results.append(result)
        if len(results) <= write_curves:
            curve_path = curve_dir / f"rank_{rank:03d}_{result.topology}_hex_curve.csv"
            with curve_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["displacement_mm", "reaction_force_N", "strain_energy_J"])
                writer.writeheader()
                writer.writerows(payload["curve"])  # type: ignore[arg-type]
    out_csv = out_dir / "hex_fem_candidates.csv"
    fields = list(results[0].to_row().keys()) if results else list(HexFEMResult.__dataclass_fields__.keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.damage_proxy + item.convergence_error):
            writer.writerow(result.to_row())
    summary = {
        "top_csv": str(top_csv),
        "out_csv": str(out_csv),
        "curve_dir": str(curve_dir),
        "evaluated": len(results),
        "config": asdict(config),
        "material": asdict(material),
        "best": min((item.to_row() for item in results), key=lambda x: float(x["damage_proxy"]) + float(x["convergence_error"])) if results else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_hex_convergence_study(
    row: dict[str, str],
    out_dir: Path,
    resolutions: list[int],
    base_config: HexFEMConfig | None = None,
    material: PA12MaterialCard | None = None,
) -> dict[str, object]:
    out_dir = ensure_dir(out_dir)
    base_config = base_config or HexFEMConfig()
    material = material or PA12MaterialCard()
    params = DesignParams.from_row(row)
    results = []
    for resolution in resolutions:
        config = replace(base_config, resolution=resolution)
        result, _ = run_hex_fem_compression(params, config=config, material=material, rank=int(float(row.get("rank", 0) or 0)))
        results.append(result)
    finest = results[-1] if results else None
    rows = []
    for result in results:
        rel_force = 0.0
        rel_energy = 0.0
        if finest is not None:
            rel_force = abs(result.reaction_force_N - finest.reaction_force_N) / max(1.0, abs(finest.reaction_force_N))
            rel_energy = abs(result.strain_energy_J - finest.strain_energy_J) / max(1e-6, abs(finest.strain_energy_J))
        item = result.to_row()
        item["relative_force_error_vs_finest"] = rel_force
        item["relative_energy_error_vs_finest"] = rel_energy
        rows.append(item)
    out_csv = out_dir / "hex_convergence.csv"
    fields = list(rows[0].keys()) if rows else list(HexFEMResult.__dataclass_fields__.keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"out_csv": str(out_csv), "resolutions": resolutions, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def calibrate_pa12_from_curve(curve_csv: Path, out_json: Path, area_mm2: float = ENVELOPE_MM * ENVELOPE_MM) -> dict[str, object]:
    rows = []
    with Path(curve_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disp = float(row.get("displacement_mm", row.get("disp_mm", 0.0)) or 0.0)
            force = float(row.get("force_N", row.get("reaction_force_N", 0.0)) or 0.0)
            rows.append((disp, force))
    if len(rows) < 3:
        raise ValueError(f"need at least 3 rows in {curve_csv}")
    rows.sort()
    early = [(d, f) for d, f in rows if 0.1 <= d <= 2.0]
    if len(early) >= 2:
        d0, f0 = early[0]
        d1, f1 = early[-1]
        stiffness = (f1 - f0) / max(1e-9, d1 - d0)
        elastic_modulus = max(200.0, min(2600.0, stiffness * ENVELOPE_MM / max(1.0, area_mm2)))
    else:
        elastic_modulus = PA12_ELASTIC_MODULUS_MPA
    plateau = [f for d, f in rows if 5.0 <= d <= 25.0]
    plateau_force = sum(plateau) / len(plateau) if plateau else max(f for _, f in rows)
    yield_stress = max(8.0, min(65.0, plateau_force / max(1.0, area_mm2) * 6.0))
    card = PA12MaterialCard(
        elastic_modulus_MPa=elastic_modulus,
        yield_stress_MPa=yield_stress,
        source=f"calibrated from {curve_csv}",
    )
    card.to_json(out_json)
    payload = {"out_json": str(out_json), "material": asdict(card), "plateau_force_N": plateau_force}
    return payload


def compare_fem_to_reference(fem_csv: Path, reference_csv: Path, out_dir: Path) -> dict[str, object]:
    out_dir = ensure_dir(out_dir)

    def load(path: Path):
        with Path(path).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    fem_rows = load(fem_csv)
    ref_rows = load(reference_csv)
    if not fem_rows or not ref_rows:
        raise ValueError("both FEM and reference CSVs need rows")
    common = sorted(set(fem_rows[0].keys()) & set(ref_rows[0].keys()))
    numeric = []
    for key in common:
        try:
            float(fem_rows[0][key])
            float(ref_rows[0][key])
            numeric.append(key)
        except Exception:
            pass
    metrics = []
    for key in numeric:
        n = min(len(fem_rows), len(ref_rows))
        diffs = []
        refs = []
        for i in range(n):
            pred = float(fem_rows[i][key])
            ref = float(ref_rows[i][key])
            diffs.append(pred - ref)
            refs.append(ref)
        rmse = math.sqrt(sum(x * x for x in diffs) / max(1, len(diffs)))
        mae = sum(abs(x) for x in diffs) / max(1, len(diffs))
        rel = mae / max(1e-9, sum(abs(x) for x in refs) / max(1, len(refs)))
        metrics.append({"field": key, "rmse": rmse, "mae": mae, "relative_mae": rel})
    out_csv = out_dir / "fem_reference_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "rmse", "mae", "relative_mae"])
        writer.writeheader()
        writer.writerows(metrics)
    summary = {"fem_csv": str(fem_csv), "reference_csv": str(reference_csv), "out_csv": str(out_csv), "metrics": metrics}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
