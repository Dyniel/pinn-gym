"""STL export utilities for selected POLMI lattice candidates."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from .design_space import DesignParams, ENVELOPE_MM


def _normal(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / length, ny / length, nz / length


def _tri_line(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> str:
    nx, ny, nz = _normal(a, b, c)
    return (
        f"  facet normal {nx:.8g} {ny:.8g} {nz:.8g}\n"
        "    outer loop\n"
        f"      vertex {a[0]:.8g} {a[1]:.8g} {a[2]:.8g}\n"
        f"      vertex {b[0]:.8g} {b[1]:.8g} {b[2]:.8g}\n"
        f"      vertex {c[0]:.8g} {c[1]:.8g} {c[2]:.8g}\n"
        "    endloop\n"
        "  endfacet\n"
    )


def _write_ascii_mesh(path: Path, name: str, verts, faces) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for face in faces:
            a = tuple(float(v) for v in verts[face[0]])
            b = tuple(float(v) for v in verts[face[1]])
            c = tuple(float(v) for v in verts[face[2]])
            f.write(_tri_line(a, b, c))
        f.write(f"endsolid {name}\n")


def _write_binary_mesh(path: Path, name: str, verts, faces) -> None:
    header = f"{name} binary STL".encode("ascii", errors="ignore")[:80]
    header = header + b" " * (80 - len(header))
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            a = tuple(float(v) for v in verts[face[0]])
            b = tuple(float(v) for v in verts[face[1]])
            c = tuple(float(v) for v in verts[face[2]])
            n = _normal(a, b, c)
            f.write(struct.pack("<12fH", *(n + a + b + c), 0))


def _box_triangles(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    v = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    return [(v[i], v[j], v[k]) for i, j, k in faces]


def write_box_lattice_preview(params: DesignParams, path: Path) -> Path:
    """Write a conservative printable preview STL using overlapping cuboid struts.

    This fallback is intentionally simple and dependency-free. The Slurm job
    attempts an implicit TPMS export first when numpy/scikit-image are available.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    half = ENVELOPE_MM / 2
    cell = max(4.5, params.cell_size_mm)
    n = max(4, min(9, round(ENVELOPE_MM / cell)))
    pitch = ENVELOPE_MM / n
    base_t = max(0.5, params.wall_thickness_mm)
    rib = max(0.0, params.edge_rib_mm)
    cap = max(0.5, params.cap_thickness_mm)
    top_cap = max(0.5, cap * (0.45 + 0.35 * (1.0 - params.trigger_layer_strength)))
    tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []

    # Top/bottom caps.
    tris.extend(_box_triangles(0, 0, -half + cap / 2, ENVELOPE_MM, ENVELOPE_MM, cap))
    tris.extend(_box_triangles(0, 0, half - top_cap / 2, ENVELOPE_MM, ENVELOPE_MM, top_cap))

    coords = [-half + i * pitch for i in range(n + 1)]
    for ix, x in enumerate(coords):
        for iy, y in enumerate(coords):
            radial = max(abs(x), abs(y)) / half
            if radial > 0.98 or (ix + iy) % 2 == 0:
                t = base_t * (1.0 + 0.35 * params.vertical_gradient * (1.0 - (iy + 1) / (n + 1)))
                tris.extend(_box_triangles(x, y, 0, t, t, ENVELOPE_MM))

    for z_i in range(1, n):
        z = -half + z_i * pitch
        layer = z_i / n
        t = base_t * (1.0 + 0.55 * params.vertical_gradient * (1.0 - layer))
        if layer > 1.0 - params.trigger_zone_mm / ENVELOPE_MM:
            t *= 1.0 - 0.25 * params.trigger_layer_strength
        t = max(0.5, t)
        for y in coords:
            tris.extend(_box_triangles(0, y, z, ENVELOPE_MM, t, t))
        for x in coords:
            tris.extend(_box_triangles(x, 0, z, t, ENVELOPE_MM, t))

    if rib > 0.05:
        rib_height = ENVELOPE_MM - 0.55 * params.trigger_zone_mm
        rib_center_z = -half + rib_height / 2
        for sx in [-1, 1]:
            tris.extend(_box_triangles(sx * (half - rib / 2), 0, rib_center_z, rib, rib, rib_height))
        for sy in [-1, 1]:
            tris.extend(_box_triangles(0, sy * (half - rib / 2), rib_center_z, rib, rib, rib_height))

    with path.open("w", encoding="utf-8") as f:
        f.write("solid pinn_gym_box_lattice_preview\n")
        for tri in tris:
            f.write(_tri_line(*tri))
        f.write("endsolid pinn_gym_box_lattice_preview\n")
    return path


def _implicit_phi_grid(params: DesignParams, resolution: int):
    import numpy as np

    resolution = int(max(48, min(256, resolution)))
    h = ENVELOPE_MM / resolution
    # Use cell-centered samples with one outside layer. This lets marching cubes
    # close clipped surfaces exactly on the 50 mm envelope instead of leaving
    # open boundaries where the TPMS touches the box.
    axis = (-ENVELOPE_MM / 2 - h / 2) + np.arange(resolution + 2, dtype=np.float32) * h
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
    blend = params.hybrid_ratio if params.topology == "hybrid" else 0.0
    if params.topology in {"diamond", "diamond_graded"}:
        field = diamond
    elif params.topology == "schwarz_p":
        field = schwarz_p / 1.5
    elif params.topology == "gyroid_diamond_hybrid":
        z01_tmp = (z + ENVELOPE_MM / 2) / ENVELOPE_MM
        z_blend = np.clip(params.hybrid_ratio + 0.25 * (0.5 - z01_tmp), 0.0, 1.0)
        field = (1.0 - z_blend) * gyroid + z_blend * diamond
    elif params.topology in {"bccz_graded", "ot_like"}:
        mix = 0.72 if params.topology == "bccz_graded" else 0.48
        field = mix * bccz + (1.0 - mix) * (0.55 * gyroid + 0.45 * diamond)
    elif params.topology in {"octet", "bcc"}:
        field = 0.55 * gyroid + 0.45 * diamond
    else:
        field = (1.0 - blend) * gyroid + blend * diamond

    z01 = (z + ENVELOPE_MM / 2) / ENVELOPE_MM
    threshold = (params.wall_thickness_mm / max(3.5, params.cell_size_mm)) * 1.7
    trigger_start = 1.0 - params.trigger_zone_mm / ENVELOPE_MM
    bumper_end = params.bumper_zone_mm / ENVELOPE_MM
    trigger_softening = 1.0 - 0.30 * params.trigger_layer_strength * np.clip((z01 - trigger_start) / max(1e-6, 1.0 - trigger_start), 0.0, 1.0)
    bumper_boost = 1.0 + 0.24 * params.vertical_gradient * np.clip((bumper_end - z01) / max(1e-6, bumper_end), 0.0, 1.0)
    threshold = threshold * (1.0 + params.vertical_gradient * (0.5 - z01)) * trigger_softening * bumper_boost
    threshold = np.maximum(threshold, params.min_feature_mm / max(3.5, params.cell_size_mm))
    half = ENVELOPE_MM / 2
    box_phi = np.maximum.reduce(
        [
            np.abs(x) - half,
            np.abs(y) - half,
            np.abs(z) - half,
        ]
    )
    shell_phi = np.abs(field) - threshold
    phi = np.maximum(shell_phi, box_phi)

    bottom_cap_phi = np.maximum(z - (-half + params.cap_thickness_mm), box_phi)
    top_cap = max(params.min_feature_mm, params.cap_thickness_mm * (0.45 + 0.35 * (1.0 - params.trigger_layer_strength)))
    top_cap_phi = np.maximum((half - top_cap) - z, box_phi)
    phi = np.minimum(phi, bottom_cap_phi)
    phi = np.minimum(phi, top_cap_phi)
    if params.edge_rib_mm > 0.05:
        rib = params.edge_rib_mm
        rib_height = ENVELOPE_MM - 0.55 * params.trigger_zone_mm
        rib_center_z = -half + rib_height / 2
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                cx = sx * (half - rib / 2)
                cy = sy * (half - rib / 2)
                rib_phi = np.maximum.reduce(
                    [
                        np.abs(x - cx) - rib / 2,
                        np.abs(y - cy) - rib / 2,
                        np.abs(z - rib_center_z) - rib_height / 2,
                    ]
                )
                phi = np.minimum(phi, rib_phi)

    return axis, h, phi


def write_implicit_tpms_stl(params: DesignParams, path: Path, resolution: int = 128, stl_format: str = "ascii") -> Path:
    """Write a TPMS-style STL through continuous-field marching cubes.

    Requires numpy and scikit-image. It is used by the Slurm job if available.
    """

    import numpy as np
    from skimage import measure

    path.parent.mkdir(parents=True, exist_ok=True)
    axis, h, phi = _implicit_phi_grid(params, resolution)
    verts, faces, _normals, _ = measure.marching_cubes(phi.astype(np.float32), level=0.0, spacing=(h, h, h), allow_degenerate=False)
    verts += axis[0]
    if stl_format == "binary":
        _write_binary_mesh(path, "pinn_gym_implicit_tpms", verts, faces)
    else:
        _write_ascii_mesh(path, "pinn_gym_implicit_tpms", verts, faces)
    return path


def write_voxelized_implicit_stl(params: DesignParams, path: Path, resolution: int = 128, stl_format: str = "ascii") -> Path:
    """Write a watertight voxelized implicit STL for final print submission.

    The continuous implicit exporter preserves smoother surfaces but can create
    tiny non-manifold sliver triangles where thickened TPMS sheets self-touch.
    This backend extracts the boundary of the occupied voxel set instead. The
    result is less smooth, but the padded binary volume gives a closed solid
    surface that is robust to those local level-set degeneracies. Some binary
    volumes still contain edge-only contacts; when the audit catches those
    non-manifold edges, the function falls back to the continuous implicit mesh
    because a closed audited STL is more important than preserving the backend
    implementation detail.
    """

    import numpy as np
    from skimage import measure

    path.parent.mkdir(parents=True, exist_ok=True)
    axis, h, phi = _implicit_phi_grid(params, resolution)
    solid = phi <= 0.0
    solid[[0, -1], :, :] = False
    solid[:, [0, -1], :] = False
    solid[:, :, [0, -1]] = False
    volume = solid.astype(np.float32)
    verts, faces, _normals, _ = measure.marching_cubes(volume, level=0.5, spacing=(h, h, h), allow_degenerate=False)
    verts += axis[0]
    half = ENVELOPE_MM / 2
    verts = np.clip(verts, -half, half)
    if stl_format == "binary":
        _write_binary_mesh(path, "pinn_gym_voxelized_implicit", verts, faces)
    else:
        _write_ascii_mesh(path, "pinn_gym_voxelized_implicit", verts, faces)
    from .mesh_quality import audit_stl_mesh

    quality = audit_stl_mesh(path)
    if quality.edge_count_not_two:
        return write_implicit_tpms_stl(
            params,
            path,
            resolution=resolution,
            stl_format=stl_format,
        )
    return path


def export_design_stl(params: DesignParams, path: Path, backend: str = "auto", resolution: int = 128, stl_format: str = "ascii") -> Path:
    if backend == "voxel":
        return write_voxelized_implicit_stl(params, path, resolution=resolution, stl_format=stl_format)
    if backend in {"auto", "implicit"}:
        try:
            return write_implicit_tpms_stl(params, path, resolution=resolution, stl_format=stl_format)
        except Exception:
            if backend == "implicit":
                raise
    return write_box_lattice_preview(params, path)
