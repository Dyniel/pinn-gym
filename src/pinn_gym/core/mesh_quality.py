"""STL mesh quality checks used before expensive physical validation."""

from __future__ import annotations

import json
import math
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class MeshQuality:
    path: str
    exists: bool
    file_size_bytes: int
    stl_format: str
    facets: int
    vertices: int
    min_x_mm: float | None
    max_x_mm: float | None
    min_y_mm: float | None
    max_y_mm: float | None
    min_z_mm: float | None
    max_z_mm: float | None
    bbox_x_mm: float | None
    bbox_y_mm: float | None
    bbox_z_mm: float | None
    open_edges: int | None
    overused_edges: int | None
    edge_count_not_two: int | None
    degenerate_facets: int
    watertight_by_edges: bool | None
    within_envelope: bool | None
    error: str | None = None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def _is_binary_stl(path: Path) -> bool:
    size = path.stat().st_size
    if size < 84:
        return False
    with path.open("rb") as f:
        header = f.read(84)
    if len(header) < 84:
        return False
    tri_count = struct.unpack("<I", header[80:84])[0]
    return 84 + tri_count * 50 == size


def _iter_ascii_triangles(path: Path) -> Iterator[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    current: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("      vertex") or line.lstrip().startswith("vertex"):
                parts = line.split()
                if len(parts) >= 4:
                    current.append((float(parts[-3]), float(parts[-2]), float(parts[-1])))
                    if len(current) == 3:
                        yield current[0], current[1], current[2]
                        current = []


def _iter_binary_triangles(path: Path) -> Iterator[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    with path.open("rb") as f:
        f.seek(80)
        count = struct.unpack("<I", f.read(4))[0]
        for _ in range(count):
            data = f.read(50)
            if len(data) != 50:
                break
            values = struct.unpack("<12fH", data)
            yield (
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            )


def _quantize_vertex(vertex: tuple[float, float, float], tolerance_mm: float) -> tuple[int, int, int]:
    scale = 1.0 / tolerance_mm
    return tuple(int(round(x * scale)) for x in vertex)


def _bounds_update(bounds: list[float], vertex: tuple[float, float, float]) -> None:
    x, y, z = vertex
    bounds[0] = min(bounds[0], x)
    bounds[1] = max(bounds[1], x)
    bounds[2] = min(bounds[2], y)
    bounds[3] = max(bounds[3], y)
    bounds[4] = min(bounds[4], z)
    bounds[5] = max(bounds[5], z)


def _triangle_area2(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> float:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return nx * nx + ny * ny + nz * nz


def audit_stl_mesh(
    path: Path,
    envelope_mm: float = 50.0,
    edge_check: bool = True,
    tolerance_mm: float = 1e-5,
    max_edge_facets: int = 3_500_000,
) -> MeshQuality:
    path = Path(path)
    if not path.exists():
        return MeshQuality(
            path=str(path),
            exists=False,
            file_size_bytes=0,
            stl_format="missing",
            facets=0,
            vertices=0,
            min_x_mm=None,
            max_x_mm=None,
            min_y_mm=None,
            max_y_mm=None,
            min_z_mm=None,
            max_z_mm=None,
            bbox_x_mm=None,
            bbox_y_mm=None,
            bbox_z_mm=None,
            open_edges=None,
            overused_edges=None,
            edge_count_not_two=None,
            degenerate_facets=0,
            watertight_by_edges=None,
            within_envelope=None,
            error="missing file",
        )

    try:
        stl_format = "binary" if _is_binary_stl(path) else "ascii"
        triangles = _iter_binary_triangles(path) if stl_format == "binary" else _iter_ascii_triangles(path)
        edge_counts: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = defaultdict(int)
        bounds = [math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf]
        facets = 0
        degenerate = 0
        for tri in triangles:
            facets += 1
            for vertex in tri:
                _bounds_update(bounds, vertex)
            if _triangle_area2(*tri) <= tolerance_mm * tolerance_mm:
                degenerate += 1
            if edge_check and facets <= max_edge_facets:
                q = [_quantize_vertex(vertex, tolerance_mm) for vertex in tri]
                for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[0])):
                    if a > b:
                        a, b = b, a
                    edge_counts[(a, b)] += 1
        if facets == 0:
            raise ValueError("no facets found")
        if edge_check and facets <= max_edge_facets:
            open_edges = sum(1 for value in edge_counts.values() if value == 1)
            overused_edges = sum(1 for value in edge_counts.values() if value > 2)
            edge_count_not_two = sum(1 for value in edge_counts.values() if value != 2)
            watertight = edge_count_not_two == 0
        else:
            open_edges = None
            overused_edges = None
            edge_count_not_two = None
            watertight = None
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        half = envelope_mm / 2.0
        within = min_x >= -half - tolerance_mm and max_x <= half + tolerance_mm and min_y >= -half - tolerance_mm and max_y <= half + tolerance_mm and min_z >= -half - tolerance_mm and max_z <= half + tolerance_mm
        return MeshQuality(
            path=str(path),
            exists=True,
            file_size_bytes=path.stat().st_size,
            stl_format=stl_format,
            facets=facets,
            vertices=facets * 3,
            min_x_mm=min_x,
            max_x_mm=max_x,
            min_y_mm=min_y,
            max_y_mm=max_y,
            min_z_mm=min_z,
            max_z_mm=max_z,
            bbox_x_mm=max_x - min_x,
            bbox_y_mm=max_y - min_y,
            bbox_z_mm=max_z - min_z,
            open_edges=open_edges,
            overused_edges=overused_edges,
            edge_count_not_two=edge_count_not_two,
            degenerate_facets=degenerate,
            watertight_by_edges=watertight,
            within_envelope=within,
        )
    except Exception as exc:
        return MeshQuality(
            path=str(path),
            exists=True,
            file_size_bytes=path.stat().st_size,
            stl_format="unknown",
            facets=0,
            vertices=0,
            min_x_mm=None,
            max_x_mm=None,
            min_y_mm=None,
            max_y_mm=None,
            min_z_mm=None,
            max_z_mm=None,
            bbox_x_mm=None,
            bbox_y_mm=None,
            bbox_z_mm=None,
            open_edges=None,
            overused_edges=None,
            edge_count_not_two=None,
            degenerate_facets=0,
            watertight_by_edges=None,
            within_envelope=None,
            error=repr(exc),
        )


def audit_stl_directory(stl_dir: Path, pattern: str = "*.stl", **kwargs: object) -> list[MeshQuality]:
    return [audit_stl_mesh(path, **kwargs) for path in sorted(Path(stl_dir).glob(pattern))]


def write_mesh_quality_report(results: Iterable[MeshQuality], out_json: Path) -> Path:
    rows = [item.to_row() for item in results]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_json
