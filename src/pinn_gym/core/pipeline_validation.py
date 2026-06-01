"""End-to-end artifact validation for POLMI search and physics runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import ensure_dir


@dataclass
class ValidationCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _count_csv(path: Path) -> tuple[int, list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = sum(1 for _ in reader)
        return rows, list(reader.fieldnames or [])


def _read_first_rows(path: Path, limit: int = 500) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows


def _finite(value: str | None) -> bool:
    try:
        x = float(value if value is not None else "")
    except ValueError:
        return False
    return math.isfinite(x)


def _csv_check(
    checks: list[ValidationCheck],
    path: Path,
    name: str,
    required_columns: list[str] | None = None,
    min_rows: int = 1,
    required: bool = True,
) -> None:
    if not path.exists():
        checks.append(ValidationCheck(name, "error" if required else "warning", f"missing {path}"))
        return
    try:
        rows, columns = _count_csv(path)
    except Exception as exc:
        checks.append(ValidationCheck(name, "error", f"cannot read CSV: {exc}"))
        return
    missing = [col for col in (required_columns or []) if col not in columns]
    status = "ok"
    message = f"{rows} rows"
    if rows < min_rows:
        status = "error" if required else "warning"
        message = f"too few rows: {rows} < {min_rows}"
    if missing:
        status = "error" if required else "warning"
        message = f"missing columns: {missing}"
    checks.append(ValidationCheck(name, status, message, {"path": str(path), "rows": rows, "columns": columns[:80]}))


def _json_check(checks: list[ValidationCheck], path: Path, name: str, required: bool = True) -> None:
    if not path.exists():
        checks.append(ValidationCheck(name, "error" if required else "warning", f"missing {path}"))
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        checks.append(ValidationCheck(name, "error", f"cannot parse JSON: {exc}"))
        return
    checks.append(ValidationCheck(name, "ok", "valid JSON", {"path": str(path), "keys": sorted(payload.keys())[:40]}))


def _top_candidate_quality(checks: list[ValidationCheck], top_csv: Path) -> None:
    if not top_csv.exists():
        return
    rows = _read_first_rows(top_csv, limit=200)
    if not rows:
        checks.append(ValidationCheck("quality:top_candidates", "error", "top candidate table is empty"))
        return
    bad = 0
    viable_lcb = 0
    target_mass = 0
    for row in rows:
        for col in ["mass_g_mean", "energy_abs_J_lcb", "failure_probability_mean", "score"]:
            if col in row and not _finite(row.get(col)):
                bad += 1
        mass = float(row.get("mass_g_mean", "inf")) if _finite(row.get("mass_g_mean")) else math.inf
        energy_lcb = float(row.get("energy_abs_J_lcb", "-inf")) if _finite(row.get("energy_abs_J_lcb")) else -math.inf
        failure = float(row.get("failure_probability_mean", "1")) if _finite(row.get("failure_probability_mean")) else 1.0
        if 20.0 <= mass <= 28.0:
            target_mass += 1
        if energy_lcb >= 35.0 and failure < 0.45:
            viable_lcb += 1
    status = "ok" if bad == 0 and viable_lcb > 0 else "warning"
    checks.append(
        ValidationCheck(
            "quality:top_candidates",
            status,
            f"{viable_lcb} LCB-viable rows, {target_mass} rows in broad mass window",
            {"sampled_rows": len(rows), "bad_numeric": bad, "viable_lcb": viable_lcb, "mass_window_rows": target_mass},
        )
    )


def _finite_columns_check(checks: list[ValidationCheck], path: Path, name: str, columns: list[str], required: bool = True) -> None:
    if not path.exists():
        return
    rows = _read_first_rows(path, limit=300)
    bad = 0
    for row in rows:
        for col in columns:
            if col in row and not _finite(row.get(col)):
                bad += 1
    status = "ok" if bad == 0 else ("error" if required else "warning")
    checks.append(ValidationCheck(name, status, f"{bad} non-finite sampled values", {"sampled_rows": len(rows), "columns": columns}))


def _summary_survivor_check(checks: list[ValidationCheck], path: Path, name: str, required: bool = False) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    survivors = payload.get("survivors")
    if survivors is None:
        return
    try:
        survivors_int = int(survivors)
    except (TypeError, ValueError):
        return
    status = "ok" if survivors_int > 0 else ("error" if required else "warning")
    checks.append(ValidationCheck(name, status, f"{survivors_int} survivor candidates", {"path": str(path)}))


def summarize(checks: list[ValidationCheck]) -> dict[str, Any]:
    counts = {"ok": 0, "warning": 0, "error": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "status": "error" if counts.get("error", 0) else "warning" if counts.get("warning", 0) else "ok",
        "counts": counts,
        "checks": [asdict(check) for check in checks],
    }


def validate_pipeline(
    run_dir: Path,
    candidates_csv: Path | None = None,
    out_dir: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(Path(out_dir) if out_dir else run_dir / "validation")
    checks: list[ValidationCheck] = []

    if candidates_csv is not None:
        _csv_check(
            checks,
            Path(candidates_csv),
            "generated:candidates",
            ["topology", "mass_g", "energy_abs_J", "failure_probability", "curve_000"],
            min_rows=100 if strict else 1,
        )

    _json_check(checks, run_dir / "metrics.json", "train:metrics", required=True)
    top_csv = run_dir / "postanalysis" / "top_candidates.csv"
    _csv_check(
        checks,
        top_csv,
        "postanalysis:top_candidates",
        ["rank", "score", "topology", "mass_g_mean", "energy_abs_J_lcb", "failure_probability_mean"],
        min_rows=20 if strict else 1,
    )
    _json_check(checks, run_dir / "postanalysis" / "summary.json", "postanalysis:summary", required=True)
    _top_candidate_quality(checks, top_csv)

    stl_dir = run_dir / "postanalysis" / "stl"
    stls = sorted(stl_dir.glob("*.stl")) if stl_dir.exists() else []
    checks.append(
        ValidationCheck(
            "postanalysis:stl",
            "ok" if stls else "warning",
            f"{len(stls)} STL files",
            {"stl_dir": str(stl_dir), "examples": [str(x) for x in stls[:5]]},
        )
    )

    physics_tables = [
        (
            "active_refinement",
            "active_refinement_candidates.csv",
            ["refine_rank", "parent_rank", "topology", "mass_g", "energy_abs_J"],
            ["mass_g", "energy_abs_J"],
        ),
        (
            "physics_gate",
            "physics_candidates.csv",
            ["rank", "topology", "physics_energy_usable_J", "physics_failure_risk"],
            ["physics_energy_usable_J", "physics_failure_risk", "physics_peak_force_N", "physics_score"],
        ),
        (
            "neural_fem_precheck",
            "voxel_fem_candidates.csv",
            ["rank", "topology", "voxel_stiffness_N_per_mm", "damage_proxy"],
            ["voxel_stiffness_N_per_mm", "damage_proxy"],
        ),
        (
            "vector_fem",
            "vector_fem_candidates.csv",
            ["rank", "topology", "reaction_force_N", "damage_proxy"],
            ["reaction_force_N", "damage_proxy", "p95_von_mises_MPa"],
        ),
        (
            "explicit_vector_impact",
            "explicit_vector_impact_candidates.csv",
            ["rank", "topology", "impact_absorbed_J", "damage_proxy"],
            ["impact_absorbed_J", "damage_proxy", "energy_balance_error_J"],
        ),
        (
            "hex_fem",
            "hex_fem_candidates.csv",
            ["rank", "topology", "reaction_force_N", "damage_proxy", "convergence_error"],
            ["reaction_force_N", "damage_proxy", "convergence_error", "elements"],
        ),
        (
            "dynamic_impact",
            "dynamic_impact_candidates.csv",
            ["rank", "topology", "impact_absorbed_J", "impact_failure_risk"],
            ["impact_absorbed_J", "impact_failure_risk", "impact_peak_force_N"],
        ),
        (
            "sensitivity",
            "sensitivity_sweep.csv",
            ["rank", "scenario", "energy_margin_J", "impact_risk"],
            ["energy_margin_J", "impact_risk", "impact_absorbed_J", "peak_force_N"],
        ),
    ]
    for subdir, filename, columns, numeric_columns in physics_tables:
        path = run_dir / subdir / filename
        _csv_check(checks, path, f"{subdir}:{filename}", columns, min_rows=1, required=False)
        _finite_columns_check(checks, path, f"{subdir}:finite", numeric_columns, required=False)
        summary_path = run_dir / subdir / "summary.json"
        _json_check(checks, summary_path, f"{subdir}:summary", required=False)
        if subdir in {"physics_gate", "dynamic_impact", "explicit_vector_impact"}:
            _summary_survivor_check(checks, summary_path, f"{subdir}:survivors", required=False)

    evidence_csv = run_dir / "evidence" / "candidate_evidence_stack.csv"
    _csv_check(
        checks,
        evidence_csv,
        "evidence:candidate_evidence_stack.csv",
        ["rank", "topology", "robust_score", "candidate_decision_pass"],
        min_rows=1,
        required=False,
    )

    convergence = sorted(run_dir.glob("hex_convergence*/hex_convergence.csv"))
    checks.append(
        ValidationCheck(
            "hex_convergence:present",
            "ok" if convergence else "warning",
            f"{len(convergence)} convergence tables",
            {"paths": [str(x) for x in convergence[:5]]},
        )
    )
    comparisons = sorted(run_dir.glob("**/fem_reference_comparison.csv"))
    checks.append(
        ValidationCheck(
            "fem_compare:present",
            "ok" if comparisons else "warning",
            f"{len(comparisons)} comparison tables",
            {"paths": [str(x) for x in comparisons[:5]]},
        )
    )

    payload = {"run_dir": str(run_dir), **summarize(checks)}
    json_path = out_dir / "pipeline_validation.json"
    md_path = out_dir / "pipeline_validation.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# POLMI Pipeline Validation",
        "",
        f"Run dir: `{run_dir}`",
        f"Status: **{payload['status']}**",
        "",
        "| Check | Status | Message |",
        "|---|---:|---|",
    ]
    for check in checks:
        lines.append(f"| `{check.name}` | {check.status} | {check.message} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["json_path"] = str(json_path)
    payload["md_path"] = str(md_path)
    return payload
