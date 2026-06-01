"""Data and run sanity checks for the POLMI pipeline."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .design_space import SCALAR_TARGET_FIELDS
from .paths import ensure_dir, project_root


@dataclass
class Check:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _is_number(value: str) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def audit_processed_data(root: Path) -> list[Check]:
    processed = root / "data" / "processed"
    checks: list[Check] = []
    expected = {
        "files_manifest.csv": 2300,
        "table_summary.csv": 1900,
        "stl_summary.csv": 26,
        "dgls_stl_geometry_features.csv": 26,
        "lattice_compression_curve_metrics.csv": 32,
        "photopolymer_specific_energy_absorption.csv": 6,
        "photopolymer_shock_response_metrics.csv": 8,
        "derived_tables_summary.json": 1,
    }
    for name, min_rows in expected.items():
        path = processed / name
        if not path.exists():
            checks.append(Check(name=f"processed:{name}", status="error", message="missing processed file"))
            continue
        if path.suffix == ".csv":
            rows = _count_csv_rows(path)
            status = "ok" if rows >= min_rows else "warning"
            checks.append(
                Check(
                    name=f"processed:{name}",
                    status=status,
                    message=f"{rows} rows",
                    details={"rows": rows, "expected_min": min_rows},
                )
            )
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                checks.append(Check(name=f"processed:{name}", status="error", message=str(exc)))
            else:
                checks.append(Check(name=f"processed:{name}", status="ok", message="valid json", details=data))

    curve_path = processed / "lattice_compression_curve_metrics.csv"
    if curve_path.exists():
        energies = []
        forces = []
        with curve_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if _is_number(row.get("energy_J", "")):
                    energies.append(float(row["energy_J"]))
                if _is_number(row.get("max_force_N", "")):
                    forces.append(float(row["max_force_N"]))
        status = "ok" if energies and min(energies) > 0 and max(energies) < 500 else "warning"
        checks.append(
            Check(
                name="sanity:compression_curves",
                status=status,
                message="energy/force ranges checked",
                details={
                    "n": len(energies),
                    "energy_min_J": min(energies) if energies else None,
                    "energy_max_J": max(energies) if energies else None,
                    "force_max_N": max(forces) if forces else None,
                },
            )
        )

    summary_path = processed / "dataset_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        large = [x for x in data if x.get("dataset") == "mendeley_72mg3x9ft2_octahedral_lpbf_large"]
        if large and large[0].get("file_count") == 39:
            checks.append(Check("sanity:large_dataset", "ok", "large Mendeley dataset extracted", large[0]))
        else:
            checks.append(Check("sanity:large_dataset", "warning", "large dataset summary missing or unexpected"))

    return checks


def audit_generated_candidates(path: Path) -> list[Check]:
    checks: list[Check] = []
    if not path.exists():
        return [Check("generated:candidates", "error", f"missing {path}")]

    rows = 0
    bad_numeric = 0
    min_feature_min = math.inf
    energy_min = math.inf
    energy_max = -math.inf
    mass_min = math.inf
    mass_max = -math.inf
    failure_max = -math.inf
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [x for x in ["topology", "min_feature_mm", *SCALAR_TARGET_FIELDS, "curve_000"] if x not in (reader.fieldnames or [])]
        if missing:
            checks.append(Check("generated:header", "error", "missing columns", {"missing": missing}))
        else:
            checks.append(Check("generated:header", "ok", "required columns present"))
        for row in reader:
            rows += 1
            for key in ["min_feature_mm", "mass_g", "energy_abs_J", "failure_probability"]:
                if not _is_number(row.get(key, "")):
                    bad_numeric += 1
                    continue
            if _is_number(row.get("min_feature_mm", "")):
                min_feature_min = min(min_feature_min, float(row["min_feature_mm"]))
            if _is_number(row.get("energy_abs_J", "")):
                value = float(row["energy_abs_J"])
                energy_min = min(energy_min, value)
                energy_max = max(energy_max, value)
            if _is_number(row.get("mass_g", "")):
                value = float(row["mass_g"])
                mass_min = min(mass_min, value)
                mass_max = max(mass_max, value)
            if _is_number(row.get("failure_probability", "")):
                failure_max = max(failure_max, float(row["failure_probability"]))

    checks.append(Check("generated:rows", "ok" if rows > 0 else "error", f"{rows} candidate rows", {"rows": rows}))
    checks.append(Check("generated:numeric", "ok" if bad_numeric == 0 else "error", f"{bad_numeric} bad numeric values"))
    checks.append(
        Check(
            "generated:physical_ranges",
            "ok" if min_feature_min >= 0.5 and 1.0 <= mass_min <= mass_max <= 120.0 and 0.5 <= energy_min <= energy_max <= 130.0 else "warning",
            "candidate physical ranges",
            {
                "min_feature_min_mm": min_feature_min,
                "mass_min_g": mass_min,
                "mass_max_g": mass_max,
                "energy_min_J": energy_min,
                "energy_max_J": energy_max,
                "failure_probability_max": failure_max,
            },
        )
    )
    return checks


def audit_run_dir(path: Path) -> list[Check]:
    checks: list[Check] = []
    if not path.exists():
        return [Check("run:dir", "error", f"missing {path}")]
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        checks.append(Check("run:metrics", "warning", "metrics.json not present yet"))
    else:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            checks.append(Check("run:metrics", "error", str(exc)))
        else:
            val_losses = [m.get("best_val_loss") for m in metrics.get("models", []) if isinstance(m.get("best_val_loss"), (int, float))]
            status = "ok" if val_losses and all(math.isfinite(v) for v in val_losses) else "warning"
            checks.append(Check("run:metrics", status, "training metrics parsed", {"best_val_losses": val_losses}))
    top_path = path / "postanalysis" / "top_candidates.csv"
    if top_path.exists():
        checks.append(Check("run:top_candidates", "ok", f"{_count_csv_rows(top_path)} selected rows"))
    return checks


def summarize_checks(checks: list[Check]) -> dict[str, Any]:
    counts: dict[str, int] = {"ok": 0, "warning": 0, "error": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "status": "error" if counts.get("error") else "warning" if counts.get("warning") else "ok",
        "counts": counts,
        "checks": [check.__dict__ for check in checks],
    }


def write_audit_report(root: Path, stage: str, checks: list[Check]) -> Path:
    out_dir = ensure_dir(root / "reports" / "audit")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"stage": stage, "timestamp": stamp, **summarize_checks(checks)}
    json_path = out_dir / f"audit_{stage}_{stamp}.json"
    md_path = out_dir / f"audit_{stage}_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [f"# POLMI Audit - {stage}", "", f"Status: **{payload['status']}**", "", "| Check | Status | Message |", "|---|---:|---|"]
    for check in checks:
        lines.append(f"| `{check.name}` | {check.status} | {check.message} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path


def run_audit(stage: str, generated_csv: Path | None = None, run_dir: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    checks = audit_processed_data(root)
    if generated_csv is not None:
        checks.extend(audit_generated_candidates(generated_csv))
    if run_dir is not None:
        checks.extend(audit_run_dir(run_dir))
    report = write_audit_report(root, stage, checks)
    summary = summarize_checks(checks)
    summary["report"] = str(report)
    return summary
