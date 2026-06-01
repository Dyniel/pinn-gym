"""Evidence stack builder for POLMI multi-fidelity candidate decisions."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import ensure_dir
from .physics import IMPACT_ENERGY_J


@dataclass(frozen=True)
class EvidenceConfig:
    target_energy_j: float = IMPACT_ENERGY_J
    safe_energy_j: float = 35.0
    peak_limit_n: float = 3500.0
    explicit_peak_limit_n: float = 4500.0
    min_crush_mm: float = 40.0
    max_failure_risk: float = 0.48
    max_damage: float = 0.88
    max_energy_balance_abs_j: float = 7.5
    min_hex_elements: int = 8
    require_explicit_fem: bool = False


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _index_by_rank(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        if "rank" not in row:
            continue
        try:
            rank = int(float(row["rank"]))
        except (TypeError, ValueError):
            continue
        out[rank] = row
    return out


def _f(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    if row is None:
        return default
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _b(row: dict[str, Any] | None, key: str) -> bool:
    if row is None:
        return False
    return _truthy(row.get(key))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _s(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.8g}"
    if value is None:
        return ""
    return str(value)


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _artifact_path(run_dir: Path, artifact_root: Path, subdir: str, filename: str) -> Path:
    candidate = artifact_root / subdir / filename
    if candidate.exists():
        return candidate
    return run_dir / subdir / filename


def _rank_from_path(path: str) -> int | None:
    match = re.search(r"rank_(\d+)_", Path(path).name)
    if not match:
        return None
    return int(match.group(1))


def _load_mesh_by_rank(run_dir: Path, artifact_root: Path) -> dict[int, dict[str, Any]]:
    for path in (
        artifact_root / "mesh_quality.json",
        artifact_root / "postanalysis" / "mesh_quality.json",
        run_dir / "postanalysis" / "mesh_quality.json",
    ):
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out: dict[int, dict[str, Any]] = {}
        for row in rows:
            rank = _rank_from_path(str(row.get("path", "")))
            if rank is not None:
                out[rank] = row
        return out
    return {}


def _aggregate_sensitivity(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    by_rank: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            rank = int(float(row["rank"]))
        except (KeyError, ValueError):
            continue
        by_rank.setdefault(rank, []).append(row)
    out: dict[int, dict[str, float]] = {}
    for rank, items in by_rank.items():
        n = max(1, len(items))
        out[rank] = {
            "sensitivity_scenarios": float(len(items)),
            "sensitivity_physics_survival_rate": sum(1 for item in items if _b(item, "survives_physics")) / n,
            "sensitivity_impact_survival_rate": sum(1 for item in items if _b(item, "survives_impact")) / n,
            "sensitivity_worst_energy_margin_J": min(_f(item, "energy_margin_J", math.inf) for item in items),
            "sensitivity_worst_impact_absorbed_J": min(_f(item, "impact_absorbed_J", math.inf) for item in items),
            "sensitivity_max_failure_risk": max(_f(item, "failure_risk", -math.inf) for item in items),
            "sensitivity_max_impact_risk": max(_f(item, "impact_risk", -math.inf) for item in items),
            "sensitivity_max_peak_force_N": max(_f(item, "peak_force_N", -math.inf) for item in items),
        }
    return out


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank_values(xs), _rank_values(ys))


def _correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pairs = [
        ("surrogate_score", "physics_score"),
        ("surrogate_score", "impact_score"),
        ("physics_score", "impact_score"),
        ("surrogate_peak_N", "dynamic_peak_N"),
        ("physics_peak_N", "dynamic_peak_N"),
        ("dynamic_absorbed_J", "explicit_absorbed_J"),
        ("dynamic_peak_N", "explicit_peak_N"),
        ("mass_g", "dynamic_absorbed_J"),
    ]
    out = []
    for left, right in pairs:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            x = _f(row, left)
            y = _f(row, right)
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
        out.append(
            {
                "metric_a": left,
                "metric_b": right,
                "n": len(xs),
                "pearson": _pearson(xs, ys),
                "spearman": _spearman(xs, ys),
            }
        )
    return out


def _robust_score(row: dict[str, object], config: EvidenceConfig) -> float:
    mass = _f(row, "mass_g", 1e3)
    physics_energy = _f(row, "physics_energy_usable_J", 0.0)
    dynamic_energy = _f(row, "dynamic_absorbed_J", 0.0)
    explicit_energy = _f(row, "explicit_absorbed_J", math.nan)
    dynamic_peak = _f(row, "dynamic_peak_N", math.inf)
    sensitivity_rate = _f(row, "sensitivity_impact_survival_rate", 0.0)
    energy_terms = [physics_energy, dynamic_energy]
    if config.require_explicit_fem:
        energy_terms.append(explicit_energy)
    finite_energy = [x for x in energy_terms if math.isfinite(x)]
    energy_reference = min(finite_energy) if finite_energy else 0.0
    score = (
        mass
        + 70.0 * max(0.0, config.target_energy_j - energy_reference)
        + 0.025 * max(0.0, dynamic_peak - config.peak_limit_n)
        + 45.0 * max(0.0, 1.0 - sensitivity_rate)
    )
    explicit_peak = _f(row, "explicit_peak_N", math.nan)
    damage = _f(row, "explicit_damage_proxy", math.nan)
    if config.require_explicit_fem:
        score += 0.010 * max(0.0, explicit_peak - config.explicit_peak_limit_n)
        score += 35.0 * (damage if math.isfinite(damage) else 1.0)
    elif math.isfinite(explicit_energy):
        score += 0.002 * max(0.0, explicit_peak - config.explicit_peak_limit_n)
        score += 6.0 * max(0.0, damage - config.max_damage)
    return score


def _dynamic_available_crush_pass(row: dict[str, object], config: EvidenceConfig) -> bool:
    if "dynamic_available_crush_pass" in row:
        return _truthy(row.get("dynamic_available_crush_pass"))
    if "dynamic_crush_distance_pass" in row:
        return _truthy(row.get("dynamic_crush_distance_pass"))
    return _f(row, "physics_collapse_mm", 0.0) >= config.min_crush_mm


def _failure_reasons(row: dict[str, object], config: EvidenceConfig) -> list[str]:
    reasons: list[str] = []
    physics_energy = _f(row, "physics_energy_usable_J", math.nan)
    dynamic_energy = _f(row, "dynamic_absorbed_J", math.nan)
    explicit_energy = _f(row, "explicit_absorbed_J", math.nan)
    energy_terms = [physics_energy, dynamic_energy]
    if config.require_explicit_fem:
        energy_terms.append(explicit_energy)
    energies = [x for x in energy_terms if math.isfinite(x)]
    if energies and max(energies) < 0.95 * config.target_energy_j:
        reasons.append("energy_shortfall")

    physics_peak = _f(row, "physics_peak_N", -math.inf)
    dynamic_peak = _f(row, "dynamic_peak_N", -math.inf)
    explicit_peak = _f(row, "explicit_peak_N", -math.inf)
    if (
        physics_peak > config.peak_limit_n
        or dynamic_peak > config.peak_limit_n
        or (config.require_explicit_fem and explicit_peak > config.explicit_peak_limit_n)
    ):
        reasons.append("peak_force")

    physics_crush = _f(row, "physics_collapse_mm", math.inf)
    explicit_crush = _f(row, "explicit_max_displacement_mm", math.inf)
    if (
        physics_crush < config.min_crush_mm
        or not _dynamic_available_crush_pass(row, config)
        or (config.require_explicit_fem and explicit_crush < config.min_crush_mm)
    ):
        reasons.append("early_bottom_out")

    explicit_damage = _f(row, "explicit_damage_proxy", 0.0)
    vector_damage = _f(row, "vector_damage_proxy", 0.0)
    if (config.require_explicit_fem and explicit_damage >= config.max_damage) or vector_damage >= config.max_damage:
        reasons.append("damage_localization")

    if row.get("mesh_known") is False:
        reasons.append("mesh_missing")
    elif row.get("mesh_pass") is not True:
        reasons.append("mesh_invalid")

    scenarios = _f(row, "sensitivity_scenarios", 0.0)
    if scenarios <= 0.0:
        reasons.append("sensitivity_missing")
    elif row.get("sensitivity_pass") is not True:
        reasons.append("sensitivity_unstable")

    if _f(row, "hex_elements", 0.0) <= 0.0:
        reasons.append("hex_missing")
    elif row.get("hex_mesh_pass") is not True:
        reasons.append("hex_invalid")

    if not reasons and row.get("candidate_decision_pass") is not True:
        reasons.append("unclassified")
    return reasons


def build_evidence_stack(
    run_dir: Path,
    out_dir: Path | None = None,
    artifact_root: Path | None = None,
    top_n: int = 500,
    config: EvidenceConfig | None = None,
) -> dict[str, object]:
    """Join all available diagnostic artifacts into one decision table."""

    config = config or EvidenceConfig()
    run_dir = Path(run_dir)
    artifact_root = Path(artifact_root) if artifact_root else run_dir
    out_dir = ensure_dir(Path(out_dir) if out_dir else artifact_root / "evidence")

    top_csv = run_dir / "postanalysis" / "top_candidates.csv"
    top_rows = _read_csv(top_csv)[:top_n]
    physics = _index_by_rank(_read_csv(_artifact_path(run_dir, artifact_root, "physics_gate", "physics_candidates.csv")))
    dynamic = _index_by_rank(_read_csv(_artifact_path(run_dir, artifact_root, "dynamic_impact", "dynamic_impact_candidates.csv")))
    explicit = _index_by_rank(
        _read_csv(_artifact_path(run_dir, artifact_root, "explicit_vector_impact", "explicit_vector_impact_candidates.csv"))
    )
    voxel = _index_by_rank(_read_csv(_artifact_path(run_dir, artifact_root, "neural_fem_precheck", "voxel_fem_candidates.csv")))
    vector = _index_by_rank(_read_csv(_artifact_path(run_dir, artifact_root, "vector_fem", "vector_fem_candidates.csv")))
    hex_fem = _index_by_rank(_read_csv(_artifact_path(run_dir, artifact_root, "hex_fem", "hex_fem_candidates.csv")))
    sensitivity = _aggregate_sensitivity(_read_csv(_artifact_path(run_dir, artifact_root, "sensitivity", "sensitivity_sweep.csv")))
    mesh = _load_mesh_by_rank(run_dir, artifact_root)

    rows: list[dict[str, object]] = []
    for top in top_rows:
        try:
            rank = int(float(top["rank"]))
        except (KeyError, ValueError):
            continue
        ph = physics.get(rank)
        dy = dynamic.get(rank)
        ex = explicit.get(rank)
        vx = voxel.get(rank)
        ve = vector.get(rank)
        hx = hex_fem.get(rank)
        se = sensitivity.get(rank, {})
        me = mesh.get(rank)

        mass = _f(top, "mass_g_mean", _f(ph, "mass_g", math.nan))
        energy_lcb = _f(top, "energy_abs_J_lcb")
        physics_energy = _f(ph, "physics_energy_usable_J")
        physics_peak = _f(ph, "physics_peak_force_N")
        physics_crush = _f(ph, "physics_collapse_mm")
        dynamic_energy = _f(dy, "impact_absorbed_J")
        dynamic_peak = _f(dy, "impact_peak_force_N")
        explicit_energy = _f(ex, "impact_absorbed_J")
        explicit_peak = _f(ex, "peak_contact_force_N")
        explicit_damage = _f(ex, "damage_proxy", 1.0)
        explicit_balance = abs(_f(ex, "energy_balance_error_J", math.inf))
        sensitivity_impact_rate = float(se.get("sensitivity_impact_survival_rate", 0.0))
        sensitivity_worst_energy = float(se.get("sensitivity_worst_impact_absorbed_J", -math.inf))

        mesh_known = me is not None
        mesh_pass = bool(mesh_known and me.get("exists") and me.get("within_envelope") is True and me.get("watertight_by_edges") is True)
        hex_elements = _f(hx, "elements", 0.0)
        hex_pass = bool(hex_elements >= config.min_hex_elements and _f(hx, "reaction_force_N", 0.0) > 0.0)

        physics_conservative = (
            physics_energy >= config.target_energy_j
            and physics_peak <= config.peak_limit_n
            and physics_crush >= config.min_crush_mm
            and _f(ph, "physics_failure_risk", 1.0) <= config.max_failure_risk
        )
        dynamic_available_crush_pass = (
            _b(dy, "impact_crush_distance_pass")
            if dy is not None and "impact_crush_distance_pass" in dy
            else physics_crush >= config.min_crush_mm
        )
        dynamic_conservative = (
            dynamic_energy >= 0.98 * config.target_energy_j
            and dynamic_peak <= config.peak_limit_n
            and dynamic_available_crush_pass
            and _f(dy, "impact_failure_risk", 1.0) <= config.max_failure_risk
        )
        explicit_conservative = (
            explicit_energy >= 0.95 * config.target_energy_j
            and explicit_peak <= config.explicit_peak_limit_n
            and explicit_damage < config.max_damage
            and explicit_balance <= config.max_energy_balance_abs_j
        )
        explicit_required_pass = explicit_conservative if config.require_explicit_fem else True
        explicit_diagnostic_warning = ex is not None and not explicit_conservative
        sensitivity_pass = sensitivity_impact_rate >= 0.8 and sensitivity_worst_energy >= 0.90 * config.target_energy_j

        row: dict[str, object] = {
            "rank": rank,
            "topology": top.get("topology", ""),
            "mass_g": mass,
            "surrogate_score": _f(top, "score"),
            "surrogate_energy_lcb_J": energy_lcb,
            "surrogate_safe_energy_pass": energy_lcb >= config.safe_energy_j,
            "surrogate_peak_N": _f(top, "force_peak_N_mean"),
            "surrogate_failure_probability": _f(top, "failure_probability_mean"),
            "surrogate_collapse_mm": _f(top, "collapse_displacement_mm_mean"),
            "physics_energy_usable_J": physics_energy,
            "physics_peak_N": physics_peak,
            "physics_collapse_mm": physics_crush,
            "physics_impact_stop_mm": _f(ph, "physics_impact_stop_mm"),
            "physics_failure_risk": _f(ph, "physics_failure_risk"),
            "physics_score": _f(ph, "physics_score"),
            "physics_conservative_pass": physics_conservative,
            "dynamic_absorbed_J": dynamic_energy,
            "dynamic_peak_N": dynamic_peak,
            "dynamic_max_displacement_mm": _f(dy, "impact_max_displacement_mm"),
            "dynamic_crush_distance_pass": _b(dy, "impact_crush_distance_pass"),
            "dynamic_available_crush_pass": dynamic_available_crush_pass,
            "dynamic_failure_risk": _f(dy, "impact_failure_risk"),
            "dynamic_energy_balance_error_J": _f(dy, "impact_energy_balance_error_J"),
            "impact_score": _f(dy, "impact_score"),
            "dynamic_conservative_pass": dynamic_conservative,
            "explicit_absorbed_J": explicit_energy,
            "explicit_peak_N": explicit_peak,
            "explicit_max_displacement_mm": _f(ex, "max_indenter_displacement_mm"),
            "explicit_damage_proxy": explicit_damage,
            "explicit_failed_springs_fraction": _f(ex, "failed_springs_fraction"),
            "explicit_energy_balance_abs_J": explicit_balance,
            "explicit_fem_score": _f(ex, "explicit_fem_score"),
            "explicit_conservative_pass": explicit_conservative,
            "explicit_required_pass": explicit_required_pass,
            "explicit_diagnostic_warning": explicit_diagnostic_warning,
            "voxel_damage_proxy": _f(vx, "damage_proxy"),
            "vector_reaction_N": _f(ve, "reaction_force_N"),
            "vector_damage_proxy": _f(ve, "damage_proxy"),
            "vector_p95_von_mises_MPa": _f(ve, "p95_von_mises_MPa"),
            "hex_elements": hex_elements,
            "hex_reaction_N": _f(hx, "reaction_force_N"),
            "hex_damage_proxy": _f(hx, "damage_proxy"),
            "hex_convergence_error": _f(hx, "convergence_error"),
            "hex_mesh_pass": hex_pass,
            "mesh_known": mesh_known,
            "mesh_pass": mesh_pass,
            "sensitivity_scenarios": se.get("sensitivity_scenarios", 0.0),
            "sensitivity_impact_survival_rate": sensitivity_impact_rate,
            "sensitivity_physics_survival_rate": se.get("sensitivity_physics_survival_rate", 0.0),
            "sensitivity_worst_energy_margin_J": se.get("sensitivity_worst_energy_margin_J", math.nan),
            "sensitivity_worst_impact_absorbed_J": sensitivity_worst_energy,
            "sensitivity_max_failure_risk": se.get("sensitivity_max_failure_risk", math.nan),
            "sensitivity_max_impact_risk": se.get("sensitivity_max_impact_risk", math.nan),
            "sensitivity_max_peak_force_N": se.get("sensitivity_max_peak_force_N", math.nan),
            "sensitivity_pass": sensitivity_pass,
        }
        row["robust_score"] = _robust_score(row, config)
        row["candidate_decision_pass"] = (
            physics_conservative and dynamic_conservative and explicit_required_pass and sensitivity_pass and mesh_pass and hex_pass
        )
        reasons = [] if row["candidate_decision_pass"] else _failure_reasons(row, config)
        row["primary_failure_reason"] = reasons[0] if reasons else ""
        row["failure_reasons"] = ";".join(reasons)
        rows.append(row)

    rows = sorted(rows, key=lambda item: _f(item, "robust_score", math.inf))
    fieldnames = list(rows[0].keys()) if rows else [
        "rank",
        "topology",
        "mass_g",
        "robust_score",
        "candidate_decision_pass",
    ]
    evidence_csv = out_dir / "candidate_evidence_stack.csv"
    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _s(row.get(key)) for key in fieldnames})

    corr_rows = _correlation_rows(rows)
    corr_csv = out_dir / "fidelity_rank_correlations.csv"
    with corr_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric_a", "metric_b", "n", "pearson", "spearman"])
        writer.writeheader()
        for row in corr_rows:
            writer.writerow({key: _s(row.get(key)) for key in ["metric_a", "metric_b", "n", "pearson", "spearman"]})

    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("primary_failure_reason") or "pass")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_csv = out_dir / "failure_reason_summary.csv"
    with reason_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["primary_failure_reason", "count"])
        writer.writeheader()
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"primary_failure_reason": reason, "count": count})

    summary = {
        "run_dir": str(run_dir),
        "artifact_root": str(artifact_root),
        "out_csv": str(evidence_csv),
        "correlations_csv": str(corr_csv),
        "failure_reason_summary_csv": str(reason_csv),
        "config": asdict(config),
        "evaluated": len(rows),
        "candidate_decision_pass": sum(1 for row in rows if row.get("candidate_decision_pass") is True),
        "physics_conservative_pass": sum(1 for row in rows if row.get("physics_conservative_pass") is True),
        "dynamic_conservative_pass": sum(1 for row in rows if row.get("dynamic_conservative_pass") is True),
        "explicit_conservative_pass": sum(1 for row in rows if row.get("explicit_conservative_pass") is True),
        "sensitivity_pass": sum(1 for row in rows if row.get("sensitivity_pass") is True),
        "mesh_pass": sum(1 for row in rows if row.get("mesh_pass") is True),
        "hex_mesh_pass": sum(1 for row in rows if row.get("hex_mesh_pass") is True),
        "failure_reason_counts": reason_counts,
        "best": rows[0] if rows else None,
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    lines = [
        "# POLMI Candidate Evidence Stack",
        "",
        f"Run dir: `{run_dir}`",
        f"Artifact root: `{artifact_root}`",
        f"Evaluated: `{len(rows)}`",
        f"Decision-pass candidates: `{summary['candidate_decision_pass']}`",
        f"Failure reason summary: `{reason_csv}`",
        "",
        "| Robust | Rank | Topology | Mass g | Phys pass | Dyn pass | Explicit pass | Sens pass | Mesh pass | Robust score |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows[:20], start=1):
        lines.append(
            f"| {i} | {row['rank']} | {row['topology']} | {_s(row['mass_g'])} | "
            f"{row['physics_conservative_pass']} | {row['dynamic_conservative_pass']} | "
            f"{row['explicit_conservative_pass']} | {row['sensitivity_pass']} | "
            f"{row['mesh_pass']} | {_s(row['robust_score'])} |"
        )
    summary_md = out_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_json)
    summary["summary_md"] = str(summary_md)
    return _json_safe(summary)  # type: ignore[return-value]
