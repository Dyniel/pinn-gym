"""Active refinement around already-promising POLMI parameter families."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .design_space import (
    SAFE_ENERGY_J,
    DesignParams,
    candidate_header,
    clamp,
    design_to_candidate_row,
    normalize_zones,
    pseudo_response,
)
from .paths import ensure_dir


TARGET_FAMILIES = {
    "schwarz_p",
    "diamond_graded",
    "gyroid_diamond_hybrid",
    "bccz_graded",
    "ot_like",
    "hybrid",
    "diamond",
}


@dataclass(frozen=True)
class ActiveRefinementConfig:
    seed: int = 20260514
    max_seeds: int = 80
    variants_per_seed: int = 24
    iterations: int = 2
    target_mass_low_g: float = 23.0
    target_mass_high_g: float = 30.0
    target_energy_lcb_J: float = SAFE_ENERGY_J
    mutation_scale: float = 0.16


def _float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    try:
        return float(row.get(name, default) or default)
    except ValueError:
        return default


def _seed_score(row: dict[str, str], config: ActiveRefinementConfig) -> float:
    mass = _float(row, "mass_g_mean", _float(row, "mass_g", 0.0))
    energy_lcb = _float(row, "energy_abs_J_lcb", _float(row, "energy_abs_J", 0.0))
    failure = _float(row, "failure_probability_mean", _float(row, "failure_probability", 0.35))
    collapse = _float(row, "collapse_displacement_mm_mean", _float(row, "collapse_displacement_mm", 0.0))
    peak = _float(row, "force_peak_N_mean", _float(row, "force_peak_N", 5000.0))
    plateau = max(1.0, _float(row, "force_plateau_N_mean", _float(row, "force_plateau_N", peak)))
    peak_ratio = _float(row, "peak_plateau_ratio_mean", peak / plateau)
    progressive = _float(row, "progressive_crush_score_mean", _float(row, "progressive_crush_score", 0.0))
    family_bonus = -5.0 if row.get("topology") in TARGET_FAMILIES else 12.0
    if config.target_mass_low_g <= mass <= config.target_mass_high_g:
        mass_penalty = 0.0
    else:
        target_mid = 0.5 * (config.target_mass_low_g + config.target_mass_high_g)
        mass_penalty = abs(mass - target_mid)
    return (
        family_bonus
        + 3.0 * mass_penalty
        + 30.0 * max(0.0, config.target_energy_lcb_J - energy_lcb)
        + 40.0 * failure
        + 4.0 * max(0.0, 40.0 - collapse)
        + 0.020 * max(0.0, peak - 1400.0)
        + 15.0 * max(0.0, peak_ratio - 1.35)
        + 12.0 * max(0.0, 0.70 - progressive)
    )


def _mutate(params: DesignParams, rng: random.Random, scale: float) -> DesignParams:
    def jitter(value: float, span: float, low: float, high: float) -> float:
        return clamp(value + rng.gauss(0.0, span * scale), low, high)

    min_feature = jitter(params.min_feature_mm, 0.12, 0.50, 0.86)
    wall = max(min_feature, jitter(params.wall_thickness_mm, 0.22, min_feature, 1.45))
    topology = params.topology
    if topology not in TARGET_FAMILIES:
        topology = rng.choice(sorted(TARGET_FAMILIES))
    elif rng.random() < 0.18:
        topology = rng.choice(["schwarz_p", "diamond_graded", "gyroid_diamond_hybrid", "bccz_graded", "ot_like"])
    trigger = jitter(params.trigger_zone_mm, 2.0, 7.0, 15.0)
    plateau = jitter(params.plateau_zone_mm, 3.5, 22.0, 35.0)
    bumper = jitter(params.bumper_zone_mm, 2.2, 8.0, 16.0)
    trigger, plateau, bumper = normalize_zones(trigger, plateau, bumper)
    return replace(
        params,
        topology=topology,
        cell_size_mm=jitter(params.cell_size_mm, 0.80, 4.2, 9.2),
        wall_thickness_mm=wall,
        min_feature_mm=min_feature,
        vertical_gradient=jitter(params.vertical_gradient, 0.22, 0.00, 1.00),
        density_bias=jitter(params.density_bias, 0.045, -0.04, 0.16),
        cap_thickness_mm=jitter(params.cap_thickness_mm, 0.18, 0.42, 1.05),
        edge_rib_mm=jitter(params.edge_rib_mm, 0.25, 0.0, 1.05),
        trigger_layer_strength=jitter(params.trigger_layer_strength, 0.22, 0.20, 1.00),
        trigger_zone_mm=trigger,
        plateau_zone_mm=plateau,
        bumper_zone_mm=bumper,
        hybrid_ratio=jitter(params.hybrid_ratio, 0.22, 0.0, 1.0),
        anisotropy_xy=jitter(params.anisotropy_xy, 0.10, 0.78, 1.28),
    )


def _refinement_score(params: DesignParams, config: ActiveRefinementConfig) -> float:
    response = pseudo_response(params)
    mass = float(response["mass_g"])
    energy = float(response["energy_abs_J"])
    peak = float(response["force_peak_N"])
    plateau = max(1.0, float(response["force_plateau_N"]))
    early_energy = float(response["early_energy_20mm_J"])
    peak_ratio = float(response["peak_plateau_ratio"])
    progressive = float(response["progressive_crush_score"])
    failure = float(response["failure_probability"])
    collapse = float(response["collapse_displacement_mm"])
    mass_window_penalty = max(0.0, config.target_mass_low_g - mass) + max(0.0, mass - config.target_mass_high_g)
    return (
        mass
        + 18.0 * mass_window_penalty
        + 35.0 * max(0.0, config.target_energy_lcb_J - energy)
        + 55.0 * failure
        + 7.0 * max(0.0, 40.0 - collapse)
        + 0.035 * max(0.0, peak - 1400.0)
        + 20.0 * max(0.0, peak_ratio - 1.35)
        + 1.2 * max(0.0, early_energy - 0.48 * max(energy, 1.0))
        + 16.0 * max(0.0, 0.70 - progressive)
    )


def run_active_refinement(top_csv: Path, out_dir: Path, config: ActiveRefinementConfig | None = None) -> dict[str, object]:
    """Create a focused local-search batch around progressive-crush elite families."""

    config = config or ActiveRefinementConfig()
    out_dir = ensure_dir(out_dir)
    with Path(top_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    filtered = [row for row in rows if row.get("topology") in TARGET_FAMILIES]
    if not filtered:
        filtered = rows
    seeds = sorted(filtered, key=lambda row: _seed_score(row, config))[: config.max_seeds]
    rng = random.Random(config.seed)
    active: list[tuple[float, DesignParams, int]] = []
    for seed_row in seeds:
        parent_rank = int(float(seed_row.get("rank", len(active) + 1)))
        parent = DesignParams.from_row(seed_row)
        active.append((_refinement_score(parent, config), parent, parent_rank))
        frontier = [parent]
        for iteration in range(config.iterations):
            scale = config.mutation_scale / (1.0 + 0.35 * iteration)
            candidates = [_mutate(item, rng, scale) for item in frontier for _ in range(config.variants_per_seed)]
            ranked = sorted(candidates, key=lambda params: _refinement_score(params, config))
            frontier = ranked[: max(2, min(8, len(ranked)))]
            for params in frontier:
                active.append((_refinement_score(params, config), params, parent_rank))

    ranked_active = sorted(active, key=lambda item: item[0])
    out_csv = out_dir / "active_refinement_candidates.csv"
    fieldnames = ["refine_rank", "parent_rank", "refine_score"] + candidate_header()
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for refine_rank, (score, params, parent_rank) in enumerate(ranked_active, start=1):
            row = design_to_candidate_row(params)
            row.update({"refine_rank": str(refine_rank), "parent_rank": str(parent_rank), "refine_score": f"{score:.8g}"})
            writer.writerow(row)

    best = []
    for score, params, parent_rank in ranked_active[:20]:
        response = pseudo_response(params)
        best.append(
            {
                "parent_rank": parent_rank,
                "topology": params.topology,
                "mass_g": response["mass_g"],
                "energy_abs_J": response["energy_abs_J"],
                "peak_plateau_ratio": response["peak_plateau_ratio"],
                "collapse_displacement_mm": response["collapse_displacement_mm"],
                "progressive_crush_score": response["progressive_crush_score"],
                "failure_probability": response["failure_probability"],
                "score": score,
            }
        )
    summary = {
        "top_csv": str(top_csv),
        "out_csv": str(out_csv),
        "config": asdict(config),
        "seed_rows": len(seeds),
        "refined_rows": len(ranked_active),
        "target_families": sorted(TARGET_FAMILIES),
        "best": best,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# POLMI Active Refinement",
        "",
        "Scope: local refinement around literature-guided progressive-crush families.",
        f"Target mass window: `{config.target_mass_low_g:.1f}-{config.target_mass_high_g:.1f} g`",
        f"Target conservative energy: `E_lcb >= {config.target_energy_lcb_J:.1f} J`",
        "",
        "| Rank | Parent | Topology | Mass g | Energy J | Collapse mm | Peak/plateau | Failure | Score |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, item in enumerate(best, start=1):
        md.append(
            f"| {i} | {item['parent_rank']} | {item['topology']} | {float(item['mass_g']):.2f} | "
            f"{float(item['energy_abs_J']):.2f} | {float(item['collapse_displacement_mm']):.1f} | "
            f"{float(item['peak_plateau_ratio']):.2f} | {float(item['failure_probability']):.3f} | {float(item['score']):.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary
