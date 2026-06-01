"""Post-training search and candidate selection."""

from __future__ import annotations

import csv
import heapq
import json
import random
from pathlib import Path
from typing import Any

from .design_space import (
    SAFE_ENERGY_J,
    SCALAR_TARGET_FIELDS,
    DesignParams,
    row_to_named_features,
    sample_designs,
)
from .paths import ensure_dir
from .stl import export_design_stl


def _load_checkpoints(run_dir: Path, device: str):
    import torch

    from .torch_models import PolmiSurrogate

    checkpoints = sorted(run_dir.glob("model_seed*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no model_seed*.pt checkpoints in {run_dir}")
    loaded = []
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
        model = PolmiSurrogate(
            input_dim=len(ckpt["feature_names"]),
            output_dim=len(ckpt["target_names"]),
            hidden_dim=int(ckpt["config"]["hidden_dim"]),
            blocks=int(ckpt["config"]["blocks"]),
            dropout=0.0,
        ).to(torch_device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        loaded.append((model, ckpt))
    return loaded, torch_device


def _score_candidate(pred: dict[str, float]) -> float:
    energy_lcb = pred["energy_abs_J_lcb"]
    mass = pred["mass_g_mean"]
    failure = pred["failure_probability_mean"]
    uncertainty = pred["energy_abs_J_std"]
    collapse = pred["collapse_displacement_mm_mean"]
    peak = pred["force_peak_N_mean"]
    plateau = max(1.0, pred.get("force_plateau_N_mean", peak))
    peak_ratio = pred.get("peak_plateau_ratio_mean", peak / plateau)
    early_energy = pred.get("early_energy_20mm_J_mean", 0.55 * energy_lcb)
    progressive = pred.get("progressive_crush_score_mean", 0.0)
    return (
        mass
        + 35.0 * max(0.0, SAFE_ENERGY_J - energy_lcb)
        + 95.0 * failure
        + 3.0 * uncertainty
        + 9.0 * max(0.0, 23.0 - mass)
        + 1.2 * max(0.0, mass - 30.0)
        + 7.0 * max(0.0, 40.0 - collapse)
        + 0.035 * max(0.0, peak - 1400.0)
        + 24.0 * max(0.0, peak_ratio - 1.35)
        + 1.4 * max(0.0, early_energy - 0.48 * max(energy_lcb, 1.0))
        + 18.0 * max(0.0, 0.70 - progressive)
    )


def _prediction_dict(target_names: list[str], mean_values: list[float], std_values: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in SCALAR_TARGET_FIELDS:
        if name not in target_names:
            continue
        idx = target_names.index(name)
        mean_value = mean_values[idx]
        std_value = max(0.0, std_values[idx])
        if name == "failure_probability":
            mean_value = max(0.0, min(1.0, mean_value))
        elif name in {
            "mass_g",
            "relative_density",
            "energy_abs_J",
            "force_peak_N",
            "force_plateau_N",
            "early_energy_20mm_J",
            "peak_plateau_ratio",
            "progressive_crush_score",
            "collapse_displacement_mm",
        }:
            mean_value = max(0.0, mean_value)
        if name == "progressive_crush_score":
            mean_value = max(0.0, min(1.0, mean_value))
        out[f"{name}_mean"] = mean_value
        out[f"{name}_std"] = std_value
        if name == "energy_abs_J":
            out[f"{name}_lcb"] = mean_value - 1.64 * std_value
    if "energy_abs_J_lcb" not in out:
        out["energy_abs_J_lcb"] = out.get("energy_abs_J_mean", 0.0)
        out["energy_abs_J_std"] = out.get("energy_abs_J_std", 0.0)
    if "force_plateau_N_mean" not in out:
        out["force_plateau_N_mean"] = max(1.0, out.get("force_peak_N_mean", 1.0))
    if "peak_plateau_ratio_mean" not in out:
        out["peak_plateau_ratio_mean"] = out.get("force_peak_N_mean", 1.0) / max(1.0, out["force_plateau_N_mean"])
    if "early_energy_20mm_J_mean" not in out:
        out["early_energy_20mm_J_mean"] = 0.55 * out.get("energy_abs_J_lcb", 0.0)
    if "progressive_crush_score_mean" not in out:
        out["progressive_crush_score_mean"] = 0.0
    return out


def run_postanalysis(
    run_dir: Path,
    search_n: int = 250_000,
    seed: int = 20260501,
    batch_size: int = 8192,
    top_k: int = 100,
    export_stl_count: int = 10,
    stl_backend: str = "auto",
    stl_resolution: int = 128,
    stl_format: str = "ascii",
    device: str = "cuda",
) -> dict[str, Any]:
    import torch

    loaded, torch_device = _load_checkpoints(run_dir, device=device)
    first_ckpt = loaded[0][1]
    target_names = list(first_ckpt["target_names"])
    model_feature_names = list(first_ckpt["feature_names"])
    out_dir = ensure_dir(run_dir / "postanalysis")
    stl_dir = ensure_dir(out_dir / "stl")
    rng = random.Random(seed)
    heap: list[tuple[float, int, dict[str, str]]] = []
    counter = 0

    for start in range(0, search_n, batch_size):
        n = min(batch_size, search_n - start)
        designs = sample_designs(n, seed=rng.randint(0, 2_000_000_000))
        rows = [d.to_row() for d in designs]
        x = torch.tensor([row_to_named_features(r, model_feature_names) for r in rows], dtype=torch.float32, device=torch_device)

        preds = []
        with torch.no_grad():
            for model, ckpt in loaded:
                xz = (x - ckpt["x_mean"].to(torch_device)) / ckpt["x_std"].to(torch_device)
                yz = model(xz)
                y = yz * ckpt["y_std"].to(torch_device) + ckpt["y_mean"].to(torch_device)
                preds.append(y)
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0).cpu()
        std = stack.std(dim=0, unbiased=False).cpu()
        for i, params in enumerate(designs):
            pred = _prediction_dict(target_names, mean[i].tolist(), std[i].tolist())
            score = _score_candidate(pred)
            row = params.to_row()
            row.update({k: f"{v:.8g}" for k, v in pred.items()})
            row["score"] = f"{score:.8g}"
            row["rank_source"] = "ensemble_lcb"
            item = (-score, counter, row)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            counter += 1
        if start == 0 or (start + n) % max(batch_size, search_n // 10) == 0:
            print(f"[postanalysis] searched {start+n}/{search_n}", flush=True)

    selected = [item[2] for item in sorted(heap, key=lambda x: -x[0])]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = str(rank)

    fieldnames = ["rank", "score"] + list(selected[0].keys() - {"rank", "score"}) if selected else ["rank", "score"]
    preferred = [
        "rank",
        "score",
        "topology",
        "cell_size_mm",
        "wall_thickness_mm",
        "min_feature_mm",
        "vertical_gradient",
        "density_bias",
        "cap_thickness_mm",
        "edge_rib_mm",
        "trigger_layer_strength",
        "trigger_zone_mm",
        "plateau_zone_mm",
        "bumper_zone_mm",
        "hybrid_ratio",
        "anisotropy_xy",
        "mass_g_mean",
        "energy_abs_J_mean",
        "energy_abs_J_std",
        "energy_abs_J_lcb",
        "failure_probability_mean",
        "collapse_displacement_mm_mean",
        "force_peak_N_mean",
        "force_plateau_N_mean",
        "early_energy_20mm_J_mean",
        "peak_plateau_ratio_mean",
        "progressive_crush_score_mean",
    ]
    extras = [x for x in selected[0].keys() if x not in preferred] if selected else []
    fieldnames = preferred + extras
    top_csv = out_dir / "top_candidates.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    exported = []
    for row in selected[:export_stl_count]:
        params = DesignParams.from_row(row)
        stl_path = stl_dir / f"rank_{int(row['rank']):03d}_{params.topology}.stl"
        export_design_stl(params, stl_path, backend=stl_backend, resolution=stl_resolution, stl_format=stl_format)
        exported.append(str(stl_path))

    md_lines = [
        "# POLMI Postanalysis",
        "",
        f"Search candidates: `{search_n}`",
        f"Top K: `{top_k}`",
        f"Exported STL: `{len(exported)}`",
        "",
        "| Rank | Score | Topology | Mass g | E LCB J | E mean J | Failure | Collapse mm |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected[:20]:
        md_lines.append(
            "| {rank} | {score} | {topology} | {mass_g_mean} | {energy_abs_J_lcb} | "
            "{energy_abs_J_mean} | {failure_probability_mean} | {collapse_displacement_mm_mean} |".format(**row)
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    payload = {"top_csv": str(top_csv), "exported_stl": exported, "search_n": search_n, "top_k": top_k}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
