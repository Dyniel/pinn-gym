# Reproducing the paper

These are the exact commands behind the figures and tables in the Scientific
Reports manuscript. They assume a checkout of `pinn-gym` and an HPC cluster
with Slurm. For a CPU-only smoke check, use
[`configs/smoke.yaml`](../configs/smoke.yaml).

## Hardware assumptions

- CPU node with ≥ 16 cores and 48 GB RAM for the `build` stage.
- One A100 (40 GB) per material for the `train` stage. Five materials run
  comfortably as a Slurm job array.
- A single A100 (or any GPU with ≥ 16 GB) for `evaluate`.

## End-to-end run

```bash
# From a fresh checkout:
python -m pip install -e ".[torch,dev]"

# Submit the whole pipeline. The Slurm template reads the YAML and dispatches
# stages with the correct resource profile.
sbatch slurm/pinn_gym.slurm configs/sr_full.yaml
```

Output lands in `runs/sr_full_<timestamp>/`. Every figure in the manuscript
maps to a file under `runs/sr_full_*/figures/`; every table maps to a CSV
under `runs/sr_full_*/tables/`.

## Stage-by-stage (for debugging)

```bash
# 1. Candidate pools + oracle (CPU, ~12 h on 16 cores).
pinn-gym run configs/sr_full.yaml --stage build

# 2. Train per-material PINNs + pooled model (GPU array).
pinn-gym run configs/sr_full.yaml --stage train

# 3. Evaluate + transfer matrix (GPU, ~30 min).
pinn-gym run configs/sr_full.yaml --stage evaluate

# 4. Re-render plots without re-running anything else.
pinn-gym run configs/sr_full.yaml --stage plots

# 5. STL audit on the top-k feasible designs.
pinn-gym run configs/sr_full.yaml --stage audit
```

Each subsequent invocation creates a **new** run directory; pin a specific run
by passing `run.output_root` in the YAML or by editing the resolved
`manifest.json` of an existing run.

## Figure → file map

| Figure | File | Stage |
|---|---|---|
| Force-displacement overlays per material | `figures/curves_overlay.{pdf,png}` | plots |
| Curve NRMSE distribution per method | `figures/nrmse_distribution.{pdf,png}` | plots |
| Energy-integral MAE per method × material | `figures/energy_error.{pdf,png}` | plots |
| Violation-rate bars | `figures/violation_bars.{pdf,png}` | plots |
| Precision@k vs k | `figures/precision_at_k.{pdf,png}` | plots |
| Regret@k vs k | `figures/regret_at_k.{pdf,png}` | plots |
| Robustness/sensitivity survival | `figures/robustness_survival.{pdf,png}` | plots |

## Table → file map

| Table | File |
|---|---|
| Per-method, per-material metrics | `tables/method_metrics.csv` |
| Cross-material transfer matrix | `tables/transfer_metrics.csv` |
| STL audit results | `tables/mesh_quality.csv` |
| Markdown summary printed in the paper appendix | `tables/summary.md` |

## Random seeds

All randomness flows from `run.seed`. The paper run uses `seed: 20260520`.
Different seeds will perturb individual numbers but should preserve the
qualitative ordering of methods within each material card.

## Data availability

`data/dummy/` ships a tiny sample sufficient to exercise the build stage on a
laptop. The full processed datasets (~30 GB) are not in git; see
[`../data/README.md`](../data/README.md) for the symlink / download procedure.
