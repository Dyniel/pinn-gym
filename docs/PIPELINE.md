# Pipeline

This document describes what each stage of `pinn-gym run config.yaml` does,
the artefacts it writes, and how the stages depend on each other.

## Run directory layout

Every invocation creates a fresh directory:

```
runs/<name>_<YYYYMMDD_HHMMSS>/
├── manifest.json         # full resolved config + version + timestamp
├── datasets/             # build stage output
├── checkpoints/          # train stage output
├── figures/              # plots stage output
├── tables/               # evaluate stage output (CSV + Markdown)
└── logs/                 # per-stage stdout/stderr capture
```

`manifest.json` is the source of truth: it contains the validated, fully
materialised configuration so that runs can be replayed exactly.

## Stages

```
build → train → evaluate → plots
                        ↘ audit
```

`build` is a pure CPU stage and must run first when datasets are not already
present. `train`, `evaluate`, `plots`, and `audit` can be re-run on existing
datasets using `--stage` overrides.

### 1. `build`

For each material preset:

1. Sample a material-aware candidate pool of size `train_n + eval_n` with a
   per-material RNG.
2. Score every candidate with the declared progressive-crush oracle.
3. Score the dynamic impact response with the rigid-indenter integrator.
4. Write `datasets/<preset>/train.csv`, `datasets/<preset>/eval.csv`, and a
   `summary.json` with feasibility statistics.

The oracle is pure-Python and embarrassingly parallel across designs;
`candidate_pool.oracle_workers` controls the multiprocessing pool size.

### 2. `train`

For each method in `train.methods` × each material × (optionally pooled):

1. Read `datasets/<preset>/train.csv`.
2. Construct dimensionless features per material card.
3. Train the model with the configured loss weights.
4. Write `checkpoints/<method>/<preset>/model.pt` and training history.

Baselines (`random`, `lightest`, `pseudo_bootstrap`) write a trivial checkpoint
that captures their decision rule, but no actual gradient step runs.

This is the only GPU-heavy stage. On Slurm, use the array template
[`slurm/pinn_gym.slurm`](../slurm/pinn_gym.slurm) which launches one task per
material.

### 3. `evaluate`

1. Load each method's checkpoint, run forward passes on the eval pool, predict
   force curves and integrate energies.
2. Compute curve RMSE, NRMSE, energy MAE, violation rate, predicted-feasible
   rate, oracle-feasible rate, precision@k, regret@k, robustness/sensitivity
   survival.
3. Stratify every metric by material card. Macro-averages are reported only
   when each card has both feasible and infeasible examples.
4. If `include_transfer_matrix=true`, evaluate each per-material model on every
   other material's eval pool and emit a transfer matrix.
5. Write `tables/method_metrics.csv`, `tables/transfer_metrics.csv`, and
   `tables/summary.md`.

### 4. `plots`

Renders the configured figures from the evaluate-stage CSVs. No model
inference happens here; if you want to refresh plots after evaluate, run:

```bash
pinn-gym run configs/sr_full.yaml --stage plots
```

### 5. `audit`

For the top-k feasible candidates per material:

1. Export an STL using the configured backend (`voxel` or `implicit`).
2. Audit watertightness, edge consistency, envelope, and degeneracies.
3. Write `tables/mesh_quality.csv` and store STLs under `audit/<preset>/`.

## Material cards

A material card is a declared numerical object, not a universal truth about a
polymer. It includes:

- density (`rho_kg_m3`)
- elastic modulus / initial stiffness
- plateau / crush stress model
- densification onset and tangent
- strain-rate amplification
- failure strain (used by the material-aware crush target)
- manufacturing constraints (minimum feature size, process tolerance)
- uncertainty ranges for robustness sweeps

Cards live in `pinn_gym.core.materials`. Add a new card by writing a JSON file
and listing its path in `materials.custom_cards`.

## Non-claims

`pinn-gym` reports numerical-oracle behaviour, not certified FEM or
experimentally validated survival. The progressive-crush oracle represents
distributed crushing and densification, not local failure events.

Cross-material transfer is presented as a separate experiment and labelled as
such in every figure and table. The pipeline does **not** generalise a model
trained on one material's pool to other materials as the main benchmark.

## POLMI Port Status

The scientific core has been ported from the original POLMI challenge
workspace into `pinn_gym.core`. The public CLI and YAML schema now call the
ported modules directly.

Reviewer-response helpers are available through:

```bash
pinn-gym reviewer-pack --run-dir simulations/results/reviewer_experiments_21392657 --out reviewer_pack
pinn-gym verify-pack reviewer_pack
```

`pinn-gym run --dry-run` remains useful for CI because it validates the config,
creates the resolved run directory and writes a manifest without requiring GPU
or Slurm access.
