# pinn-gym

**A reproducible numerical gym for physics-informed design of printable,
energy-absorbing lattice metamaterials.**

The repository contains the public, lightweight code path. Heavy artefacts
from the POLMI challenge workspace are published separately in the
`v0.1.0-full-dump` GitHub release so a clone stays small while the complete
run can still be reconstructed.

## Reviewer Quickstart

```bash
git clone https://github.com/Dyniel/pinn-gym
cd pinn-gym
python -m pip install -e ".[dev]"
pytest -q
pinn-gym validate configs/smoke.yaml
```

Build the canonical reviewer pack from an unpacked full dump:

```bash
pinn-gym reviewer-pack \
  --run-dir simulations/results/reviewer_experiments_21392657 \
  --out reviewer_pack
pinn-gym verify-pack reviewer_pack
```

Download the full dump from the release:

```bash
gh release download v0.1.0-full-dump \
  --repo Dyniel/pinn-gym \
  --dir full_dump
cd full_dump
sha256sum -c FULL_DUMP_SHA256SUMS
```

See [docs/FULL_DUMP.md](docs/FULL_DUMP.md) for split-archive reconstruction
and [docs/REVIEWER_PACK.md](docs/REVIEWER_PACK.md) for the exact reviewer
tables.

## What It Does

`pinn-gym` runs a declared numerical-oracle benchmark:

1. Sample material-aware lattice candidate pools.
2. Score force-displacement curves with a reduced-order progressive-crush oracle.
3. Train per-material and pooled PINN or MLP surrogates.
4. Report curve nRMSE, energy error, violation rate, precision@k and regret@k.
5. Run reviewer experiments: seed repeats, loss-weight ablations and pooled tuning.
6. Export and audit STL geometry when requested.

The main claim is numerical-oracle scoped. This is not a certified FEM or
experimental validation package.

## Common Commands

```bash
# Fast schema check.
pinn-gym validate configs/smoke.yaml

# Prepare a run directory and manifest without compute.
pinn-gym run configs/smoke.yaml --dry-run

# Laptop-scale smoke run.
pinn-gym run configs/smoke.yaml

# Paper-grade Slurm run.
sbatch slurm/pinn_gym.slurm configs/sr_full.yaml
```

The full run writes `runs/<name>_<timestamp>/` with datasets, checkpoints,
tables, figures, logs and optional mesh audits. Large outputs are intentionally
gitignored.

## Repository Layout

```text
pinn-gym/
├── configs/          YAML entrypoints for smoke and SR-scale runs
├── docs/             Reproduction, reviewer-pack and full-dump notes
├── slurm/            Cluster launcher
├── src/pinn_gym/     CLI, config schema, pipeline and scientific core
├── tests/            Unit and smoke tests ported from the POLMI workspace
└── runs/             Local run output, ignored except .gitkeep
```

## Full Dump

The release assets contain the bulky material needed to recreate the working
workspace: submission packs, reviewer pack, data, geometry, reports and
simulation outputs. The git repository stores manifests and instructions, not
multi-GB generated artefacts.

## License

MIT. See [LICENSE](LICENSE).
