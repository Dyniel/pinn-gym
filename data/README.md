# Data

`pinn-gym` operates on three classes of data, only one of which is in this
repository.

```
data/
├── README.md        ← this file
├── dummy/           ← tiny synthetic sample, committed to git
├── raw/             ← gitignored — downloaded source datasets
├── extracted/       ← gitignored — unpacked archives
└── processed/       ← gitignored — derived tables consumed by build stage
```

## `dummy/`

A handful of synthetic rows that exercise the CSV schema and the basic
audit path without any real measurements. Use it for:

- CI smoke tests (`pytest tests/`).
- Verifying that a fresh checkout loads without errors.
- Documentation examples.

The dummy sample is intentionally tiny (≤ 1 MB) so it fits in git. **It is not
sufficient to reproduce paper numbers.** For a publishable run you need the
full processed datasets — see [Full datasets](#full-datasets) below.

## Full datasets

The build stage of `pinn-gym` generates per-material candidate pools *in
silico* from the declared numerical oracle; it does **not** require external
measured data. If you only want to reproduce the manuscript's numerical
results, you can skip this section: a successful

```bash
pinn-gym run configs/sr_full.yaml --stage build
```

will populate `runs/sr_full_*/datasets/` from scratch.

However, the literature-grounding step in our methodology uses six public
datasets to anchor the material cards and to scrub the candidate pool against
empirically observed force-displacement curves. To reproduce that step, fetch
the originals into `data/raw/`:

| Subfolder | Source | License |
|---|---|---|
| `mendeley_ptmr5ggz74_lattice_compression` | Mendeley `ptmr5ggz74` | CC BY 4.0 |
| `mendeley_72mg3x9ft2_octahedral_lpbf_large` | Mendeley `72mg3x9ft2` | CC BY 4.0 |
| `mendeley_n38x2tfzk7_dual_graded_lattices` | Mendeley `n38x2tfzk7` | CC BY 4.0 |
| `mendeley_r85v44bh2m_foam_filled_lattices` | Mendeley `r85v44bh2m` | CC BY 4.0 |
| `mendeley_x3zcxr6nxx_alveolar_hollow_lattice` | Mendeley `x3zcxr6nxx` | CC BY 4.0 |
| `mendeley_fhv5y4tzgc_shock_photopolymer` | Mendeley `fhv5y4tzgc` | CC BY 4.0 |
| `github` | Misc. open lattice geometry repos | per-repo |

The manifest of expected files lives in
[`scripts/download_datasets.py`](../scripts/download_datasets.py) once the
scripts are ported from the `polmi` codebase.

### On HPC: symlink instead of download

The datasets are ~30 GB and you almost certainly already have them on a shared
scratch space. Symlink instead of duplicating:

```bash
ln -s /path/to/shared/scratch/polmi_data/raw       data/raw
ln -s /path/to/shared/scratch/polmi_data/extracted data/extracted
ln -s /path/to/shared/scratch/polmi_data/processed data/processed
```

The symlinked subdirectories are ignored by git via `.gitignore`, so no
accidental commits.

## Regenerating the dummy sample

```bash
python scripts/make_dummy_data.py
```

This will recreate `data/dummy/` from a deterministic seed once the core
sampler has been migrated. Until then, the committed dummy files in
`data/dummy/` are hand-written placeholders matching the documented CSV
schema.
