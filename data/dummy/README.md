# dummy/

Tiny synthetic sample used by tests and documentation. Hand-written rows; do
not use for any scientific claim.

Layout mirrors what the `build` stage produces:

```
dummy/
├── README.md
└── pa12/
    ├── train.csv         ← 6 rows × 16 columns
    ├── eval.csv          ← 3 rows × 16 columns
    └── summary.json      ← per-material feasibility counts
```

Columns (matches `pinn_gym.core.design_space.candidate_header`, truncated to
`curve_points=8` for compactness in git):

```
topology, unit_cell_mm, strut_diameter_mm, layers, mass_g,
peak_force_N, energy_J, feasible,
curve_000, curve_001, ... curve_007
```

To regenerate after the core migration:

```bash
python scripts/make_dummy_data.py
```
