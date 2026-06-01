# Configuration reference

A pinn-gym run is fully described by one YAML file. This document is the
field-by-field reference. The canonical commented template is
[`configs/default.yaml`](../configs/default.yaml). The schema is enforced by
[`src/pinn_gym/config.py`](../src/pinn_gym/config.py); unknown keys are
rejected so that typos surface immediately.

## Top-level sections

| Section | Purpose |
|---|---|
| `run` | Orchestration: run name, seed, stages, device. |
| `materials` | Material cards under test. |
| `candidate_pool` | Sampler + declared numerical oracle. |
| `train` | PINN / baseline training. |
| `evaluate` | Ranking, transfer matrix, metrics. |
| `plots` | Figure generation. |
| `audit` | STL watertightness + envelope. |

## `run`

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | str | `"run"` | Output goes to `runs/<name>_<timestamp>/`. |
| `seed` | int | `20260520` | Master RNG seed shared by every stage. |
| `stages` | list | all 5 | Subset of `[build, train, evaluate, plots, audit]` in order. |
| `device` | str | `"cuda"` | `"cuda"` or `"cpu"`. Build/audit are CPU regardless. |
| `output_root` | str/null | `null` | Override the default `runs/` root. |

Resume an interrupted run by passing only the remaining stages:

```bash
pinn-gym run configs/sr_full.yaml --stage evaluate,plots,audit
```

## `materials`

| Field | Type | Default | Notes |
|---|---|---|---|
| `presets` | list[str] | `[pa12, pla, petg, tpu, pa_cf]` | Built-in material cards. |
| `material_aware_crush_target` | bool | `true` | Per-material feasibility gate. **Keep `true`** unless intentionally running a PA12-only pilot. |
| `custom_cards` | list[str] | `[]` | Extra material JSON files; their basenames become preset names. |

The five built-in presets cover one ductile (`pa12`), two brittle thermoplastics
(`pla`, `petg`), one elastomer (`tpu`), and one fibre-filled (`pa_cf`). Each
card declares density, stiffness, plateau stress, densification, failure
strain, and manufacturing constraints. See
[`docs/PIPELINE.md`](PIPELINE.md#material-cards) for definitions.

## `candidate_pool`

Drives `pinn_gym.core.sr_benchmark.build_sr_dataset`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `train_n` | int | `4000` | Training-pool size **per material**. |
| `eval_n` | int | `1200` | Evaluation-pool size **per material**. |
| `layers` | int | `96` | Crush-front discretisation. |
| `steps` | int | `320` | Quasi-static integration steps per design. |
| `max_displacement_mm` | float | `50.0` | Stroke envelope. |
| `oracle_workers` | int | `0` | 0 = auto = `min(cpu_count, 8)`. |
| `dynamic_amplification` | float | `1.16` | Strain-rate factor applied to the static curve. |
| `yield_scale` | float | `0.08` | Plateau-to-yield ratio used by the layered model. |
| `fixture_peak_force_limit_n` | float | `3500.0` | Peak-force violation threshold. |
| `target_min_crush_mm` | float | `40.0` | Used only when `material_aware_crush_target=false`. |

## `train`

Drives `pinn_gym.core.sr_benchmark.train_sr_models`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `methods` | list[str] | `[pinn_full, pinn_energy, mlp_softplus]` | One of `KNOWN_METHODS`. Baselines `random`, `lightest`, `pseudo_bootstrap` skip the actual GPU training step. |
| `pooled` | bool | `true` | Train an additional pooled multi-material model. |
| `epochs` | int | `200` | Training epochs per model. |
| `batch_size` | int | `1024` | |
| `rows_per_material` | int | `50000` | Subsample cap per material. |
| `hidden_dim` | int | `256` | MLP width. |
| `blocks` | int | `4` | Residual block depth. |
| `lr` | float | `1.0e-3` | |
| `weight_decay` | float | `1.0e-5` | |
| `loss_weights.boundary` | float | `0.5` | F(0)=0 enforcement weight. |
| `loss_weights.energy` | float | `1.0` | Energy-integral loss weight. |
| `loss_weights.peak` | float | `0.2` | Peak-force soft bound. |
| `loss_weights.monotonicity` | float | `0.5` | Monotonic-increase prior on the plateau. |
| `loss_weights.smoothness` | float | `0.1` | Second-difference smoothness. |
| `peak_soft_bound` | float | `1.05` | Relative peak-force soft bound. |
| `monotonic_strain_after` | float | `0.05` | Strain after which monotonicity is enforced. |

### Known methods

- `random` — baseline, no training.
- `lightest` — baseline, no training.
- `pseudo_bootstrap` — empirical-curve baseline, no training.
- `mlp_softplus` — plain MLP with non-negative force output.
- `pinn_energy` — energy-integral PINN.
- `pinn_full` — curve + energy + monotonicity + peak + smoothness PINN.

## `evaluate`

Drives `pinn_gym.core.sr_benchmark.evaluate_sr_run`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `precision_ks` | list[int] | `[1, 3, 5, 10, 25, 50]` | All entries must be positive. |
| `target_energy_j` | float\|"auto" | `"auto"` | `"auto"` uses the oracle's `IMPACT_ENERGY_J`. |
| `peak_limit_n` | float | `3500.0` | Peak-force violation threshold. |
| `min_crush_mm` | float | `40.0` | Minimum crush stroke for feasibility. |
| `curve_limit_mm` | float | `40.0` | Truncation point for curve metrics. Must be ≤ `candidate_pool.max_displacement_mm`. |
| `include_transfer_matrix` | bool | `true` | Cross-material transfer experiment. Heavy; disable for smoke runs. |

## `plots`

Drives `pinn_gym.core.sr_plots.render_sr_figures`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `formats` | list[str] | `[pdf, png]` | Subset of `[pdf, png, svg]`. |
| `figures` | list[str] | all 7 | Subset of `KNOWN_FIGURES`. |
| `dpi` | int | `150` | Use `300` for journal submissions. |

### Known figures

- `curves_overlay`
- `nrmse_distribution`
- `energy_error`
- `violation_bars`
- `precision_at_k`
- `regret_at_k`
- `robustness_survival`

## `audit`

Drives `pinn_gym.core.mesh_quality.audit_stl_directory`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `stl_export_count` | int | `3` | Top-k geometries exported per material. `0` disables. |
| `stl_backend` | str | `"voxel"` | `"voxel"` or `"implicit"`. |
| `stl_resolution` | int | `144` | Voxel grid resolution. |
| `stl_format` | str | `"binary"` | `"binary"` or `"ascii"`. |
| `warn_only` | bool | `false` | If `true`, audit failures do not fail the run. |

## Validation

```bash
pinn-gym validate configs/sr_full.yaml
```

Returns a JSON summary on success, or a `pydantic.ValidationError` listing
every violation on failure. Validation is fast (no compute) and safe to run in
CI.

## Generating a default config

```bash
pinn-gym write-default my_run.yaml
```

Writes every field at its default value. Useful as a starting point when
forking a custom run.
