# Reviewer Pack

The canonical reviewer artefacts come from:

```text
simulations/results/reviewer_experiments_21392657
```

That directory is part of the `v0.1.0-full-dump` release, not the git clone.
After unpacking the full dump, regenerate the compact publication pack with:

```bash
pinn-gym reviewer-pack \
  --run-dir simulations/results/reviewer_experiments_21392657 \
  --out reviewer_pack
pinn-gym verify-pack reviewer_pack
```

`reviewer_pack/` contains:

- `SUMMARY.md`: auto-filled headline numbers read from the run.
- `tables/pooled_regret_per_card.csv`: pooled model regret and ranking metrics.
- `tables/baselines_per_card.csv`: random and oracle baselines.
- `tables/seed_repeats_ci.csv`: repeated-seed mean and standard deviation.
- `tables/loss_weight_ablation.csv`: physics-loss weight ablation.
- `tables/pooled_tuning.csv`: pooled-model capacity and optimizer grid.
- `raw_per_stage/`: unfiltered CSV copies for traceability.
- `SHA256SUMS`: deterministic checksum manifest for the pack.

The working interpretation file from the challenge workspace is archived in
the release asset `reviewer-pack-20260527.tar.zst`. It documents the main
finding: the original 512x7 pooled PINN overfits and collapses feasibility,
while a compact 384x6 model restores curve fidelity and finite regret on
cards with feasible pools.

All numbers are scoped to the declared progressive-crush numerical oracle.
They are not experimental validation.
