# Full Dump Release

The git repository is intentionally lightweight. Large CSVs, STL files,
checkpoints and historical run outputs are published in the
`v0.1.0-full-dump` GitHub release.

## Download

```bash
mkdir -p full_dump
gh release download v0.1.0-full-dump \
  --repo Dyniel/pinn-gym \
  --dir full_dump
cd full_dump
sha256sum -c FULL_DUMP_SHA256SUMS
```

## Reassemble Split Archives

Archives larger than the release comfort limit are split deterministically into
`*.partNNN` files. Reassemble before extraction:

```bash
for first in *.tar.zst.part000; do
  base="${first%.part000}"
  cat "${base}".part* > "${base}"
done
```

`FULL_DUMP_SHA256SUMS` verifies downloaded upload files, including split
parts. To verify a reassembled logical archive, compare it with the
`logical-split` row in `FULL_DUMP_MANIFEST.tsv`:

```bash
sha256sum full-dump-data.tar.zst full-dump-simulations-results.tar.zst
awk -F '\t' '$5 == "logical-split" {print $4 "  " $1}' FULL_DUMP_MANIFEST.tsv
```

## Extract

Run extraction from the root of a fresh clone:

```bash
tar --zstd -xf full_dump/reviewer-pack-20260527.tar.zst
tar --zstd -xf full_dump/submission-sr-20260519-pinn.tar.zst
tar --zstd -xf full_dump/submission-polmi-20260518.tar.zst
tar --zstd -xf full_dump/full-dump-data.tar.zst
tar --zstd -xf full_dump/full-dump-geometry.tar.zst
tar --zstd -xf full_dump/full-dump-reports.tar.zst
tar --zstd -xf full_dump/full-dump-simulations-generated.tar.zst
tar --zstd -xf full_dump/full-dump-simulations-results.tar.zst
```

If an archive was split, use the reassembled `.tar.zst` filename in the command
above.

## Manifest

`FULL_DUMP_MANIFEST.tsv` records the asset filename, source path, byte size and
SHA256 digest. The manifest is generated before release upload from
`/users/scratch1/dancies/pinn_gym_release_assets`.

## Expected Assets

- `reviewer-pack-20260527.tar.zst`
- `submission-sr-20260519-pinn.tar.zst`
- `submission-polmi-20260518.tar.zst`
- `full-dump-data.tar.zst`
- `full-dump-geometry.tar.zst`
- `full-dump-reports.tar.zst`
- `full-dump-simulations-generated.tar.zst` or split parts
- `full-dump-simulations-results.tar.zst` or split parts
- `FULL_DUMP_SHA256SUMS`
- `FULL_DUMP_MANIFEST.tsv`
