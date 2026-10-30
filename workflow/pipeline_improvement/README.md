# Legacy archive against the current pipeline

Builds the tables behind `notebooks/pipeline_improvement_public_vs_main.ipynb`. The archived
legacy pipeline (the shot-first FileDB `public/<shot>/`) and the current one (HSDS `main`) are
scored by the same validation kernels, slice by slice.

| script | does |
|---|---|
| `build_comparison_tables.py` | per shot: pairs the legacy and current EFIT slices by time, then writes the fit (m-file), virial and Grad-Shafranov (g-file), Thomson (g-file against HSDS `public` Thomson), vacuum (HSDS diagnostics, stored and common wall model) and legacy DCON/RDCON rows; resumable, shardable |
| `make_snapshot.py` | merges the shards, reduces the vacuum table to one row per source, shot, wall model and family, and writes the gzip snapshot with `snapshot_manifest.json` |

## Inputs

- **Legacy archive.** `/srv/vest.filedb/public` was deleted on 2026-10-01. Its copy survives on
  3dst-nas, share `vest_db_backup`, which is mounted read-only on vestserver at
  `/mnt/vest-backup` (not persistent across reboots). The restored selective copy is
  `/mnt/backup/vaft/public_restore_20261021`. Either root works as `--legacy-root`; they hold the
  same files.
- **Current pipeline.** `--current-filedb /srv/vest.filedb` for the EFIT g/m/a-files. A slice
  counts as current only when its time is in HSDS `main`'s equilibrium.
- **Stability atlas.** `--atlas-dir ~/runs/campaign/atlas/stability`, copied as is.

## Run (vestserver, isolated worktree)

```bash
. ~/work/pipeline-improvement/env.sh
python workflow/pipeline_improvement/build_comparison_tables.py \
    --legacy-root /mnt/backup/vaft/public_restore_20261021 --current-filedb /srv/vest.filedb \
    --atlas-dir ~/runs/campaign/atlas/stability --output ~/runs/pipeline_improvement/tables/efit \
    --shots-file shots_efit.txt --checks fit virial thomson stability --shard 0/4
python workflow/pipeline_improvement/make_snapshot.py \
    --tables ~/runs/pipeline_improvement/tables/efit ~/runs/pipeline_improvement/tables/vacuum \
    --output notebooks/data/pipeline_improvement_public_vs_main
```

HSDS is read with `cache="off"` and `--hsds-sleep` seconds between reads, because the single
service node returns 503 to everyone under an unthrottled bulk read.
