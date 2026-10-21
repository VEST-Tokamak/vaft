# Legacy archive against the current pipeline: snapshot

The tables behind `notebooks/pipeline_improvement_public_vs_main.ipynb`. They were built on vestserver on
2026-10-21 by `workflow/pipeline_improvement/build_comparison_tables.py` and merged by `make_snapshot.py`.
`snapshot_manifest.json` records the producer commit of every run shard, the sources, the row counts and the
SHA-256 of each file.

**Sources**

| Side | Data |
|---|---|
| Legacy | Shot-first FileDB `public/<shot>/`, read from 3dst-nas share `vest_db_backup` (the syncthing mirror of the deleted `/srv/vest.filedb/public`), mounted read-only on vestserver. Restored copy: `/mnt/backup/vaft/public_restore_20261021`. |
| Current | `/srv/vest.filedb/efit/magnetic/<shot>/`, kept only at HSDS `main` equilibrium times. |
| Diagnostics (vacuum check, Thomson) | HSDS `public` and `main`. |

**Scope**

- **Equilibrium checks:** 1,472 shots, every shot whose HSDS `main` holds an equilibrium and that also exists in `public`.
- **Vacuum check:** 2,279 shots that hold magnetics and PF data in both sources.
- **Thomson check:** the 57 shots whose HSDS `public` product holds `thomson_scattering`.
- **Stability, current side:** the Tier A stability atlas v2 (`stability_current_*`), copied unchanged.

**Files**

| File | One row per | Notes |
|---|---|---|
| `slices.csv.gz` | side and slice | Pairs share `pair_id`; matched by time, within 0.5 ms. |
| `fit.csv.gz` | side and slice | m-file relative RMS per family. `*_common` columns cover the channels both sides fitted. |
| `constraint_points.csv.gz` | fitted channel | Subsample of 150 paired slices. |
| `virial.csv.gz` | side and slice | Volume integral against the pair_13 closure, Grad-Shafranov residual, a-file scalars. |
| `thomson.csv.gz` | side and slice | g-file against HSDS `public` Thomson. `no_thomson` rows mark shots without it. |
| `vacuum_by_family.csv.gz` | source, shot, wall model and family | Medians over channels both sources evaluated. |
| `stability_legacy*.csv.gz` | DCON/RDCON run, or RDCON surface | Read from the legacy `linear_stability` text output. |
