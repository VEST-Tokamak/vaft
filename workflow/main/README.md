# VEST `workflow/main`

This is the maintained OMAS-first VEST pipeline. All local
paths must come from `vaft.database.filedb.FileDB`; rules must not reconstruct paths or
refer to the legacy `/srv/vest.filedb/public/{shot}` layout.

Set `VAFT_FILEDB_DIR` to the deployment root, add shots to `config.yaml`, and run:

```bash
snakemake --snakefile workflow/main/Snakefile --cores 1
```

Set `raw_source` to an archived file or a `{shot}` path template for offline,
reproducible runs. If it is null, the raw stage exports the shot from the VEST
SQL database. The current DAG finalizes raw, versioned static, diagnostic, and
eddy-current products. Later migration issues extend the same DAG with EFIT,
CHEASE, stability, and publication stages.

Machine selection is explicit. The legacy 43017 and 45967 transitions are
retained, while the corrected PF6/PF7 geometry boundary at 45958 creates a
separate 45958–45966 era. Every stage writes a deterministic `manifest.json`
beside its output and records unavailable diagnostic components rather than
inventing zero waveforms.

The legacy server remains a read-only reference. Audit a mounted or copied
shot-first tree without modifying it:

```bash
python -m vaft.cli filedb audit /path/to/legacy --target-root /path/to/FileDB
```
