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

For manual, database-backed re-exports, use the serial command below instead
of launching multiple Snakemake jobs. It holds a per-FileDB lock, opens only
one SQL connection, retries one shot at a time, and preserves an existing good
dump until a replacement completes:

```bash
python -m vaft.cli raw-redump \
  --filedb-root "$VAFT_FILEDB_DIR" \
  --shot-range 29350 48823 \
  --attempts 3 --retry-delay 30 --inter-shot-delay 2
```

The products are written directly to
`raw/{shot}/vest_{shot}_daq_raw.json.gz` and
`raw/{shot}/vest_{shot}_daq_manifest.json`. Existing valid dumps are skipped;
pass `--force` to re-export them. Add `--source-template
'/archive/vest_{shot}_daq_raw.json.gz'` to redump from a local archive without
contacting SQL. With neither `--shots` nor `--shot-range`, the command exports
the fixed supported range `29350–48823`. It does not discover or query shot
lists before exporting.

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
