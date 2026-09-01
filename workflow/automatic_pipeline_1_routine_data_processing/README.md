# VEST routine data-processing pipeline

The maintained VEST production pipeline. `workflow/main` was folded into this
workflow by PR #121; this is the single Snakemake DAG that carries a shot from
the raw DAQ export through to linear-MHD stability:

```text
raw → static → diagnostics → eddy → constraints → k-file → EFIT
                                                     ↓
                                       CHEASE → DCON/RDCON/STRIDE → mhd_linear
                                                     ↓
                                                 gpec_ideal
```

Every product path is resolved by `PipelinePaths` in `paths.py`, never
reconstructed by hand. Two layouts are selectable through the `layout` config
key: the legacy `shot_first` tree (default, diffable against the read-only
legacy server) and the canonical `filedb` grammar from
`vaft.database.filedb.FileDB`. Issue #138 removes `shot_first` once #137
finishes validating the canonical schema.

```bash
make run                 # snakemake --cores 30 --configfile config.yaml
```

## HSDS sources

Each analysis lineage is stored in its own named HSDS source, so two valid
representations of the same IDS never overwrite each other (issue #56).
`vaft.database` writes `main` unless a call names another source;
`chease-mhd-stability` takes the CHEASE-refined equilibrium and the linear-MHD
results that follow from it. `python -m vaft.cli summary sources` prints the
catalog. The pre-VAFT `public` namespace stays readable and is never written,
migrated or deleted.

`hsload` does not create a top-level folder, so each source has to be
provisioned once by an HSDS administrator before anything can be written
into it:

```bash
hstouch -u <admin> -o <owner> /main/
hstouch -u <admin> -o <owner> /chease-mhd-stability/
```

Writing into a namespace that does not exist fails with `MissingSourceError`
naming exactly that command, rather than an opaque uploader exit code. Nothing
in VAFT creates a top-level namespace, and no probe asks the server to.

## Legacy reference

The legacy server remains a read-only reference. Audit a mounted or copied
shot-first tree without modifying it:

```bash
python -m vaft.cli filedb audit /path/to/legacy --target-root /path/to/FileDB
```
