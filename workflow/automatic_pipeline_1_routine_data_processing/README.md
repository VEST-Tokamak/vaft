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
provisioned once by an HSDS administrator before anything can be written into
it. Set the owner in a variable and confirm it before running anything —
HSDS accepts any string as an owner without checking that the account exists,
and ownership cannot be changed afterwards:

```bash
OWNER=admin                     # the account the pipeline runs as
echo "owner will be: $OWNER"

hstouch -o "$OWNER" /main/      # trailing slash: folder, not domain
hstouch -o "$OWNER" /chease-mhd-stability/
```

Two things that are easy to get wrong and expensive to undo:

- The **trailing slash** is what makes these folders. `hstouch /main` creates a
  single HDF5 file called `main`, and every later write fails in a way that
  reads like a permissions problem.
- `-u` is a **credential override**, not "run as admin"; drop it when
  `~/.hscfg` already authenticates as the right account. Only `-o` needs admin
  rights, and it is the flag that silently accepts a placeholder.

Repairing either means `hsdel` and recreate, which is safe only while the
namespace is still empty.

Writing into a namespace that does not exist fails with `MissingSourceError`
naming exactly that command, rather than an opaque uploader exit code. Nothing
in VAFT creates a top-level namespace, and no probe asks the server to.

The full operator procedure — preflight, ACLs, source-policy verification,
canonical bootstrap, smoke test, rollback and an acceptance checklist — is in
[DEPLOYMENT.md](DEPLOYMENT.md).

## HSDS replication

Replication copies a finalized canonical product into a named source; FileDB
stays authoritative. It is off by default (`hsds.replicate: true` in
`config.yaml`) and requires `layout: filedb`.

Each stage replicates on its own, carrying only the IDS subtree it owns, so a
shot appears in a source with whatever it actually produced — a vacuum shot with
diagnostics and eddy and no equilibrium, a shot whose EFIT never converged with
everything upstream of it. Where a stage goes and what of it travels is
`vaft.database.sources.STAGE_REPLICATION`; the workflow never decides either.

Three states stay distinct, and none implies the next:

```text
local product completed  →  replicated to HSDS  →  round-trip validated
```

The second and third live in `omas/{stage}/{shot}/metadata/replication.json`,
which is the replication rule's own Snakemake output. A product on disk therefore
never implies it reached HSDS. Re-running is cheap: a record is reused only while
its stored hash still matches the current product, so a rebuilt product is re-sent
and an unchanged one is not — without replaying the solver stage behind it.

```bash
python replicate_to_hsds.py --shot 39915 --stage efit --filedb-root "$VAFT_FILEDB_DIR"
```

## Legacy reference

The legacy server remains a read-only reference. Audit a mounted or copied
shot-first tree without modifying it:

```bash
python -m vaft.cli filedb audit /path/to/legacy --target-root /path/to/FileDB
```
