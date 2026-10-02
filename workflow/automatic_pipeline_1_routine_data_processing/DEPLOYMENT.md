# Enabling HSDS replication

Operator procedure for bringing the `main` and `chease-mhd-stability` namespaces
online. Written against HSDS 0.9.0.alpha0 and h5pyd 0.20.0.

Every check in step 1 and step 3 is read-only. The first command that changes
anything on the server is in step 2.

---

## Corrections worth reading first

**A deployment written before #813 must be migrated before its next pipeline
run.** Stage products are now resolved as `{stage}.json.gz`, and a tree written
earlier holds `{stage}.json`. The resolver asks for a name that is not there, so
Snakemake sees no output for any stage of any shot and rebuilds every one of
them from raw -- `run_efit_reconstruction` and `run_chease` included, which is
days of solver time and exactly what the change set exists to avoid. Nothing
warns you: a missing target is indistinguishable from work not yet done.

So migrate first, with the pipeline stopped: `python -m vaft.cli filedb
migrate-products "$VAFT_FILEDB_DIR"` prints the plan, and `--apply` performs it
one stage at a time. It rewrites the files already on disk and re-runs no
physics. The procedure, including the separately gated deletion of what it
supersedes, is under "Migrating stored products onto their declared container"
below.

**The trailing slash decides folder versus domain.** `hstouch /main` creates a
*domain* — a single HDF5 file called `main`. `hstouch /main/` creates a *folder*.
Replication needs a folder, and a domain at that path fails every later write in
a way that reads like a permissions problem.

**`hstouch -u` is a credential override, not "run as admin".** `-u`/`-p`/
`--api_key` override the credentials in `~/.hscfg`. Only `-o/--owner` relates to
admin, and it is the flag that *requires* admin rights. When `~/.hscfg` already
authenticates as the intended owner, both flags can be dropped.

**HSDS does not validate the owner string.** `hstouch -o OWNER /main/` silently
succeeds and leaves the namespace owned by a user called `OWNER`. This has
happened once on this deployment. Ownership cannot be changed afterwards —
h5pyd has no `chown`, and `hstouch` opens with `mode='x'` — so the only repair is
to delete and recreate while the namespace is still empty.

**h5pyd folder probes do not create anything.** A missing folder logs
`folder put status_code: 404` on its way to raising. The message is misnamed:
`Folder.__init__` resolves `mode=None` to `"r"`, and its create branch is gated
on `w`/`w-`/`x`. `hsls` and `h5pyd.Folder(..., mode="r")` are both safe for
existence checks.

---

## 1. Preflight

Run from the machine that will run the pipeline, as the account that will run it.

```bash
hsinfo
```

Expect `server state: READY` and a `username` line. The parenthetical is the
resolved role — `username: admin (admin)`. If it does not say `admin`, you cannot
run step 2's `-o`; ask an administrator to.

Configuration comes from `~/.hscfg` (`hs_endpoint`, `hs_username`, and one of
`hs_password` / `hs_api_key`), or from `HS_ENDPOINT` / `HS_USERNAME` /
`HS_PASSWORD`. `hsconfigure` writes the file interactively. VAFT reads no HSDS
settings of its own — it inherits whatever h5pyd resolves.

### Which account is which

| Role | Who | Why |
| --- | --- | --- |
| Acting admin | The identity in `~/.hscfg` | Only an admin may pass `-o`. Used once, in step 2. |
| Owner | The account the pipeline runs as | Owns the namespace afterwards and needs write access without an ACL grant. Match `/public/`, whose owner is `admin`, only if the pipeline genuinely runs as `admin`. |

### Confirm the namespaces do not already exist

```bash
hsls /main/
hsls /chease-mhd-stability/
```

404 is the expected, wanted answer. Or through the same call VAFT itself makes:

```bash
python -c '
import h5pyd, logging; logging.disable(logging.WARNING)
for ns in ("main", "chease-mhd-stability", "public"):
    try:
        n = len(list(h5pyd.Folder(f"/{ns}/", mode="r")))
        print(f"/{ns}/  EXISTS ({n} entries)")
    except Exception as e:
        print(f"/{ns}/  absent: {e}")
'
```

If one already exists, do not run step 2 — `hstouch` opens with `mode='x'` and
will refuse, which is the safe outcome. Verify its owner and ACLs against step 2
instead.

---

## 2. Create the namespaces

```bash
OWNER=admin
echo "owner will be: $OWNER"

hstouch -o "$OWNER" /main/
hstouch -o "$OWNER" /chease-mhd-stability/
```

### If the owner is already wrong

Repair it while the namespaces are still empty; the cost is zero now and
unbounded later. Verify the entry count is `0` first, and never delete a
populated namespace to fix its owner.

```bash
hsdel /main/ /chease-mhd-stability/
OWNER=admin
hstouch -o "$OWNER" /main/
hstouch -o "$OWNER" /chease-mhd-stability/
```

### ACLs

The pipeline account needs **create, read, update and delete**. Delete matters:
`hsload` with no flags opens the target with `mode="w"`, so re-replicating a
stage *replaces* its domains. An account with create and delete but not update
cannot reliably overwrite.

```bash
# Both namespaces -- they are separate domains and an ACL on one
# does not reach the other.
hsacl /main/ +crud g:editors
hsacl /chease-mhd-stability/ +crud g:editors

hsacl /main/
hsacl /chease-mhd-stability/
hsacl /main/ "$OWNER"
```

`Unexpected error: Not Found` from `hsacl <domain> <user>` means that user has
*no ACL entry* on that domain — not that the domain is missing. A server-
configured superuser bypasses ACLs and will still be able to write, which is how
an owner mistake stays invisible until someone runs the pipeline as anyone else.

Leave `default` at read-only, as `/public/` has it.

### Verify

```bash
python -c '
import h5pyd, logging; logging.disable(logging.WARNING)
for ns in ("main", "chease-mhd-stability"):
    f = h5pyd.Folder(f"/{ns}/", mode="r")
    print(ns, "class=", getattr(f, "_obj_class", None), "owner=", f.owner)
'
```

Both must report `class= folder`. Anything else means the trailing slash was
lost; `hsdel` and redo.

---

## 3. Verify the source policy

Client-side, no server contact.

```bash
python - <<'PY'
from vaft.database import sources as s
from vaft.database.sources import ReadOnlySourceError

print("read-only:", [x.name for x in s.known_sources() if not x.writable])
try:
    s.resolve("public", writable=True); print("FAIL: public accepted a write")
except ReadOnlySourceError:
    print("ok: public refuses writes")

# A stage that is one solve's result goes to one source per stability product,
# so it has no destination until the product is named.
PER_PRODUCT = {"mhd_linear": ("dcon-peeling", "dcon-kink", "rdcon", "stride")}

for stage in s.replicable_stages():
    for product in PER_PRODUCT.get(stage, (None,)):
        dest = s.source_for_stage(stage, product=product)
        s.resolve(dest, writable=True)
        print(f"ok: {stage:12s} -> {dest}")
PY
```

Expect `read-only: ['public', 'chease-mhd-stability', 'magnetic-efit']` and one
`ok:` line per stage, four for `mhd_linear` (one per stability product).

```bash
# user= cannot falsify the destination -- must raise before connecting
python -c '
import omas, vaft.database.ods as m
try:
    m.save_ods(omas.ODS(), 39915, source="main", user="public")
    print("FAIL")
except ValueError as e:
    print("ok:", str(e)[:70])
'

# the one raw-h5pyd write path carries its own gate
python -c '
from vaft.database.utils import processed_registry_uri as u
print("read :", u("public"))
try:
    u("public", writable=True); print("FAIL")
except Exception as e:
    print("write:", type(e).__name__)
'

# and the corrective updaters refuse to be pointed at public
VAFT_HSDS_SOURCE=public python -c '
import sys; sys.path.insert(0, "workflow/automatic_pipeline_2_corrective_data_update")
try:
    import update_thomson_scattering_and_core_profile; print("FAIL")
except Exception as e:
    print("ok:", type(e).__name__)
'
```

> **`public`'s read-only guarantee is client-side.** `default` has read alone,
> but `admin`, `g:admins` and `g:editors` hold update and delete on `/public/`,
> and the pipeline authenticates as `admin`. Only VAFT refuses. To enforce it
> where a stray script cannot bypass it, drop the pipeline account's write bits —
> but only once the corrective updaters are running against `main`, or you will
> break them:
>
> ```bash
> hsacl /public/ -ud admin
> hsacl /public/
> ```

---

## 4. Configure the pipeline

In `config.yaml`:

```yaml
layout: filedb            # required; shot_first is refused
base_dir: ${VAFT_FILEDB_DIR}

hsds:
  replicate: true         # was false
  attempts: 3
  retry_delay: 5.0
```

`VAFT_FILEDB_DIR` must point at the canonical FileDB root, and h5pyd must resolve
credentials as in step 1. There is no source name to configure: each stage's
destination comes from `vaft.database.sources.STAGE_REPLICATION`.

### Environment the pipeline needs, and why

Four settings that are not obvious and each of which fails in a way that does not
name itself:

| Setting | Why |
| --- | --- |
| `PATH=~/.local/bin:$PATH` | The replicator shells out to `hsload`/`hsget`. A non-interactive SSH shell excludes `~/.local/bin`, and the failure surfaces as a replication error, not a missing-tool error. |
| `--scheduler greedy` | snakemake 7.32.4 calls `pulp.list_solvers`, which **no released pulp version exposes** — they all have `listSolvers`. The call sits in a `try/except ImportError`, so it works only when pulp is *absent*; installing pulp turns a caught error into an uncaught `AttributeError`. The greedy scheduler avoids the ILP path entirely. |
| `conda: null` | `config.yaml` sets `conda: vaft`, and snakemake's job bookkeeping shells out to `conda env export`. In a non-interactive shell conda is not on `PATH`, so the job succeeds and then the run dies during bookkeeping. |
| `--unlock` after an interruption | A killed run leaves a snakemake directory lock. The next run refuses to start until it is cleared. |

### External codes

| Code | Location | Build |
| --- | --- | --- |
| EFIT | `~/git/efit/build-linux/efit/efit` | `cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_STANDARD_LIBRARIES="-llapack -lblas"` — the BLAS reference must be *appended*, since `CMAKE_EXE_LINKER_FLAGS` places flags before the libraries and cannot resolve `liblapack`'s `dscal_`. |
| CHEASE | `~/work/chease_1/chease` | prebuilt |
| GPEC | `~/git/GPEC/bin/` | see below |

```bash
FC=gfortran LAPACKHOME=/usr \
NETCDF_FORTRAN_HOME=/usr/lib/x86_64-linux-gnu NETCDFINC=/usr/include \
FFLAGS="-fallow-argument-mismatch -O2" \
OMPFLAG=-fopenmp RECURSFLAG=-frecursive LDFLAGS=-fopenmp make all
```

Three things have to be true and each fails opaquely:
`-fallow-argument-mismatch` (gfortran 10+ makes argument mismatches fatal, so
`dcon_interface.mod` is never produced and every dependent package cascades);
`NETCDF_FORTRAN_HOME` rather than `NETCDFHOME` (the makefile reads only the
former, and Debian puts `libnetcdff` in the multiarch path); and no stale
zero-length binaries from a previous failed link, which `make` treats as up to
date and silently skips.

### ideal-GPEC is not part of a routine run

`gpec.modules` defaults to `[dcon, rdcon, stride, gpec]`. The fourth is
ideal-GPEC, which costs **~2 hours per shot** and which `build_mhd_linear` waits
on, because it depends on every configured (code, mode). It is issue #95 scope
and is **not replicated** — `STAGE_REPLICATION` marks `gpec_ideal` as
`deferred_to: "#95"` — so including it gates the whole stability branch behind
work that never reaches HSDS. Over a 5000-shot range that is the difference
between weeks and years.

```yaml
gpec:
  modules: [dcon, rdcon, stride]
```

```bash
# shot_first must be refused before the DAG is built
snakemake --snakefile Snakefile \
  --config layout=shot_first hsds='{"replicate": true}' --list
```

Expect `WorkflowError: hsds.replicate requires layout: filedb`, raised while the
Snakefile is read. With replication correctly enabled the same `--list` shows
five `replicate_*_to_hsds` rules; with `replicate: false`, none.

---

## 4.5. Bootstrap a canonical product

**Required if the pipeline has only ever run `layout: shot_first`.** Replication
consumes *canonical* products; without one, step 5 fails at `No stage manifest` —
not because replication is broken, but because there is nothing to replicate.

Shot **39915** is the migration fixture. Do not hand-copy products into canonical
paths: the canonical layout expects a stage `manifest.json` the legacy tree does
not carry in that form, and a fabricated manifest would assert a provenance
nobody produced. Rebuild instead — it is the path production takes, and it
validates the layout switch at the same time.

No SQL re-export is needed. The raw stage re-exports from an existing archive,
validating the shot number and field mapping and writing a genuine manifest.

```bash
ls -l /srv/vest.filedb/public/39915/diagnostics/vest_39915_daq_raw.json.gz
```

`$VAFT_FILEDB_DIR` is the canonical root; `raw/`, `omas/`, `efit/`, `chease/` and
`gpec/` are created directly beneath it. It must be distinct from the legacy
tree — if the shot-first root is `/srv/vest.filedb/public`, then
`/srv/vest.filedb` works and the two sit side by side. The audit is read-only:

```bash
python -m vaft.cli filedb audit /srv/vest.filedb/public --target-root "$VAFT_FILEDB_DIR"
```

### Retiring `chease-mhd-stability`

The combined source is superseded: the refinement lives in `main/chease` and
each stability product in `main/chease/{product}`. Retiring it is gated, because
part of what it holds **cannot be copied forward**.

A faithful copy of the stability half would have to say which product each
result came from, and the combined product does not record it:

- DCON's two edge treatments write the same fields at the same
  `(time_slice, position)`. One run happened; nothing on disk says whether it
  was `dcon-peeling` or `dcon-kink`.
- RDCON's and STRIDE's rational surfaces are appended to one `ntms` AOS, and the
  `<solver name=...>` fragment goes to the whole IDS rather than to each
  surface.

Copying it anyway would write provenance to HSDS that nothing can verify. So the
stability results are **re-run**, not moved, and the deletion gate checks for
that:

```python
from vaft.database.retirement import plan_retirement, copy_refinement, delete_retired_source

report = plan_retirement(shots)          # reads only
report.deletable                          # every shot accounted for?
report.blocking                           # the ones that are not, and why
```

Per shot, two questions: has the refinement reached `main/chease`, and has the
stability stage been re-run into the per-product sources.

```bash
# 1. copy each refinement forward (read back and verified before it reports success)
python -c "from vaft.database.retirement import copy_refinement; copy_refinement(39915, apply=True)"

# 2. re-run the stability stage for the shot, which populates main/chease/{product}

# 3. re-plan, review, and only then delete
python -c "..."     # plan_retirement(shots) -> review report.to_dict()
hsdel /chease-mhd-stability/
```

Deletion is the administrator's own command. `delete_retired_source` validates
the plan and prints it, but will not remove a namespace itself: that is not
something a library call should be able to do as a side effect, however clean
the plan looks. It also refuses an **empty** report — "no shots were examined"
and "every shot is safe" are different findings.

### Checking a product's ancestry

Each stage records the digest of what it consumed and of what it produced, so
"which EFIT produced this stability result" is a comparison rather than an act
of faith:

```text
EFIT       equilibrium.code.parameters...artifacts.gfile.sha256
             |
CHEASE     manifest["input"][i]["sha256"]          what it read
           manifest["input"][i]["output_sha256"]   what it wrote
             |
stability  GPECSuiteResult.input_equilibrium_sha256
```

```python
from vaft.database.provenance import verify_chain

report = verify_chain(
    chease_manifest=json.loads(chease_manifest_path.read_text()),
    stability_input_sha256=run["input_equilibrium_sha256"],
    shot=39915,
)
report.verified          # every link checked and agreed
report.broken            # digests disagree -- products of different runs
report.unrecorded        # no digest -- written before the chain existed
```

A **mismatch** means the stability cell consumed an equilibrium this CHEASE
stage did not produce; a CHEASE rerun between the refinement and the solve is
enough to cause it, and paths cannot see it. An **unrecorded** link is not a
pass: products from before the chain carry `""`, and reporting those as verified
would make the check useless exactly on the archive that most needs it.

The verifier compares recorded digests and never re-hashes the tree. Re-hashing
answers "do these files agree with each other now" — weaker, different, and
impossible once the upstream file has been archived off the host.

### Relocating a canonical root written before the lineage segments

A canonical tree written before `family`/`refinement`/`product` became path
segments (#527) resolves to nothing afterwards, and nothing warns about it: the
resolver asks for `efit/magnetic/{shot}` and returns a path, the data is at
`efit/{shot}`, and the path is simply empty. Deployments created before that
change need their subtrees moved once.

Dry run first — it prints the plan and changes nothing:

```bash
python -m vaft.cli filedb relocate "$VAFT_FILEDB_DIR"
```

Read the plan, then apply it:

```bash
python -m vaft.cli filedb relocate "$VAFT_FILEDB_DIR" --apply
```

What it does and does not do:

- **Directories are renamed, never copied.** No file in the tree is opened,
  read, hashed or written, so an interrupted run leaves each subtree either
  wholly moved or wholly where it was — never half written.
- **The whole `gpec/{code}` subtree moves at once**, so every `(shot, mode)` cell
  under it travels in one rename rather than one per cell.
- **A DCON cell keeps the legacy `dcon` product** rather than becoming
  `dcon-peeling` or `dcon-kink`. The old tree recorded no edge treatment, so
  neither branch can be asserted, and guessing would put provenance on disk that
  the move cannot verify.
- **Stages that belong to no family are untouched** — `raw/`, `omas/static/`,
  `omas/diagnostics/`, `omas/eddy/`, `omas/impa/`.
- **An occupied destination stops the whole run**, before anything moves. Two
  subtrees claiming one path were written by different runs, and which is
  authoritative is not a question this can answer. Resolve it by hand first.
- **Re-running is a no-op.** A shot directory is all digits and a lineage segment
  never is, so the two grammars cannot be confused and a second `--apply` moves
  nothing.

The dry run exits non-zero when the plan has collisions, so a script can gate on
it rather than parsing the JSON.

### Migrating stored products onto their declared container

A deployment written before #813 holds products the resolver no longer asks for
(`eddy.json` where it now wants `eddy.json.gz`) and eddy products carrying the
whole diagnostics product they were solved against. **Do this before the
deployment's next pipeline run.** On the new code every shot's Snakemake target
is the gzipped name, which does not exist in such a tree, so a single
`snakemake` invocation rebuilds every eddy product from scratch — the physics
re-run this migration exists to avoid.

```bash
python -m vaft.cli filedb migrate-products "$VAFT_FILEDB_DIR"
python -m vaft.cli filedb migrate-products "$VAFT_FILEDB_DIR" --stage diagnostics --apply
```

Then, once that stage's report is clean, delete what it superseded and move to
the next stage:

```bash
python -m vaft.cli filedb sweep-products "$VAFT_FILEDB_DIR" --stage diagnostics
python -m vaft.cli filedb sweep-products "$VAFT_FILEDB_DIR" --stage diagnostics --apply
python -m vaft.cli filedb migrate-products "$VAFT_FILEDB_DIR" --stage eddy --apply
```

What it does and does not do:

- **Neither half re-runs any physics.** A container change re-encodes bytes that
  are already correct; the eddy change is the same projection
  `replication._project` has always applied on the way to HSDS.
- **HSDS is unaffected and needs no re-upload.** `_project` has always stripped
  the non-owned IDS before publishing, so no replica changes. The replication
  *records* do change hash, so the next replication run re-sends unless you
  accept that cost knowingly — one full re-send, otherwise harmless.
- **Files are read, decoded and rewritten** — unlike `relocate`, which only
  renames. Atomicity is bought per product: a temporary in the same `output/`
  directory, fsync, verify by re-reading, `os.replace`, then fsync the
  directory. An interrupted run leaves every product wholly old or wholly new,
  and its temporary named in `orphan_temporaries` on the next pass.
- **The originals are not deleted by the migration.** `sweep-products` is a
  separate step, and it is what makes everything before it reversible: until it
  runs, undoing the migration is deleting the new product.
- **Migrate and sweep one stage at a time when disk is tight.** Writing every
  new product before deleting any original peaks at about 1107 G against the
  1126 G free on this deployment; alternating per stage never exceeds 1058 G.
- **An eddy original whose shot has no diagnostics product is refused.** The
  dropped IDS are recoverable from that shot's diagnostics product and the era's
  static product — but that recoverability is a precondition of the deletion,
  not a property of it, and without it the original is the only local copy of
  that shot's magnetics.
- **The manifest is rewritten, and says so.** `output` describes the file and is
  updated; `input` describes the run and is never touched. A `migration` block
  carries `previous_output.sha256`, which is how a downstream manifest's now
  dangling `input.diagnostics_sha256` stays joinable.
- **Re-running is a no-op**, and `--verify-shape` proves it by opening each
  migrated product rather than inferring it from the file name.
- **The pipeline must be idle.** Check with
  `pgrep -af 'snakemake|replicate_to_hsds|generate_(eddy|diagnostics)_ods'`, the
  workflow's `.snakemake/locks`, and
  `find "$VAFT_FILEDB_DIR/omas" -name '*.json' -newermt '-10 minutes'`. Setting
  `hsds: replicate: false` for the window is belt and braces.

The dry run exits non-zero when the plan cannot be applied, and `sweep-products`
exits non-zero when anything was refused, so both can be scripted on.

Keep the bootstrap config separate from production, with replication off:

```bash
cd workflow/automatic_pipeline_1_routine_data_processing

cat > bootstrap-39915.yaml <<'YAML'
base_dir: ${VAFT_FILEDB_DIR}
layout: filedb
shots: [39915]
raw:
  mode: archive
  archive_template: /srv/vest.filedb/public/{shot}/diagnostics/vest_{shot}_daq_raw.json.gz
hsds:
  replicate: false
YAML
```

Ask for one product by path; Snakemake works backwards to it through
raw → static → diagnostics → eddy. Nothing downstream runs, so no EFIT or CHEASE
binary is needed. Shot 39915 resolves to machine era `vest-pre-43017-pf1906`.

Ask the resolver for the path rather than spelling it: the container is declared
in `OMAS_PRODUCT_SUFFIX`/`OMAS_PRODUCT_SUFFIXES`, and a literal here goes stale
the next time it moves — silently, because Snakemake reports a path it has no
rule for as a missing target rather than as a wrong name.

```bash
EDDY=$(python -c 'import os; from vaft.database.filedb import FileDB; \
print(FileDB(os.environ["VAFT_FILEDB_DIR"]).omas_product("eddy", shot=39915))')

snakemake --snakefile Snakefile --configfile bootstrap-39915.yaml --dry-run "$EDDY"
snakemake --snakefile Snakefile --configfile bootstrap-39915.yaml --cores 4 "$EDDY"
```

```bash
python - <<'PY'
import json, os
from vaft.database.filedb import FileDB
from vaft.database.replication import REPLICABLE_STATUSES
db = FileDB(os.environ["VAFT_FILEDB_DIR"])
for stage in ("diagnostics", "eddy"):
    product = db.omas_product(stage, shot=39915)
    manifest = db.omas_manifest(stage, shot=39915)
    status = json.loads(manifest.read_text()).get("status") if manifest.exists() else None
    print(f"{stage:12s} product={product.exists()} manifest={manifest.exists()} "
          f"status={status!r} replicable={status in REPLICABLE_STATUSES}")
PY
```

Both must report `product=True manifest=True replicable=True`. A `status` of
`partial` is fine and still replicable — some diagnostic components were
unavailable, which is ordinary. Only `skipped`, `blocked`, `failed` or
`no_output` block step 5.

The archive is read and copied, never moved; `/srv/vest.filedb/public/` is
unchanged by this step.

---

## 5. Smoke test — one stage, one shot

Do not run the pipeline. Replicate a single stage, so the blast radius is one IDS
in one shot folder.

`eddy` is the best first candidate: it owns exactly one IDS (`pf_passive`), so a
mistake is one domain, and it exercises the projection that keeps a stage from
overwriting its neighbours.

```bash
python replicate_to_hsds.py --shot 39915 --stage eddy --filedb-root "$VAFT_FILEDB_DIR"
```

Expect `shot 39915 eddy -> main (validated): pf_passive`.

```bash
python - <<'PY'
import h5pyd, json, os, logging; logging.disable(logging.WARNING)
print(sorted(h5pyd.Folder("/main/39915/", mode="r")))
from vaft.database.filedb import FileDB
db = FileDB(os.environ["VAFT_FILEDB_DIR"])
print(json.dumps(json.loads(
    db.omas_replication_record("eddy", shot=39915).read_text()), indent=2))
PY
```

| Check | Expected |
| --- | --- |
| Remote folder | `/main/39915/` lists `pf_passive.h5`, `dataset_description.h5`, `master.h5` |
| Owned IDS only | **No** `magnetics.h5` — the eddy stage solves against it but does not own it, and its product no longer carries it |
| Record state | `"state": "validated"`, `round_trip.passed = true` |
| Provenance | `"source": "main"`, `"remote_uri": "hdf5://main/39915/"`, a `product_sha256` |
| Manifest | unchanged — it describes production, not replication |

> **Per-shot folders are created by the writer.** `hsload` does not create a
> missing folder -- it fails with
> `Domain: hdf5://main/39915/dataset_description.h5 not found` -- so every
> write path (`save_ods`, the native IDS writer, replication) now creates
> `/<source>/<shot>/` itself under the shot's master lock, right before its
> first upload (`vaft.database.utils.ensure_shot_folder`). An existing folder
> costs one GET and is never touched; a second host losing the create race
> (409) counts as success. Checked against this deployment on 2026-10-01 as
> `admin` with h5pyd 0.24.0: the folder is created, a repeat is a no-op, and
> `hsload` into it succeeds.
>
> The folder is owned by the writing account, so that account needs create
> permission in the source folder -- `admin`, the production writer, has it.
> A writer without it gets an error naming the `hstouch` that provisions the
> shot. `provision_hsds_shots.sh` still does that for a range, and is now only
> needed for a writer without create permission:
>
> ```bash
> ./provision_hsds_shots.sh main 39000 45000
> ```

### Then prove the merge preserves what was already there

The property most worth confirming on a real server, because failing it is
silent: the files stay in the folder and simply stop being visible to the eager
reader.

```bash
python replicate_to_hsds.py --shot 39915 --stage efit --filedb-root "$VAFT_FILEDB_DIR"

python - <<'PY'
import tempfile, pathlib
from vaft.database.transport import run_hsget
from vaft.database.staging import external_h5_links
with tempfile.TemporaryDirectory() as d:
    m = run_hsget("hdf5://main/39915/master.h5", pathlib.Path(d) / "master.h5")
    print(external_h5_links(m))
PY
```

Expect both `pf_passive.h5` and `equilibrium.h5`. If only the last appears, stop:
the master merge is not working and further replication will keep hiding earlier
stages.

---

## 6. Idempotency and retry

```bash
python replicate_to_hsds.py --shot 39915 --stage eddy --filedb-root "$VAFT_FILEDB_DIR"
```

Expect `already replicated to main; product unchanged` and no upload. Reuse is
not "the record exists": the recorded `product_sha256` must still match the
current local product *and* the state must satisfy the run's contract. `--force`
overrides.

| `state` | Meaning | What a rerun does |
| --- | --- | --- |
| `validated` | Sent, read back, compared clean | Nothing |
| `replicated` | Bytes are on the server; the comparison failed or was skipped. `error` says which. | Re-sends and re-checks; with `--no-validate`, treats it as done |
| `failed` | The write itself did not complete | Re-sends |

A `replicated` record with an `error` is the case to look at by hand: the data is
there but did not match what was sent.

A retry merges against the **pre-write** master, not against its own failed
attempt — the master is fetched once, before the first attempt, and reused. This
is covered by `test_the_previous_master_is_captured_once_not_per_attempt`; there
is nothing to check by hand.

```bash
# Optional: exercise the retry path against a stopped endpoint.
HS_ENDPOINT=http://127.0.0.1:1 python replicate_to_hsds.py \
  --shot 39915 --stage eddy --filedb-root "$VAFT_FILEDB_DIR" \
  --attempts 2 --retry-delay 1
```

Expect two logged attempts, a `state: "failed"` record, and a non-zero exit.

### Regenerating products after a machine-history change

A release that moves a machine-era boundary changes what the mapper builds for
the shots on the moved side, and nothing re-runs them on its own. 0.8.0 is such
a release (#956/#961): the wall-2409 loops now start at shot 43017
(`WALL_GEOMETRY_2409_FIRST_SHOT`) and the PF-2507 geometry at 45968 rather than
45958 (`PF_GEOMETRY_2507_FIRST_SHOT`), so every diagnostics, eddy, EFIT and
downstream product for a shot **>= 43017** built before 0.8.0 differs from what
the release mapper produces: 43017–45957 gain the 15 wall loops, 45958–45967
additionally change PF geometry (2507 → 1906), and >= 45968 change the wall.
Shots below 43017 are unaffected.

Nothing has to be deleted. A shot's static input path is derived from its era
name, so a shot on the moved side now resolves to a static product that did not
exist before; Snakemake 7.32 reruns a job whose input set changed (`input` is in
its rerun triggers) and everything downstream follows, replication included.
But only for shots that are in a run: the worker processes new shots, so list
the affected shots explicitly (`shots:` in the config, in batches) after
deploying.

A stale product is identifiable from its manifest without reading the data: the
diagnostics and eddy manifests record `machine_version` (the era name the
product was built under -- the retired names `vest-43017-45957-pf1906`,
`vest-45958-45966-pf2507` and `vest-45967-plus-pf2507` no longer resolve, and
`machine_era()` refuses them) and `input.static_sha256` (the hash of the static
product it was built from). A product whose `machine_version` is not one of
`VEST_MACHINE_ERAS`, or whose `static_sha256` differs from the current static
product's, is one the release will rebuild.

---

## 7. Enable pipeline replication

Only after steps 5 and 6 pass.

```bash
make run
```

Start with one shot in `config.yaml`, confirm both namespaces populate, then
widen.

| Stage | Target source | Owned IDS |
| --- | --- | --- |
| `diagnostics` | `main` | magnetics, pf_active, tf, barometry, spectrometer_uv, langmuir_probes |
| `eddy` | `main` | pf_passive |
| `efit` | `main` | equilibrium |
| `chease` | `chease-mhd-stability` | equilibrium |
| `mhd_linear` | `chease-mhd-stability` | mhd_linear, ntms |

`static` is not shot-replicated — it is versioned by machine era and its geometry
travels inside the diagnostics product. `gpec_ideal` has a declared destination
but no rule; it remains issue #95 and refuses with a message naming it.

Expect a shot's presence to differ by stage. A vacuum shot appears in `main` with
diagnostics and eddy and no equilibrium; a shot whose EFIT never converged keeps
everything upstream of it. That is the intended model, not a partial failure.

---

## 8. Rollback and recovery

```yaml
hsds:
  replicate: false
```

One key. The rules leave the DAG and the records leave the target set; nothing
local is touched and nothing remote is removed. Local processing is unaffected —
replication was never a precondition for it.

To recover a failed or half-validated replica: read the record's `state` and
`error`. `failed` — re-run; the write did not complete and there is nothing to
clean up. `replicated` with an error — the data is on the server but did not
match; inspect before re-running, since a mismatch may mean the local product
changed under you. To force a clean re-send, delete the record and re-run, or
pass `--force`.

**Safe to delete:** `replication.json` (derived state; costs one re-send), and a
domain this pipeline wrote under `/main/` or `/chease-mhd-stability/` provided
you also delete the matching record.

**Never delete:** anything under `/public/`. `master.h5` in a populated shot
folder — the eager reader resolves the shot's contents from it and there is no
rebuild path; re-replicate rather than hand-editing it. The stage
`manifest.json` — it describes production, and the stage would have to re-run.

No recovery step touches `public`. It is not written, migrated or deleted by any
part of this system, and it is not a runtime dependency. If a recovery procedure
seems to call for modifying it, the procedure is wrong. The one remaining read is
the corrective updaters' one-time bootstrap of their shot registry, which opens
`/public/processed_shots.h5` in `"r"`.

---

## 9. Acceptance checklist

- [ ] Server reachable, role confirmed — `hsinfo` → `READY`, `username: … (admin)`
- [ ] Both namespaces exist as folders, correctly owned — `_obj_class == "folder"`
- [ ] Pipeline account holds create, read, update, delete — `hsacl /main/`
- [ ] `public` refuses writes — `resolve("public", writable=True)` raises
- [ ] Every destination resolves to a writable named source — one `ok:` per stage
- [ ] `user=` cannot mislabel a destination — raises before connecting
- [ ] The registry cannot be pointed at `public` — `ReadOnlySourceError` at import
- [ ] `shot_first` refused before the DAG is built — `WorkflowError`
- [ ] A canonical product exists for the fixture shot — `omas/eddy/39915/…`, replicable
- [ ] One stage replicated and validated — `state == "validated"`
- [ ] Only owned IDS travelled — no `magnetics.h5` from the eddy stage (the eddy product is its own projection, so replication is an identity on it)
- [ ] Every canonical product is under its declared container — `migrate-products` reports `pending: 0` and `--verify-shape` reports no failures
- [ ] A second stage did not hide the first — `external_h5_links` lists both
- [ ] A rerun reused the record — no upload
- [ ] A new shot's folder was created by its first write — no `hstouch`
- [ ] `/public/` unchanged throughout — entry count and modified timestamp

---

## 10. Running the new-shot worker

`vaft pipeline-worker` watches the VEST SQL `shot` table and runs this pipeline on each new shot
(issue #58). It only calls Snakemake, so everything above still applies, including section 7's
requirement that one shot is replicated before replication is enabled for all of them.

### Configure

```bash
cp worker.example.yaml /srv/vaft/worker.yaml   # outside the repository
$EDITOR /srv/vaft/worker.yaml                  # first_shot, cores, run_timeout, paths
```

- **SQL credentials must already be provisioned** for the account the worker runs as. They live in
  `~/.vest/database_raw_info.yaml` together with `~/.vest/encryption_key.key`. If they are missing, the
  worker refuses to start. It does not fall back to `raw.setup_raw_db()`, because that prompt would
  block forever under a service with no terminal.
- **`pipeline_config` must set `raw.mode: sql`.** For each run the worker replaces `shots:` and forces
  `conda: null`, and it keeps that run's config next to the run log under
  `log_dir/runs/<run_id>/`.
- **Paths may use environment variables**, e.g. `${VAFT_CHECKOUT}` in the example. An unset variable
  stops the worker instead of silently creating a new state file under a literal `${...}` directory.
- **The worker reproduces Snakemake's config merge.** Snakemake merges `pipeline_config` over the
  workflow's own `config.yaml`, and the worker checks and harvests against that same merged result.
- **`first_shot` is where the worker's responsibility starts.** Shots below it belong to batch
  regeneration and the worker never looks at them.

### What a cycle does

1. **Detect.** Every shot in SQL above the watermark is inserted into the state file. Inserting a
   shot that is already there does nothing, so polling the same shot twice changes nothing.
2. **Wait for the upload.** The DAQ writes one complete field per SQL row. A shot goes ahead as soon
   as either condition holds:
   - its inventory contains every field the previous shot had, and no new field has arrived for
     30 s (fields within one core upload arrive at most ~10 s apart); or
   - no field has been uploaded for `quiet_seconds` (default 600).

   On VEST (shots 48800–48916) the 177 core fields arrive 126–196 s after the shot record, so a
   routine shot starts about 2–3 minutes after it is fired, before the next one. Pressure, Plasma
   Current and the 6 kW ECH powers form a separate group. It lands up to 261 s after the core
   fields, or a day later, or never. While it is missing, the quiet limit decides.
3. **Classify.** The classifier runs before anything is scheduled:
   - no waveform fields → `excluded`, and Snakemake never sees the shot;
   - required raw fields missing → only the raw dump is archived;
   - otherwise → the whole pipeline runs.

   Fields that arrive after a shot was processed are handled by the re-check below, so none of these
   verdicts is final.

4. **Run.** One `snakemake` process runs in this directory with `--keep-going --rerun-incomplete
   --scheduler greedy`. HSDS concurrency is the pipeline config's `hsds.concurrency`; replications of
     one shot no longer race for its `master.h5` (#913, section 11).
   - A manual run holding the directory lock makes the worker **busy**: it tries again next cycle
     and does not count an attempt.
   - The worker runs `--unlock` only when no live Snakemake process is working in the workflow
     directory. It checks by command line and cwd, via psutil or `/proc`, not by a recorded pid,
     which a reboot can hand to an unrelated process. A lock whose holder cannot be ruled out is
     left alone.
5. **Harvest.** Each shot's state comes from the products on disk, not from Snakemake's exit code.
   The worker reads every declared stage manifest's `status`, every replication record's `state`, and
   the raw preflight's exclusion list. It also checks that each required validation plot exists.

   | Shot state | Meaning |
   | --- | --- |
   | `completed` | Every declared product succeeded. |
   | `partial` | Every product exists, but some are intentionally incomplete (a vacuum shot's EFIT, a no-output stability cell, …). Snakemake will not rebuild them, so they are not retried. |
   | `excluded` | The classifier, the raw preflight or an operator ruled the shot out. |
   | `failed` | A declared product is missing. The shot is retried on the next cycle, and Snakemake rebuilds only what is missing. |
   | `gave_up` | The shot has used `max_attempts` runs and waits for an operator. |

6. **Re-check for late fields.** For `recheck_seconds` (3 days) after processing, the worker compares
   each shot's SQL inventory with a baseline. The baseline is the fields SQL listed when the shot
   settled plus the fields in the dump manifest's `inventory`. A field the dump failed to load is
   therefore not mistaken for a late arrival. When new fields have arrived, the worker reprocesses
   the shot:
   - It moves the shot's raw dump and manifest to `log_dir/superseded/<shot>/<time>/`. They are
     moved, never deleted.
   - It returns the shot to `detected`, so it is classified and run again.
   - Snakemake then re-exports the raw dump and reruns every product downstream of it, because
     that input is newer. **This includes HSDS replication, which replaces what was published.**

   Reprocessing happens at most `max_reprocess` times per shot (3 by default). Later arrivals are
   recorded as a `late_fields_ignored` event. An operator-excluded shot is never reprocessed.

### Run as a service

```ini
# /etc/systemd/system/vaft-pipeline-worker.service
[Unit]
Description=VAFT new-shot pipeline worker
After=network-online.target

[Service]
User=vaft
Environment=VAFT_FILEDB_DIR=/srv/vest.filedb
Environment=VAFT_CHECKOUT=/home/vaft/git/vaft
Environment=VAFT_WORKER_CONFIG=/srv/vaft/worker.yaml
ExecStart=/opt/conda/envs/vaft/bin/vaft pipeline-worker run
KillSignal=SIGTERM
TimeoutStopSec=300
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

- **Stopping.** SIGTERM lets the running Snakemake finish its bookkeeping and release the lock.
- **A run that has to be SIGKILLed** leaves the directory lock behind. The worker removes it at once,
  because that lock is provably its own.
- **Restarting after a crash or a reboot.** The worker puts every shot left `running` back into the
  queue and removes a lock that no live Snakemake holds.
- **One worker per state file.** A second worker exits immediately.

### Operate

```bash
vaft pipeline-worker status                  # watermark, counts, failed/gave_up shots
vaft pipeline-worker status --shot 48950     # one shot: stages and event log
vaft pipeline-worker retry --shot 48950      # fresh attempt budget
vaft pipeline-worker exclude --shot 48950 --reason "calibration shot"
vaft pipeline-worker run --once              # a single cycle, e.g. to test a config
```

- **`retry` on a shot the classifier excluded or limited to the raw dump** sends it back to be
  classified again, because SQL may have finished writing it since. For a raw-only shot, delete its
  raw dump and manifest first. They are Snakemake outputs, so otherwise the truncated dump would be
  reused as is.
- **The watermark assumes VEST numbers shots in acquisition order.** A row inserted later below the
  watermark is not detected.
- **Monitoring.** Monitoring code (#1347) reads the same file through
  `vaft.database.worker.read_worker_state`.

---

## 11. One writer per shot master (#913)

Every HSDS write of a shot replaces its `master.h5` with the stored master plus the write's own IDS.
Before #913 two overlapping writes of one shot both read the old master, and whichever replaced it
last dropped the other's links. On 2026-09-17 this hid six IDS of shots 39241 and 39620 behind
replication records that said `passed: true`.

The write path now does two things:

- **Holds a per-shot lock** across the read, merge and replace. Replication holds it from the
  capture of the previous master through its safety-net merge. Every VAFT writer on this host is
  covered: the routine pipeline, the corrective updaters, the new-shot worker and the maintenance
  repairs. Different shots never wait for each other. The lock files live in `$VAFT_HSDS_LOCK_DIR`,
  default `/tmp/vaft-hsds-locks`. The location is deliberately fixed rather than `$TMPDIR`, so that
  two writers always meet at the same lock. If that directory exists but this account cannot create
  files in it (made by another account without `chmod 1777`), the writer locks in a per-user
  directory under the temp root instead and warns once, naming both directories: its writes are then
  serialized only against this account's. Fix the mode, or point `$VAFT_HSDS_LOCK_DIR` at a shared
  directory for every writer. Do not give the worker's service unit `PrivateTmp=yes` without doing
  the same: a private `/tmp` is a lock no manual updater on the host shares.
- **Re-reads the stored master immediately before replacing it**, so a link added in the meantime is
  kept. This also applies to a plain `vaft.database.save`, which used to replace the master with one
  naming only its own IDS.

What it does **not** cover:

- writers on another host;
- `hsload` run by hand;
- Windows, where the lock is a no-op; the first write of a process warns (`RuntimeWarning`) that it is not enforced there.

The re-read narrows those windows; it cannot close them.

With the race gone, `hsds.concurrency` can go back above 1. The new-shot worker no longer forces
`--resources hsds=1`.

### Audit and repair

```bash
vaft maintenance audit-masters --shots 39000-48916 --report audit.json          # read-only
vaft maintenance audit-masters --shots 39241 39620 43245 44148 --apply          # relink
```

Each unlinked file is downloaded and read before it is judged. Each shot is reported as one of:

| Status | Meaning |
| --- | --- |
| `complete` | The master links every stored IDS file. |
| `links_missing` | Files are stored that the master does not link. `--apply` relinks them, under the shot's lock. |
| `stubs_unlinked` | The only files the master does not link hold no value at all -- every leaf an IMAS fill, every array of structures empty. Nothing is hidden, and `--apply` leaves them unlinked. 39240, 43245, 44148, 44453 and 44604 are like this: an empty `equilibrium.h5`. |
| `no_master` | Files are stored but there is no master. Nothing can be copied from it, so re-replicate the shot. |
| `absent` | No such shot folder, or one holding only derived images. |
| `unreadable` | Listing or reading failed. The error is in the report. |

The command exits non-zero while any shot is `links_missing`, `no_master` or `unreadable`; `stubs_unlinked` is not a failure. A `links_missing` shot can carry stubs too -- the report lists them under `stubs`, and `--apply` links only the files under `missing`.

