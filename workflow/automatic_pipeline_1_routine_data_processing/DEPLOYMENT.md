# Enabling HSDS replication

Operator procedure for bringing the `main` and `chease-mhd-stability` namespaces
online. Written against HSDS 0.9.0.alpha0 and h5pyd 0.20.0.

Every check in step 1 and step 3 is read-only. The first command that changes
anything on the server is in step 2.

---

## Corrections worth reading first

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

for stage in s.replicable_stages():
    dest = s.source_for_stage(stage)
    s.resolve(dest, writable=True)
    print(f"ok: {stage:12s} -> {dest}")
PY
```

Expect `read-only: ['public']` and one `ok:` line per stage.

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

```bash
EDDY="$VAFT_FILEDB_DIR/omas/eddy/39915/output/eddy.json"

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
| Owned IDS only | **No** `magnetics.h5` — the eddy product carries it, but eddy does not own it |
| Record state | `"state": "validated"`, `round_trip.passed = true` |
| Provenance | `"source": "main"`, `"remote_uri": "hdf5://main/39915/"`, a `product_sha256` |
| Manifest | unchanged — it describes production, not replication |

> **Per-shot folders must be provisioned. This is settled, not open.** HSDS does
> not auto-create them: `hsload` fails with
> `Domain: hdf5://main/39915/dataset_description.h5 not found` until the folder
> exists. Provision one shot with `hstouch -o "$OWNER" /main/39915/`, or a whole
> range with the script below. It is idempotent — `hstouch` opens with `mode='x'`
> and refuses an existing folder, which the script counts rather than treats as
> an error.
>
> ```bash
> ./provision_hsds_shots.sh main 39000 45000
> ./provision_hsds_shots.sh chease-mhd-stability 39000 45000
> ```
>
> A shot needs a folder in **every** source it replicates into, so a full
> backfill is one call per source.

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
- [ ] Only owned IDS travelled — no `magnetics.h5` from the eddy stage
- [ ] A second stage did not hide the first — `external_h5_links` lists both
- [ ] A rerun reused the record — no upload
- [ ] Per-shot folder behaviour recorded — auto-created, or `hstouch` needed
- [ ] `/public/` unchanged throughout — entry count and modified timestamp
