# Deploying pipeline 2 and its worker

Pipeline 2 builds the Thomson lineage for shots that were processed earlier. This
document is about running it unattended: what must be in place, how the worker
decides what to run, and how to operate it. The pipeline itself is described in
[`README.md`](README.md) and `docs/_guide/Pipelines.md`; the routine pipeline's
[`DEPLOYMENT.md`](../automatic_pipeline_1_routine_data_processing/DEPLOYMENT.md)
still applies for everything they share (HSDS credentials, `hsload`/`hsget` on the
`PATH`, the FileDB layout, section 7's rule that one shot is replicated before
replication is enabled for all).

## 1. What must exist first

- **Pipeline 1's products for the shots.** The magnetic EFIT (`omas/efit/magnetic`)
  gives the equilibrium that maps Thomson, the geometry for the resistive estimate
  and the pressure for the inferred `T_i`; the diagnostics product gives the plasma
  onset for the transient charge states. A shot without them records the dependent
  stages as `unavailable`.
- **The uploads** in `external.data_root`: `Shot{n}_v{v}.mat` and the other Thomson
  names (`vaft.machine_mapping.thomson_scattering.thomson_file_shot` is the
  authority), and `CES_{n}.mat` or `IDS_{n}.mat` for the ions.
- **EFIT** for the kinetic lineages: `$EFIT` or `$EFITHOME`, or `kinetic.executable`.
- **OPEN-ADAS ADF11 files** for the charge states of carbon and oxygen. They are
  fetched on first use, so a server without network access needs the cache warmed
  and `composition.adf11_cache_dir` pointing at it.

## 2. Configure the pipeline

The deployed pipeline configuration is merged over `config.yaml`, which supplies
every key it omits. State what you intend rather than inheriting:

```yaml
base_dir: ${VAFT_FILEDB_DIR}
layout: filedb
external:
  data_root: /srv/vest.diagnostic
  ces_options: ids
hsds:
  replicate: false          # turn on after the check in section 3
```

The worker refuses to start without `external.data_root` and with any layout but
`filedb`.

## 3. Check replication once before enabling it

`core_profiles` in `main` is owned by four stages that replicate one after the
other (`core_profiles`, `composition`, `ti_inferred`, `resistive_zeff`), each
replacing the IDS the previous one wrote; the Snakefile makes each wait for the
earlier stage's replication record. Before turning replication on for many shots,
replicate **one** shot to a scratch source, read it back, and confirm that it
carries `ion[]`, `zeff`, `t_i_average` (where inferred) and
`global_quantities.z_eff_resistive`, and that the measured `core_profiles` of a
shot whose composition was skipped is still intact. This is the same discipline as
section 7 of the routine deployment, and it is the one check no offline test makes.

## 4. The worker

```bash
cp corrective.example.yaml /srv/vaft/corrective.yaml    # outside the repository
$EDITOR /srv/vaft/corrective.yaml                       # first_shot, cores, paths, hold_file
vaft corrective-worker run --config /srv/vaft/corrective.yaml --once    # one cycle, then read it
vaft corrective-worker run --config /srv/vaft/corrective.yaml           # the service
```

**What a cycle does.**

1. *Recover* what a killed worker left running (its shots go back to pending,
   forced) and `--unlock` the workflow directory only if no Snakemake works in it.
2. *Scan* the upload directory. Each shot has a signature: its files' names, sizes
   and modification times. A shot not seen before, or whose signature changed, is
   pending. A scan that cannot read the directory is reported and the queue still
   runs.
3. *Run* the pending shots whose newest file is at least `quiet_seconds` old, up to
   `max_shots_per_run`, in one Snakemake call.
4. *Harvest*: read each stage's manifest, and with `hsds.replicate` its replication
   record, for the stages in scope. The exit code is never consulted.

**A shot's state.**

| State | Meaning |
|---|---|
| `pending` | an upload is new or changed, or a retry is due |
| `running` | in the current run |
| `completed` | at least one stage produced a product |
| `unavailable` | every stage said so (`unavailable`/`no_output`); nothing to retry |
| `failed` → `pending` | a stage `failed`, or a manifest or replication record is missing or unreadable; retried, forced |
| `gave_up` | `max_attempts` runs still failed; waits for `retry` |
| `excluded` | an operator said never; survives a changed upload |

**Why a finished shot is forced.** The two root rules have no input files, so
Snakemake treats an existing output as up to date even after the upload it was
built from was replaced, and a stage that failed leaves its manifest behind. A
changed or failed shot is therefore run with `--forcerun generate_thomson_ods
generate_ces_ods`, which reruns everything downstream. A busy workflow lock or a
requested stop releases the shots without counting an attempt.

**What it does not notice.** A change in *pipeline 1's* products. If the magnetic
EFIT of a shot is regenerated, its pipeline 2 products still describe the old one;
run `vaft corrective-worker retry --shot N` to force it.

## 5. One writer per shot

The routine worker and any regeneration of EFIT or the diagnostics write products
of the same shots. `master.h5` writes are serialised per `(source, shot)` by a lock
(#913), so they cannot corrupt each other, but two jobs deciding what a shot's
products should be is still one too many. While a regeneration owns a shot range:

```bash
touch ${VAFT_FILEDB_DIR}/pipeline/corrective/HOLD    # the file named by hold_file
```

The worker keeps scanning and recording uploads and starts no run; remove the file
and everything pending runs. `first_shot` and `last_shot` bound the range the
worker owns at all.

## 6. Operate

```bash
vaft corrective-worker status --config corrective.yaml                 # counts, and what needs attention
vaft corrective-worker status --config corrective.yaml --shot 48226    # stages, events
vaft corrective-worker retry   --config corrective.yaml --shot 48226   # fresh budget, forced
vaft corrective-worker exclude --config corrective.yaml --shot 48226 --reason "wrong file"
```

`status` reads the state file read-only, so it is safe while the service runs. Each
run keeps its exact Snakemake configuration and log under `log_dir/runs/<run_id>/`.
As a service, run it under the account that owns the FileDB and has the HSDS
credentials, with `Restart=on-failure`; the worker holds a lock on its state file
and a second one refuses to start.

## 7. Limits worth knowing

- **No disk guard.** Unlike the routine worker it does not pause on low free space.
  Its products are small (JSON ODS), but the kinetic lineages leave EFIT work
  directories while they run.
- **Figures are not judged.** `plots.enabled` schedules one optional figure set per
  stage; a missing figure never changes a shot's state. A figure whose data is
  present but which fails to render fails its rule, and only that.
- **A shot with no composition gets no inferred `T_i`.** The ion density is half of
  the formula, so it is not guessed with `n_i = n_e`. That includes shots whose
  transient composition could not be timed (no diagnostics product).
- **The resistive `Z_eff` is one number per discharge window**, written at every
  usable kinetic slice, and only when the fit is not on a bound.
