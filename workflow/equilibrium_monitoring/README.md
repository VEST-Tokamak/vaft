# Equilibrium monitoring

Collect inputs, attempted time points, failed outputs and verification/validation
metrics before tuning the reconstruction. Collection is read-only; it does not
run EFIT, change quality gates, publish products or schedule a background job.

## Two storage modes, one schema

```sh
# Published data only. Every requested shot remains in shots.csv.
python -m vaft.cli summary monitor --shot-range 38950:39050 \
  --mode hsds --source main --output ./monitor-hsds --figures

# Run on vestserver, or use a mounted FileDB. This also queries HSDS.
python -m vaft.cli summary monitor --shot-range 38950:39050 \
  --mode server --source main --filedb /srv/vest.filedb \
  --output ./monitor-server --figures
```

`--vacuum` adds the canonical PF-only vacuum benchmark, including its assessment
window and wall-history/excitation evidence. It re-solves the vacuum response and
can be expensive; it is off by default. `--no-magnetics` skips signal QA.
The server path reads current canonical FileDB products, manifests, submitted
constraints and explicitly listed k-files. Superseded generations are not scanned.
If filenames in a k-file manifest refer to unavailable files, that is an explicit
error in `shots.csv`, never a silent disappearance of the shot.

The family is explicit (`--family magnetic|electron|kinetic`). Omitting `--source` selects `main`, `electron-efit` or `kinetic-efit` for the
chosen family. An explicit source is honored; no missing lineage falls back to
`main`. Corrective-stage attempt verdicts retain a `stage_` prefix; their success
is not interpreted as a scientifically usable EFIT slice. FileDB is a local/mounted filesystem interface, not an SSH client. Missing
or malformed local metadata are recorded in the shot error column. An unreachable
HSDS service stops the run instead of masquerading as an empty database.

## Output contract (schema 1)

- `shots.csv`: one storage observation per requested shot; coverage, stage/runtime
  status, recorded attempt/output counts, replication state and read/analysis errors.
- `attempts.csv`: union of collection statuses, submitted constraint times, listed
  k-file times and existing equilibrium times. Failures without g-files survive.
  Inputs without a recorded execution status are `unrecorded`, not solver failures.
- `checks.csv`: canonical verification/validation results, per slice, whole product,
  whole-shot signal QA, or vacuum window. These scopes are not interchangeable.
- `metrics.csv`: numeric leaves of those results and collection metrics for direct
  statistical querying. Units of individual leaves remain defined by the provider;
  the headline check unit in checks.csv does not apply to every nested metric.
- `channels.csv`: measured/reconstructed/sigma/z fit evidence, signal health and
  optional vacuum residuals. Original definitions and channel details are retained.
- `inputs.csv`: constraint-stage snapshots and parsed k-file namelists, with
  hashes. K-file namelists are the exact submitted inputs; the constraint-stage ODS can
  precede preset sigma-floor changes and is labelled separately. Diagnostic QA
  refers to the observed diagnostic product; without recorded input hashes it is
  not proof of the exact diagnostics generation used by a historical EFIT run.
- `population.json`: requested-shot/recorded-attempt denominators and failure-code counts by backend.
- `manifest.json`: schema, VAFT version, query selection, options and generation time.

For large scans, use `--batch-size 100` to write `shots_START_END/` partitions
and release each batch before the next. Each partition has its own population
counts and figures; concatenate their CSVs for a full-cohort analysis, keeping the
backend/source/family keys. The Python API returns an in-memory result, so callers
should likewise pass bounded shot batches.

The output directory is a replaceable snapshot, suitable for repeated/manual
monitoring. Keep dated directories when comparing generations. JSON cells are
strict JSON (nonfinite values become null). ODS data are never written back.

## Denominators, joins and provenance

Group statistical populations by backend, source and family. Server mode retains
separate HSDS and FileDB observations: their products may be different generations,
so pooling them would count some shots twice and misattribute failures. `product_id`
identifies recorded collection evidence; `attempt_id` adds the EFIT millisecond.
Joins use time, never list position; duplicate millisecond keys yield an explicit
`analysis_error`. No nearby-time interpolation is used to hide unmatched records.

HSDS absence is **unknown**, not evidence of failure/nonexecution. A published
product can contain recorded failed attempts even when those instants have no
reconstruction. Without a collection payload, published slices are `outputs_only`;
a full attempted-time denominator requires server evidence. Likewise, zero usable
recorded statuses does not mean every output is bad if statuses were never recorded.

Temporal continuity is evaluated on the whole product, not on isolated slices.
Independent validation stays separate from fit residuals. Vacuum evidence refers
to its assessment window, not every k-file instant. For statistical inference,
use shot-level aggregates or resample whole shots, not independent time points.
The generated plots are exploratory distributions, not causal tests or automatic
acceptance thresholds. Missing inputs and provider errors are retained.

## Lazy access and future HSDS migration

The HSDS adapter uses `database.open` and closes each IDS handle after reading.
It snapshots only the requested analysis subtrees, bounded to one shot. Physics
validation requires full 2-D equilibrium fields, signal QA requires waveforms,
and the vacuum benchmark needs circuit geometry: lazy access does not make those
reads free. `io_metrics_json` records available local client instrumentation.
No camera IDS is opened. Current lazy HSDS access supports occurrence 0 only.

`EvidenceBackend.read(shot) -> ShotEvidence` is the storage boundary. A future
attempt-only HSDS occurrence/source adapter can return collection statuses,
submitted constraints, k-file snapshots and provenance through the same contract.
The collector, exported tables and plots remain unchanged; it can then replace
the FileDB adapter. No DD occurrence or failure-slot schema is invented here.

```python
from vaft.database import equilibrium_summary
result = equilibrium_summary([39915, 48224], mode="server", filedb="/srv/vest.filedb")
result.export("./monitor")
# New storage adapters can be supplied through backends=[...] for migration.
```
