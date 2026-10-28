# HSDS I/O benchmark

A reproducible, **read-only** benchmark of VAFT's HSDS access paths, plus the
opt-in instrumentation it is built on. It is also the measurement used when
profiling and optimising the lazy, prefetch and eager access paths.

## Run

```bash
python workflow/hsds_benchmark/run_benchmark.py \
    --shots 39915,41524 --source main \
    --scenarios lazy_scalar,lazy_large,prefetch,eager_canonical,eager_h5image,eager_full \
    --clients 1,2,4 --repeat 3 --output runs/hsds-bench/<label>
```

HSDS endpoint and credentials come from `.hscfg` / `HS_*`, as for any VAFT
read (`vaft.database.hscfg`). On a checkout whose environment has an editable
VAFT installed from elsewhere, put the intended tree first on the import path:
the runner records `vaft_commit` and `vaft_dirty`, so check them.

## Scenarios

| scenario | access | what it stresses |
|---|---|---|
| `lazy_scalar` | `open()` | per-slice global quantities and per-coil/probe leaves, one selection each: scalar-heavy traversal |
| `lazy_large` | `open()` | every slice's 2-D `psi` and every probe's field: large-array selections |
| `prefetch` | `open()` + `prefetch(ids)` | bulk read per IDS, then the scalar traversal |
| `eager_canonical` | `load(transport="canonical")` | hsget of each IDS domain; **cold** (empty cache) and **warm** (same cache, second run) |
| `eager_h5image` | `load(transport="h5image")` | derived image download in 4 MiB slices; cold and warm |
| `eager_full` | `load()` (default transport) | the path a user gets without options; cold and warm |

Each client runs in its own fresh process (`spawn`, one `Process` per client,
not a pool). With `--clients N`, N processes are released together by a
barrier, each with its own cache directory; a group whose clients did not get
distinct PIDs is recorded as an error rather than timed.

## Output

* `results.jsonl` -- one record per client run: scenario, shot, clients,
  client index, repeat, cache state, `wall_s`, `peak_rss_mb`, the I/O summary,
  probe digests and any error (a failed client is a result, not a crash).
* `summary.json` -- `context` (run ID, VAFT commit and dirty flag,
  VAFT/h5pyd/omas/Python versions, host, HSDS `getServerInfo`, each probed
  domain's `modified` stamp, UTC start and end) and `groups`: p50/p95 wall time,
  median requests / observed bytes / subprocess seconds, max peak RSS, and the
  correctness check, per (scenario, shot, clients, cache state).

### What is observed, and what is not

| metric | source | caveat |
|---|---|---|
| request count, by method | `requests.Session.send` while recording | in-process h5pyd traffic only |
| response bytes | `Content-Length` of each response | `responses_without_length` counts the ones without; never estimated |
| header latency p50/p95 | time to response headers | h5pyd streams bodies (`stream=True`), so body transfer is in `wall_s`, not here |
| subprocess seconds / bytes | `transport._run` around hsget/hsload | their HTTP traffic is in another process and not counted as requests; bytes are the `hsstat` total (download) or local file size (upload); a **failed** hsget/hsload is not recorded (its time is still in `wall_s`) |
| wall time | `perf_counter` around the scenario | module imports happen before timing starts |
| peak RSS | `getrusage` of the client process | per client |

### Correctness

Every scenario reads the same probe values (equilibrium time and slice
scalars, `psi` in the large set, Ip, PF coil currents, probe positions and
fields). `eager_canonical` is the reference; every other run's digests for the
same paths must match. Per group the summary reports:

* `mismatched_paths` -- values that differ from eager canonical;
* `missing_paths` -- values eager canonical read that this run should have read
  (scalar scenarios are not expected to read `psi` or probe fields) but did not;
* `probe_errors` -- reads that failed for a reason other than the path being
  absent (`KeyError`/`IndexError` mean "not in this shot"; anything else is
  counted, never treated as absent);
* `reference_available` -- whether an eager canonical run succeeded for the
  shot at all (without one, nothing was compared);
* `canonical_disagreement` -- paths on which eager canonical runs disagree with
  each other.

The large probe set is a superset of the small one, so lazy scalar values are
compared too.

### Correlating with server metrics

While recording, every request carries `X-VAFT-Run-ID: <run_id>`. Together
with `started_utc` / `ended_utc` in `summary.json`, that selects the run's
lines in HSDS service-node logs and the server's metrics for the same window.

## Instrumentation in your own code

```python
from vaft.database.instrumentation import record_io

with record_io(run_id="my-check") as io:
    ods = vaft.database.open(39915)
    ods["equilibrium.time"]
print(io.summary())
```

Outside `record_io` nothing is patched; behaviour is unchanged.

## Safety

* Read-only: no scenario writes to HSDS.
* **Production:** low client counts (≤ 4), and not while a production
  regeneration or backfill is writing (the numbers would be disturbed and so
  would the run). The 1/2/4/8-client matrix and the HSDS 1.x comparison belong
  on staging (vest-server#2).
* h5pyd stays pinned at 0.24.0 (`pyproject.toml`); 1.0.0 silently empties
  attributes on this HSDS (Open: #1271, blocks the 1.x upgrade).

## First smoke result (2026-10-11, vestserver, production HSDS 0.9.0.alpha0, 4 nodes, h5pyd 0.24.0)

Shot 39915, one client, one repeat, taken while a production backfill was
running, so it shows the instrumentation working, not a baseline:

| scenario | wall | requests | observed bytes | hsget | probes vs eager canonical |
|---|---|---|---|---|---|
| eager_canonical, cold | 16.7 s | 25 | 0.38 MB | 9.1 s | 143/143 agree |
| eager_canonical, warm | 4.4 s | 25 | 0.38 MB | 0 s | 143/143 agree |
| lazy_scalar | 2.2 s | 148 | 0.41 MB | -- | agree |
| prefetch | 4.1 s | 213 | 29.4 MB | -- | agree |

## Baseline (2026-10-11 19:58–20:09 KST, vestserver, production HSDS 0.9.0.alpha0, 4 nodes)

No production writer was running. Shots 39915 and 41524, 1/2/4 clients, 3 repeats, 378 client runs,
runner at 565b4b4a9. Wall-time p50 in seconds for shot 39915 (41524 is within about 20 %):

| scenario | 1 client | 2 clients | 4 clients | requests | observed MB | peak RSS MB |
|---|---|---|---|---|---|---|
| lazy_scalar | 0.77 | 0.93 | 1.17 (p95 2.96) | 148 | 0.41 | 169 |
| lazy_large | 1.17 | 1.39 | 2.49 | 309 | 3.9 | 175 |
| prefetch | 1.30 | 1.67 | 2.62 | 213 | 29.4 | 212 |
| eager_canonical, cold | 8.33 (hsget 3.8) | 9.87 | 13.80 (hsget 7.5) | 25 + hsget | 0.38 | 342 |
| eager_canonical, warm | 2.97 | 3.39 | 3.81 | 25 | 0.38 | 341 |
| eager_full, cold | 7.26 | 7.91 | 10.15 | 79 | 10.5 | 347 |
| eager_full, warm | 3.24 | 3.68 | 5.56 | 50 | 0.76 | 346 |
| eager_h5image | -- | -- | -- | -- | -- | -- |

* Correctness: every value agrees with eager canonical (0 mismatched, 0 missing). The canonical
  repeats agree with each other.
* `eager_h5image` failed in every run: `H5ImageUnavailableError` (the derived image is missing or
  stale on production for these shots). It cannot be measured until the images exist.
* A warm eager load still makes 25 (canonical) or 50 (full) requests and takes about 3 s, which is
  roughly 4x a lazy scalar read.
* The smoke run during a production backfill took 16.7 s for eager canonical cold, twice the 8.3 s
  here. Contention matters more than client count up to 4.
