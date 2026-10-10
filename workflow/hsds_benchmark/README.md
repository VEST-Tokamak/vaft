# HSDS I/O benchmark (#1869)

A reproducible, **read-only** benchmark of VAFT's HSDS access paths, plus the
opt-in instrumentation it is built on. Part of the database I/O umbrella #1868;
#1870 uses it to profile and optimise.

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

Each client runs in a fresh process (`spawn`). With `--clients N`, N processes
start together, each with its own cache directory.

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
| subprocess seconds / bytes | `transport._run` around hsget/hsload | their HTTP traffic is in another process and not counted as requests; bytes are the `hsstat` total (download) or local file size (upload) |
| wall time | `perf_counter` around the scenario | module imports happen before timing starts |
| peak RSS | `getrusage` of the client process | per client |

### Correctness

Every scenario reads the same probe values (equilibrium time and slice
scalars, `psi` in the large set, Ip, PF coil currents, probe positions and
fields). `eager_canonical` is the reference; every other run's digests for the
same paths must match, and `mismatched_paths` lists any that do not. The large
probe set is a superset of the small one, so lazy scalar values are compared
too.

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
  attributes on this HSDS (#1271).

## First smoke result (2026-10-11, vestserver, production HSDS 0.9.0.alpha0, 4 nodes, h5pyd 0.24.0)

Shot 39915, one client, one repeat, taken while a production backfill was
running, so it shows the instrumentation working, not a baseline:

| scenario | wall | requests | observed bytes | hsget | probes vs eager canonical |
|---|---|---|---|---|---|
| eager_canonical, cold | 16.7 s | 25 | 0.38 MB | 9.1 s | 143/143 agree |
| eager_canonical, warm | 4.4 s | 25 | 0.38 MB | 0 s | 143/143 agree |
| lazy_scalar | 2.2 s | 148 | 0.41 MB | -- | agree |
| prefetch | 4.1 s | 213 | 29.4 MB | -- | agree |
