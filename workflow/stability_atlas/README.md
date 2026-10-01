# Stability atlas (lane N, #1429)

This directory holds the drivers for the n-resolved Tier A stability atlas: DCON ideal δW, RDCON/STRIDE Δ′, Mercier/ballooning flags and QA, keyed by (shot, time, EFIT lineage, n).

| script | issue | what it does |
|---|---|---|
| `scan_controls.py` | #141 | Narrow numerical-control scan. It changes one axis at a time from the packaged templates. Each variant gets its own `templates_dir`, and each (equilibrium, variant, module, n) runs as its own suite call. Resumable. Writes `scan_controls.csv`. |

Runs go on vestserver against GPEC e68d7ac2 or later, with explicit `GPECHOME` and BLAS threads capped at 2. Outputs live under `~/runs/campaign/lane-n/`, never in the production FileDB. Results and decisions are recorded in #141 and in the lane log #1448.
