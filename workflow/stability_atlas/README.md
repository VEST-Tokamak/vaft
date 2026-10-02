# Stability atlas (lane N, #1429)

This directory holds the drivers for the n-resolved Tier A stability atlas: DCON ideal δW, RDCON/STRIDE Δ′, Mercier/ballooning flags and QA, keyed by (shot, time, EFIT lineage, n).

| script | issue | what it does |
|---|---|---|
| `scan_controls.py` | #141 | Narrow numerical-control scan. It changes one axis at a time from the packaged templates. Each variant gets its own `templates_dir`, and each (equilibrium, variant, module, n) runs as its own suite call. Resumable. Writes `scan_controls.csv`. |
| `select_slices.py` | #1429, #1331 | Tier A good/admissible slices for both lineages. Writes `slices.csv`. An electron-kinetic slice takes the label of the magnetics-only slice at exactly its own time (Δt = 0), and is dropped when no such slice exists. |
| `run_batch.py` | #1429 | Per slice: source g-file (an electron-EFIT ODS is converted with `vaft.data.eqdsk.from_omas`, matched by time), CHEASE refinement with pipeline 1's script, then DCON (full edge at mpsi 256/512, truncated edge at 256), RDCON and STRIDE (mpsi 256/512). Resumable; solver scratch is pruned after each job. |
| `build_atlas.py` | #1429, #142 | Reads the run directories and writes `atlas_n.csv` (one row per shot, time, lineage, n), `atlas_surfaces.csv` (one row per rational surface and solver) and `schema.json`. |

Runs go on vestserver against GPEC e68d7ac2 or later. The calling environment sets `GPECHOME` explicitly and caps BLAS/OpenMP threads at 2; the scripts do not set them. Outputs live under `~/runs/campaign/lane-n/`, never in the production FileDB. Results and decisions are recorded in #141 and in the lane log #1448.

Decisions the atlas encodes:
- `mpsi` 256 is reported; 512 is the convergence check (#141).
- DCON runs with `bal_flag=t` and `termbycross_flag=f`. The packaged `dcon.in` stops at unconfirmed sign flips of `crit`, so confirmed Newcomb crossings are read from `dcon.out` instead.
- Full-edge and truncated-edge energies are separate columns (#792).
- A surface's Δ′ is resolved when mpsi 256 and 512 agree in sign and within 20 %. Per-n Δ′ summaries use resolved interior surfaces only.
- RDCON and STRIDE are never combined, and nothing is combined across n.

Memory: RDCON at n=2 peaks at about 25 GB RSS per process; DCON's `match` companion grows faster than linearly with the number of harmonics (n=6, mpsi 512: 9.4 GB at mpert 99, over 21.7 GB at mpert 132), so DCON reserves 0.7 GB·n². Size `--workers` for RDCON phases from that number, not from the core count.

Mode coverage (#1429): RDCON and STRIDE are run for n = 1, 2 only; the DCON family (full and truncated edge, mpsi 256/512) for n = 1..6. `build_atlas.py --modes 1 2 3 4 5 6` reports RDCON/STRIDE as NOT_APPLICABLE for n >= 3.
