# The VEST coupling asset: exact reciprocity, with provenance (issues #373, #347)

`vaft/data/geometry/VEST_em_coupling_pf_versions.npz` holds the five mutual
inductance matrices every eddy-current solve, wall-mode basis and plasma-free
benchmark reads through `vaft.machine_mapping.em_coupling`:

| key | shape | copied verbatim |
|---|---|---|
| `mutual_active_active_1906` / `_2507` | 10 × 10 | yes (symmetric to 1e-15) |
| `mutual_passive_active_1906` / `_2507` | 950 × 10 | yes |
| `mutual_passive_passive` | 950 × 950 | **repaired** |
| `provenance` | JSON string | written by this tool |

## The defect, and what it was

A mutual-inductance matrix is reciprocal: `M_ij == M_ji` exactly. The
committed `mutual_passive_passive` was asymmetric by 1.27e-3 (relative) on
331,200 of 902,500 entries, and until #347 it was loaded verbatim; since #347
the mapper folded it to `(M + Mᵀ)/2` on load with a warning, and develop's
wall-mode check refuses any product that still carries the raw matrix.

The asset was extracted from the legacy VFIT products, whose kernel
`getMutualInductanceCoil.m` reads

```matlab
if fix(Coil1_Material)==2 || fix(Coil1_Material)==2   % sic
    mu_r=1.04;
```

The second test was meant to be `Coil2_Material` (the commented-out line
above it says so), so the SUS316LN factor 1.04 was applied only when the SUS
conductor was the *first* argument, and a full `i, j` double loop then stored
the SUS-side value at `[SUS, W]` and the W-side value at `[W, SUS]`. The
matrix says exactly this: loops 0–719 are SUS (resistivity 7.8e-7) and
720–949 tungsten (5.6e-8) in `VEST_static_geometry.json.gz`; both
within-material blocks are symmetric to zero; every cross-material pair
differs by the factor 1.04 to 1e-12; the asymmetric count is 2 × 720 × 230.

## The repair

`regenerate_passive_coupling.py --repair` sets the cross block to the
SUS-side value (μ_r = 1.04 whenever either conductor is SUS), which is the
donor's evident intent and the factor `vaft.omas.process_wrapper` already
applies to every non-W11 loop. Nothing else changes: within-material blocks
and the active-side matrices are copied bit for bit. The result is symmetric
to exactly 0, and the asset gains a `provenance` record (generator, date,
git commit, source and static-geometry SHA-256, the factor, the convention,
the input and output asymmetry, the legacy source digests). The mapper
surfaces that record into `em_coupling.code.parameters` and says
"reciprocity exact" instead of "symmetrized on load".

This is a documented repair of one material factor, **not** a regeneration
from geometry. `vaft.formula.green.mutual_inductance` / `self_inductance`
could rebuild all five matrices symmetric by construction, but the donor's
own symmetrized products differ from this asset by up to 4 %, so that is a
different, larger decision and out of scope here.

The tool refuses anything it cannot explain: a non-constant cross ratio,
asymmetry inside a material block, a non-finite entry, an active-side matrix
that is not reciprocal. It is idempotent: `--repair` on the repaired asset
writes nothing.

```bash
PYTHONPATH=. python workflow/em_coupling/regenerate_passive_coupling.py --verify
PYTHONPATH=. python workflow/em_coupling/regenerate_passive_coupling.py --repair --dry-run
```

| | SHA-256 |
|---|---|
| before | `2d645cffaa6350955fb131779c6f5129c6c478ff0b480136d7686fef717fe819` |
| after | `1c391442762fb1c9563eb5fc62fef35ee4e002c7cda8afc06968539a5e2a6f05` |

## What it changes in production

The repaired matrix differs from the load-time average the pipeline has used
since #347 by 6.4e-4 of max|M|, in the cross block only. PF-driven wall
currents (`benchmark_wall_currents`, plasma-free) on the packaged shots:

| shot | max \|ΔI\| / max\|I\| | rms | loops moving > 1 % | slowest wall τ |
|---|---|---|---|---|
| 39915 | 1.02e-3 | 3.3e-4 | 1 (W11, 1.0 %) | 7.187 ms → 7.187 ms |
| 41524 | 6.5e-4 | 2.2e-4 | 0 | unchanged |
| 41672 | 7.0e-4 | 3.5e-4 | 0 | unchanged |

Tungsten loops move most (≤ 1.0 %), SUS loops ≤ 0.35 %.

## The packaged samples

39915 carries no `em_coupling` and reconstructs it on load; unchanged. 41524
and 41672 shipped the raw pre-repair matrix inside their frozen pipeline
product, which the wall-mode check refused. Their `imas.nc` are regenerated
with `em_coupling` re-mapped from the repaired asset
(`generation.em_coupling` in each manifest records the asset digest, the
replaced asymmetry, and that the passive-loop currents remain those the
frozen eddy stage solved against the old matrix).
