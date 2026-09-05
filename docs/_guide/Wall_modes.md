---
title: Wall eigenmodes
author: VEST team
date: 2026-09-03 10:00
category: guide
layout: post
mermaid: false
permalink: /reference/wall-modes/
guide:
  architecture: The segment-wise eigenbasis of the VEST passive wall, and the reduced operators built from it.
  prerequisites: A loaded ODS with pf_passive geometry and em_coupling (the packaged samples work offline).
  expected: How to build, inspect, project onto and reduce with the wall-mode basis; what it guarantees and what it refuses.
---

# Wall eigenmodes

The reduced vessel-wall contract is owned by VFIT
([VEST-Tokamak/vfit#8](https://github.com/VEST-Tokamak/vfit/issues/8)); this
page describes the VAFT implementation
([#473](https://github.com/VEST-Tokamak/vaft/issues/473)). The principle in one
line: **local eigenbasis, global electromagnetic dynamics**.

## What it is

The passive wall is ~950 filament loops. They are grouped into physical
segments — by loop name, split where a structure has a genuine gap in Z — and
each segment gets its own L/R eigenbasis. The bases are block-assembled into
`V_seg`, and every reduced operator is then a projection of the **full**
matrices, so the mutual inductance between segments survives in the
off-diagonal blocks of the reduced inductance.

```python
import vaft
from vaft.omas.process_wrapper import compute_impedance_matrices_ods, compute_wall_mode_basis_ods
from vaft.process.wall_modes import reduced_operators, reduce_response, select_slowest, check_wall_mode_basis

ods = vaft.omas.sample.sample_ods()          # the packaged 39915 machine
basis = compute_wall_mode_basis_ods(ods)     # 19 segments, 950 modes, ~0.15 s

basis.n_modes()                               # (240, 10, 10, 8, 8, 23, 23, 230, ...)
basis.labels()[:3]                            # (('W1', 0), ('W1', 1), ('W1', 2))
basis.segment("W1").tau[:3]                   # slowest local decay times [s]

R_mat, L_mat, M_mat = compute_impedance_matrices_ods(ods, [])   # code naming: M_mat is the inductance
keep = select_slowest(basis, 40)              # forty slowest modes, wherever they live
ops = reduced_operators(basis, R_mat, M_mat, L_mat, keep)
ops.L_r.shape, ops.R_r                        # (40, 40), identity
G_red = reduce_response(G_full, basis, keep)  # any (n_obs, 950) response -> (n_obs, 40)
check_wall_mode_basis(basis, R_mat, M_mat)["coupling"]["W9_L-W8_L"]   # 0.91: strongly coupled halves
```

## Conventions

| item | rule |
|---|---|
| segmentation | by `pf_passive.loop.*.name`, split at Z gaps > 1.5 × median loop height (`vest-name-zgap-1.5-v1`); ordered by first element index; W11 included |
| eigenproblem | per segment, `S_g = R^{-1/2} L R^{-1/2}`, `eigh`; eigenvalues are the decay times, descending |
| normalization | R-orthonormal: `vᵀ R v = 1`, `vᵀ L v = τ`; amplitudes in √W (`aᵀa` is the ohmic dissipation) |
| sign | largest-magnitude component positive |
| projection | `a = V_segᵀ R I_w` — exact in the retained subspace; **not** `V_segᵀ I_w` |
| reduced operators | `R_r = I` (computed and checked), `L_r = V_segᵀ L V_seg`, `M_r = V_segᵀ M_src`, `G_red = G V_seg` |
| indexing | `(segment_id, k)`, segment-major; per-segment orders `M_repr = (M_1, …, M_G)` |

The basis chooses no reduced order; `select_slowest` and `select_tau_range`
exist for the study that will (vfit #10).

## What it refuses

A non-diagonal or non-positive resistance, an inductance block that is not
positive definite, a condition number above 1e12, an inductance asymmetric by
more than 1e-6 (relative), and a near-degenerate pair of decay times inside a
segment (relative gap below 1e-6; pass `on_cluster="warn"` to record instead).
The packaged coupling asset is asymmetric by 1.27e-3 (#347); the mapper
symmetrizes it on read, and an artifact materialized before that must be
re-mapped: `compute_wall_mode_basis_ods(ods, remap_em_coupling=True)`.

## Provenance

`basis.provenance` records the segmentation version and digest, the resistance
calibration in force (#308), the input asymmetry, the normalization, sign rule
and mode order; `basis.digest()` fingerprints the modes. With
`record=True` the identity is appended to `em_coupling.code.parameters` as
`wall_mode_*` lines — never under `pf_passive`, whose loops are compared
bitwise against the reference geometry. The basis is computed on demand, not
shipped; `to_npz` / `from_npz` exist for export.

## Related

`wall_time_constants` in the plasma-free benchmark now reads the same pencil
at full rank, so the QA and the basis cannot report two different walls.
TokaMaker's `eig_wall` (#232) remains an independent global reference on its
own mesh (6.88 ms vs 7.19 ms here; W11 excluded there).
