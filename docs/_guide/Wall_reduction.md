---
title: Reduced-wall order study
author: VEST team
date: 2026-09-04 18:00
category: guide
layout: post
mermaid: false
permalink: /reference/wall-reduction/
guide:
  architecture: How many wall modes the VEST vessel needs, decided by response fidelity rather than by decay time, and checked against measured magnetics.
  prerequisites: The wall eigenmode basis (see Wall eigenmodes) and a loaded ODS with pf_active waveforms, pf_passive, em_coupling, magnetics and the limiter outline.
  expected: How to run the full-versus-reduced convergence study, read off a representation order, compare full and reduced walls with the data, and find plasma-free shots.
---

# Reduced-wall order study

The wall eigenmode basis ([Wall eigenmodes](/reference/wall-modes/),
[#473](https://github.com/VEST-Tokamak/vaft/issues/473)) can represent the
vessel current in any number of modes. This page is about *how many, and
which* — the representation order `M_repr = (M_1, …, M_G)` that
[VEST-Tokamak/vfit#10](https://github.com/VEST-Tokamak/vfit/issues/10) asks
for, implemented in `vaft.validation.wall_reduction`
([#494](https://github.com/VEST-Tokamak/vaft/issues/494)).

Three layers, kept apart on purpose:

| layer | question | function |
|---|---|---|
| full ↔ reduced | how much of the wall's own response does a truncated basis miss? | `order_convergence` |
| representation order | which is the smallest order within the tolerances an application sets? | `representation_order` |
| measurement ↔ model | do full and reduced walls disagree with the data alike (vessel-model error) or differently (reduction error)? | `experimental_comparison` |

Metrics only, no verdicts: tolerances are the study's and are reported with
the result.

## The reduced circuit

With the projected operators of the basis the reduced wall obeys

```
L_r da/dt + R_r a = -M_r dI_coil/dt,        I_w = V a
```

integrated by the same routine as the full wall
(`vaft.process.electromagnetics.solve_eddy_currents`), so the only difference
between the two solutions is the retained subspace. Keeping every mode
reproduces the 950-loop solve to 1e-12; the error of any truncation is the
basis span, not the closure.

```python
from vaft.process import wall_modes as wm
from vaft.validation import wall_reduction as wr

system = wr.wall_system(ods)                # R, L, source coupling, basis, PF drive
keep = wm.select_by_score(system["basis"], wm.mode_scores(system["basis"], system["R_mat"],
                          system["M_mat"], system["L_mat"], drive=system["drive"],
                          time=system["time"])["output_weight"], 76)
ops = wm.reduced_operators(system["basis"], system["R_mat"], system["M_mat"], system["L_mat"], keep)
a, I_w = wm.solve_reduced_eddy(ops, system["drive"], system["time"], V=system["basis"].V(keep))
```

## Which modes: rankings, not the spectrum

`mode_scores` returns one value per mode under four rankings — `tau` (the
decay time), `drive_gain` (quasi-static amplitude a unit source ramp excites,
times observability; needs no drive), `response_energy` (rms projected
amplitude under a drive) and `output_weight` (energy times observability).
`select_by_score` keeps the top `M` across segments; `allocate_per_segment`
grows a per-segment allocation greedily until a dissipation or output
tolerance is met, adding modes where the remaining error is largest.

On the packaged wall (shot 39915, its PF programme, 62 probes and 11 flux
loops), relative error of the wall term at the probes:

| retained | slowest (`tau`) | one per segment | `output_weight` |
|---|---|---|---|
| 19 | 28 % | 15 % | 7.4 % |
| 76 | 24 % | 6.1 % | 1.2 % |
| 152 | 5.6 % | 3.4 % | 0.26 % |

Decay time alone is a poor guide: the driven response is not sparse in eigen
coordinates, and a whole-wall eigenbasis behaves the same as the segment-wise
one, so the penalty is intrinsic to eigen coordinates rather than to the
segmentation.

## Moment patterns: the enrichment the contract does not yet include

The response to the coils lives in the block Krylov space
`span{R⁻¹M, (R⁻¹L)R⁻¹M, …}` — the coil-controllable subspace — and
`moment_patterns` returns an R-orthonormal basis of it, block by block, built
as a block Arnoldi iteration in the R inner product. The first block is the
resistive limit, the wall current a constant ramp settles into; each further
block is the next inductive correction (the transfer function's moments at
zero frequency).

| patterns | probe error, PF programme | probe error, step drive |
|---|---|---|
| 10 (resistive limit) | 1.7 % | — |
| 19 | 0.9 % | 25 % |
| 76 | 1e-4 | 0.6 % |

Ten drive-independent vectors do what 76 eigenmodes do, and 76 do what no
eigen truncation reaches. These patterns are global, not segment-tagged, so
they are outside the reduced-wall contract (vfit #8); the study reports them
as the alternative the downstream EFIT issues can decide on.
`combined_operators` projects the wall onto any R-orthonormal basis, eigen
or not.

## Running the study

```python
observation = wr.observation_set(ods, n_coils=system["n_coils"])   # probes, loops, boundary ring, grid
rows = wr.order_convergence(system, observation=observation)        # every (rule, order, drive)
order = wr.representation_order(rows, {"probe": 0.01, "flux_loop": 0.01, "grid_psi": 0.01})
```

`order_convergence` scores the drive-dependent rankings on the PF programme
and then tests every selection on three drives — the programme, the loudest
coil alone, and a step on it — so a row for `step` says how a selection made
for the programme transfers to a transient it never saw. Rows carry the
Euclidean and dissipation-weighted current error, the error per segment, the
relative error at each observation class, the probe peak error and the cost.

Two plots draw the rows: `plot_passive_structure_overview_wall_reduction`
(error against retained order, one panel per metric) and
`plot_passive_structure_field_wall_reduction` (the wall's flux on the
equilibrium region — full, reduced or their difference).

## Against the measurements

```python
result = wr.experimental_comparison(ods, {"tau_19": wm.select_slowest(system["basis"], 19),
                                          "moments_30": wm.moment_patterns(system["R_mat"],
                                                        system["M_mat"], system["L_mat"], 3)})
```

This runs the [#190](https://github.com/VEST-Tokamak/vaft/issues/190)
plasma-free benchmark for the full PF-only wall and for each reduced wall
over the same window, and reports two blocks per model: `measurement`
(residual against the data) and `reduction` (distance from the full model
at the same channels). On shot 39915 the full wall improves the coil-only
residual by a median 91 %, and every reduced wall — even the nineteen
slowest modes, 6 % from the full wall term at the probes — improves it by
the same 91 % to within a point: the residual that remains is the vessel
model's, not the truncation's.

`find_plasma_free_shots` classifies database shots as `plasma_free`,
`plasma`, `undriven` or `daq_missing` from their plasma-current and coil
peaks, with the evidence, so dedicated vacuum shots can be brought into the
comparison; the plasma-free interval of a mapped shot is judged by
`vaft.validation.vacuum_benchmark.plasma_free_interval` as for any other case.
