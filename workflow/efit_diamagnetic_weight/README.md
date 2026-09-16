# Is the diamagnetic constraint reachable, and is the pressure real? (#386)

#649 stopped two ill-conditioned virial checks from gating acceptance and the
yield went from 0 of 31 to 30 of 77. That makes this the moment of maximum risk
of reading "more slices pass" as "the reconstructions are good". #386 records
the reason to doubt it: the reconstructed pressure sits roughly ninety times
below what force balance requires, and the one measurement that would pin
pressure — the diamagnetic loop — carries no weight.

This workflow asks whether that is a **weighting** problem or a **data**
problem, and it exists because the obvious experiment has already been run and
could not have detected the answer.

## Why #663's answer does not settle it

#663 swept `objective_scales["diamagnetic_flux"]` from ×1 to ×10,000 on all
three reference shots and classified the family as inactive within a produced
solution. That is a correct measurement of what it measured. It is not a
statement about the diagnostic, because the sweep never left the region where
the row is inactive by construction.

EFIT processes a statistical row as `FWT/sigma`:

| step | where | value |
| --- | --- | ---: |
| `SIGDLC = weight × legacy_weight_scale × 1000` | `vaft/code/efit/kfile.py` | `1e7` mWb |
| `sigdia = 1e-3 × abs(SIGDLC)` | `data_input.F90:2297` | `1e4` Wb |
| `fwtdlc = fwtdlc / sigdia**nsq`, `nsq = 1` | `data_input.F90:2782`, `:109` | — |
| row `= fwtdlc × rspdlc`, rhs `= −fwtdlc × diamag` | `response_matrix.F90:2192` | — |

So the processed row weight VEST submits is `FWTDLC/sigdia = 1e-4`. The stored
3 % measurement error would give `2.3e4`. **The gap is 8.4 decades, and the
×1…×10,000 ladder covers four of them** — every one inside the inactive region.
#664's `10^0`–`10^4` ladder sits in the same place.

The instrument this workflow adds is
`constraints.uncertainty_scales["diamagnetic_flux"]`, which divides `SIGDLC`
for that family alone. It ships at `1.0` for every family, so an unscaled
k-file is byte-identical to the one the writer has always produced.

### Why the sigma and not the weight

Both reach the same row weight. Scaling `FWT*` to `1e8` would additionally
write a value far outside anything EFIT was exercised with and would perturb
the branches that test `fwtdlc` directly (`response_matrix.F90:2840`,
`chkerr.f90:61`). Scaling the sigma moves the row weight and nothing else — and
it is the quantity that is actually wrong, and the one a production
uncertainty decision would set.

`legacy_weight_scale` cannot serve: it is shared by every family's legacy
uncertainty, so moving it moves them all. `uncertainty_mode="standard_deviation"`
cannot serve either, for the same reason — it replaces the legacy uncertainty
for all five families at once, so a run under it differs from the baseline on
five axes and attributes nothing. Where each rung sits relative to the real
measurement error is reported per slice instead, as `sigma_ratio_to_stored`.

## The ladder

Ten rungs, one decade apart, carrying the processed row weight from `1e-4` to
`1e5`:

| rung | SIGDLC [mWb] | processed row weight | |
| --- | ---: | ---: | --- |
| `legacy_sigma` | `1e7` | `1e-4` | today's production configuration |
| `sigma_x1e1` … `sigma_x1e3` | `1e6` … `1e4` | `1e-3` … `1e-1` | |
| `sigma_x1e4` | `1e3` | `1` | the far end of #663, in row-weight terms |
| `sigma_x1e5` … `sigma_x1e7` | `100` … `1` | `10` … `1e3` | |
| `sigma_x1e8` | `0.1` | `1e4` | near the stored 3 % error (`2.3e4`) |
| `sigma_x1e9` | `0.01` | `1e5` | past it |

Everything else is the frozen #579/#663 configuration: the `(2,2)` zero-edge
profile, the qualified seed, the numerical controls, the grid, the cadence, and
unit objective scales. One axis at a time, which is the rule #588 established
for the seed study.

## What is asked, in order

1. **Reachability** — at what processed row weight does the diamagnetic
   residual first stop being ignored? Reported on the residual itself, not on
   geometry, because a constraint can be active and still not move the
   boundary.
2. **Consequence** — does the reconstructed pressure follow? `p_axis`, `wmhd`
   and `betap`, **on slices common to every rung**. Population change is
   reported separately and never mixed in. #663 had to publish a correction for
   reporting a smaller aggregate residual at high weight that turned out to be
   population selection; comparing only common slices is what makes that
   mistake impossible rather than merely unlikely.
3. **Direction** — does it move *toward* a measurement the fit never used?

Classification reuses #663's own thresholds, imported from its module rather
than restated: LCFS median ≥ 5 mm, area or volume ≥ 2 %, another family's
residual or χ² ≥ 20 %, acceptance ≥ 10 pp. The two studies therefore compose.
The response floor on the residual is `1e-3`: #663 measured invariance at
`1e-7` through its whole ladder and a real but sub-threshold 0.5 % response at
one slice, so the floor sits three orders from each.

Both chi-squares are kept. The m-file's is `(physical residual/sigma)²` with
the submitted weight divided out; the solver's own contribution is
`(weight × residual)²`. #663's correction notes these are different quantities.

## The kinetic cross-check

`thomson_pressure_check.py` runs no EFIT and answers question 3. **39915 is the
only shot that can be asked**: its ten Thomson samples run 0.308–0.317 s inside
a 0.306–0.331 s EFIT window (`workflow/efit_reference_set/README.md`). 41524
and 41672 carry no packaged kinetic data.

The test is **one-sided, and stays one-sided**. Thomson measures electrons, so
`n_e k T_e` is a lower bound on the total pressure. An electron pressure far
*above* the reconstructed total is decisive — no unmeasured ion population can
make a total smaller. An electron pressure *below* it says nothing at all. The
registry already encodes that asymmetry
(`vaft/validation/registry.py`, `independent_validation.thomson_pressure`).

The packaged `g039915.00319` cannot be used: at 0.319 s it is 2 ms from the
nearest Thomson sample, beyond the half-cadence tolerance the validator
applies. The reconstructions this reads are on a 1 ms grid that lands on the
sample times exactly.

## Running it

The kinetic check first — it needs no EFIT and may answer the question on its
own:

```bash
PYTHONPATH=$PWD python workflow/efit_diamagnetic_weight/thomson_pressure_check.py --baseline /scratch/baseline --shot 39915
```

Then the ladder:

```bash
PYTHONPATH=$PWD EFITHOME=/path/to/efit-install python workflow/efit_diamagnetic_weight/diamagnetic_weight_scan.py --output /scratch/dia-weight --shots 39915,41524,41672 --workers 8
```

`--workers N` runs that many rungs at once. Every EFIT process gets its own
working directory and `run_efit` runs with `cwd=workdir`, so the pool is safe.
**Runtimes from a run with `--workers > 1` are not comparable between rungs** —
scheduling noise reads as a configuration effect. Measure runtime with
`--workers 1`.

Each rung caches on its configuration hash, so an interrupted scan resumes and
a changed configuration re-runs rather than reusing a stale result.

Generated numerical products belong in the chosen output directory and are not
version-controlled with the workflow.

### Two things about the build that will otherwise waste a run

**Keep `--output` short.** `read_limiter.f90:29` declares the limiter filename
as `character*100`, so `INPUT_DIR` plus `lim.dat` must fit in 100 characters.
A longer path is silently truncated and every slice fails to open the limiter,
which surfaces as `0 produced` on every rung with no other symptom. `table_di2`
itself is `character(256)` and prints in full, so the run *looks* correctly
configured. `/scratch/dia-weight` is fine; a deep temporary directory is not.

**The m-file is not required, and is not available.** `write_m.F90` writes
NetCDF and this EFIT build is configured `ENABLE_NETCDF=OFF`, so no
`m0*.nc` appears and #663's `constraint_audit` — which reads `fwtdia` and
`chidflux` from it — attaches nothing. The diamagnetic row is therefore
reconstructed from the a-file's `cdflux` and the k-file's `DFLUX`/`SIGDLC`/
`FWTDLC`, which EFIT always writes:

```
measured   = DFLUX * 1e-3                 [Wb]   data_input.F90:2296
sigma      = |SIGDLC| * 1e-3              [Wb]   data_input.F90:2297
row weight = FWTDLC / sigma                      data_input.F90:2782
residual   = cdflux - measured            [Wb]   beta_li.F90:787,832
```

`cdflux` is EFIT's *exact* forward flux, not the linearised row the fit is
driven by, so this residual is the same quantity `chidflux` scores
(`beta_li.F90:840`). Where a NetCDF-enabled build *does* write an m-file, its
`fwtdia` is read as a cross-check and any disagreement is recorded per slice
as `row_weight_agreement` rather than averaged away.

A slice whose row could not be read is counted in `diamagnetic_row_read` and
never treated as a slice that did not respond — the verdict refuses to
conclude anything when no rung's row could be read at all.

## Result

Run on 39915, 41524 and 41672 at 1 ms cadence, ten rungs each, EFIT commit
`4d10ed5`. Medians over the slices that produced an equilibrium:

| rung | row weight | 39915 produced/accepted | 41524 | 41672 | common-slice residual change |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_sigma` | `1e-4` | 18 / 14 | 11 / 6 | 32 / 21 | baseline |
| `sigma_x1e1` | `1e-3` | 18 / 14 | 11 / 6 | 33 / 22 | `0` … `1.5e-9` |
| `sigma_x1e2` | `1e-2` | 18 / 14 | 11 / 6 | 32 / 21 | `0` |
| `sigma_x1e3` | `1e-1` | 18 / 14 | 11 / 6 | 33 / 22 | `0` … `2.2e-9` |
| `sigma_x1e4` | `1` | 10 / 10 | 3 / 3 | 13 / 13 | `0` … `1.4e-8` |
| `sigma_x1e5` | `10` | 6 / 6 | 2 / 2 | 8 / 8 | `0` |
| `sigma_x1e6`–`x1e9` | `1e2`–`1e5` | 4 / 4 | 2 / 2 | 7 / 7 | `0` |

**`w_activation`: never, on all three shots.** Across nine decades of processed
row weight the common surviving slices are unchanged to `1e-8` — on most rungs
bit-identical. Tracing one slice (39915 at 315 ms, measured `-1.4275e-3` Wb):
`cdflux` is `1.8764e-3` Wb at both `1e-4` and `1e-1`, equal to eight
significant figures, with `p_axis` `35.5588207` against `35.5588206`.

**`w_fail`: `sigma_x1e4` or `sigma_x1e5`.** What ends the ladder is not a
diverging fit but a **collapse to the null solution**: at row weight `1` the
slices that had a real plasma fail in the boundary tracer — *"First and last
contour points are too far apart"*, *"Less than 3 contour points found"*,
*"Number of contour points greater than max allowed"* — with `chi2_final`
around `4.5e-8` and `gs_error` near `0.29`, which is the collapse signature
`vaft/code/efit/termination.py` defines. 39915 at 315 ms goes from `flagged`,
`exit=iconvr=2`, 11 iterations at `1e-1` to `collapsed`, `exit=solver_error`,
9 iterations at `1`.

**The 100 % acceptance at the top of the ladder is an artifact and must not be
read as quality.** The count of degenerate slices — `cdflux == 0` and
`p_axis == 0`, the sub-`CUTIP` vacuum returns — is constant across the whole
ladder (4 on 39915, 2 on 41524, 6 on 41672). High weight does not create them;
it removes everything else, until at `sigma_x1e9` on 39915 all four surviving
slices are vacuum. Vacuum solutions pass the acceptance gate trivially.

So between inert and collapse there is **no usable band**, and the transition
takes less than one decade. The diamagnetic constraint cannot be made to
constrain a VEST reconstruction by weighting. **The pressure deficit of #386 is
not a weighting problem.**

A mechanism worth testing rather than asserting: the row is confined to the
FF' columns (`response_matrix.F90:2192-2196` writes only
`(nbase+1):(nbase+kffcur)`), and #663 found the VEST solution is selected by Ip
plus PF-current anchoring. If FF' is effectively pinned by those, the
diamagnetic row can push on it and be either overruled or fatal, with nothing
in between — which is what this ladder measures.

### The sign convention, on all three shots

The signed measurement reached `DFLUX` on **18 of 18, 11 of 11 and 32 of 32**
compared slices. #385's convention holds on 41524 and 41672, which the
packaged-sample regression never covered.

### Against Thomson

Three reconstructions fell inside the Thomson window (39915 at 315, 316 and
317 ms, five channels each). The electron pressure alone is **2.04x, 3.01x and
4.56x** the reconstructed pressure at the sampled positions, decisive on two
of the three. Electrons are a lower bound, so an excess cannot be explained by
unmeasured ions: **the pressure is real and the fit is missing it. Not a data
problem either.**

State the factor carefully. This is 2-4.6x *at the Thomson channel positions*,
not the ~90x #386 records — that figure is a volume-integral beta_p against a
virial estimate, a different quantity measured a different way. Thomson
confirms the direction and the reality of the deficit; it does not
independently confirm its size.

The six highest rungs kept no slice inside the Thomson window and say nothing.
This arm falls silent exactly where the ladder is most aggressive.

### One reporting caveat

`chi_squared` and `chi_squared_reweighted` are equal throughout these runs,
because `FWTDLC = 1` makes the processed weight exactly `1/sigma` and the two
definitions coincide. They are kept separate because #663's correction is
about configurations where the submitted `FWT` is not one; here the
distinction is real but degenerate.

## Scope

Characterization only. No production default changes: `uncertainty_scales`
ships at `1.0` for every family and the baseline rung is today's configuration
exactly. Choosing a real `SIGDLC` is a diagnostic-owner decision — loop
calibration and TF pickup residual — per #386, and this study can only bracket
it.

One consequence worth knowing before running: adding `uncertainty_scales` to
`EFITConstraintConfig` changes `EFITScientificConfig.sha256`, so result
directories cached by earlier #663/#664 runs will re-run rather than resume.
That is the cache behaving correctly, not a regression.

## What this study is not

- Not the #659 linearisation bias. That is a ~2 % effect on `beta_p` and its
  sign is *upward*, while #386's deficit is a factor of ninety downward; the
  two are not the same root cause. It is measured from stored reconstructions
  and reported on that issue.
- Not the virial gate. #649's decision stands: `delbp` and `dbpli` cannot
  decide acceptance at `A ≈ 1.4`, and nothing here re-enables them.
