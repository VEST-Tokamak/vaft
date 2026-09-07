# How a VEST EFIT fit terminates today (#171)

Issue #171 asks for VEST's EFIT solver configuration to be characterized and
then stated explicitly instead of inherited. This directory is the
characterization half. It changes nothing: it runs the routine configuration
over each reference shot's full constraint window at the routine cadence and
records *why* every slice stopped where it did.

```bash
PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \
    python workflow/efit_numerics/baseline_termination.py \
        --output /scratch/baseline --markdown /scratch/baseline.md
```

Shots default to the reference set's magnetics arm
(`workflow/efit_reference_set/`). The remote and raw databases are not
readable from a developer machine today, so the baseline is what the packaged
products support; widening it is follow-up work, not a blocker.

## The unit is a discharge, not a slice

A slice that fails in the boundary finder during the ramp and a slice that
fits in the flat-top are different facts about the same configuration, and a
summary that averaged them would report neither. So every slice is classified
before it is counted:

- **phase** — plasma or vacuum, by the writer's own `CUTIP` current cut, so
  the classification is the one EFIT already applies.
- **exit path** — `iconvr=2` (the chi-square criterion), exhaustion of
  `MXITER`, or a solver error.
- **acceptance failures** — the numbered criteria `chkerr` reports, so "did
  not converge" is replaced by which of EFIT's twenty-two criteria was
  violated.
- **null solutions** — the residual falling below 1e-5 while the
  Grad-Shafranov error stays above 0.1. That is not a fit, and counting it as
  one would flatter every summary it entered.

## Two things about the log that mislead if taken at face value

**The `chi2` in the iteration line is not the fit's chi-square.** After the
first step it collapses to ~1e-7 on slices whose a-file reports 200. The
baseline therefore keeps the first and last values under their own names and
takes EFIT's own answer, the a-file's `chisq`, as the chi-square. Anything
that reported the log value as the chi-square would report convergence where
there is none.

**A slice with no a-file is not a slice with a bad fit.** Roughly half the
plasma slices never write one, and a summary over a-files alone silently
drops them. The log is what says how many slices there were.

## What the baseline found

The routine configuration, over each reference shot's whole discharge, with
the channel decisions the diagnostics assessment implies:

| shot | window [s] | plasma slices | wrote an a-file | collapsed | exits on chi-square | exits on a solver error | accepted | iterations (med) | a-file chi2 (med) | GS error (med) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 39915 | 0.306–0.331 | 22 | 7 | 9 | 13 | 9 | **0** | 11 | 74.4 | 0.0089 |
| 41524 | 0.315–0.336 | 21 | 5 | 13 | 8 | 13 | **0** | 11 | 131 | 0.317 |
| 41672 | 0.312–0.352 | 34 | 19 | 7 | 27 | 7 | **0** | 11 | 22.8 | 0.0082 |

Every plasma slice on every shot sits above the 5 kA cut, so none of this is a
vacuum artefact. **No slice on any shot is accepted.** The acceptance failures
are dominated by #21 (Bp not consistent) and #5 (minor radius out of range),
with #1 (chi-square above SAICON's 80) on a minority.

Gating the condemned channels changed one shot and not the others: on 39915 it
moved the median chi-square from 59.1 to 74.4 and cost one a-file, while 41524
and 41672 were unchanged to three significant figures. The condemned probes
matter on one discharge and not measurably on the other two, which is worth
knowing before a study attributes a difference to a setting.

The two conclusions that shape what should be scanned next:

**The fit never reaches the tolerance it is given.** Every k-file writes
`ERROR = 1e-5`. The Grad-Shafranov error at exit is 0.008 to 0.3, three to
four orders above it, and the loop leaves on the chi-square criterion after
eleven iterations on all three shots. `ERRMIN` and `SAICON` — the two settings that decide this
— were not written at all until #171's configuration work, so EFIT's own
defaults of 1e-2 and 80 have been in force throughout. That is the mechanism,
and it is now settable.

**Termination is not the only thing failing.** Only 7 of 22, 5 of 21 and 19 of
34 plasma slices produce an equilibrium at all; the rest end in the boundary
finder or collapse to a null solution, neither of which a termination setting
can fix. Those belong to initialization (#196) and to the
computational domain (#459), and separating them from the termination
question is a precondition for scanning either. A scan that varied `ERRMIN`
across slices that never produced an equilibrium would measure nothing.

## Why no slice is accepted: the envelope, not the termination settings

`chkerr` does not only ask whether the fit converged. It tests every solution
against the `&incheck` bounds EFIT reads from `mhdin.dat` in the table
directory, and those bounds are dimensioned for a machine. EFIT's compiled-in
values are DIII-D's (`aminor_min = 30 cm`, `rcntr_min = 90 cm`); the packaged
VEST table softens them part-way, to 25 cm and 30 cm. VEST does not reach
them. `acceptance_envelope.py` audits the bounds against what the reference
set actually produced:

| a-file scalar | failure | packaged bound | VEST median | VEST range | rejected by the bound |
| --- | --- | --- | --- | --- | --- |
| `aminor` | #5 | 25 … 75 | 23.1 | 7.6 … 29.4 | **21 of 38** |
| `rcntr` | #7 | 30 … 160 | 33.5 | 18.0 … 39.8 | **9 of 38** |
| `rcurrt` | #8 | 30 … 160 | 31.8 | 17.9 … 37.0 | **11 of 38** |
| `elong`, `zcntr`, `zcurrt`, `li`, `qstar` | #6, #9, #10, #2, #13 | — | — | — | 0 of 38 |

Counts exclude collapsed reconstructions, which write zeros and are correctly
rejected by any floor. **A VEST plasma is rejected for being small**, and no
termination setting can move that.

The proposed envelope is derived from the machine, never fitted to a
discharge — the same rule the Green-table work had to keep.
`vest_acceptance_envelope` takes the limiter outline from the canonical static
ODS (R 0.104–0.760 m, Z ±1.185 m) and the grid, and sets `aminor_max` to half
the limiter's radial extent, `aminor_min` to five grid cells (below which a
boundary is not resolved), and the centre and centroid bounds to the limiter
inset by `aminor_min`. It rejects none of the 38 reconstructions. The physics
bounds — `li`, `betap`, `qstar`, elongation — and the consistency tolerances
are left exactly as they were: what those should be is a separate argument,
and moving them alongside the geometry would confound the two.

`write_mhdin` emits an `&incheck` block when an envelope is passed, so a
generated table carries the VEST envelope while the bundled table keeps its
own. The Green tables themselves are byte-identical either way.

### Measured: the envelope works, and it is not enough

The baseline was rerun against a table directory identical to the packaged one
byte for byte except for its `&incheck` block — same Green tables, same
geometry, same constraints, same executable.

**The three geometric failures disappear completely.** #5, #7 and #8 are gone
from every shot; nothing else moved. The derived envelope does exactly what it
was meant to and has no side effects.

**Acceptance does not change.** Not one slice becomes acceptable, because
failure **#21 fires on every slice that produces an equilibrium** — 7 of 7,
5 of 5, 19 of 19 — and one failure is enough to set `lflag`.

| | packaged envelope | VEST envelope |
| --- | --- | --- |
| a-files written | 7 / 5 / 19 | 7 / 5 / 19 |
| accepted | 0 | 0 |
| failures #5, #7, #8 | 21, 9, 11 slices | **none** |
| failure #21 | every slice | every slice |

### Measured again, with the virial gate ignored (issue #649)

The virial consistency checks are ill-conditioned at VEST's aspect ratio and
cannot decide the question they are asked, so they no longer gate acceptance.
Rerunning with that policy and nothing else changed:

| | packaged envelope | + VEST geometry | + virial gate ignored |
| --- | --- | --- | --- |
| a-files written | 31 | 31 | 31 |
| **accepted** | **0** | **0** | **18** |
| failures #5, #7, #8 | 41 slices | none | none |
| failures #20, #21 | every slice | every slice | none |
| failure #1 | 13 slices | 13 slices | 13 slices |

Per shot: 4 of 7 on 39915, 2 of 5 on 41524, 12 of 19 on 41672. **These are the
first accepted VEST reconstructions in this line of work.**

What remains is a single criterion, and it is the right one: **#1, chi-square
above `SAICON = 80`**, on exactly the 13 slices that are not accepted. Every
other criterion passes on every slice. The acceptance question has gone from
"nothing passes, for six different reasons, three of them arithmetic" to "one
well-posed fit-quality threshold rejects 13 of 31" — which is precisely what
the termination scan is for, and it is now a meaningful scan.

**What this does not say.** An accepted slice is not a correct one. Poloidal
beta is still about 0.007 across all of them, and #386's pressure deficit is
untouched by any of this. Acceptance now means "converged, with a geometry the
machine can contain and a chi-square under threshold", which is what EFIT's
criteria were always meant to mean, and no more.

### What #21 actually is

`beta_li.F90:874` computes it as the relative disagreement between the virial
estimate of beta_p — from the Shafranov integrals and the **measured**
diamagnetic flux — and the equilibrium's own beta_p:

```
sbpp  = (s1 + s2*(1 - rttt/rcentr))/2 - exmui     ! exmui carries the measured diamagnetic flux
delbp = |(sbpp - betap)/sbpp|                     ! rejected when >= delbp_diff = 0.08
```

Across the 31 equilibria it runs 0.43 to 4.27, median **0.85** — eleven times
its tolerance. That is not a tolerance that needs adjusting. It is issue #386:
the reconstruction's beta_p is about 0.007 while the virial estimate from the
measured diamagnetic flux is of order one, because under `legacy_weight` the
diamagnetic constraint is emitted weightless and the fit never constrains
pressure.

This is the pressure deficit #386 records, and it is real. But `delbp` cannot
measure it: `sbpp` is a difference of two terms near 0.8 leaving 0.01 to 0.2,
negative on three slices, with a median cancellation of twelvefold; and the
second branch divides by `alpha - 1`, which VEST measures at 0.22 to 0.51 and
falling. Issue #649 records the analysis and the decision: at VEST's aspect
ratio these two checks do not gate acceptance. The quantities are still
computed and written to the a-file, so nothing is lost to anyone who wants
them — only the automatic rejection stops.

**#386 therefore no longer blocks acceptance**, though it remains an open
defect and every accepted slice still carries essentially no pressure.

## What the termination settings can and cannot do

Reading `fit.F90` and `response_matrix.F90` against the baseline narrows this
sharply. Under `ICONVR = 2` the exit test is five nested conditions, and two
of them are inert for VEST: `saisq <= SAICON` and `|saisq - saiold| <= 0.10`
both compare the ~1e-7 linear-solve residual, not the physical chi-square. The
exit therefore reduces to a hard-coded minimum of eight iterations **and**
`errorm <= ERRMIN`, which is why every slice leaves at about eleven.

Two consequences for any scan:

- **`ERRMIN` is the only active gate.** The `ERROR = 1e-5` the k-file writes is
  compared at `fit.F90:236`, but the path it gates also requires
  `chisq <= SAIMIN`, and `ERRMIN` at 1e-2 always fires first.
- **`SAICON` cannot bind while `NXITER = 1`**, because `chisqr` then runs only
  on the first outer pass and `chisq` holds the linear residual thereafter.
  Raising `NXITER` restores the physical chi-square to the test as a side
  effect, so the two must be scanned together, never `SAICON` alone.

## What comes next, in order

1. **Termination scan** — now well posed. One criterion rejects the remaining
   13 slices, chi-square above `SAICON`, and the settings that bear on it are
   exactly the ones this issue names.
2. **#386 in parallel** — the pressure deficit no longer blocks acceptance but
   is still a defect: every accepted slice carries a poloidal beta of about
   0.007 against a virial estimate of order one.
3. **Termination scan, in detail** — `ERRMIN` first, since it is the only active gate,
   then `NXITER` with `SAICON` (which cannot bind without it), then `MXITER`
   and `RELAX`. Keep `MXITER × NXITER` below 500: EFIT indexes its
   per-iteration diagnostic arrays by the cumulative counter against a
   compiled-in bound of 515 with no runtime clamp.
3. **Then #196**, holding the pinned configuration fixed, on the slices this
   baseline shows failing in initialization.
4. **Then #468, #459 and #579**, each with the other two fixed.
