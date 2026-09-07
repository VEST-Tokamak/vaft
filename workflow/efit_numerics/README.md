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

## What comes next, in order

1. **Termination scan** — vary `ERRMIN`, `SAICON`, `NXITER`, `MXITER` and
   `RELAX` over the slices that do produce an equilibrium, and find the
   configuration under which the reference case converges. `EFITNumericsConfig`
   now expresses all of them, and the scientific hash moves with them.
2. **Then #196**, holding the pinned configuration fixed, on the slices this
   baseline shows failing in initialization.
3. **Then #468, #459 and #579**, each with the other two fixed.
