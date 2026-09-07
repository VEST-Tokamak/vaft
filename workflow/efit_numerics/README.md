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

## The first-slice seed and the convergence basin (#588)

`seed_basin.py` varies the seed and nothing else — no `ICINIT`, no termination
setting — because #588 requires that seed geometry, temporal continuation and
stopping criteria stay separable or nothing can be attributed. Three
discharges, the routine seed as row one, one axis at a time.

### The seed could not be varied independently, and that is the first finding

`EFITInitializationConfig.rzero` drove **three** namelist quantities: `RELIP`
(where the seed ellipse sits), `RZERO` (the reference major radius) and
`RCENTR`, which sets `BTOR = (B_t R)_measured / RCENTR`. A radial sweep
therefore moved the seed, the normalisation and the vacuum toroidal field
together. `ellipse_rzero` now drives `RELIP` alone, and setting it to `None`
restores the old coupling, so the pre-#588 k-file is still reproducible
byte-for-byte. The results below are from the corrected sweep; the first one
was discarded.

### The basin is narrow, and the routine seed sits near its outboard edge

Routine totals across the three discharges: 31 equilibria, 18 accepted.

| seed change | equilibria | accepted | slices recovered | slices lost |
| --- | --- | --- | --- | --- |
| `ellipse_rzero` 0.30 (inboard) | **38** | **21** | 13 | 4 |
| `ellipse_rzero` 0.35 | 35 | 19 | 10 | 5 |
| routine, 0.40 | 31 | 18 | — | — |
| `ellipse_rzero` 0.45 | 22 | 19 | 6 | 16 |
| `ellipse_rzero` 0.50 (outboard) | 15 | 15 | 3 | 20 |
| `zzero` −0.05 m | 18 | 7 | 4 | 17 |
| `minor_radius` 0.20 | 34 | 22 | 4 | 2 |

Moving the seed 5 cm outboard costs slices on every shot; 10 cm outboard
halves the yield. Moving it 5–10 cm inboard gains on every shot. A 5 cm
**vertical** offset costs 13 slices on 41672 alone, which is worth stating
because that axis was expected to do nothing by symmetry.

The asymmetry has a physical reading: the reconstructions put VEST's plasma
centre at R ≈ 0.32 m, so a seed at 0.40 m starts outboard of where the plasma
actually is, and pushing it further out leaves the basin.

### The collapse block is not a seed problem

Of the 29 slices that produce nothing, **only four are ever recovered by any
seed**, and every one of them sits at the trailing edge of its block —
position 0 or 1 from the end, adjacent to where the fit starts working:

| shot | collapse block | ever recovered |
| --- | --- | --- |
| 39915 | 307–315 (9 slices) | 315 |
| 41524 | 315–327 (13) | 327 |
| 41672 | 315–321 (7) | 320, 321 |

The seed can move the boundary between the failing and working phases by a
slice or two. It cannot touch the block. Across a threefold range in minor
radius, a factor of two in elongation, ±10 cm radially and ±5 cm vertically,
86 % of the collapse block is untouched. **That population belongs to #459**,
with the 17 `findax` losses, not to initialization.

### Refining the region, and the new default

The coarse sweep is on a 5 cm grid, which is too coarse to say where the good
region begins or ends — a single best point on that grid is not an optimum.
So the radial axis was re-run at 2 cm over the same three discharges and the
same 77 plasma slices. The rerun of 0.30 m reproduces the coarse row exactly,
so the differences between neighbouring points are solver behaviour, not run
noise.

| `ellipse_rzero` | equilibria | accepted | recovered | lost |
| --- | --- | --- | --- | --- |
| 0.28 | 31 | 14 | 9 | 7 |
| 0.30 | 38 | 21 | 13 | 4 |
| **0.32** | **39** | **22** | 11 | **2** |
| 0.34 | 33 | 18 | 6 | 3 |
| 0.36 | 34 | 19 | 8 | 4 |
| routine, 0.40 | 31 | 18 | — | — |

Two things are established and one is not. Established: the whole 0.30–0.36 m
region beats the routine seed, and below 0.30 m the gain disappears — 0.28 m
is back to the routine's 31 equilibria with four fewer accepted. Not
established: a point inside the region. The response is not smooth, 0.34 m
falls back to 33 while 0.36 m recovers to 34, and one or two slices out of 77
is not a resolvable difference.

**The routine default is now `ellipse_rzero = 0.32 m`, and only that field.**
`rzero` stays at 0.40 m, so `RZERO`, `RCENTR` and the `BTOR` derived from it
are untouched; `test_the_inboard_seed_default_moves_relip_and_nothing_else`
compares the two k-files line by line and fails if anything else moves. The
argument for 0.32 m specifically is physical rather than fitted: the reference
set's own reconstructions put the current centroid at a median of 31.8 cm and
the boundary centre at 33.5 cm, so 0.40 m seeds the ellipse at the *vessel's*
centre and asks the solver to walk inboard on every slice. That it is also the
best measured point is corroboration, not the argument. Read no more precision
into the second digit than 77 slices from three discharges can carry.

One trap found while making the change: `generate_constraints_ods` writes its
own hard-coded `RELIP = 0.4` into the ODS `code.parameters` tree, beside
`RZERO`, `AELIP` and `EELIP`. The k-file writer overrides all of them from the
configuration — which is why the sweep varied anything at all — but nothing
said so, so that is pinned too.

### What this leaves for #196

Temporal continuation still has a case — a warm start might carry a converged
solution into the block from the working side, which is a different mechanism
from re-seeding. But #588's premise, that the first slice must be seeded into
a good basin before continuation can help, is now answered: the basin is
narrow and off-centre, and a better seed is available and justified.
