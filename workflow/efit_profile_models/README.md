# EFIT profile-model uncertainty (#579)

The first pass is measured and recorded below; its table is
`test/data/efit_profile_model_study.json`. A new run writes to the requested
output directory and is not version-controlled with the workflow.

This workflow varies EFIT's fitted `P'` and `FF'` representation while holding
the numerical, spatial, temporal, diagnostic, and initialization choices fixed.
It is an uncertainty study, not a search for the lowest chi-square model.

The baseline is explicit:

- `ellipse_rzero = 0.32 m`, independently of `RZERO = RCENTR = 0.4 m`;
- `ICINIT = 2`, so every slice uses the same independent ellipse seed;
- `ERRMIN = 1e-2`, `SAICON = 80`, `ICONVR = 2`, `NXITER = 1`;
- 129 x 129 packaged response tables with canonical VEST geometry and the VEST
  acceptance envelope;
- 1 ms reconstruction cadence and a 0.5 ms constraint-averaging window;
- the same upstream channel-quality decisions and constraint values for every
  model.

Run the five-model pilot over the reference set:

```bash
cd <checkout> && EFITHOME=~/git/efit/vaft-install python - <<'EOF'
import os, runpy, sys
root = os.getcwd()
sys.path.insert(0, root)
import vaft; assert vaft.__file__.startswith(root), vaft.__file__
sys.argv = ["profile_model_study.py", "--output", "/scratch/efit-profile-models"]
runpy.run_path("workflow/efit_profile_models/profile_model_study.py", run_name="__main__")
EOF
```

Running the script by path instead would put its own directory at `sys.path[0]`,
so an editable install elsewhere — the main checkout, when this is a worktree —
would supply `vaft` and the study would silently measure the wrong tree.
`PYTHONPATH` does not fix that, which is why the launcher asserts on
`vaft.__file__`. Pass `--shots` to narrow; the default is every reference-set
discharge with a packaged product.

Add the `(2,3)`, `(3,2)`, and `(3,3)` order cases with `--full-matrix`. Every
requested time remains in the output, including collapsed and missing-output
slices. Quantitative profile and geometry differences use paired times where
both the candidate and `(2,2)` baseline produced an equilibrium.

The raw EFIT files remain beneath `shot_<shot>/<model>/`. The JSON report keeps
the resolved configuration and digest, the outcome of every requested time,
a-file metrics, diagnostic-resolved m-file fit measures, the boundary extent,
the edge-current and oscillation diagnostics, and every paired difference. The
1-D profiles and the LCFS polygon those were computed from are released before
the report is written — they are about twenty times the rest of it — and stay
in the g-files and the per-model cache; `--keep-arrays` writes them into the
report instead. Re-running the command resumes completed model runs whose
scientific digest and analysis schema match.

The reported reference-R current profile is reconstructed from the g-file as
`j_phi(R_axis, psi) = R_axis p'(psi) + FF'(psi) / (mu_0 R_axis)`. It is a
consistent decomposition diagnostic, not a flux-surface average.

Interpretation limits are explicit:

- the study needs an EFIT built against netCDF. `write_m` is behind
  `#ifdef USE_NETCDF`, and a build without it writes a-files and g-files
  normally while silently omitting every m-file — so every chi-square in the
  report comes out empty and nothing else looks wrong. The run refuses at the
  first model rather than reporting a scan that answers nothing;
- absolute pressure and `beta_p` are not qualified while #386/#659 remain open;
- q comparisons exclude approximately `psi_N < 0.05` because of #317;
- a model that produces fewer equilibria is not compared on its easier subset
  without also reporting the changed outcome population;
- the fit comparison uses EFIT's flux-loop plus magnetic-probe chi-square from
  the m-file; the reported total is dominated by the model-invariant plasma-
  current term and is retained only as a diagnostic;
- a smaller magnetic chi-square is not evidence of a better model when
  geometry, condition number, or temporal jitter deteriorates;
- `unchanged` in the signed comparison means the two models agreed to within
  `1e-6` relative, which is the files' own precision and not a physical
  tolerance. A half-percent change counts as a change, and how big a change is
  gets read off the signed median rather than off the counts;
- the model-induced spread is reported over the times **every** model produced
  an equilibrium. A sigma over whatever happened to reconstruct at each slice
  is a sigma over a different ensemble at every slice, which is the selection
  effect that made `NXITER = 3` look like an improvement in #171. The
  all-available figures are kept in the JSON and are not the headline, and the
  model that empties the common population is named rather than dropped;
- the chi-square is not reduced by the fitted coefficient count. EFIT does not
  report how many free parameters it used, and
  `magnetics_chisq_per_active_signal` divides by the number of active signals,
  not by degrees of freedom.

---

# What the first pass found

Nine models over 39915, 41524 and 41672 — 77 plasma slices, one axis at a time,
everything outside the profile block frozen and hashed. The table is
`test/data/efit_profile_model_study.json`.

| model | KPPCUR | KFFCUR | edge | FWTBP | equilibria | accepted | collapsed | no output | condno median | runtime |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `p11_zero` | 1 | 1 | zero | 0 | 41/77 | 28 | 29 | 7 | 1.7e9 | 28 s |
| **`p22_zero`** | **2** | **2** | **zero** | **0** | **50/77** | **30** | **22** | **5** | **5.0e9** | **29 s** |
| `p32_zero` | 3 | 2 | zero | 0 | 49/77 | 29 | 22 | 6 | 5.2e10 | 26 s |
| `p23_zero` | 2 | 3 | zero | 0 | 54/77 | 30 | 18 | 5 | 6.6e9 | 32 s |
| `p24_zero` | 2 | 4 | zero | 0 | 54/77 | 30 | 18 | 5 | 4.5e10 | 40 s |
| `p33_zero` | 3 | 3 | zero | 0 | 54/77 | 30 | 18 | 5 | 4.4e11 | 32 s |
| `p22_zero_fwtbp` | 2 | 2 | zero | 1 | 50/77 | 30 | 22 | 5 | 5.8e11 | 27 s |
| `p22_free` | 2 | 2 | free | 0 | 52/77 | 24 | **10** | 15 | 1.8e9 | 44 s |
| `p22_free_fwtbp` | 2 | 2 | free | 1 | 48/77 | 28 | 20 | 9 | 1.0e12 | 37 s |

## The chi-square cannot rank these models, and that has to be said first

EFIT's chi-square for a VEST reconstruction is the plasma-current term. Not
mostly — entirely. Over the 497 reconstructions with an m-file:

| term | median | share of the total |
| --- | ---: | ---: |
| plasma current (`chipasma`) | 27.3 | **1.0000000000** |
| flux loops + probes (`saisil` + `saimpi`) | 8.0e-8 | 1.5e-9 |
| PF currents (`chifcc`) | 2.1e-22 | — |
| diamagnetic flux (`chidflux`) | 1.1e-13 | — |

On 488 of the 497 the reported total equals the plasma-current term to every
digit written; the nine exceptions are one slice — 41524 at 336 ms — seen once
per model. The a-file `chisq` that `SAICON = 80` accepts or rejects is that
same number. **VEST's acceptance criterion is testing how well EFIT reproduces
`Ip`, and nothing else.**

The mechanism is the uncertainty each channel is given, and it is traceable
end to end. Under `uncertainty_mode = "legacy_weight"` the writer emits
`BITMPI = weight / VBIT * legacy_weight_scale`, so with the ODS probe weight of
0.1, `VBIT = 10` and a scale of 1e4 the k-file carries `BITMPI = 100`.
`data_input.F90:2686-2690` then takes

```
sigmpi = max(SERROR * |measured|, |BITMPI| * VBIT)
```

which is `max(0.0005 * 0.05, 100 * 10) = 1000` — **a probe measuring 0.05 T is
given an uncertainty of 1000 T**, and a flux loop measuring 0.003 Wb an
uncertainty of 100 Wb. `Ip` gets `BITIP = 1000`, hence 10 kA against a measured
81 kA, so its residual is the only one that survives normalization.

The magnetics are not being reproduced, either. Across the baseline's 3657
active probe readings the calculated value differs from the measured one by a
median of **30%**, and across its 682 flux-loop readings by **9%**. They are
badly fitted and weighted out of the objective at the same time, and no number
EFIT reports shows either: the misfit is divided into invisibility by the
uncertainty, and the uncertainty itself is only in the k-file.

This is #663's premise, now measured rather than asserted. It is also the
missing half of #171's result: `MXITER` from 25 to 200 left the chi-square
identical because the chi-square is not measuring the fit.

**Everything below therefore reads the magnetic chi-square as a relative
quantity only.** The per-channel weights are the same for every model, so
ratios between models are meaningful; the absolute scale, and its distance
from `SAICON`, are not.

## Q1 — does more profile freedom lower the residual? No; it raises it

| candidate | common slices | magnetic chi-square | LCFS RMS | \|Δarea\| | \|Δvolume\| | \|Δli\| | \|Δq95\| | edge current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `p11_zero` | 41 | **−8%** | 27 mm | 18% | 26% | 32% | 0.3% | −54% |
| `p32_zero` | 49 | −0.4% | **1 mm** | **0.5%** | 0.6% | 0.2% | 0.4% | +1% |
| `p23_zero` | 49 | +6% | 20 mm | 11% | 15% | 17% | 2.0% | +53% |
| `p24_zero` | 49 | +11% | 32 mm | 17% | 24% | 26% | 3.4% | +102% |
| `p33_zero` | 50 | +6% | 19 mm | 10% | 15% | 17% | 2.1% | +53% |
| `p22_zero_fwtbp` | 49 | −1% | 3 mm | 0.9% | 1.2% | 0.2% | 0.7% | +0.01% |
| `p22_free` | 40 | +12% | **68 mm** | **32%** | **45%** | **53%** | 14% | **+871%** |
| `p22_free_fwtbp` | 44 | +12% | 66 mm | 32% | 45% | 53% | 14% | +871% |

Every model with more freedom than `(2,2)` fits the magnetics **worse**, and
does so on *every* compared slice — 0 of 49 improved for `p23_zero`, 0 of 49
for `p24_zero`, 0 of 50 for `p33_zero`, 0 of 40 for `p22_free`. The one model
with *less* freedom, `p11_zero`, is better on every slice: 41 of 41.

A least-squares fit cannot have a larger residual at its optimum for having
more parameters, so this is not overfitting failing to pay off — it is the
solver spending the freedom somewhere else. The plasma-current term says where:
it is **bit-identical on 360 of 371 compared slices**, and the remaining eleven
differ at 1e-8, below the file's own precision. So the term that carries all
the weight does not move, the term that moves carries none of the weight, and
the direction of the trade is set by the weighting rather than by the physics.

**The order that matters is `KFFCUR`, alone.** `p32_zero` adds a P′ coefficient
and moves the boundary by a millimetre; `p23_zero` adds an FF′ coefficient and
moves it by two centimetres; `p33_zero` adds both and is indistinguishable from
`p23_zero` on every column. At β_p ≈ 0.007 the `R p′` term barely reaches
`j_φ`, so P′ order is free and inert — which is a statement about VEST's
pressure (#386) as much as about the basis.

**And `KFFCUR` has not converged by three.** The fourth point was added for
exactly this question: `(2,4)` moves the boundary 32 mm where `(2,3)` moved it
20 mm, doubles the edge current where `(2,3)` raised it by half, and fits
monotonically worse. There is no plateau in the range VEST can support.

That is exactly the **model-dependent regime** #579 named: the magnetic fit
stays similar while area, volume and `li` keep changing — the quantities are
not determined by this reconstruction at this complexity. The **robust
plateau** it hoped for does not exist on the FF′ axis, and the **pathological
regime** arrives alongside rather than after it (see the q profile below).

## Q2 — what the freedom costs

**Not acceptance.** 30 slices accepted at `(2,2)`, `(2,3)`, `(2,4)` and `(3,3)`
alike; 29 at `(3,2)`. The acceptance criterion is the `Ip` term, so a profile
model cannot move it, and that constancy is the same fact as the section above
rather than independent evidence.

**Conditioning, by two orders of magnitude.** `FWTBP = 1` takes the median
condition number from 5.0e9 to 5.8e11, and `(3,3)` to 4.4e11 — the response
matrix is becoming singular in the added directions, which is what "freedom the
magnetics cannot support" looks like numerically.

**An oscillating q.** P′ and FF′ stay monotone everywhere (`pprime_turns ≤ 1`),
but the q profile turns 9 to 11 times at `(2,3)`, `(2,4)` and `(3,3)`, and up
to 15 with the free edge, against 3 at `(2,2)` and 0 at `(1,1)`. The pathology
is in the derived profile, not in the fitted coefficients, so a check that read
only `pprime`/`ffprime` monotonicity would have called every model clean.

**Runtime.** `(2,3)` costs 10% more than `(2,2)`, `(2,4)` 38%, the free edge
52%. Measured serially, and this study has no parallel mode on purpose: #171
recorded a recommendation drawn from scheduling noise in a run that used one.

## Q3 — does the right order change with the discharge phase?

No, but the *uncertainty* does, consistently and in the expected direction.
The sign of every model's effect is the same in every cohort; only its size
moves. Across the shots, ramp-up is the least determined phase and ramp-down
the most:

| shot | phase | times (all nine models) | σ area | σ volume | σ li | σ q95 |
|---|---|---:|---:|---:|---:|---:|
| 39915 | quasi-stationary | 4 | 18.5% | 28.7% | 31.6% | 7.2% |
| 39915 | ramp-down | 4 | 16.1% | 24.1% | 29.5% | 4.3% |
| 41524 | ramp-down | 7 | 16.2% | 24.6% | 30.2% | 6.3% |
| 41672 | ramp-up | 3 | 18.3% | 28.5% | 32.0% | 7.9% |
| 41672 | quasi-stationary | 7 | 17.6% | 27.0% | 31.2% | 6.1% |
| 41672 | ramp-down | 10 | 16.1% | 24.2% | 30.0% | 5.3% |

The ordering holds on every shot that has more than one cohort, but the spread
between cohorts is two percentage points on a quantity whose absolute spread is
seventeen. **The phase changes the answer far less than the model does**, and
three ramp-up times is not a population to build a phase-dependent
recommendation on. 41524 contributes no quasi-stationary row: only one of its
plasma times is flat, and no time there carries all nine models.

## Q4 — the uncertainty, which was the original ask

Over the times where **every** model produced an equilibrium:

| shot | plasma times | all nine | σ area | σ volume | σ li | σ q95 | σ beta_p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 39915 | 22 | 8 | 17.4% | 26.5% | 30.9% | 5.1% | 66.8% |
| 41524 | 20 | 7 | 16.2% | 24.6% | 30.2% | 6.3% | 61.3% |
| 41672 | 35 | 20 | 17.2% | 26.3% | 31.0% | 5.8% | 66.9% |

**Plasma area carries a 17% model-form uncertainty and volume 25%, from the
profile representation alone, at a chi-square that does not move at all.** The
three discharges agree to within a percentage point on every quantity, which is
a stronger statement than any one of them: this is a property of reconstructing
VEST from magnetics, not of a discharge.

`q95` is the exception at 5-6%, and it is the one quantity the magnetics do
pin. `li` at 31% and `beta_p` at 60-70% are not measurements in this
configuration; `beta_p` was already unqualified under #386/#659 and this is a
second, independent reason.

The population is honest: the strict and all-available figures differ by under
a percentage point (17.2% against 16.4% on 41672), so the selection effect the
`complete` population exists to exclude is small here. It is reported anyway,
with the leave-one-out counts, because it had to be checked rather than assumed.

## The free edge is the first thing in this series to move the collapse block

`PCURBD = FCURBD = 0` takes collapsed slices from **22 to 10** — twelve of them
become real equilibria (`collapsed → flagged`) across the three discharges. The
seed (#588), the domain and the grid (#459) each left that block untouched;
this is the first lever that does not.

It is not a free win and should not be read as one. The same change costs six
accepted slices and four flagged ones to `no_output` (acceptance 30 → 24), and
the equilibria it produces sit **68 mm** from the baseline boundary with 32%
more area and an edge current 8.7× higher. Whether those are better
reconstructions is not answerable from the magnetic residual, for the reason in
the first section. What the result does say is that the hard zero-edge
assumption is implicated in the null solutions, which is a lead for #459's
remaining block that nothing else has produced.

`FWTBP = 1` on top of the free edge cancels most of it — 20 collapsed instead
of 10 — while raising the condition number to 1e12. On the zero-edge baseline
`FWTBP` does nearly nothing to the geometry (3 mm, 0.9% area) at a hundredfold
cost in conditioning. There is no case for it here.

## What this does not say

- **Not which model VEST should use.** The comparison that would decide it —
  which model fits the magnetics best — is available only as a relative
  quantity, on channels weighted out of EFIT's objective by ~2e4 each. The
  ranking would be of a residual the solver is not minimizing. That decision
  waits on #663.
- **Not that `(2,2)` is wrong.** It is the point the other eight are measured
  against, and nothing here dislodges it. `(3,2)` is indistinguishable from it
  and `(1,1)` is clearly worse — 9 fewer equilibria, 7 more collapses.
- **Not a pressure result.** Every model carries essentially no pressure
  (#386/#659); the 60-70% spread on `beta_p` is spread in a quantity that is
  near zero in all nine.
- **Not build-independent in its absolute counts.** The baseline model on
  41672 produced 32 of its 41 requested slices under a Linux build of EFIT and
  33 under the macOS build these numbers come from — same commit, same inputs,
  one slice of difference. Every comparison here is within one build; an
  absolute count quoted across builds is not safe.
- **Not an answer about `KPPFNC`/`KFFFNC`.** Both are held at 0, the polynomial
  basis, exactly as #579 scoped. That `KFFCUR` has not converged by four is an
  argument for asking the basis question, not for raising the polynomial order
  further.

## Handover

- **#663** — which constraint families carry information. This study measured
  its premise: the magnetics contribute 1.5e-9 of the objective because each
  channel is handed an uncertainty four orders above its signal. The
  `legacy_weight` path and `BITMPI = weight/VBIT*1e4` are where to start.
- **#459** — the free edge moves the collapse block that seed, domain and grid
  did not. Twelve slices.
- **#386/#659** — P′ order is inert because there is no pressure to represent.
