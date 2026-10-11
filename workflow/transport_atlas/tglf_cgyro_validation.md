# TGLF against CGYRO: linear validation versus nonlinear saturation validation (#1900)

TGLF is a quasilinear model with two parts that can each fail on their own:

    flux = sum over ky and modes of  W_QL(linear eigenmode) x phi_sat(saturation rule) x QLA

`vaft.code.gacode.tglf.spectral` rebuilds this product from TGLF's own output files and
reproduces `sum_flux_spectrum` to machine precision on all 240 sensitivity runs (#1932).
CGYRO can check each factor, but each check uses a different kind of CGYRO run and
answers a different question.

## Linear validation: is TGLF's eigenmode right?

| | |
|---|---|
| CGYRO run | linear, one ky per run, initial value until the growth rate converges (`workflow/gyrokinetic/run_linear.py`) |
| TGLF run | `USE_TRANSPORT_MODEL=F`, the **same** local input, the same ky |
| compared | growth rate gamma, real frequency omega (sign = mode direction), ky of gamma_max, field structure |
| tests | TGLF's linear physics: its eigensolver, trapped-particle and collision models (XNU_MODEL), the gyro-fluid closure |
| does not test | any SAT rule, and therefore no flux magnitude |
| cost | ~core-minutes per (state, surface, ky); the whole grid is affordable |
| verdict | per ky: `converged` CGYRO gamma vs TGLF gamma; `max_time` = marginal, `decayed` = stable (never a growth rate) |

Three traps:

- **TGLF's linear model depends on SAT_RULE.** `tglf_startup.f90` fixes several linear
  controls by SAT rule:
  - SAT2/3: XNU_MODEL = 3, WDIA_TRAPPED = 1, UNITS = CGYRO;
  - SAT0: UNITS = GYRO, XNU_MODEL = 2;
  - SAT1: XNU_MODEL = 2 and keeps the input's UNITS.

  A linear TGLF-vs-CGYRO comparison therefore validates one *preset*. It must name that
  preset (`sensitivity.csv` columns `*_effective`); it does not validate "TGLF".
- **UNITS=CGYRO moves the ky points** by `grad_r0`; it does not change ky's
  normalisation. Compare at the same physical ky, not the same index.
- **CGYRO's frequency sign** depends on IPCCW/BTCCW. Convert with the direction that
  CGYRO prints (`workflow/gyrokinetic/README.md`).

## Nonlinear saturation validation: is TGLF's flux level right?

| | |
|---|---|
| CGYRO run | nonlinear, many ky and kx, run until the flux is statistically stationary |
| TGLF run | the routine transport-model run, one per SAT rule and field model |
| compared | time-averaged Q_e, Q_i, Gamma_e in gyro-Bohm units, with an uncertainty |
| tests | the product W x phi_sat x QLA: saturation level, its ky shape, and the QL amplitude |
| cost | ~3 core-h per a/c_s on VEST surfaces; ~1000 a/c_s (~3000 core-h) per surface |
| verdict | only with batch means over several bursts, a resolution check (n_radial, n_tor) and a locality check (l_corr / L_x, l_corr / L_n) |

What makes it a different measurement:

- **Stationarity is not guaranteed in a short run.** On 39915 r/a 0.7 (Lane Y, #1484),
  zonal-flow / turbulence bursts every ~100-130 a/c_s moved 50 a/c_s windows between
  0.26 and 15 Q_GB. Only the t = 100-1000 batch mean (3.04 ± 0.70 Q_GB) is a result.
- **The gap mixes linear and saturation errors.** TGLF's linear drive was already above
  CGYRO's on that surface (median gamma ratio 1.79). A flux ratio TGLF/CGYRO is therefore
  not a saturation-rule error by itself. The linear comparison on the same surface and
  preset has to be stated alongside it.
- **The SAT pairs do not split the TGLF side the way they split each other.**
  - SAT0-SAT1 and SAT2-SAT3 share the linear physics; it is measured, in `sat_pairs.csv`
    column `linear_identical`. A CGYRO flux can therefore rank the saturation closure
    within a pair.
  - SAT1-SAT2 changes the linear presets and the saturation rule at once. That pair
    cannot attribute a CGYRO disagreement to either part (`comparison = combined_model`).
- **Locality is part of the result.** epsilon_local = l_corr / L_n = 0.58 on 39915 r/a 0.7
  (`local_questionable`). Any flux-tube number on VEST surfaces carries that caveat.

## Which states get CGYRO

Linear CGYRO is cheap. It runs on every #1482 sensitivity state at r/a 0.6, 0.7 and 0.8,
ES and EM (Lane Y, `gyrokinetic_linear/`).

Nonlinear CGYRO costs ~3000 core-h per surface, so it is limited to **representative**
and **disagreement** surfaces. They are chosen from the SAT spread
`max(Q_tot) / min(Q_tot)` over SAT0-3 (EM-BPER, `sensitivity.csv`, 21 surfaces with all
four solved and positive). Marginal surfaces are excluded (min Q_tot < 0.1 Q_GB or
gamma_max < 0.05 c_s/a): near threshold, the ratio measures how close the surface is
to marginal, not how much the rules disagree.

| role | surface (EM-BPER) | SAT spread | Q_tot SAT0..3 (Q_GB) | status |
|---|---|---|---|---|
| representative | 39915 @ 0.317 s, magnetics, r/a 0.7 | 5.6 (median 5.2) | 10.4 / 26.3 / 57.8 / 36.3 | done: CGYRO 3.04 ± 0.70 (Lane Y, #1484) |
| disagreement, same state | 39915 @ 0.317 s, magnetics, r/a 0.8 | 11.9 | 40 / 137 / 480 / 226 | proposed |
| disagreement, other shot | 42962 @ 0.332 s, magnetics, r/a 0.8 | 22.9 | 1.5 / 3.4 / 35.4 / 7.1 | proposed (gamma_max 3.8 c_s/a: check whether the drive is ion- or electron-scale on the linear grid first) |

The two proposed surfaces have different purposes:

- 39915 r/a 0.8 changes only the radius from the completed run. It keeps the state, the
  preset and the locality context, so a change in the CGYRO/TGLF ratio there reads as a
  radial trend rather than as a different discharge.
- 42962 r/a 0.8 is the largest disagreement among surfaces that are not marginal.

Lane Y owns the CGYRO allocation and decides; this table is the selection rule and its
current output. It is regenerated from `sensitivity.csv` when the state set grows
(#1900 item 5).

Excluded as marginal:
- 40330 r/a 0.6 (spread 63, Q_tot 0.01-0.67);
- 42962 magnetics r/a 0.3 (spread 1215, Q_tot 0.01-7.0);
- 40330 r/a 0.7 (Q_tot ≤ 0.11).
