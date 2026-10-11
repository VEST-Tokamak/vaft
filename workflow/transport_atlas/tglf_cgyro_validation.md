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
| cost | ~0.3-1 core-h per (state, surface, field model, ky) as measured by Lane Y (~400-500 core-h for 360 runs); the whole grid is affordable |
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
| cost | ~3 core-h per a/c_s on VEST surfaces; Lane Y ran 39915 r/a 0.7 to ~1000 a/c_s (~3000 core-h) to average several bursts |
| verdict | only with batch means over several bursts, a resolution check (n_radial, n_tor) and a locality check (l_corr / L_x, l_corr / L_n) |

What makes it a different measurement:

- **Stationarity is not guaranteed in a short run.** On 39915 r/a 0.7 (Lane Y, #1484),
  zonal-flow / turbulence bursts every ~100-130 a/c_s moved 50 a/c_s windows between
  0.26 and 15 Q_GB. A single window is not a result; Lane Y reports the t = 100-1000
  batch mean (3.04 ± 0.70 Q_GB), still provisional until the resolution check
  (n_radial 128 / n_tor 24) is done.
- **The gap mixes linear and saturation errors.** On the linear grid, TGLF's growth
  rates were above CGYRO's (median TGLF/CGYRO ratio 1.79 over the 82 converged points;
  Lane Y, #1484). On 39915 r/a 0.7, Lane Y read the nonlinear gap as partly linear and
  mostly saturation. So a TGLF/CGYRO flux ratio is not a saturation-rule error by itself;
  the linear comparison on the same surface and preset has to be stated alongside it.
- **The SAT pairs do not split the TGLF side the way they split each other.**
  - SAT2-SAT3 always share the linear physics. SAT0-SAT1 share it only when the input
    has UNITS=GYRO, because SAT1 keeps the input's units. Either way it is measured, in
    `sat_pairs.csv` column `linear_identical` (30/30 for both pairs on the v1 inputs,
    which are GYRO). Where it holds, a CGYRO flux can rank the saturation closure
    within the pair.
  - SAT1-SAT2 changes the linear presets and the saturation rule at once. That pair
    cannot attribute a CGYRO disagreement to either part (`comparison = combined_model`).
- **Locality is part of the result.** epsilon_local = l_corr / L_n = 0.58 on 39915 r/a 0.7
  (`local_questionable`). Every other surface needs its own locality check before its
  flux-tube number is read as a result.

## Which states get CGYRO

Linear CGYRO is cheap. It runs on every #1482 sensitivity state at r/a 0.6, 0.7 and 0.8,
ES and EM (Lane Y, `gyrokinetic_linear/`).

Nonlinear CGYRO costs thousands of core-h per surface, so it is limited to
**representative** and **disagreement** surfaces. The rule below is this note's proposal;
its thresholds are choices, not values taken from #1484.

1. **Eligible.** Every one of SAT0-3 (EM-BPER, `sensitivity.csv`) must have:
   - Q_tot >= 0.2 Q_GB;
   - an *ion-scale* growth rate `gamma_max_ion_scale` (ky rho_s <= 1) >= 0.1 c_s/a.

   Nonlinear CGYRO at ion scale cannot test a surface whose TGLF drive is electron-scale
   or near threshold. Near threshold, the spread measures how close the surface is to
   marginal, not how much the rules disagree. 13 of the 21 surfaces with four positive
   fluxes are eligible.
2. **Spread.** `max(Q_tot) / min(Q_tot)` over SAT0-3. The eligible median is 2.0.
3. **Representative:** an eligible surface near the median spread with a robust drive.
   **Disagreement:** the largest eligible spreads.

| role | surface (EM-BPER) | SAT spread | Q_tot SAT0..3 (Q_GB) | min ion-scale gamma_max (c_s/a) | status |
|---|---|---|---|---|---|
| representative | 39916 @ 0.321 s, magnetics, r/a 0.5 | 1.98 | 6.1 / 8.6 / 12.1 / 9.7 | 0.31 | proposed |
| disagreement | 39915 @ 0.317 s, magnetics, r/a 0.7 | 5.55 | 10.4 / 26.3 / 57.8 / 36.3 | 0.38 | run by Lane Y: 3.04 ± 0.70 Q_GB, provisional (resolution check pending) |
| disagreement | 39915 @ 0.317 s, magnetics, r/a 0.8 | 11.9 | 40 / 137 / 480 / 226 | 0.80 | proposed |
| disagreement | 39916 @ 0.321 s, magnetics, r/a 0.8 | 10.4 | 136 / 353 / 1417 / 701 | 1.35 | proposed |

Why these surfaces:

- **39915 r/a 0.7** is in the upper part of the eligible spread range, not at its median.
  The completed run is therefore a disagreement case, not a representative one.
- **39915 r/a 0.8** is the same state one surface further out. Profiles and locality
  change with radius, so it needs its own locality check. The equilibrium, composition
  and preset are shared with the completed run.
- **39916 r/a 0.8** gives the second-largest eligible spread on another shot.
- **39916 r/a 0.5** sits at the median spread with a clear ion-scale drive.

**Lane Y owns the CGYRO allocation and decides.** This table is the rule's current
output. It is regenerated from `sensitivity.csv` when the state set grows (#1900 item 5).

Not eligible, although their raw spread is large:
- 40330 r/a 0.6 (spread 63; Q_tot 0.01-0.67, ion-scale gamma_max <= 0.05);
- 42962 magnetics r/a 0.3 (spread 1215; Q_tot 0.01-7.0);
- 42962 magnetics r/a 0.8 (spread 22.9). gamma_max sits at ky rho_s ~ 24, i.e. electron
  scale; the ion-scale gamma_max is 0.08-0.20. This agrees with #1484, which found 42962
  near-marginal at ion scale.
