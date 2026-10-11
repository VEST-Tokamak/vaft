# Nonlinear CGYRO validation plan for the #1900 surfaces

This plan follows [`tglf_cgyro_validation.md`](tglf_cgyro_validation.md), which defines
the selection rule:
- eligible surfaces need ion-scale gamma_max >= 0.1 c_s/a and Q_tot >= 0.2 Q_GB under
  every SAT rule;
- the representative surface sits near the median SAT spread;
- the disagreement surfaces have the largest spreads.

**Nothing here is launched.** Running it needs an allocated core-hour budget, and Lane Y
owns the CGYRO allocation. The tooling is Lane Y's nonlinear workflow (#1656).

## Linear and nonlinear stay separate

| step | measures | needs | decides |
|---|---|---|---|
| L (linear) | gamma(ky), omega(ky) and the dominant branch: CGYRO against TGLF on the same local input, TGLF in **both** linear presets (SAT0/1: XNU_MODEL 2, GYRO; SAT2/3: XNU_MODEL 3, WDIA_TRAPPED 1, CGYRO units) | ~1 core-h per (ky, field) | which TGLF linear preset is closer; the size of the linear error on each surface |
| N (nonlinear) | time-averaged Q_e, Q_i, Gamma_e with uncertainty | 10^3 core-h per surface | the TGLF/CGYRO flux ratio, read against step L's linear error on the same surface |

A step-N ratio is reported only next to the step-L result on the same surface and preset.
The flux gap is never attributed to "saturation" when step L shows a linear gap.

## Candidates

TGLF EM-BPER numbers come from `sensitivity.csv` and `sat_pairs.csv` (#1932, v1, GACODE
b493397). Linear CGYRO numbers come from Lane Y's `gyrokinetic_linear/linear_summary.csv`.
The electromagnetic flux fraction f_em is at most 0.018 on every candidate, so EM-BPER
(A_parallel) is kept only for comparability with the completed run, not because it is
needed.

| surface | role | Q_tot SAT0/1/2/3 (Q_GB) | Q_e/Q_i SAT0 → SAT3 | ion-scale gamma_max, omega (SAT0/1 → SAT2/3 preset) | linear CGYRO / TGLF gamma_max (EM) |
|---|---|---|---|---|---|
| 39916 @ 0.321 s, r/a 0.5 | representative | 6.1 / 8.6 / 12.1 / 9.7 | 1.83 → 1.63 | 0.43, +0.13 → 0.31, +0.18 | **not on the linear grid (r/a 0.6-0.8 only)** |
| 39915 @ 0.317 s, r/a 0.7 | disagreement (completed) | 10.4 / 26.3 / 57.8 / 36.3 | 1.90 → 1.91 | 0.52, +0.11 → 0.38, +0.22 | 0.37 (branch electron/electron) |
| 39915 @ 0.317 s, r/a 0.8 | disagreement | 40 / 137 / 480 / 226 | 1.67 → 2.57 | 1.06, +0.04 → 0.80, +0.41 | 0.56 (electron/electron) |
| 39916 @ 0.321 s, r/a 0.8 | disagreement | 136 / 353 / 1417 / 701 | 1.53 → 1.94 | 1.74, **-0.28 → 1.35, +0.22** | 0.71 (EM ion/ion; ES CGYRO electron vs TGLF ion) |

Frequencies use the convention that negative means the ion diamagnetic direction.

### What each run is meant to resolve

- **39916 r/a 0.5 (representative).** The four rules agree to within a factor of 2. If
  CGYRO lies far below all four, as on 39915 r/a 0.7, the overprediction is common to
  every SAT closure; it is not a matter of which rule to pick. This is what makes the
  completed run's result general or specific. It needs step L first.
- **39915 r/a 0.8 (same state as the completed run).** Only the radius changes: same
  equilibrium, composition and preset, with the drive up 2x and the spread up 2x. The
  questions are whether the CGYRO/TGLF ratio follows the spread, and whether Q_e/Q_i
  favours the SAT0/1 partition (1.7) or the SAT3 one (2.6). The partition is
  independent of the overall level, so it is the cleanest discriminator here.
- **39916 r/a 0.8 (other shot, near a branch boundary).** The two TGLF linear presets
  disagree on the *direction* of the dominant ion-scale mode: omega is -0.28 for
  SAT0/1 and +0.22 for SAT2/3. Linear CGYRO also changes branch between ES and EM. Step L
  says which preset's mode is right. Step N says which branch carries the flux. A
  saturation rule cannot be judged on a surface where its linear input has the wrong
  mode.

### Which SAT rules are compared, and how far the comparison can go

All four rules are set against CGYRO, but they are ranked only within pairs that share
measured linear physics (`sat_pairs.csv` `linear_identical`):
- SAT0 against SAT1: saturation-model sensitivity, valid at UNITS=GYRO, which these inputs
  use;
- SAT2 against SAT3: quasilinear/saturation closure, including SAT3's QLA factor.

SAT1 against SAT2 is a combined model change. A CGYRO result can say which of the two
fluxes is closer, but not whether the linear part or the saturation part made the
difference.

## Execution (proposed, not run)

1. **Step L on 39916 r/a 0.5.** Run `workflow/gyrokinetic/run_linear.py` on the same
   resolved state (Lane T's `resolve_transport_state`, the `state_identity` equal to the
   #1482 one), ky rho_s 0.1-6, ES and EM, plus TGLF linear in both presets.
   About 24 CGYRO runs, about 25 core-h.
2. **Step L, second preset, on the existing grid.** Rerun TGLF linear only (seconds)
   with the SAT2/3 preset on all three r/a 0.6-0.8 candidates. Lane Y's TGLF linear
   column is a single preset.
3. **Step N**, with the completed run's settings so that all four are comparable:
   `N_RADIAL=64, N_TOROIDAL=16, KY=0.1, BOX_SIZE=3, N_THETA=24, N_XI=16, N_ENERGY=8,
   DELTA_T=0.002 (adaptive), N_FIELD=2`. Input from `vaft.code.gacode.cgyro` on the same
   local input as TGLF. Requeue-safe parsing from #1656. Restart in segments of at most
   48 h.
4. **Resolution check** on one disagreement surface (39915 r/a 0.8, as the most strongly
   driven surface of the completed state): `N_RADIAL=128, N_TOROIDAL=24` over at least
   300 a/c_s after saturation. The completed run's own resolution check (#1484) comes
   first; if it moves Q by more than its error, these settings change before step N
   starts.

## Budget (estimate, from the completed run at ~3 core-h per a/c_s on 32 cores)

| item | simulated time | core-h |
|---|---|---|
| step L, 39916 r/a 0.5 | - | ~25 |
| step N, 39916 r/a 0.5 (gamma ~0.3-0.4, bursts ~100+ a/c_s as on 39915 r/a 0.7) | 1000 a/c_s | ~3000 |
| step N, 39915 r/a 0.8 and 39916 r/a 0.8 (gamma ~0.8-1.7: faster saturation, possibly a smaller adaptive time step) | 400-600 a/c_s each | ~1200-3600 each |
| resolution check, one surface | ~300 a/c_s at ~2-3x cost | ~2000-2700 |
| **total** | | **~7500-13000** |

## Acceptance criteria (per surface, before any SAT statement)

1. **Stationarity.** Batch means over at least 5 blocks of at least 2 burst periods. The
   block-size and start-time choices agree within 1 standard error (as #1484 did with
   9×100 vs 3×300 a/c_s). The transient before saturation (first 100 a/c_s, or until the
   zonal/turbulent ratio settles) is excluded and stated.
2. **Resolution.** The refined run's mean lies within the base run's 1 standard error;
   otherwise the base setting is not accepted.
3. **Locality.** Report l_corr/L_x (box adequate if < 0.3) and epsilon_local =
   l_corr/L_n. epsilon_local > 0.5 is labelled `local_questionable` and travels with the
   number; it is not a rejection.
4. **Linear context.** Step L's converged gamma ratio and branch are stated beside the
   flux ratio. A surface whose dominant branch differs between CGYRO and the TGLF preset
   is not used to rank that preset's saturation rule.
5. **The statement made.** TGLF/CGYRO flux ratios with propagated uncertainty, for Q_e,
   Q_i and Q_e/Q_i. Rules are ranked only within linear-identical pairs, and no
   preferred configuration is declared from fewer than three accepted surfaces.
