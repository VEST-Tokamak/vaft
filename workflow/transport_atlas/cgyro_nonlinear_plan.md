# Nonlinear CGYRO validation plan for the #1900 surfaces

The surfaces come from the selection rule in
[`tglf_cgyro_validation.md`](tglf_cgyro_validation.md):
- **Eligible:** ion-scale gamma_max >= 0.1 c_s/a and Q_tot >= 0.2 Q_GB under every SAT
  rule.
- **Representative:** near the median SAT spread.
- **Disagreement:** the largest spreads.

**Nothing here is launched.** Running it needs an allocated core-hour budget, and Lane Y
owns the CGYRO allocation, the tooling (#1656) and the locality QA used below.

## Linear and nonlinear stay separate

| step | measures | cost | decides |
|---|---|---|---|
| L (linear) | gamma(ky), omega(ky) and the dominant branch. CGYRO is compared with TGLF on the same local input, with TGLF in **both** linear presets: SAT0/1 (XNU_MODEL 2) and SAT2/3 (XNU_MODEL 3, WDIA_TRAPPED 1, CGYRO units). | ~0.3-1 core-h per (ky, field) | which TGLF linear preset is closer; the size of the linear error on each surface |
| N (nonlinear) | time-averaged Q_e, Q_i, Gamma_e with uncertainty | 10^3-10^4 core-h per surface | the TGLF/CGYRO flux ratio, read beside step L's linear error on the same surface |

Rules for combining the two steps:
- A step-N ratio is reported only next to the step-L result on the same surface and
  preset.
- A flux gap is never called a saturation error when step L shows a linear gap on that
  surface.
- In step L, only `converged` CGYRO growth rates count. A ky point that ends `max_time`
  (marginal) or `decayed` (stable) gives no gamma and no branch.

## Candidates

TGLF EM-BPER numbers come from `sensitivity.csv` and `sat_pairs.csv` (#1932, v1, GACODE
b493397). Linear CGYRO numbers come from Lane Y's `gyrokinetic_linear/linear_summary.csv`.
Shear and box size come from each surface's own `input.tglf`, using
L_x = BOX_SIZE / (KY s) at KY = 0.1. Frequencies are negative in the ion diamagnetic
direction.

| surface | role | Q_tot SAT0/1/2/3 (Q_GB) | Q_e/Q_i SAT0 / SAT1 / SAT3 | TGLF ion-scale gamma_max, omega: SAT0/1 preset; SAT2/3 preset | linear CGYRO gamma_max (EM) / TGLF gamma_max (EM); branches | q, s | BOX_SIZE for L_x 40 rho_s (L_x at box 3) |
|---|---|---|---|---|---|---|---|
| 39916 @ 0.321 s, r/a 0.5 | representative | 6.1 / 8.6 / 12.1 / 9.7 | 1.83 / 1.82 / 1.63 | 0.43, +0.13; 0.31, +0.18 | **not on the linear grid** (r/a 0.6-0.8 only) | 2.18, 0.31 | 1.3 (96) |
| 39915 @ 0.317 s, r/a 0.7 | disagreement; run, **provisional** (resolution check pending) | 10.4 / 26.3 / 57.8 / 36.3 | 1.90 / 1.84 / 1.91 | 0.52, +0.11; 0.38, +0.22 | 0.19 / 0.52 = 0.37; electron/electron | 3.74, 0.71 | 2.8 (42, as run) |
| 39915 @ 0.317 s, r/a 0.8 | disagreement | 40 / 137 / 480 / 226 | 1.67 / 1.63 / 2.57 | 1.06, +0.04; 0.80, +0.41 | 0.61 / 1.08 = 0.56; electron/electron | 4.37, 1.73 | 6.9 (17) |
| 39916 @ 0.321 s, r/a 0.8 | disagreement | 136 / 353 / 1417 / 701 | 1.53 / 1.49 / 1.94 | 1.74, **-0.28**; 1.35, **+0.22** | 1.24 / 1.74 = 0.71; ion/ion (CGYRO ES: electron) | 3.41, 2.22 | 8.9 (14) |

**Electromagnetic runs are required, not optional.** The EM flux fraction in TGLF is
small (f_em <= 0.018). But electrostatic nonlinear CGYRO with kinetic electrons is
unusable at ky_min <= 0.2: the omega_H branch runs away, and finite beta removes it
(#1484). So step N runs EM.

### What each run is meant to resolve

- **39916 r/a 0.5, representative.**
  - The four SAT rules agree to within a factor of 2 here.
  - On 39915 r/a 0.7 the provisional CGYRO flux lies below every rule: 0.3x SAT0,
    0.05-0.12x SAT1-3.
  - This run tests whether CGYRO again falls below every rule on a surface where the rules
    agree. One more surface makes the 0.7 result recur or not; it cannot make it general.
  - It needs step L first.
- **39915 r/a 0.8, same state as the 0.7 run.** Same equilibrium, composition and preset,
  but the drive is 2x higher and so is the spread.
  - Does the CGYRO/TGLF ratio follow the spread?
  - Does Q_e/Q_i favour the SAT0/1 partition (1.6-1.7) or SAT3's (2.6)? The partition
    does not depend on the flux level, so it discriminates here. It does not at r/a 0.7,
    where all rules give about 1.9.
  - The shear is high (s = 1.73), so this run needs a box 2.5x the 0.7 run's.
- **39916 r/a 0.8, other shot.**
  - The two TGLF linear presets disagree on the *direction* of the dominant ion-scale
    mode: -0.28 against +0.22.
  - Linear CGYRO's branch changes between ES and EM.
  - Step L decides which preset's mode is right before any saturation rule is judged on
    this surface.
  - Lane Y's first nonlinear attempt here (#1484) ran away in an undersized box: BOX_SIZE 4
    gave L_x = 15 rho_s, the adaptive dt fell 7x, and the run was cancelled. A rerun needs
    BOX_SIZE ~9.

### Which SAT rules are compared, and how far the comparison can go

All four rules are set beside CGYRO. They are *ranked* only within pairs that share
measured linear physics (`sat_pairs.csv` `linear_identical`):
- **SAT0 vs SAT1:** saturation model; valid at UNITS=GYRO, which these inputs use.
- **SAT2 vs SAT3:** quasilinear/saturation closure, including SAT3's QLA.

SAT1 vs SAT2 is a combined model change. CGYRO can say which of the two fluxes is closer,
but not whether the linear part or the saturation part made the difference.

## Execution (proposed, not run)

1. **Step L on 39916 r/a 0.5.** Run `workflow/gyrokinetic/run_linear.py` on the same
   resolved state (Lane T's `resolve_transport_state`, with `state_identity` equal to the
   #1482 one) over ky rho_s 0.1-6, EM and ES, plus TGLF linear in both presets. About 24
   CGYRO runs, about 10-25 core-h.
2. **Second preset on the existing grid.** Rerun TGLF linear only (seconds each) with the
   SAT2/3 preset on all three r/a 0.6-0.8 candidates. Lane Y's TGLF linear column holds a
   single preset.
3. **Step N.** Use the completed run's velocity and theta resolution and time stepping
   (`input.cgyro` of 39915 r/a 0.7):
   - `N_ENERGY=8, N_XI=16, N_THETA=24, N_TOROIDAL=16, KY=0.1, N_FIELD=2`;
   - adaptive dt (`DELTA_T_METHOD=1`, `DELTA_T=0.002` initial), default `AMP`.

   The box is sized **per surface** to L_x ~ 40 rho_s. `N_RADIAL` scales with
   `BOX_SIZE` to keep kx_max at the completed run's ~4.8:

   | surface | BOX_SIZE | N_RADIAL |
   |---|---|---|
   | 39916 r/a 0.5 | 1.3 | ~32 |
   | 39915 r/a 0.8 | 7 | ~150 |
   | 39916 r/a 0.8 | 9 | ~192 |

   Each run is segmented: CGYRO counts `MAX_TIME` from the restart, so each segment's
   target is set explicitly (#1484). The parsing is requeue-safe (#1656). After the first
   100 a/c_s, Lane Y's locality QA (#1656) runs. If it reports `box_too_small`, the box is
   enlarged before more time is spent.
4. **Resolution and box checks.**
   - The 0.7 run's own resolution check (Lane Y, #1484: `N_RADIAL=128, N_TOROIDAL=24`)
     comes first. If it moves Q by more than the criterion below, these settings change
     before step N starts.
   - One r/a 0.8 surface also gets a second box size (L_x ~55 rho_s). The box-dependence
     of the saturated flux there is not known.

## Budget (estimate)

The scale is the 0.7 run: ~3 core-h per a/c_s at N_RADIAL 64 on 32 ranks. Cost scales
with N_RADIAL. The strongly driven 0.8 surfaces may also lose up to ~2x to a smaller
adaptive dt. Durations are set by stationarity criterion 1 (below), with a hard cap.

| item | core-h per a/c_s | simulated time | core-h |
|---|---|---|---|
| step L, 39916 r/a 0.5 | - | - | 10-25 |
| step N, 39916 r/a 0.5 (N_RADIAL ~32) | ~1.5-3 | 700-1200 (bursts ~100 a/c_s, as on 0.7) | ~1000-3600 |
| step N, 39915 r/a 0.8 (N_RADIAL ~150) | ~7-14 | 400-700 | ~2800-9800 |
| step N, 39916 r/a 0.8 (N_RADIAL ~192) | ~9-18 | 400-700 | ~3600-12600 |
| second box size, one r/a 0.8 surface | ~10-20 | 400 | ~4000-8000 |
| **total (Lane Y's pending 0.7 resolution check not included)** | | | **~11000-34000** |

Recommended order, so that each stage can stop the next:
1. step L, everywhere;
2. 39915 r/a 0.8, the same state as the completed run, so the comparison isolates radius;
3. 39916 r/a 0.5;
4. 39916 r/a 0.8 last: it is the most expensive, and its mode identity must be settled by
   step L first.

## Acceptance criteria (per surface, before any SAT statement)

1. **Stationarity.**
   - The burst period tau_b is measured from the zonal/turbulent |phi|^2 ratio. Use its
     dominant period, or, if there is no clear peak, the 1/e decay time of its
     autocorrelation.
   - The transient ends at t0, the later of 10/gamma_lin and the first time after which
     the running mean over tau_b stays within a factor of 2 of the final mean.
   - The run continues, in segments, until [t0, T] holds at least 5 blocks of at least 2
     tau_b each, up to the cap in the budget table.
   - Batch means with block lengths 2 tau_b and 3 tau_b, and start times t0 and
     t0 + (T - t0)/4, must agree within 1 standard error. A run that reaches the cap
     without meeting this is reported as `not_stationary`, not averaged.
2. **Resolution and box.** The refined (or larger-box) run gets at least 3 blocks. Its mean
   must agree with the base run's within 2 combined standard errors. Otherwise the base
   setting is not accepted on that surface.
3. **Locality** (Lane Y's definitions, #1484/#1656):
   - box adequacy l_corr/L_x: < 0.25 adequate, < 0.5 marginal;
   - epsilon_local = l_corr / min(L_Ti, L_Te, L_n, L_q): < 0.1 ok, < 0.3 marginal, else
     `local_questionable`.

   A `box_too_small` run is not accepted. A `local_questionable` run is accepted but keeps
   the label on every number it produces. TGLF is equally local, so the label limits what
   either code says about VEST.
4. **Linear context.**
   - Step L's converged gamma ratio and branch are stated beside the flux ratio.
   - A surface whose dominant converged branch differs between CGYRO and a TGLF preset is
     not used to rank that preset's saturation rule.
   - A surface with no converged CGYRO gamma near the TGLF peak gives a flux ratio only,
     with no attribution.
5. **The statement made.**
   - TGLF/CGYRO flux ratios with propagated uncertainty, for Q_e, Q_i and Q_e/Q_i, on each
     accepted surface (criteria 1-3).
   - Rules are ranked only within linear-identical pairs.
   - A preference between the two members of a pair is stated only if it holds on every
     accepted surface where the pair is ranked, and it is quoted with the number of those
     surfaces.
