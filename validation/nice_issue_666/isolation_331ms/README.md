# Focused diagnosis: 41672 at 331 ms

See also the [COCOS-specific re-audit](COCOS_AUDIT.md), which finds exact
COCOS 11/7 native-array equivalence and rejects global diagnostic sign flips.

2026-09-09. No additional shot/time scan, production adapter change, upstream
patch, or acceptance-threshold change was made. Experiments use fresh native
directories under `/tmp/nice331-isolation-20260909`; the v5 baseline remains
untouched. These are diagnostic controls, not accepted reconstructions.

## Findings

1. **VacTH fails on the measured inputs, not universally on this geometry.**
   Stopping after VacTH reproduces Ip = -54,813.6 A, R-barycenter = NaN,
   Z-barycenter = -3.25316 m. No nonlinear solver has run at this point.
   With exactly the same diagnostic geometry, uncertainties, contour, basis,
   and COCOS settings, a known positive-current loop at (0.4 m, 0) and zero
   active currents gives Ip = 125,807 A versus 125,798.123 A injected, and
   barycenter (0.399969 m, 4.57901e-6 m). Maximum diagnostic discrepancy is
   0.003463 sigma (B-pol) and 0.001707 sigma (flux). This argues against a
   universal probe-sign or 2-pi error in the tested forward/VacTH path; it does
   not independently validate experimental calibration or all plasma profiles.

2. **VacTH's invalid result is passed downstream.** The local source
   `src/vacth_solver.cpp:2574` takes sqrt(tmpr2/Ip + R0^2) without checking
   the radicand. `src/solver_recon_aux.cpp:140` computes a distance from the
   resulting NaN barycenter; `err > maxdist` is false, so the re-centering
   branch is skipped. The original harmonic solution continues to provide
   Dirichlet and Neumann boundary data. A zero return code for the VacTH-only
   control is not scientific success.

3. **The two-step direct initialization is numerically insufficient, but not
   the sole cause.** Baseline direct residuals go 0.0342846 -> 29030;
   computed current becomes 9.12056 MA for requested 125.8 kA.
   Allowing ten direct steps reaches residual 9.22308e-14 and the requested
   current. Nevertheless subsequent reconstruction ends with NaN objectives.
   Its final axis is at (0.155575 m, -1.16156 m), not evidence of a good seed.
   One direct step also fails; switching the two-step initializer to Picard
   yields final magnetic cost 333733 and residual 0.687156. None is accepted.
   The direct Newton source applies an undamped `X = Xold + deltaX` update
   (`src/solver_dir.cpp:808`); the two-step jump is observed, not inferred.

4. **The nonlinear magnetic objective is a boundary objective in mode 1.**
   `src/solver_recon_SQP2.cpp:1693` fits VacTH-derived Neumann data under fixed
   Dirichlet data. Direct B-pol/flux response matrices are used in mode 2,
   not the selected `algoMeshDomain=1`. `_PrepareNeumannRelated` builds new
   boundary uncertainties using `neumannPercent=0.1`; it does not propagate
   the full original diagnostic covariance. Thus magnetic cost 333473 is
   not the sum of the original diagnostic normalized residual squares, and
   matching EFIT's diagnostic conditioning does not make these objectives
   statistically identical. This corrects an overly broad interpretation of
   the earlier matched-input study, not its failure classification.

## Independent plasma-response checks

The known-loop VacTH boundary matches analytic Green response with relative
L2 errors 0.1100% in psi/(2*pi) and 0.05851% in tangential B. This tests the
boundary handoff's units and orientation independently of nonlinear solving.

Sampling the same analytic flux on the existing NICE mesh and applying the
source's P1 derivative formulas gives 7.496% relative L2 B-pol interpolation
error (max 36.77 diagnostic sigma), and 0.9855% flux error (max 0.01525 sigma).
At boundary edge midpoints the tangential-field error is 12.49% relative L2,
0.613 RMS / 2.722 maximum in the mode-1 boundary sigma. These are manufactured
solution interpolation checks, not a demonstrated runtime assembly bug.
The direct diagnostic matrix is not the active mode-1 objective. Mesh accuracy
is a secondary concern distinct from the already-passing active-coil response.

An independent nonnegative-current least-squares model subtracts the recorded
fixed passive/active contributions and fits the remaining plasma response
using Green filaments. It retains the actual per-channel sigma and includes
the measured Ip with its recorded uncertainty (not a hard current equality).

| Spatial support | Combined-fit Ip | B-pol RMS sigma | Flux RMS sigma |
| --- | ---: | ---: | ---: |
| 15 x 17 central grid | 129223 A | 57.48 | 0.792 |
| 25 x 49 grid clipped to the full limiter | 128658 A | 50.78 | 0.744 |

The full-support flux-only fit reaches 0.591 sigma RMS and Ip = 125780 A.
The worst combined-fit residuals are zero-based probe indices 46 (-318.8
sigma), 45 (-141.3), 22 (+115.3), 23 (-102.7), and 24 (+76.3). These sampled
positive-current models are not exhaustive proofs of inconsistency and do not
identify a particular bad sensor. They do locate the dominant conflict in
the B-pol data/prescribed-response combination rather than the five flux loops.
For example, probe 46 (MagneticFieldProbe_C4-06) has inferred plasma Bz
+0.0871752 T after subtracting the recorded prescribed fields.

## Next bounded work

Before any new scan, audit the calibration/orientation, validity and local
conditioning of the identified B-pol channels, and independently check the
fixed passive and active subtraction against their measured time traces.
Separately, require finite/physical VacTH Ip and barycenter before generating
boundary conditions; validate a converged direct seed before SQP. These are
recommendations, not fixes applied in this diagnostic turn. More nonlinear
iterations alone did not resolve the focused failure.

`native_controls.json` records the controlled-run summaries. The accompanying
scripts reproduce the native controls and independent Green/P1 checks with
the local source and v5 inputs (`PYTHONPATH=.`). The native-control script
refuses existing output case directories; choose a new ROOT for replay.
The synthetic source is a forward-mapping control, not a force-balanced
Grad-Shafranov equilibrium or an EFIT reference.
