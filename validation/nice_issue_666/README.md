# Issue #666: corrected adapter, reconstruction unresolved

Follow-up: [331 ms initialization/response/solver isolation](isolation_331ms/README.md)
records the 2026-09-09 diagnostic controls. In particular, mode 1's reported
magnetic cost is a VacTH-derived boundary objective, not the original
diagnostic chi-square. No new time scan or production fix was made in that
follow-up.

The [synthetic-equilibrium isolation](synthetic_equilibria/README.md) adds
noiseless Solov'ev and local TokaMaker controls. VacTH recovers both below
0.03 sigma, and both full reconstructions pass the current numerical-convergence
stage. Their 33--39% plasma-current loss is retained as a deferred equilibrium-
quality issue and no longer blocks this stage's success status.

The corrected local study completed on 2026-09-08. **Zero of 34 reference
slices meets scientific acceptance. Issue #666 is not complete.**

| Shot | Requested | Native equilibrium tables | Accepted |
| --- | ---: | ---: | ---: |
| 39915 | 9 | 2 | 0 |
| 41524 | 6 | 0 | 0 |
| 41672 | 19 | 0 | 0 |

At **41672, exactly 331 ms**, the corrected contour meshes without a reported
node/edge mismatch and NICE executes 30 reconstruction iterations. Final
relative residual is **1.01219**, versus the unchanged **1e-10** target;
magnetic cost is **333473**, above NICE's existing validity limit **10000**.
NICE rejects the plasma and omits equilibrium tables. VacTH initialization
already reports a nonfinite radial barycenter; it is retained as failure
evidence, not hidden by the later finite iteration residuals.

All enabled magnetic diagnostics pass the actual native active-coil response
comparison: worst discrepancy across the study is **0.0057792 sigma**, below
the unchanged 0.1-sigma criterion. The focused case's maximum is
**0.00075548 sigma**. Fixed passive response is subtracted exactly once.
Raw/conditioned values, errors, weights, corrections, source/executable hashes,
input hashes, and full logs are preserved in the JSON records.

`collected.json` is the authoritative recollection with explicit COCOS-11
flux/profile conversion, verified initializer residuals, full failure logs,
and available equilibrium comparisons. The 39915 outputs at 319 and 323 ms
are unconverged; comparisons to stored EFIT references are diagnostic only.
Their final magnetic diagnostic tables contain zero measured/computed values
because of NICE's uninitialized standalone `signBp/signF`; those tables are
rejected rather than reported as zero residuals. Stored EFIT references were
not regenerated with the newly shared conditioning; this limitation is
explicit in each comparison.

`focus.json`, `39915.json`, `41524.json`, `41672.json`, and `families.json`
preserve original run records. `overall.json` lists every requested time.
Four reduced-family configurations are rejected before execution; the two
full-magnetic variants run and fail. Diamagnetics is explicitly unsupported,
so “Full” does not imply that a diamagnetic observation was fitted.

## Exact-time EFIT

The local installed EFIT binary and hydrated local Green tables support an
attempt at 331 ms. The matched canonical constraints generated k041672.00331
(and the auxiliary 332 ms slice). EFIT reaches iteration 101 at 331 ms, reports
`1st separatrix point is off grid`, and produces no g-file despite exiting
zero. See `efit_exact.json` and `efit_logs.json`. There is **no usable exact-time
EFIT reference**; the existing 333 ms slice is never labeled an exact match.
The EFIT installation is a locally modified build, recorded by its own
`vaft-external-install.json`; NICE remains at the pinned, unmodified revision.

## Reproduction and evidence lineage

Entrypoint: `python -m vaft.code.nice.validate_reference`; full invocation is
in `vaft/code/nice/README.md`. Native run directory:
`/tmp/nice-666-corrected-v5`. NICE revision:
`7ad1ea8f3da4fee25a61a7c2c01b1773db5f4906`, executable
`/tmp/nice-build-clang2/nice_recon` (AppleClang 21, Release, Eigen 3,
SuiteSparse, packaged compatibility header and cassert force-included).

- `../nice_issue_666_superseded`: original wrong-COCOS/intersecting-contour study.
- `../nice_issue_666_intermediate_v3`: corrected COCOS but passive-conductor
  intersection remained; first EFIT attempt encountered LFS pointer tables.
- `../nice_issue_666_intermediate_v4`: broad contour cleared conductors but
  enclosed adjacent active rectangles, causing mesh construction timeouts.
  The window study was interrupted; these are not final results.
- Current v5 contour threads the VEST gaps outside passives and excludes active
  coils. No thresholds were relaxed to produce an accepted result.

The remaining numerical failure is unresolved. Successful execution, passing
input checks, or available but unconverged equilibrium tables do not close it.
