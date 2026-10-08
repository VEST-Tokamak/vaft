# COCOS re-audit: 41672 at 331 ms

Performed 2026-09-14 with the canonical ODS, pinned NICE revision
`7ad1ea8f3da4fee25a61a7c2c01b1773db5f4906`, AppleClang 21, Eigen 3, and
the production VEST adapter settings. This audit changes no production input,
threshold, or reconstruction report.

## Source-level transformation chain

The effective configuration is COCOS 11 in all three relevant XML fields:
`inCOCOS=11`, `outCOCOS=11`, `useNewCOCOSManager=1`, and `inoutCOCOS=11`.
Preparation rejects any disagreement. With positive 125798 A Ip and positive
B0, NICE's `ManageCOCOS` maps COCOS 11 to its internal NCIpp convention as:

| Quantity | Input -> NICE factor |
| --- | ---: |
| Ip, B0, active-coil current, B-pol scalar | +1 |
| psi / flux-loop scalar | -1 / -1 |
| p-prime, FF-prime | -2 pi / -2 pi |

The two “-1” entries in the psi/flux row are different operations: a full psi
profile has the additional `1/(2*pi)` scale, while the standalone flux-loop
reader applies only its scalar sign. VEST's calibrated loop value is written
with `flux_loop_input_sign=-1`; NICE then applies `CocosInToNice_Fmeas=-1`.
The resulting internal measurement has the same sign as NICE's Green-function
flux. The synthetic-loop recovery and equivalent-COCOS test below verify this
end to end, rather than relying only on the table.

Probe angles are not pre-flipped by VAFT. For COCOS 11,
`aux_inout.cpp:619-649` changes alpha to `-alpha`. NICE subsequently evaluates
`Br*cos(alpha_internal) + Bz*sin(alpha_internal)`. This equals VAFT/IMAS
`Br*cos(alpha) - Bz*sin(alpha)`. Applying an additional scalar or angle sign
correction in the adapter would therefore be wrong.

## Equivalent COCOS encoding test

The same 331 ms physical input was executed twice in VacTH-only mode:

1. production COCOS 11 encoding;
2. native COCOS 7 encoding with the file-level transformations analytically
   inverted (probe angle negated and flux-loop file value negated).

Both executions produced exactly the same reported barycenter, inferred Ip,
B-pol/flux cost, and return code. All six compared arrays were bit-for-output
identical (`max_abs=0`, `relative_l2=0`): measured and computed B-pol,
measured and computed flux loops, Dirichlet psi, and Neumann field.

This is strong evidence that the COCOS 11 input chain itself is internally
equivalent to NICE's native convention. Both cases reproduce the same physical
failure (Ip = -54813.6 A and NaN radial barycenter), so changing the COCOS label
or adding a compensating sign cannot repair it.

## Sign-matrix falsification test

The prepared file values were independently multiplied by +/-1 for each
diagnostic family, retaining COCOS 11 inside NICE:

| B-pol | Flux | VacTH Ip [A] | B RMS [T] | Flux RMS [Wb] | B cost | Flux cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| +1 | +1 | -54813.6 | 0.0298776 | 0.0559801 | 39538.4 | 518.125 |
| +1 | -1 | -52592.4 | 0.0294829 | 0.0587375 | 39399.3 | 567.229 |
| -1 | +1 | -865371 | 0.0437907 | 0.120577 | 78558.6 | 2573.07 |
| -1 | -1 | -902580 | 0.0447104 | 0.128280 | 80688.2 | 2895.61 |

The production combination has the lowest total data cost. Flipping flux alone
slightly lowers B cost through the coupled fit but increases total cost and
does not restore Ip/barycenter. Either B-pol flip causes a large degradation.
This rules out the simple global B-pol sign, global flux sign, and combined
global-sign hypotheses for this failure. It does not rule out individual
channel orientation/calibration defects, which remain the leading data-side
hypothesis from the plasma-response audit.

## Standalone output caveat

NICE explicitly documents that non-IMAS text equilibrium tables remain in
internal NCIpp. The adapter therefore converts accepted native equilibrium
quantities back to COCOS 11: `psi *= -sign(Ip)*2*pi`, derivative quantities by
the reciprocal factor, and Ip/B0/q/current signs using the source input signs.
Regression coverage exercises these factors.

There is a separate pinned-source defect in final diagnostic tables: the new
standalone reader does not initialize `dataReconstruction.signBp/signF`, while
`solver_out_full_recon.cpp` multiplies final diagnostic values by those fields.
The adapter does not guess; it accepts a final diagnostic table only when its
measured column verifies a unique +/-1 mapping to the manifest. This output
defect cannot explain the earlier VacTH NaN because it is used after solving.

## Conclusion

No remaining evidence supports a global COCOS mismatch as the cause of the
41672/331 ms reconstruction failure. The current COCOS handling is verified
by source tracing, a synthetic positive-current recovery, a full sign matrix,
and exact COCOS 11/7 native-array equivalence. The investigation should stay
focused on individual B-pol conditioning/prescribed-response consistency and
VacTH's unchecked invalid barycenter, followed by solver initialization—not
on another global sign or 2-pi adjustment.

Artifacts: `cocos_sign_matrix.json`, `cocos_equivalence.json`, and their two
reproduction scripts. Fresh native directories were used under `/tmp`; the
accepted-study artifacts were not overwritten.
