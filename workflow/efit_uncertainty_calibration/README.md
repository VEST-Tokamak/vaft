# EFIT uncertainty calibration: at what σ are the magnetics informative and the fit convergent? (#891)

`sigma_calibration.py` scans the submitted σ of the magnetic families under
`uncertainty_mode = "standard_deviation"` and reports the **range** of rungs
that satisfy all of #891's criteria. It never reports the rung with the
smallest χ², because σ is χ²'s denominator: a wide enough σ always wins that
contest by throwing the information away.

```bash
python workflow/efit_uncertainty_calibration/sigma_calibration.py \
    --output /scratch/sigma --table /scratch/sigma/efit_sigma_calibration.json
```

## Contract

- **Uncertainty model.** `standard_deviation` σ with binary weights (#921).
- **Gating.** The diagnostics stage as shipped: C4-04 is a recorded fault (#988) and the array review is on (#1003).
- **Numerics.** NXITER 1, the packaged 129 table after #970's limiter move, the nine #924 slices, ERRMIN 1e-2 and 1e-4.
- **EFIT.** 3b5dae5, the #918-consistent build, executable sha256 `4a4e645e…`.
- **Ladder.** Probe σ alone, loop σ alone, and both together, each × 1, 2, 4, 8, 16, 32 through `uncertainty_scales`. Each is run with and without a floor of 2 % of the family's median |m|.
- **Profile basis.** (1,1) and (2,1). The routine (2,2) does not settle once the magnetics carry weight (#1027).
- **χ² target.** `SAICON = N + 3√(2N)` per slice, where N counts the fitted probes, loops, Ip and PF. Under a statistical σ the legacy 80 lies below the expected χ² (#1027).
- **Diamagnetic flux.** Held inactive; it gets its own axis (#386).
- **Initialization.** Every run is a cold start from the same seed ellipse, one EFIT call per slice in an emptied workdir. There is no continuation along the ladder and no start from the legacy solution.
  - Each run records an initialization fingerprint, and a run that differs from the first is refused.
  - On the recorded run all 1,188 runs carry the same fingerprint (`070f8231…`).
- **Reference branch.** One legacy-weight run per basis serves as the reference branch for displacement, nothing more.

**Operating-range rule.** A rung is in the range only if all four hold:

1. The tight fit converges on every slice.
2. Both magnetic families' reduced χ² lie in [0.5, 5].
3. The probe residual is within 1.5× of that at the narrowest converging σ.
4. The median vertical drift between the loose and the tight stop is ≤ 5 mm.

## Result (2026-09-21, 1,188 runs, 6.4 h serial): no rung satisfies all four

The record is `test/data/efit_sigma_calibration.json`. In the table below:
- "converged" counts slices at ERRMIN 1e-4;
- the reduced χ² values are medians over the converged slices;
- "residual" is the median probe |m − r|/|m|;
- "to legacy" is the median LCFS RMS distance to the legacy reference.

| rung | converged | probe χ²r | loop χ²r | residual | drift | to legacy |
|---|---|---|---|---|---|---|
| legacy reference (1,1) | 6/9 | – | – | 25.7 % | 8.2 mm | – |
| (1,1) ×1 (the current σ) | 0/9 | – | – | – | – | – |
| (1,1) ×8, floor 2 % | 5/9 | 1.03 | 0.14 | 8.5 % | 5.2 mm | 95 mm |
| (1,1) ×16, no floor | 5/9 | 0.81 | 0.10 | 8.3 % | 3.9 mm | 79 mm |
| (1,1) ×16, floor 2 % | **8/9** | 0.45 | 0.06 | 9.9 % | 9.3 mm | 94 mm |
| (1,1) ×32, no floor | **8/9** | 0.25 | 0.08 | 14.3 % | 2.9 mm | 53 mm |
| (1,1) probe ×16 only, floor 2 % | 4/9 | 0.25 | 1.15 | 8.5 % | 6.2 mm | 73 mm |
| (2,1) ×16, floor 2 % | 6/9 | 0.20 | 0.03 | 7.7 % | 3.8 mm | 87 mm |
| (2,1) ×32, floor 2 % | 7/9 | 0.10 | 0.01 | 9.7 % | 5.2 mm | 90 mm |
| legacy reference (2,1) | 6/9 | – | – | 24.7 % | 8.1 mm | – |

### Why no range

- **Criterion (i) fails for every rung, on one slice.** The best rungs, (1,1) at ×16 and ×32 on both families, converge on 8 of 9 slices. The one they lose, 41672 @ 342, takes EFIT's `iconvr = 2` exit and then fails in `findax` ("1st separatrix point is off grid"), so it writes no equilibrium. That is a boundary/separatrix failure after the fit, not a non-convergence. It is also the one slice that legacy (1,1) reconstructs and the statistical σ does not.
- **Criterion (iii) cannot be applied.** No rung converges everywhere, so there is no "narrowest converging" residual to compare against.
- **The two families want different widening.**
  - Where the probes reach χ²r ≈ 1 (×8–×16), the loops sit at 0.06–0.14: their σ is widened far more than their residual needs.
  - With the loops left at ×1, their χ²r is 1.2–2.4, but only 3–4 of 9 slices converge.
  - The diagonal ladder, which moves both families together, therefore cannot place both in [0.5, 5] at once.

### What the scan does establish

- **The current σ is too tight by roughly an order of magnitude.** At ×1 no slice converges. Convergence appears from ×8 and is widest at ×16–×32 on both families.
- **With the magnetics weighted, the fit follows the data 2–3× more closely than the legacy branch.** The probe residual is 8–14 % against 25 %.
- **It lands on a different equilibrium.** The boundary sits a median 5–9 cm from the legacy-weight reference on the same basis. So under legacy weighting EFIT settles on a branch the magnetics do not select. The routine products use legacy weighting too, on the (2,2) basis, and whether they are displaced by a similar amount is not measured here.
- **The vertical drift of #924 shrinks but does not vanish.** It is 3–9 mm on the best rungs, against 8 mm for legacy.
- **(1,1) is more robust than (2,1).** It converges on at least as many slices on 31 of the 32 rungs; the exception is probe ×4 alone (0 against 1).

### Next

1. **Decouple the families.** Run a 2-D probe × loop ladder, or a loop σ fixed near ×1–×2 while the probes scan. Both the probe and the loop χ²r can only reach [0.5, 5] if they are widened independently.
2. **Treat 41672 @ 342's `findax` failure as a separate defect,** with the same class as the 46091 `bound` failures. Also report the range over the other 8 slices, stated as such.
3. **The 5–9 cm displacement from the legacy-weight branch** is the headline for #924 and for downstream consumers. It needs to be measured against the routine (2,2) products, and checked against an independent boundary measurement (for example a camera or limiter contact), before any production σ is adopted.
