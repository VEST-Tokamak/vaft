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

## Weight study with the diamagnetic flux fitted: the study's rules (stage 2 onward)

`weight_scan.py` runs every constraint time of 39915, 41524 and 41672 (89 slices) under each setting and judges each slice with `criteria.py`. The diamagnetic flux became fittable once #1196 corrected its sign. These rules were adopted on 2026-09-28 and are the rules the study reports against.

### Per slice

1. **Admissible.** This is a veto: a slice that fails it is not a reconstruction, whatever its χ². It must meet all of:
   - converged, with p ≥ 0 everywhere;
   - βp > 0 and W > 0;
   - q95 > 2;
   - **0.3 ≤ Ip_MHD / Ip_measured ≤ 1 + 2σ_Ip**, with σ_Ip the relative Ip σ the fit was given on that slice. In the ramp-up the current on closed flux surfaces can be only 30–80 % of the Rogowski's Ip. An earlier convergence study that lowered the Ip input showed it. So the reconstruction may fall well short of the measurement, but may exceed it only within its own Ip σ.
     - Until 2026-09-29 the upper edge was 1.03. In stage 3 every Ip-ratio failure was on that side (1.03–1.17, none below 0.3), and at a 5 % Ip σ a ratio of 1.04–1.07 is z ≈ 0.9–1.3. The edge was tighter than the σ itself.
     - The routine's legacy weight is not a σ, so the routine is judged at the 5 % reference (edge 1.10).
2. **Measurement.** Each family is judged against the σ EFIT fitted with.
   - **Probes and flux loops:** reduced χ² in [0.5, 2].
   - **Ip and the diamagnetic flux:** one channel each, judged by **|z| ≤ 2**. A single channel's χ² is one z². Even with the right σ it lands in [0.5, 2] only 32 % of the time, and two such families together only 10 %.
   - **PF currents:** reported, not graded. They are fitted against a 10⁻⁴ relative σ.
   - A family the setting holds inactive is not graded.
3. **Virial.** |ln(βp from the pressure integral / βp from the pair_13 closure)| ≤ ln 2.
   - pair_13 is the RT-free closure. The others are ill-conditioned at VEST's aspect ratio (#649).
   - The slice is **indeterminate**, not failed, when the pair_13 denominator is below 0.1 or when either βp is ≤ 0. The admissibility veto catches the second case.
4. **Grad–Shafranov.** Relative residual ≤ 0.05. The threshold is provisional.
5. **Thomson: physical consistency, not fit quality.** This applies only where Thomson samples lie in the slice window.
   - An ohmic VEST plasma carries no fast-ion pressure, so p = p_e (1 + f_i T_i/T_e).
     - f_i = n_i,tot/n_e ≤ 1, because impurities dilute the ions.
     - T_i ≤ T_e, because the ions are heated only by the electrons.
   - The band is therefore **p_e ≤ p_recon ≤ 2 p_e**, i.e. ln(Σp_e / Σp_recon) ∈ [−ln 2, 0].
   - Criteria version 1 (before 2026-10-02) used [1, 3] p_e; every evaluation records `criteria_version`.

A slice is **good** when it is admissible, the measurement criterion passes, and neither the virial nor the Grad–Shafranov criterion fails. A criterion that cannot be evaluated on a slice is reported and never counted against it.

**Thomson does not enter `good`.** It is reported beside it as `physically_consistent` (`None` where there is no Thomson), so a numerically sound fit that disagrees with Thomson stays visible as exactly that.
- Thomson is never a target that a σ or a setting is chosen by.
- Choosing by it would make it a hidden fitting constraint, and it would stop being an independent check.

Across settings each slice is labelled (`criteria.slice_labels`, fit quality only; the good settings that pass or fail Thomson are listed beside it as `consistent` / `inconsistent`): **good** when some study setting is good there, **admissible** when some is admissible but none good, otherwise **unreconstructible** — no setting in the study gives a physical magnetics-only reconstruction of it. Such a slice is reported as that, not forced. The routine's verdict is shown beside the label and never counts towards it.

### Per setting: the σ is calibrated, not gridded

**Stage 3 (`--stage 3`) grids only the axes that need a judgement:**
- profile basis (1,1), (2,1), (1,2);
- diamagnetic σ ×16, ×64 or off, relative to its stored 3 %. At ×1–×4 EFIT does not solve;
- Ip σ ×1, ×4, ×10, i.e. 5 %, 20 %, 50 %.

**The probe and loop σ are solved for in each cell:**
- Start at probe ×4 and loop ×1, with a 2 % floor.
- After each sweep, update m ← m·√(median χ²r) per family. The median is over the admissible slices, or over the converged ones if none is admissible.
- Stop when both medians lie in **[0.8, 1.25]**, or after four rounds.
- At fixed residuals χ²r scales as 1/m², so one step suffices; a few are needed when the fit moves.

The record keeps every round.

**Stage 4 (`--stage 4`, 2026-09-29) replaces stage 3's exit and axes.**
- *The exit.* Stage 3 let SAICON = N + 3√(2N) gate EFIT's exit. At a calibrated σ most slices fit the probes at χ²r ≈ 7, so they ran all 514 iterations with ψ converged to ~2×10⁻⁸ and were reported unconverged. The median that set σ then came only from the few slices that fitted: the calibration was circular. Stage 4 writes SAICON = 10¹⁰, so EFIT stops on ERRMIN and a χ² stall. χ² is judged by the criteria above, not by the exit.
- *The axes.* Diamagnetic and Ip σ moved nothing in stage 3 and are fixed at ×16 and 20 %.
- *The bases.* (1,1), (2,1), (1,2), (1,3), (2,2). In stage 3 the (1,1) basis pinned li at 0.88–1.04 (routine: 0.64–0.75), and 20 of its ψ-converged slices had βp < 0 and W < 0. The magnetics fix Λ = βp + li/2, and with li pinned the only way to a smaller Λ is negative pressure. The extra FF′ terms let the current profile broaden instead. A positivity constraint was not taken: in (1,1) it would only pin βp at 0 and move the misfit into the probes.
- *A round with no converged slice* (a loose fit can lose the boundary in `bound`) steps the multipliers back to the geometric midpoint of the last two rounds. It starts at probe ×8, loop ×2, for at most six rounds.

### Result and the working setting (2026-09-29)

Stage 4 over 77 plasma slices; the other 12 are vacuum, |Ip| < 15 kA:

| basis | converged | admissible | good | probe / loop σ | |
|---|---|---|---|---|---|
| (1,1) | 40 | 17 | 6 | ×4.46 / ×1.65 | li pinned, 17 slices with βp < 0 and W < 0 |
| **(2,1)** | 20 | 16 | 4 | ×3.62 / ×2.15 | Thomson p_recon/p_e 1.5–1.9 |
| (1,2) | 25 | 19 | 5 | ×3.79 / ×1.95 | Thomson p_recon/p_e 2.7–3.7 |
| (1,3), (2,2) | 3, 6 | 1, 1 | 1, 1 | — | diverge (ψ residual ~0.4) |

**The working setting is (2,1)**, with probe ×3.62, loop ×2.15, diamagnetic ×16, Ip σ 20 %, the ψ-only exit and ERRMIN 10⁻⁴. `--stage 5` runs it.

**Stage 5** asked whether the unreconstructible ramp-up and ramp-down slices are a tolerance problem. They are not:
- ERRMIN 10⁻³ gives the same 20 / 16 / 4 on half the iterations.
- But βp and W then move by up to 9 %.

**Every symptom follows the shot's magnetics quality** (`workflow/magnetics_quality`). As the condemned probes go 2 → 10 → 17 across 39915 → 41524 → 41672:
- the outboard witnesses fall 20 → 14 → 11;
- (2,1) convergence falls 13/22 → 5/20 → 2/35;
- probe χ²r rises 1.1 → 5.7 → 12.7.

So the calibrated σ is effectively 39915's flat-top σ. The basis choice rests on 39915.

### How it runs

- **Parallelism.** Slices run in parallel (`--workers`, 24 on vestserver, which leaves 8 of 32 cores to the HSDS pipeline).
  - Each task is one slice: its constraints are built once, then its settings run in turn. The per-slice checkpoint makes a run resumable.
  - The cold-start contract still holds. Every non-reference run must start from one initialization fingerprint. Workers cannot share the running baseline, so the fingerprints are checked when the records are merged, and a disagreement aborts the run.
- **What was measured, and what was ruled out.** Stage 2 took ~9 s per run on one core, and ~97 % of it was EFIT.
  - EFIT's MPI splits the time slices of one call. With one slice per call, a process pool gives the same parallelism without coupling runs.
  - Moving to Python 3.14 would change the ~4 % spent in Python, not the EFIT.

### Stages 1 and 2, as run

**Stage 1** (2026-09-28, 1,424 runs):
- **Setup:** three bases × diamagnetic σ ×1/×4/×16/×64/off, probe ×16 and loop ×2.
- **Result:** no good slice under the old rules.
- **What it showed:**
  - The diamagnetic row is unsolvable at ×1–×4.
  - Probe χ²r was about 0.1 and loop χ²r about 0.3, so both σ were too wide.
  - The routine reconstruction is admissible but 2–4.5× below p_e on the Thomson slices.

**Stage 2** (probe × loop × diamagnetic × Ip σ, 49 settings) was stopped after 478 of 4,361 runs. It still used the per-slice chi-square band on single channels.
