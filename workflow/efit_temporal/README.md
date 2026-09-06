# EFIT temporal resolution: cadence and averaging window (issue #468)

`run_cadence_study.py` varies the two temporal controls of a VEST EFIT run —
the reconstruction cadence `dt_EFIT` and the constraint-averaging half-window
`w` (#433) — one at a time, under one fixed and serialized numerical
configuration, and records what the solver reported per slice: `jflag` /
`lflag` from the a-file, the iteration count and final Grad–Shafranov
residual from EFIT's terminal log (the local build has no NetCDF, so no
m-file), the total χ², the boundary-finder errors, and the wall-clock cost.
Convergence first; equilibrium smoothness is a later question.

```bash
PYTHONPATH=. python workflow/efit_temporal/run_cadence_study.py \
    --eddy-ods 39915_eddy.json --shot 39915 \
    --efit ~/git/efit/build-mac/efit/efit --output study/ \
    --cadences 1.0,0.4,0.2,0.08 --windows 0.5 --window-scan 1.0,0.2,0.1
```

Cadences and windows are in milliseconds and snap to the 40 µs diagnostics
grid. The input product's own `equilibrium` is discarded and the constraints
rebuilt for every case; channel selection is the diagnostics-stage
assessment's verdict alone. `study.json` carries the resolved
`EFITScientificConfig` and its digest, every setting held fixed, and the
per-slice rows; `study.md` is the summary table.

Two things had to change in the k-file writer before sub-millisecond slices
could be studied at all: EFIT reads the slice time as `ITIME` [ms] +
`ITIMEU` [µs] and names its outputs by the exact microsecond, and the writer
neither wrote `ITIMEU` nor named its k-files that way (it truncated the
remainder to 0.1 ms), so slices closer than a millisecond collided and were
mislabelled. Whole-millisecond slices are unchanged.

## First pass: 39915, the full eddy product, packaged 129×129 tables

Local build `~/git/efit/build-mac` (no NetCDF), `EFITScientificConfig()`
defaults, independent slices, wall currents from the eddy stage, plasma
window 0.3063–0.3308 s from `plasma_timing`. Slices whose constraint current
is below EFIT's 50 kA `CUTIP` (the ramp before 0.310 s and after 0.326 s) are
counted separately: EFIT does not attempt them.

| case | cadence [ms] | window [ms] | slices | ≥ 50 kA | a-files | jflag = 1 | boundary errors | iterations | GS residual (median) | χ² (a-files, median) | EFIT s/slice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dt1ms_w0.5ms | 1.00 | 0.50 | 25 | 15 | 7 | 0 | 5 | 11 | 0.0087 | 69 | 0.27 |
| dt0.4ms_w0.5ms | 0.40 | 0.50 | 62 | 39 | 16 | 0 | 13 | 11 | 0.0087 | 80 | 0.28 |
| dt0.2ms_w0.5ms | 0.20 | 0.50 | 123 | 78 | 31 | 0 | 27 | 11 | 0.0088 | 80 | 0.32 |
| dt0.08ms_w0.5ms | 0.08 | 0.50 | 307 | 196 | 80 | 0 | 68 | 11 | 0.0090 | 80 | 0.43 |
| dt1ms_w1ms | 1.00 | 1.00 | 25 | 15 | 5 | 0 | 5 | 11 | 0.0088 | 87 | 0.54 |
| dt1ms_w0.2ms | 1.00 | 0.20 | 25 | 15 | 5 | 0 | 5 | 11 | 0.0088 | 87 | 0.43 |
| dt1ms_w0.1ms | 1.00 | 0.10 | 25 | 15 | 5 | 0 | 5 | 11 | 0.0088 | 87 | 0.39 |

(`study_39915_first_pass.md` is the script's own table for the same run; its
"converged" column counts `jflag = 1` over all slices including the below-cut
ones, which is why it shows 1–6 there.)

What the first pass says:

- **The temporal controls are not what limits this reconstruction.** At
  every cadence and every window, every slice above the current cut runs
  exactly 11 outer iterations, ends at a Grad–Shafranov residual of 0.009,
  and never sets `jflag = 1`. Finer cadence adds slices between the same
  slices and reproduces their behaviour; the window from 0.1 to 1 ms moves
  the median χ² by a few percent and nothing else.
- **The ramp fails in the boundary finder**, not in the fit: every slice from
  0.310 to 0.314 s (and a third of all slices above the cut at 0.08 ms)
  ends with "first and last contour points are too far apart" and writes no
  a-file. That is a plasma-shape / limiter question for #459 and the
  initialization policy of #196, not a temporal one.
- **The flat-top does not meet the termination criterion under this
  configuration.** χ² runs from 128 at 0.317 s down to 3.6 at 0.324 s against
  `SAICON = 80`, the residual sits just under `ERRMIN = 0.01`, and the loop
  stops at 11 iterations without `jflag`. The stored pipeline product for the
  same shot (`vaft/data/efit/a039915.00319`) has `jflag = 1` at χ² = 77.6
  from a different EFIT build and table set. Which settings differ is exactly
  what #171 is for.

So the cadence question cannot be answered yet: under the configuration the
study can currently hold fixed, no cadence converges, and a comparison of
non-converged slices measures the configuration, not the sampling. The
harness, the k-file fix and this baseline are the deliverable; the study
proper starts once #171 pins a configuration under which the reference case
converges, and then reruns this exact command.

Cost, for planning: 0.3–0.4 s per slice on this machine, roughly linear in
slice count (307 slices at 0.08 ms took 2.2 min).
