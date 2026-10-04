# Direct fixed-boundary PF-current fit (#1608, stage 1)

`fit_free_boundary_coils(eq, machine, ...)` takes a canonical
`EquilibriumData` and ODS `pf_active` element geometry. It needs no OFT or
TokaMaker runtime. The result keeps the plasma and coil contributions to full
weber flux and tesla field at each fitted boundary/X-point sample. Static
passive vessel currents are zero in this free-space model.

The target is converted to COCOS 11. `J_phi = -2π(R p' + FF'/(μ0 R))` is
integrated over fractional LCFS grid cells, producing ring currents in A.
The COCOS 11 target has `Br=+∂ψ/∂Z/(2πR)` and `Bz=-∂ψ/∂R/(2πR)`;
the ring Green flux has the opposite derivative orientation. Returned flux
components use the Green orientation, with an arbitrary additive gauge, while
the fitted physical fields follow the target orientation. The exact kernels
used by `compute_point_response_matrices` evaluate plasma self-field and
grouped signed-turn PF responses. Finite coil rectangles use a 3×3 area
quadrature of the same kernels. Circuit variables are physical A; no
filament-turn current is exposed as a circuit current.

The objective uses the difference in full-weber flux from the first LCFS
sample, avoiding an arbitrary flux offset. `flux_normal` also fits `B·n=0`
away from X-points, which instead get `Br=Bz=0`. The default weights divide
flux rows by `|psi_boundary-psi_axis|` and field rows by peak plasma boundary
field. `regularization` is a dimensionless Tikhonov coefficient on currents
divided by `current_scale_A` (default `|Ip|/ncoil`); `current_penalties` scales
individual terms. `current_bounds` maps circuit names to `(min_A,max_A)`;
omitted bounds are unbounded. The result includes physical residuals, weighted
matrix singular values/rank/condition number, regularization norm, active
bounds, integrated `Ip`, optimizer status and acceptance status. Acceptance
checks both RMS and maximum relative flux residuals (defaults 2% and 4%),
field residuals, and an integrated plasma current within 5% of declared `Ip`.
It also requires finite caller-supplied bounds for every PF circuit: without
machine current ratings, a good numerical field fit is `bounds_unverified`,
not certified as physically realizable. A regularized rank-deficient fit may
be accepted under bounds for its field fit; rank and condition remain explicit.

## Reproduce the small cases

From the repository root:

```bash
python -m pytest test/test_equilibrium_coil_fit.py -q
PYTHONPATH=. python validation/fixed_free_1608/direct_fit.py
```

The script uses the packaged VEST `pf_active` geometry, a 33-point radial
target grid, 40 LCFS samples, `flux_normal`, and `regularization=1e-5`.
The JSON output contains fitted currents and all primary diagnostics. The
following values were measured on 2026-10-04; small platform differences in
contour extraction and numerical quadrature are expected.

| Family | Topology | Integrated/target Ip | RMS/max flux/span | RMS Bn (T) | Max saddle component (T) | Condition |
|---|---|---:|---:|---:|---:|---:|
| Solov'ev/CF | limited | 0.9929 | 0.0038 / 0.0081 | 0.0016 | — | 1.13e8 |
| Solov'ev/CF | lower single null | 0.9911 | 0.0440 / 0.1135 | 0.0047 | 0.0030 | 9.31e7 |
| Solov'ev/CF | double null | 0.9863 | 0.0064 / 0.0153 | 0.0021 | 0.0038 | 1.28e8 |
| Guazzotto–Freidberg Part 1 | limited | 0.9996 | 0.0014 / 0.0028 | 0.0001 | — | 1.67e10 |
| Guazzotto–Freidberg Part 1 | lower single null | 0.9996 | 0.0105 / 0.0230 | 0.0004 | 0.0002 | 8.31e9 |
| Guazzotto–Freidberg Part 1 | double null | 0.9996 | 0.0013 / 0.0031 | 0.0001 | 0.0001 | 7.50e9 |

The Solov'ev lower-single-null case exceeds the default flux thresholds with
this coil geometry. None of these reference runs supplies hardware current
bounds, so their physical current feasibility remains unverified. The large
condition numbers show that individual currents are sensitive even where the
boundary fit is good.
Physical field and later free-boundary closure comparisons are the validation
targets; matching another inverse method's current vector is not required.

This stage does not run a Grad-Shafranov closure. The next stages add the
independent TokaMaker `get_vfixed()` path, profile transfer, and free-boundary
forward solves. Guazzotto pressure/bootstrap surface currents and toroidal
flow are rejected because this volume-current model cannot represent them.
