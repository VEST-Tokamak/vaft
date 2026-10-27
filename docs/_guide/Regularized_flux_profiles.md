# Positive regular flux profiles (opt-in)

`vaft.formula.regularized_profiles` adds a machine-independent offline fitter.
`fit_profile` and its legacy tuple remain unchanged. Its first consumer is the
external vaft-NeurIPSequilibrium study.

## Exact representation and objective

Set x = psi_N = rho_pol^2, u = x/domain_max and
`c_j = floor + logaddexp(0, theta_j)`. Then
`F(x) = scale * sum_j B_j(u) c_j`.

- `bernstein2` / `bernstein3`: degree 2 / 3, ascending j, coefficients 3 / 4;
  `B_j(u) = binomial(n,j) u^j (1-u)^(n-j)`.
- `bspline6`: cubic, ascending basis index, knot vector
  `[0,0,0,0,0.7,0.9,1,1,1,1]`. Six positive coefficients; custom two-knot
  fractions are explicit sensitivity settings, not silently selected knots.
- Positive partition-of-unity basis on `[0,domain_max]` implies
  `F >= scale*floor`; with finite parameters softplus gives positive control
  values. No output clipping is used. Floating-point underflow at extreme
  parameter values is a numerical limitation, not a stronger positivity claim.
- F has finite derivative at zero, hence dP/drho_pol = 2 rho_pol F'(rho_pol²)
  vanishes on axis. F'(0), hollow profiles and nonzero LCFS values are allowed.
  rho_tor requires a separately qualified relation.

The objective is
`mean Huber_delta((F(x_i)-y_i)/scale) + lambda/2 * integral_0^1 (d²(F/scale)/du²)^2 du`.
Huber is `r²/2` for `abs(r)<=delta`, otherwise `delta*(abs(r)-delta/2)`.
SciPy TRF least_squares minimizes transformed data residuals
`sign(r)*sqrt(2*Huber(r))/sqrt(N)` and separate linear curvature residuals.
This preserves the quadratic penalty rather than robustifying it. Curvature
uses 32 Gauss-Legendre points per knot interval (exact for these bases).
The analytic parameter Jacobian includes the softplus sigmoid derivative.
Initialization reuses VAFT's order-2 free-polynomial fit in normalized units,
then inverse softplus of positive initial control values. Initial controls
are floored for initialization only; observations and final outputs are not.
Default ftol/xtol/gtol are 1e-9, max_nfev 2000. Condition limit is 1e10.

## Result and limitations

`fit_regularized_profile(x,y,config=RegularizedProfileConfig(...))` returns
`RegularizedProfileFit`: config, unconstrained parameters, status/reason and
per-observation residual/robust weights, rank/condition, counts/span/gaps,
optimizer termination/optimality/nfev and separate data/penalty objectives.
`evaluate` returns NaN outside the declared domain unless explicitly requested
with `allow_extrapolation=True`. Outside-domain positivity is not guaranteed.
`evaluate(...,derivative=1)` is dF/dpsi_N, not dP/drho_pol.

Inputs must be finite aligned arrays in the declared domain. No input masking,
branch selection, uncertainty estimation or physics mapping occurs here.
Caller owns measured masks; negative measured values can remain observations
of a positive model and are not silently clipped. More distinct observations
than coefficients and a full-rank, well-conditioned data design are required,
even when regularized. Saturated fitted data Jacobians fail explicitly.
Numerical scale, Huber weights, residuals and legacy zero uncertainties are
never measured sigma. Sigma-weighted fitting is outside this API's v1 scope.

A regular fit is not evidence of unobserved-core accuracy, pedestal resolution,
valid flux topology, independent equilibrium information or autograd support.
Serialized config and parameters can be evaluated independently by a future
backend; directional-gradient qualification and nuisance optimization are
separate work. No default fitting mode or machine policy changes.
