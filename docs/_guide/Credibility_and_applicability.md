---
title: Credibility and applicability
author: VEST team
date: 2026-10-05 09:00
category: guide
layout: post
permalink: /reference/credibility-applicability/
guide:
  architecture: The six credibility axes (E/T/I/A/N/V) evidence is placed on, the ordering-based applicability evaluation, and who owns what between formula, process, validation and workflow.
  prerequisites: The Computational layers page, for the Formula, Process and Code vocabulary.
  expected: Which axis a check belongs to, how an ordering becomes a margin and a status, and why neither ever runs on the default processing path.
related:
  api: [validation]
---

A VAFT result is trusted for six different reasons, and they fail independently. A reconstruction
can fit every probe and still be unidentified. A solve can converge and still violate an ordering
its model assumed. A datum can be valid and still be extrapolated. This page fixes the vocabulary for
those questions (issue #1639) and the evaluation of one of them, approximation applicability
(#1628, with the ordering quantities from #1627).

## The rule this page protects

**Scientific assessment surrounds the fast path; it never burdens it.** A process computes its
result and returns it. Assessment is a separate, optional call over that result, and so is
uncertainty propagation. Nothing on this page is imported by `vaft.process`, and
`test/test_validation_credibility.py` checks that in a clean interpreter.

## The six axes

`vaft.validation.credibility.AXES`:

| axis | question | typical evidence |
| --- | --- | --- |
| `source_evidence` (E) | Does the source datum itself provide credible evidence? | channel validity, calibration, drift, saturation |
| `transformation` (T) | How faithfully was it turned into what is consumed? | time mismatch, mapping distance, fraction extrapolated |
| `inference` (I) | Does the evidence determine the requested state? | fit residuals, Jacobian conditioning, singular values |
| `applicability` (A) | Do the model's assumptions hold for this state? | ordering margins (below) |
| `numerical` (N) | Was the calculation performed reliably? | convergence, Grad–Shafranov residual, identity closure |
| `independent_validation` (V) | Does evidence *not* used in the inference support it? | withheld diagnostic, unused identity, alternative solver |

The axes are a classification, not a result schema. A domain module still returns its own plain
dict. `Evidence(axis, key, status, metrics, reasons, cost, role)` exists only so that evidence from
different domains can be laid side by side. `compose()` returns **one status per axis** and never
one overall score. For example, *fit = pass, independent validation = not available* is a
different statement from *independent validation = fail*.

**Rules carried over from #253 and #337:**
- A metric is not a verdict, and a verdict is not a policy. Whether a result may be *used* is
  decided by the workflow.
- Missing evidence is not failure.
- A precondition ("can this algorithm run?") is not credibility ("is the result trustworthy?").

**Fitted data is not independent validation.** Every piece of evidence can carry a role from
`EVIDENCE_ROLES`, such as `used_for_inference` or `independent_validation`.
- VAFT's EFIT fits the diamagnetic flux by default (`EFITConfig.use_diamagnetic_flux`; #891,
  #1440). So when a report assessed `diagnostic_fit.diamagnetic_flux`, every check in
  `DIAMAGNETIC_CHECKS` moves off the V axis automatically.
- Name any other fitted check in `used_for_inference=`. An unknown key or a bare string is refused
  rather than ignored.

### Cost classes

`COST_CLASSES = ("cheap", "moderate", "expensive")`:
- **cheap:** read from results that already exist, such as a convergence flag or an ordering
  parameter.
- **moderate:** local computation, but no solver rerun.
- **expensive:** solver ensembles, Monte Carlo, continuation. Always opt-in.

## Pilots

| pilot | where | what it shows |
| --- | --- | --- |
| A: equilibrium reconstruction | `evidence_from_equilibrium_report` | the `validate_equilibrium` report on the axes; fit quality and plausibility (I), solution consistency (N) and independent validation (V) stay apart |
| A: the #891/#1331 study criteria | `evidence_from_efit_criteria` | criteria v2 (#1521); Thomson stays on V and never enters fit quality |
| B: virial closures | the virial checks inside pilot A | identity residual (N), leave-one-identity-out (V), closure conditioning (I) |
| C: current moments (#943) | `successive_discrepancy` | relative change of an observable along the moment order |
| D: asymptotic validity | `vaft.validation.applicability` | ordering margins and contracts, below |

**How the mappings are chosen.**
- `CATEGORY_AXES` maps each existing validation category to its default axis.
- `CHECK_AXES` records the registry checks placed elsewhere, each with its reason in a source
  comment. For example, plausibility of the inferred state is `inference`, not `numerical`.
- `CRITERIA_AXES` is the complete mapping for the study criteria. There, `virial` compares two β_p
  of the same g-file, so it is `numerical`, and only Thomson is on V.

## Applicability: margins first, labels second

A reduced model is derived under orderings. For example, ideal single-fluid MHD needs all of the
following:
- $S \gg 1$;
- $d_i/L \ll 1$;
- $\rho_i/L \ll 1$;
- $\tau_{\rm evol}/\tau_A \gg 1$.

These test different things, so an `ApproximationContract` keeps them as separate
`OrderingAssumption`s. Each assumption names its quantity, its characteristic `scale` and its
`scope`, so $d_i/a$ can never silently stand in for $d_i/\delta_{\rm layer}$.

The primary output is the continuous **ordering margin** against the assumption's threshold
$x_0$:

$$m = \log_{10}(x_0/x) \quad (x \ll x_0), \qquad m = \log_{10}(x/x_0) \quad (x \gg x_0).$$

- $m > 0$: on the permitted side, with $m$ decades to spare. `SUPPORTED` means only this:
  $x = 0.99$ against $x_0 = 1$ is supported with $m = 0.004$, so read the margin, not the label.
- $m \le 0$: violated.
- The default $x_0 = 1$ is where the expansion parameter reaches order unity. This is the only
  cutoff applied without a source. Any other threshold must name its source in `threshold_source`,
  or the assumption refuses to construct (#1639 §12).

**Statuses.** They come second, from the same vocabulary as the operational-space renderer (#1664):

| status | meaning |
| --- | --- |
| `SUPPORTED` | every assumption evaluated and satisfied |
| `OUTSIDE` | at least one evaluated assumption violated, whatever else is missing |
| `UNASSESSED` | nothing violated, but some quantity is missing, non-finite or non-positive, or the state lacks a field the contract's scope is defined on |
| `NOT_APPLICABLE` | the state is outside the contract's `applies_to` scope |

There is no `MARGINAL`: a margin band would be a cutoff nobody derived, and the margin itself is
there for anyone who wants to see how close a state is.

**Functions:**
- `evaluate_contract(contract, state)` returns every assumption's value, margin and status, plus
  the limiting assumption.
- `evaluate_population(contract, table)` keeps every row and every individual check.
- `as_evidence(result)` places a result on the A axis.

The strongest applicability evidence pairs the ordering with the convergence it predicts (#1639
§12): `successive_discrepancy([Q_reduced, ..., Q_full])`.

## Ownership

| concern | owner |
| --- | --- |
| ordering quantities ($S$, $d_i/L$, $Kn$, $\Omega_c/\nu$, …) | `vaft.formula` / `vaft.process` (#1627) |
| which orderings a model assumes (the contracts) | declared beside the model (#1627 for ordering families, #1628 for formulas, codes, workflows) |
| margin, status and composition semantics | `vaft.validation.applicability` (this page) |
| the credibility axes and `Evidence` | `vaft.validation.credibility` (this page) |
| drawing statuses on a diagram | `vaft.plot.operational_space` (#1664) |
| acceptance policy (what a study accepts) | the workflow or notebook |
| sensitivity, Jacobians, uncertainty propagation | #1642, opt-in |
| the VEST population study | `notebooks/vest_ordering_and_model_applicability.ipynb` (#1629) |

## Sensitivity, linearization and uncertainty (#1642)

The domain that owns an operation owns its derivatives. EFIT's native response matrix stays in
`vaft.code.efit.linearization`, and a process's Jacobian stays beside the process. Two shared
pieces sit around them:
- the algebra, in `vaft.formula.sensitivity`;
- what the numbers mean, in `vaft.validation.sensitivity`.

There is no `sensitivity` package, and nothing in this section runs unless it is called.

| function | layer | cost |
| --- | --- | --- |
| `finite_difference_jacobian(f, x)` | formula | 2n model calls (central) |
| `linear_covariance_propagation(J, Σ)` | formula | $J\Sigma J^{\mathsf T}$, correlations kept |
| `monte_carlo_propagation(f, μ, Σ, samples, seed)` | formula | one model call per sample: opt-in only |
| `singular_value_spectrum(J, column_scale)` | formula | SVD of $JD$, never of $J^{\mathsf T}J$ |
| `linearity_ratio(f, x, δ, J)` | formula | share of a finite response the Jacobian misses |
| `Jacobian(provenance, kind, perturbation, inputs, outputs, matrix \| jvp)` | validation | explicit or matrix-free |
| `compare_jacobians(reference, candidate, tolerance=)` | validation | one derivative verified by another |
| `compare_linear_to_monte_carlo(J Σ Jᵀ, sampled, tolerance=)` | validation | local-to-global escalation test |
| `scan_evidence(report)` | validation | a targeted scan (#1663) as model-form evidence |

**Distinctions kept apart:**
- **Provenance:** `native`, `finite_difference`, `autodiff` or `analytic`. Comparing a Jacobian
  with one of the *same* provenance is refused: that is repetition, not validation.
- **Kind:** an `observation` Jacobian (identifiability), a `governing_operator` one (uniqueness,
  branches) and a `forward` one (local sensitivity) answer different questions.
- **Perturbation:** `physical` versus `numerical`. Physical sensitivity is never mixed with
  convergence.

**Tolerances.** A comparison returns metrics. It carries a status only when the caller passes a
`tolerance=(warn, fail)`; no tolerance is invented here.

`test/test_sensitivity_contract.py` demonstrates the #1642 acceptance points at unit-test cost:

1. **A derivative checked by another method.** The exact Green's-function field (an analytic
   derivative already in VAFT, $B = \nabla\times\psi$) matches the finite-difference derivative of
   its flux to below $10^{-6}$.
2. **Local propagation against sampling.** `vaft.process.confinement.dimensionless_confinement_indices`
   already propagates the exponent covariance through a central-difference Jacobian. Its covariance
   equals the generic kernels', and it agrees with 4000-draw Monte Carlo to 5 % in standard
   deviation at $\alpha_P = -0.6$.
3. **Where the linearization fails.** Near $\alpha_P = -1$ (here $1+\alpha_P = 0.07$, about 1.4
   standard errors from the pole), the same map is off by a factor of $e^{3.8}$, as that function's
   docstring warns. The cheap `linearity_ratio` flags it before any sampling is paid for.

### Workflow graduation audit

#1642 §21 asks which workflow-held logic is reusable. These are the candidates found so far. None
is moved by this change; each move belongs to the owning lane.

| workflow logic | what is reusable | owning API |
| --- | --- | --- |
| `workflow/efit_identifiability/identifiability_study.py` (#664) | column scaling, null space, singular classes, direction linearity | already partly in `vaft.code.efit.analyze_efit_identifiability`; the generic SVD is `singular_value_spectrum` |
| `workflow/efit_uncertainty_calibration/weight_scan.py` stage 6 and `sensitivity_report.py` (#1663) | targeted scan over σ and basis; model-form spread | stays a study; read through `scan_evidence` |
| `workflow/confinement_scaling/closures.py` | bootstrap over shots | `vaft.process.confinement.bootstrap_confinement_scaling` (already graduated) |
| `vaft.process.confinement.dimensionless_confinement_indices` | inline central-difference $J\Sigma J^{\mathsf T}$ | could call the formula kernels; left untouched (Lane D) |
