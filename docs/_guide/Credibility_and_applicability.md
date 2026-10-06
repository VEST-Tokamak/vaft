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
`EVIDENCE_ROLES`, such as `used_for_inference` or `independent_validation`. Pass the checks whose
reference was fitted in `used_for_inference=`, and the adapters move them off the V axis. VAFT's
default EFIT preset fits the diamagnetic flux (#891, #1440), so its diamagnetic checks belong there.

### Cost classes

`COST_CLASSES = ("cheap", "moderate", "expensive")`:
- **cheap:** read from results that already exist, such as a convergence flag or an ordering
  parameter.
- **moderate:** local computation, but no solver rerun.
- **expensive:** solver ensembles, Monte Carlo, continuation. Always opt-in.

## Pilots

| pilot | where | what it shows |
| --- | --- | --- |
| A: equilibrium reconstruction | `evidence_from_equilibrium_report` | the `validate_equilibrium` report on the axes; fit quality (I), physical consistency (N) and independent validation (V) stay apart |
| A: the #891/#1331 study criteria | `evidence_from_efit_criteria` | criteria v2 (#1521); Thomson stays on V and never enters fit quality |
| B: virial closures | the virial checks inside pilot A | identity residual (N), leave-one-identity-out (V), closure conditioning (I) |
| C: current moments (#943) | `successive_discrepancy` | relative change of an observable along the moment order |
| D: asymptotic validity | `vaft.validation.applicability` | ordering margins and contracts, below |

**How the mappings are chosen.** `CATEGORY_AXES` maps each existing validation category to its
default axis. `CHECK_AXES` and `CRITERIA_AXES` record the checks whose axis differs from their
category's default, each with its reason in a source comment.

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

- $m > 0$: satisfied, with $m$ decades to spare.
- $m \le 0$: violated.
- The default $x_0 = 1$ is where the expansion parameter reaches order unity. This is the only
  cutoff applied without a source. Any other threshold must name its source in `threshold_source`,
  or the assumption refuses to construct (#1639 §12).

**Statuses.** They come second, from the same vocabulary as the operational-space renderer (#1664):

| status | meaning |
| --- | --- |
| `SUPPORTED` | every assumption evaluated and satisfied |
| `OUTSIDE` | at least one evaluated assumption violated, whatever else is missing |
| `UNASSESSED` | nothing violated, but some quantity is missing, non-finite or non-positive |
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
