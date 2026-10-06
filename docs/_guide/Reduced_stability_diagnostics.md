---
title: Reduced stability diagnostics
author: VEST team
date: 2026-10-06 10:00
category: guide
layout: post
permalink: /reference/reduced-stability-diagnostics/
guide:
  architecture: An inventory of VAFT's equilibrium-derived analytic, reduced and empirical MHD stability diagnostics, each with its logical status, assumptions, reference and validation path; kernels in vaft.formula.stability.
  prerequisites: None.
  expected: For each criterion what it is, what it assumes, what it proves and what it does not, where VAFT implements it, and what to compare it with.
related:
  api: [formula, diagram]
---

VAFT holds many stability-related quantities: beta limits, the Greenwald fraction, $l_i$–$q$ boundaries,
the $s$–$\alpha$ ballooning model, Suydam's and Mercier's interchange criteria, Bussac's internal kink,
$\Delta'$, and the local stability quantities DCON, RDCON and STRIDE write. They are **not**
interchangeable stable/unstable tests. Some are exact definitions, some are reduced models with strong
assumptions, some are empirical correlations of a few machines, and some are heuristics kept only for
backward compatibility. This page records which is which.

The relationship to the full solvers is

```text
equilibrium-derived reduced diagnostics
        ├── interpretation
        ├── limiting-case validation
        └── comparison  ──>  DCON / RDCON / STRIDE / GPEC
```

and not a screening gate. Satisfying Suydam, Mercier, Bussac or $s$–$\alpha$ does not prove MHD
stability. For VEST, $a/R_0$ is not small, so where a large-aspect-ratio criterion disagrees with a
full-geometry calculation, the disagreement is information about the reduced model.

![stability diagnostic taxonomy]({{ '/assets/diagrams/stability_diagnostic_taxonomy.svg' | relative_url }})

## Logical status

| Status | Meaning | Example |
| --- | --- | --- |
| exact definition | a quantity, not a criterion; true wherever it is defined | $\Delta'$ from the outer slopes, the decay index |
| reduced model | derived from MHD under a stated geometry and ordering | Suydam, circular Mercier, Bussac, $s$–$\alpha$ |
| semi-empirical | a fit to numerical stability calculations | Troyon $\beta_N$ |
| empirical | a correlation of measured discharges, machine-specific or multi-machine | Greenwald, Wesson JET $l_i$–$q_\psi$ |
| heuristic | an approximation or an unsourced rule; deprecated where it misleads | $\alpha_{crit} \approx 0.6\,\hat s$, `kink_stability_criterion` |
| solver-derived | read from DCON / RDCON / STRIDE output, not computed by VAFT | DCON $D_I$, $C_A$; RDCON $\Delta'$, $D_R$, $H$ |

A **necessary** criterion (Suydam, Mercier) proves instability when violated and proves nothing when
satisfied. A **marginal boundary** ($s$–$\alpha$, Bussac's $\beta_{p1} \approx 0.3$) separates stable from
unstable only within its model.

## Inventory

| Criterion | Problem | Geometry and ordering | Type | Status | VAFT | Primary reference | Validation path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kruskal–Shafranov, $q^*$ | external kink | cylinder; Freidberg / Menard $q^*$ for elongation | algebraic | reduced | boundaries `freidberg_2008_kink_*`, `menard_2004_qstar_*`, `low_q` | Kruskal (1954); Shafranov (1956) | DCON $n = 1$ |
| Bussac internal kink | $m = n = 1$ internal kink | large $A$, circular, parabolic $q$, $1 - q_0 \ll 1$ | algebraic $\delta W$ | reduced, marginal at $\beta_{p1} = \sqrt{13/144}$ | `bussac_poloidal_beta`, `bussac_internal_kink_energy` | Bussac et al., PRL 35 (1975) 1638 | published limit; DCON $n = 1$ on circular equilibria |
| Cheng–Furth–Boozer $l_i$–$q_a$ | ideal and resistive kinks | pressureless cylinder, current-profile family | envelope of eigenproblems | reduced (projected) | boundaries `cheng_1987_*`, `li_qa(reference="cheng_1987")` | Cheng, Furth, Boozer, PPCF 29 (1987) 351 | the published envelope |
| Wesson JET $l_i$–$q_\psi$ | disruption-free operation | JET discharges | correlation | empirical, machine-specific | `empirical_li_qa`, `li_qa(reference="wesson_1989")` | Wesson et al., NF 29 (1989) 641 | the published figure |
| Suydam | local interchange | straight cylinder, $m \to \infty$ | algebraic | reduced, necessary | `suydam_criterion` | Suydam (1958) | analytic profiles; cylindrical limit of Mercier |
| Mercier | local interchange | circular tokamak, $\epsilon \ll 1$, low $\beta$ | algebraic | reduced, necessary | `mercier_criterion_circular` | Mercier, NF 1 (1960) 47 | DCON $D_I$ where conventions overlap |
| Mercier, general | local interchange | shaped torus, flux-surface averages | surface integrals | reduced, necessary | not yet: DCON `di` is read | Mercier (1960); Glasser et al. (1975) | DCON $D_I$ |
| GGJ $D_I$, $D_R$, $H$ | ideal and resistive interchange | toroidal inner layer | algebraic in $E, F, H$ | definition | `ggj_ideal_interchange_index`, `ggj_resistive_interchange_index` | Glasser, Greene, Johnson, Phys. Fluids 18 (1975) 875 | RDCON / STRIDE `di`, `dr`, `h` |
| magnetic well | average curvature | nested surfaces, toroidal-flux label | derivative of $V'$ | definition (diagnostic, no boundary) | `magnetic_well_from_specific_volume` | Greene, Comments PPCF 17 (1997) 389 | cylinder ($W = 0$); analytic $V'$ |
| $s$–$\alpha$ ballooning | high-$n$ ballooning | shifted circles, large $A$, $n \to \infty$, ideal | ODE (Newcomb shooting) | reduced, first and second boundary | `s_alpha_marginal_alpha`, `s_alpha_ballooning_stable`, `s_alpha_ballooning_eigenmode` | Connor, Hastie, Taylor, PRL 40 (1978) 396 | the published diagram; DCON `ca1` |
| $\alpha_{crit} \approx 0.6\,\hat s$ | first ballooning boundary | as above, moderate shear | linear fit | heuristic (empirical-fit flag) | `ballooning_stability_criterion` | CHT (1978), Fig. 1 | `s_alpha_marginal_alpha` |
| curvature $\kappa_n$, $\kappa_g$, local shear | interchange and ballooning drives | general equilibrium | field-line geometry | definition | not yet | — | — |
| $\Delta'$ | tearing | outer region, any geometry | matching of an ODE solution | definition; benchmark slab / cylinder | `delta_prime_from_outer_derivatives`; RDCON / STRIDE $\Delta'$ read | Furth, Killeen, Rosenbluth (1963) | RDCON / STRIDE |
| decay index | vertical stability | external field | derivative | definition | `decay_index_from_bz` | Mukhovatov, Shafranov, NF 11 (1971) 605 | — |
| critical decay index | vertical stability with a wall | device- and wall-dependent | — | — | not yet | — | — |
| Kadomtsev mixing radius | sawtooth crash | cylinder, complete reconnection | integral | reduced | `kadomtsev_mixing_radius` | Kadomtsev (1975) | analytic $\sqrt2\,r_1$ |
| Porcelli trigger | sawtooth trigger | toroidal $\delta W$ + kinetic terms | composite | — | not yet; the old `sawtooth_stability_criterion` is a deprecated heuristic | Porcelli, Boucher, Rosenbluth, PPCF 38 (1996) 2163 | — |
| Modified Rutherford | island evolution | — | ODE | — | owned by #1031 | — | — |
| Troyon $\beta_N$ | ideal $\beta$ limit | numerical ideal stability of a family | fit | semi-empirical | boundary `troyon` | Troyon et al., PPCF 26 (1984) 209 | — |
| Greenwald, Murakami, Hugill | density limit | multi-machine | correlation | empirical | boundaries `greenwald*`, `murakami*` (Hugill coordinates: `hugill_coordinates`) | Greenwald, PPCF 44 (2002) R27 | — |

Deprecated heuristics stay for backward compatibility and warn when called: `kink_stability_criterion`
($\beta_{N,crit} = 2.8\,q_{95}$, unsourced), `sawtooth_stability_criterion`, `beta_stability_boundary` and
`plasma_stability_margins`. `power_limit_from_beta` and `power_limit_from_q` are misnamed: they are
rearranged $\beta$ and $q$ relations, not power limits. They are listed as heuristics and are not yet
deprecated.

## Suydam and Mercier

Both compare shear stabilisation, $rB^2(q'/q)^2/8\mu_0$, with the pressure drive $p'$. The toroidal
average curvature turns the drive into $p'(1 - q^2)$: outside $q = 1$ an outward pressure fall
stabilises. Both are **necessary** conditions for local interchanges.

![Suydam and Mercier]({{ '/assets/diagrams/interchange_criteria.svg' | relative_url }})

Sign conventions differ. `suydam_criterion` and `mercier_criterion_circular` are positive where the
criterion holds. The GGJ index $D_I$, which DCON writes, is **negative** where it holds, and
$D_R = D_I + (H - \tfrac12)^2 \ge D_I$, so a surface can be Mercier-stable and still resistive-interchange
unstable.

## Bussac internal kink

In a cylinder the $m = n = 1$ internal kink is marginal at leading order, and toroidicity decides it.
For a parabolic $q$ inside $r_1$, $\delta\hat W_T \propto (1 - q_0)(13/144 - \beta_{p1}^2)$, with
$\beta_{p1} = 2\mu_0(\langle p\rangle_1 - p(r_1))/B_{\theta1}^2$. A core above
$\beta_{p1} \approx 0.30$ is unstable. Shaping, other $q$ profiles and the kinetic terms of Porcelli's model
change this. Expose $\epsilon_1 = r_1/R_0$, the elongation and the triangularity alongside the result, so a
later study can see how far an equilibrium is from the model.

## Where the layers stop

The kernels here take explicit physical quantities and never read an ODS. Evaluating them on a
reconstructed equilibrium belongs to `vaft.process`:
- locating $q = m/n$ surfaces;
- profile derivatives;
- flux-surface averages, the equilibrium-native Mercier $D_I$, and curvature and local shear.

Running and reading solvers belongs to `vaft.code.gpec`. Those are the next steps of #1635 and are left to
the stability workflow. Use [MHD stability]({{ '/workflows/mhd-stability/' | relative_url }}) for the
solver side and [MHD mode representations across geometries]({{ '/reference/geometric-approximations/#mhd-mode-representations-across-geometries' | relative_url }})
for which mode family each criterion addresses.

## References

1. B. R. Suydam, Proc. 2nd UN Int. Conf. Peaceful Uses of Atomic Energy 31 (1958) 157.
2. C. Mercier, Nucl. Fusion 1 (1960) 47.
3. A. H. Glasser, J. M. Greene and J. L. Johnson, Phys. Fluids 18 (1975) 875.
4. M. N. Bussac, R. Pellat, D. Edery and J. L. Soulé, Phys. Rev. Lett. 35 (1975) 1638.
5. J. W. Connor, R. J. Hastie and J. B. Taylor, Phys. Rev. Lett. 40 (1978) 396.
6. C. Z. Cheng, H. P. Furth and A. H. Boozer, Plasma Phys. Control. Fusion 29 (1987) 351.
7. F. Porcelli, D. Boucher and M. N. Rosenbluth, Plasma Phys. Control. Fusion 38 (1996) 2163.
8. J. M. Greene, Comments Plasma Phys. Control. Fusion 17 (1997) 389.
9. J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014).
