---
title: Asymptotic orderings
author: VEST team
date: 2026-10-05 16:00
category: guide
layout: post
permalink: /reference/asymptotic-orderings/
guide:
  architecture: The asymptotic ordering parameters reduced plasma models expand in - Lundquist number, inertial lengths, gyroradii, Knudsen numbers, magnetization, evolution times - as explicit-length kernels in vaft.formula.ordering, kept separate from performance and similarity quantities.
  prerequisites: None.
  expected: What each ordering parameter tests, which length or species it must name, which model assumption it bears on, and a worked timescale hierarchy.
related:
  api: [formula, diagram]
---

Some dimensionless numbers say how well a plasma performs, or where it sits in similarity space
($\beta_N$, $\rho_*$, $\nu_*$). Others are the small or large parameters a model's derivation expands
in. A Lundquist number tests whether resistive diffusion is slow compared with Alfvénic dynamics.
$d_i/L$, $\rho_i/L$, a Knudsen number or $\Omega\tau$ each test a different reduction. This page covers
the second kind: **asymptotic ordering parameters** (#1627). The question they answer is which reduced
models the scale separation of a given state justifies.

The kernels are in [`vaft.formula.ordering`]({{ '/reference/formula/ordering/' | relative_url }}). Their
inputs come from VAFT's Alfvén speed, Spitzer resistivity, gyrofrequency and Larmor radius formulas, and
the inertial length calls the plasma frequency. The
characteristic length, the resistivity and the collision time are always explicit arguments, so a global
scale cannot be silently swapped for a layer scale.

## The parameters and what they test

| Quantity | Kernel | Tests | Must name |
| --- | --- | --- | --- |
| $\tau_A = L/v_A$, $\tau_R = \mu_0L^2/\eta$ | `alfven_time`, `resistive_diffusion_time` | the two ends of the MHD hierarchy | $L$, $\eta$ model, $Z_{eff}$, $\ln\Lambda$ |
| $S = \tau_R/\tau_A$ | `lundquist_number` | resistivity perturbatively small **at that scale** | $L$: global $a$ or a layer width |
| $R_m = \mu_0VL/\eta$ | `magnetic_reynolds_number` | flux freezing in a flow | a measured or modelled $V$ |
| $d_i/L$, $d_e/L$ | `inertial_length` | Hall and electron-inertia corrections | species; global $a$ vs layer $\delta$ |
| $\rho_s/L_T$, $\rho_i/L$ | `sound_gyroradius`, `particle.larmor_radius` | finite-Larmor-radius corrections | the local gradient length |
| $Kn = \lambda/L$ | `thermal_speed`, `braginskii_*_collision_time`, `mean_free_path`, `knudsen_number` | local collisional fluid closure | species; $L_T$, $L_n$ or $L_\parallel$ |
| $\chi = \lvert\Omega\rvert\tau$ | `magnetization` | strongly magnetized transport | species |
| $\tau_{evol}/\tau_A$ | `evolution_time` | the plasma evolves through a sequence of equilibria | the state variable ($I_p$, $W$, axis position) |

Some distinctions must not be blurred:
- **A large global $S$ does not make resistivity irrelevant in a tearing layer.** The layer's own $S$, and
  $d_i/\delta$, $\rho_s/\delta$, decide that.
- **Global $\rho_*$ is a similarity coordinate. Local $\rho_i/L_T$ is an FLR ordering.** They are not
  interchangeable.
- **$\tau_{evol}/\tau_A \gg 1$ is physics.** It says nothing about whether an EFIT run converged, and
  convergence says nothing about it.
- **$Kn$ and $\Omega\tau$ are independent.** A plasma can be strongly magnetized and collisionless along
  the field at the same time.

## A worked hierarchy

![timescale hierarchy]({{ '/assets/diagrams/timescale_hierarchy.svg' | relative_url }})

`timescale_hierarchy` evaluates the hierarchy of one illustrative low-field spherical-tokamak state rather
than assuming it. Every time comes from a formula kernel; only the pulse length is an input.
- **Clearly ordered:** $S$ and $\tau_{evol}/\tau_A$, both about $10^4$.
- **Of order one, so not ordered:**
  - $\tau_{pulse}/\tau_R \approx 0.7$. Resistive relaxation has to be checked, not assumed either way.
    $\tau_R$ carries no geometric factor, and the slowest cylindrical mode decays $j_{01}^2 \approx 5.8$
    times faster.
  - $d_i/a \approx 0.3$, so Hall corrections are not globally negligible.

These numbers come from an illustrative state, not a measured discharge. Answering the same questions
across the VEST database is the ordering atlas of #1629.

## Ordering contracts

Whether an ordering holds depends on the model that assumes it. The contracts are registered in
`vaft.validation.orderings.CONTRACTS` as `ApproximationContract` objects of
`vaft.validation.applicability` (#1639). They are evaluated with `evaluate_contract` for one state and
`evaluate_population` for a table, whose columns are the names in `ORDERING_QUANTITIES`.

![ordering contracts]({{ '/assets/diagrams/ordering_contract_map.svg' | relative_url }})

The quantities are grouped by the scale they are taken on, because the same ratio means different things
on different scales. A global $d_i/a$, a mode's $k\,d_i$ and a tearing layer's $d_i/\delta$ are three rows:
$d_i/a$ can be small while $d_i/\delta \sim 1$, and single-fluid MHD then fails in the layer.

| Group | Quantities |
| --- | --- |
| foundational | $\lambda_D/L$ (quasineutrality), $\omega/\Omega_{ci}$ (low frequency). They hold almost everywhere in a fusion core, so they rarely discriminate |
| global | $S$, $d_i/a$, $\epsilon = a/R_0$, $\beta$, $\beta/\epsilon$ |
| equilibrium profile | $\rho_i/L_{T_i}$, $\rho_s/L_{T_e}$, $\lambda_{e,i}/qR$, $\Omega_{ce}\tau_e$, $\Omega_{ci}\tau_i$, $\nu_{*e}$, $\nu_{*i}$, $\Delta_{b,i}/L_p$, $M = U/v_{ti}$, $M_A = U/v_A$, $\lvert p_\perp - p_\parallel\rvert/p$ |
| perturbation ($k$) | $k_\perp\rho_i$, $k\,d_i$, $k\,d_e$, $\omega\tau_i$, $k_\parallel/k_\perp$, $\delta n/n$ |
| inner layer ($\delta$) | $d_i/\delta$, $d_e/\delta$, $\rho_s/\delta$ |
| time history | $\tau_{evol}/\tau_A$, $\tau_{age}/\tau_R$, $\tau_{transp}/\tau_{turb}$ |

| Contract | Needs |
| --- | --- |
| `ideal_single_fluid_mhd` | $S \gg 1$; $d_i/a$, $k\,d_i$, $k_\perp\rho_i$, $\rho_i/L_{T_i}$, $\lvert p_\perp - p_\parallel\rvert/p \ll 1$; quasineutral and low frequency. Not $\tau_{evol}/\tau_A \gg 1$: ideal MHD describes Alfvénic dynamics, $\omega\tau_A \sim 1$ |
| `resistive_mhd` | the same without $S$, and with $d_i/\delta$, $\rho_s/\delta \ll 1$ in the layer it resolves. In a strong guide field $\rho_s$ is the layer's two-fluid scale |
| `hall_mhd` | $k\,d_e$, $d_e/\delta \ll 1$; quasineutral. $k\,d_i$ and $d_i/\delta$ are not ordered: Hall MHD exists to keep them of order one |
| `braginskii_two_fluid` | $\lambda_{e,i}/qR \ll 1$, $\Omega_{ce}\tau_e$, $\Omega_{ci}\tau_i \gg 1$, $\rho_i/L_{T_i} \ll 1$, $\omega\tau_i \ll 1$; quasineutral |
| `flr_small_fluid` | $\rho_i/L_{T_i}$, $\rho_s/L_{T_e}$, $k_\perp\rho_i \ll 1$ |
| `low_beta_reduced_mhd` | $\epsilon$, $\beta/\epsilon$, $k_\parallel/k_\perp \ll 1$ ($\beta \sim \epsilon^2$) |
| `high_beta_reduced_mhd` | $\epsilon$, $\beta$, $k_\parallel/k_\perp \ll 1$ ($\beta \sim \epsilon$, so $\beta/\epsilon$ is not ordered) |
| `drift_kinetic` | $\rho_i/L_{T_i}$, $k_\perp\rho_i$, $\omega/\Omega_{ci} \ll 1$; quasineutral |
| `gyrokinetic_delta_f` | $\rho_i/L_{T_i}$, $k_\parallel/k_\perp$, $\omega/\Omega_{ci}$, $\delta n/n$, $M \ll 1$; quasineutral. $k_\perp\rho_i$ is not ordered, and that separates gyrokinetics from drift kinetics and MHD |
| `local_neoclassical` | $\rho_i/L_{T_i}$, $\Delta_{b,i}/L_p$, $M \ll 1$; fails in a pedestal or barrier |
| `banana_regime_neoclassical` | $\nu_{*e}$, $\nu_{*i} \ll 1$ |
| `pfirsch_schlueter_neoclassical` | $\lambda_{e,i}/qR \ll 1$, i.e. $\hat\nu \sim qR/\lambda \gg 1$ |
| `quasi_static_equilibrium` | $\tau_{evol}/\tau_A \gg 1$, $M$, $M_A \ll 1$ |
| `resistively_relaxed_current` | $\tau_{age}/\tau_R \gg 1$. An order-one value is ambiguous, because $\tau_R$ has no geometric factor |
| `gyrokinetic_transport_separation` | $\rho_i/L_{T_i} \ll 1$, $\tau_{transp}/\tau_{turb} \gg 1$ |

A contract lists only what its model requires. An empty cell means the model does not order that
quantity, which is different from requiring it to be large. The parallel Knudsen number
$\lambda/qR$, the inverse of $\hat\nu$ up to an order-one convention factor, is shared by the Braginskii closure and the Pfirsch–Schlüter regime: the
collisional closure is the Pfirsch–Schlüter ordering, and it fails in the banana regime of a hot core.
The reduced-MHD contracts list only the reduction's own orderings, so evaluate them together with the
ideal or resistive MHD contract.

Every threshold is order unity. The result is a continuous margin per ordering,
$m = \mp\log_{10}x$, and a status per contract (`SUPPORTED`, `OUTSIDE`, `UNASSESSED`, `NOT_APPLICABLE`),
never a single valid/invalid flag. A quantity a state does not carry is `UNASSESSED`, never a violation.
Signed ratios (Mach numbers, the pressure anisotropy) enter as magnitudes. An exactly zero value, as in a
static or isotropic state, has no logarithmic margin and is also `UNASSESSED`.

This covers the fluid-to-kinetic core of the hierarchy, not every subsidiary ordering of every derivation.
Full Vlasov–Maxwell kinetics assumes none of these orderings and has no contract. A gyrofluid shares the
gyrokinetic orderings and differs only in its closure. The plateau regime between banana and
Pfirsch–Schlüter has no ordering of its own.

## References

1. S. I. Braginskii, in *Reviews of Plasma Physics*, Vol. 1, Consultants Bureau (1965), p. 205.
2. J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).
3. D. Biskamp, *Magnetic Reconnection in Plasmas*, Cambridge University Press (2000).
4. H. R. Strauss, Phys. Fluids 19 (1976) 134; Phys. Fluids 20 (1977) 1354.
5. F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239.
6. P. Helander and D. J. Sigmar, *Collisional Transport in Magnetized Plasmas*, Cambridge University Press (2002).
7. E. A. Frieman and L. Chen, Phys. Fluids 25 (1982) 502.
8. I. G. Abel et al., Rep. Prog. Phys. 76 (2013) 116201.
9. F. I. Parra and P. J. Catto, Plasma Phys. Control. Fusion 52 (2010) 045004.
10. R. D. Hazeltine and J. D. Meiss, *Plasma Confinement*, Dover (2003).
11. K. V. Roberts and J. B. Taylor, Phys. Rev. Lett. 8 (1962) 197.
