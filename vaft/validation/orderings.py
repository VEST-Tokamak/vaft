"""Ordering contracts of the reduced plasma models (#1627 phase C).

Each contract names the asymptotic orderings one model is derived under, as
:class:`vaft.validation.applicability.ApproximationContract` objects, so a state
-- a summary row, a profile point, a mode, a layer, a time slice -- can be
evaluated against them with :func:`~vaft.validation.applicability.evaluate_contract`
and :func:`~vaft.validation.applicability.evaluate_population`. Nothing here
computes a quantity: the ordering parameters are :mod:`vaft.formula` kernels
applied by whoever builds the state, under the column names in
:data:`ORDERING_QUANTITIES`.

The assumptions of one model are kept separate on purpose, and so are the
scales they are taken on. A global ``d_i/a``, a mode's ``k d_i`` and a
tearing layer's ``d_i/delta`` are three quantities: the first can be small
while the last is of order one, and single-fluid MHD then fails in the layer.
Each quantity therefore belongs to one :data:`GROUPS` entry -- foundational,
global, equilibrium profile, perturbation (scope ``mode``), inner layer
(scope ``local``) or time history -- and carries its own scale.

A contract lists only what its model *requires*. An ordering the model keeps
of order one -- ``k_perp rho_i`` in gyrokinetics, ``k d_i`` in Hall MHD,
``tau_evol/tau_A`` in ideal MHD, which describes Alfvenic dynamics -- is
absent, not marked large: that absence is what separates the model from its
neighbours. Every threshold is order unity -- the point where the expansion
parameter stops separating scales -- because no other cutoff has a derivation
behind it.

Signed ratios (a Mach number, the pressure anisotropy) enter as magnitudes. A
value of exactly zero -- a static or isotropic state -- has no logarithmic
margin, and :func:`~vaft.validation.applicability.ordering_margin` reports it as
``UNASSESSED`` rather than as satisfied by an infinite margin.

This is the fluid-to-kinetic core of the hierarchy, not every subsidiary
ordering of every derivation. Full Vlasov--Maxwell kinetics assumes none of
these and has no contract; a gyrofluid shares the gyrokinetic orderings and
differs by its closure, not by an ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .applicability import ApproximationContract, OrderingAssumption

__all__ = [
    "CONTRACTS",
    "GROUPS",
    "ORDERING_QUANTITIES",
    "OrderingQuantity",
    "contract",
]

#: Where an ordering quantity lives, in the order the contract map draws them.
GROUPS = ("foundational", "global", "profile", "perturbation", "layer", "time_history")


@dataclass(frozen=True)
class OrderingQuantity:
    """What a state column means: its definition, scale, scope, group and the kernels that compute it."""

    definition: str
    scale: str
    scope: str
    group: str
    kernels: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {self.group!r}")


def _q(definition, scale, scope, group, *kernels) -> OrderingQuantity:
    return OrderingQuantity(definition, scale, scope, group, tuple(kernels))


_BRAG_E = ("ordering.thermal_speed", "ordering.braginskii_electron_collision_time", "ordering.mean_free_path",
           "ordering.knudsen_number")
_BRAG_I = ("ordering.thermal_speed", "ordering.braginskii_ion_collision_time", "ordering.mean_free_path",
           "ordering.knudsen_number")

#: Column name -> meaning. The names are the ones the #1629 ordering atlas reads.
ORDERING_QUANTITIES: Dict[str, OrderingQuantity] = {
    # foundational: true of almost every fusion plasma, so they rarely discriminate
    "debye_length_over_L": _q("lambda_D / L", "L", "profile", "foundational", "ordering.debye_length"),
    "omega_over_ion_gyrofrequency": _q("omega / Omega_ci", "Omega_ci", "mode", "foundational",
                                       "particle.gyrofrequency"),
    # global: one number per discharge state, on the minor or major radius
    "lundquist_number": _q("S = mu0 a v_A / eta", "a", "global", "global", "ordering.lundquist_number"),
    "ion_skin_depth_over_a": _q("d_i / a", "a", "global", "global", "ordering.inertial_length"),
    "inverse_aspect_ratio": _q("epsilon = a / R0", "R0", "global", "global",
                               "equilibrium.inverse_aspect_ratio_from_a_R"),
    "beta": _q("beta = 2 mu0 p / B0^2", "B0^2 / 2 mu0", "global", "global", "equilibrium.beta_t_from_n_T_B"),
    "beta_over_inverse_aspect_ratio": _q("beta / epsilon", "epsilon", "global", "global",
                                         "equilibrium.beta_t_from_n_T_B", "equilibrium.inverse_aspect_ratio_from_a_R"),
    # equilibrium profile: per flux surface, on the profile's own gradient length
    "rho_i_over_LTi": _q("rho_i / L_Ti", "L_Ti", "profile", "profile",
                         "particle.larmor_radius", "utils.normalized_gradient_scale_length"),
    "rho_s_over_LTe": _q("rho_s / L_Te", "L_Te", "profile", "profile",
                         "ordering.sound_gyroradius", "utils.normalized_gradient_scale_length"),
    "electron_parallel_knudsen_number": _q("lambda_e / (q R)", "q R", "profile", "profile", *_BRAG_E),
    "ion_parallel_knudsen_number": _q("lambda_i / (q R)", "q R", "profile", "profile", *_BRAG_I),
    "electron_magnetization": _q("|Omega_ce| tau_e", "-", "profile", "profile",
                                 "particle.gyrofrequency", "ordering.braginskii_electron_collision_time",
                                 "ordering.magnetization"),
    "ion_magnetization": _q("|Omega_ci| tau_i", "-", "profile", "profile",
                            "particle.gyrofrequency", "ordering.braginskii_ion_collision_time",
                            "ordering.magnetization"),
    "electron_collisionality": _q("nu*_e = nu_e q R / (epsilon^3/2 v_te)", "q R / epsilon^3/2", "profile",
                                  "profile", "neoclassical.electron_collisionality_sauter"),
    "ion_collisionality": _q("nu*_i = nu_i q R / (epsilon^3/2 v_ti)", "q R / epsilon^3/2", "profile", "profile",
                             "neoclassical.ion_collisionality_sauter"),
    "ion_orbit_width_over_L": _q("Delta_b,i / L_p, Delta_b = q rho_i / sqrt(epsilon)", "L_p", "profile", "profile",
                                 "particle.larmor_radius", "neoclassical.banana_width"),
    "sonic_mach_number": _q("M = U / v_ti", "v_ti", "profile", "profile",
                            "ordering.mach_number", "ordering.thermal_speed"),
    "alfven_mach_number": _q("M_A = U / v_A", "v_A", "profile", "profile",
                             "ordering.mach_number", "stability.v_alfven_from_B_n_mi"),
    "pressure_anisotropy": _q("|Delta| = |p_perp - p_par| / p, the magnitude of the kernel's signed Delta", "p",
                              "profile", "profile", "ordering.pressure_anisotropy"),
    # perturbation: per mode or fluctuation spectrum, on its wavenumber
    "k_perp_rho_i": _q("k_perp rho_i", "1 / k_perp", "mode", "perturbation", "particle.larmor_radius"),
    "k_ion_skin_depth": _q("k d_i", "1 / k", "mode", "perturbation", "ordering.inertial_length"),
    "k_electron_skin_depth": _q("k d_e", "1 / k", "mode", "perturbation", "ordering.inertial_length"),
    "omega_tau_i": _q("omega tau_i", "1 / tau_i", "mode", "perturbation",
                      "ordering.braginskii_ion_collision_time"),
    "k_par_over_k_perp": _q("k_par / k_perp", "1 / k_perp", "mode", "perturbation"),
    "fluctuation_amplitude": _q("delta n / n ~ e delta phi / T_e", "n, T_e / e", "mode", "perturbation"),
    # inner layer: a tearing layer or current sheet, on its width
    "ion_skin_depth_over_layer": _q("d_i / delta", "delta_layer", "local", "layer", "ordering.inertial_length"),
    "electron_skin_depth_over_layer": _q("d_e / delta", "delta_layer", "local", "layer", "ordering.inertial_length"),
    "rho_s_over_layer": _q("rho_s / delta", "delta_layer", "local", "layer", "ordering.sound_gyroradius"),
    # time history: one discharge's evolution against its intrinsic times
    "tau_evolution_over_tau_alfven": _q("tau_evol / tau_A", "a", "time_history", "time_history",
                                        "ordering.evolution_time", "ordering.alfven_time"),
    "tau_age_over_tau_resistive": _q("tau_age / tau_R", "a", "time_history", "time_history",
                                     "ordering.resistive_diffusion_time"),
    "tau_transport_over_tau_turbulence": _q("tau_transport / tau_turb", "L_p / v_ti", "time_history",
                                            "time_history", "ordering.evolution_time"),
}


def _assume(quantity: str, ordering: str, meaning: str) -> OrderingAssumption:
    q = ORDERING_QUANTITIES[quantity]
    return OrderingAssumption(quantity, ordering, scale=q.scale, scope=q.scope, meaning=meaning)


_QUASINEUTRAL = ("debye_length_over_L", "small", "quasineutral: no Debye-scale charge separation")
_LOW_FREQUENCY = ("omega_over_ion_gyrofrequency", "small", "slow against ion gyration")
_SCALAR_PRESSURE = ("pressure_anisotropy", "small", "a scalar pressure")

_FREIDBERG = "J. P. Freidberg, Ideal MHD, Cambridge University Press (2014), Ch. 2"
_BRAGINSKII = "S. I. Braginskii, in Reviews of Plasma Physics, Vol. 1, Consultants Bureau (1965), p. 205"
_HINTON = "F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239"
_HELANDER = "P. Helander and D. J. Sigmar, Collisional Transport in Magnetized Plasmas, Cambridge (2002), Ch. 8"
_BISKAMP = "D. Biskamp, Magnetic Reconnection in Plasmas, Cambridge University Press (2000), Ch. 1"
_WESSON = "J. Wesson, Tokamaks, 4th ed., Oxford University Press (2011), Sec. 2.11"
_STRAUSS = "H. R. Strauss, Phys. Fluids 19 (1976) 134; Phys. Fluids 20 (1977) 1354"
_HAZELTINE = "R. D. Hazeltine and J. D. Meiss, Plasma Confinement, Dover (2003), Ch. 5"
_ABEL = "I. G. Abel et al., Rep. Prog. Phys. 76 (2013) 116201"
_FRIEMAN = "E. A. Frieman and L. Chen, Phys. Fluids 25 (1982) 502"
_PARRA = "F. I. Parra and P. J. Catto, Plasma Phys. Control. Fusion 52 (2010) 045004"
_ROBERTS_TAYLOR = "K. V. Roberts and J. B. Taylor, Phys. Rev. Lett. 8 (1962) 197"


def _contract(name, model, assumptions, references, limitations=()) -> ApproximationContract:
    return ApproximationContract(name, model, tuple(_assume(*a) for a in assumptions),
                                 references=references, limitations=limitations)


#: Contract name -> contract, in the order of the hierarchy: fluid, reduced fluid,
#: kinetic, neoclassical, then the equilibrium and time-history statements.
CONTRACTS: Dict[str, ApproximationContract] = {c.name: c for c in (
    _contract(
        "ideal_single_fluid_mhd", "ideal single-fluid MHD",
        (_QUASINEUTRAL, _LOW_FREQUENCY,
         ("lundquist_number", "large", "resistive diffusion slow against Alfvenic dynamics"),
         ("ion_skin_depth_over_a", "small", "no Hall physics on the global scale"),
         ("k_ion_skin_depth", "small", "no Hall physics on the perturbation's scale"),
         ("k_perp_rho_i", "small", "no finite-Larmor-radius physics in the perturbation"),
         ("rho_i_over_LTi", "small", "no finite-Larmor-radius physics in the equilibrium"),
         _SCALAR_PRESSURE),
        (_FREIDBERG,),
        ("describes Alfvenic dynamics, omega tau_A ~ 1: slow evolution is the equilibrium-sequence contract, "
         "not a condition on MHD",
         "a large global S says nothing about a thin resistive layer",
         "the collisionality condition of the textbook derivation enters only through the scalar pressure: "
         "a collisionless plasma can still satisfy ideal MHD's perpendicular dynamics")),
    _contract(
        "resistive_mhd", "resistive single-fluid MHD",
        (_QUASINEUTRAL, _LOW_FREQUENCY,
         ("ion_skin_depth_over_a", "small", "no Hall physics on the global scale"),
         ("k_ion_skin_depth", "small", "no Hall physics on the perturbation's scale"),
         ("k_perp_rho_i", "small", "no finite-Larmor-radius physics in the perturbation"),
         ("rho_i_over_LTi", "small", "no finite-Larmor-radius physics in the equilibrium"),
         ("ion_skin_depth_over_layer", "small", "the resistive layer is wider than d_i"),
         ("rho_s_over_layer", "small", "the resistive layer is wider than rho_s: no drift-tearing physics"),
         _SCALAR_PRESSURE),
        (_BISKAMP,),
        ("no requirement on S: resistivity is kept, not ordered out",
         "the layer orderings are evaluated on the layer the model resolves; d_i/a small does not imply them",
         "in a strong guide field rho_s, not d_i, is the two-fluid scale of the layer")),
    _contract(
        "hall_mhd", "Hall (extended) MHD",
        (_QUASINEUTRAL,
         ("k_electron_skin_depth", "small", "electron inertia negligible on the perturbation's scale"),
         ("electron_skin_depth_over_layer", "small", "electron inertia negligible in the layer"),
         _SCALAR_PRESSURE),
        (_BISKAMP,),
        ("k d_i and d_i/delta are not ordered: Hall MHD exists to keep them of order one",
         "ion FLR enters at k_perp rho_i ~ k d_i sqrt(beta) and is not kept consistently")),
    _contract(
        "braginskii_two_fluid", "Braginskii magnetized two-fluid closure",
        (_QUASINEUTRAL,
         ("electron_parallel_knudsen_number", "small", "electron parallel heat flux local in the gradient"),
         ("ion_parallel_knudsen_number", "small", "ion parallel heat flux and viscosity local"),
         ("electron_magnetization", "large", "electrons gyrate many times between collisions"),
         ("ion_magnetization", "large", "ions gyrate many times between collisions"),
         ("rho_i_over_LTi", "small", "perpendicular transport local on the Larmor scale"),
         ("omega_tau_i", "small", "dynamics slow against ion collisions (Chapman-Enskog)")),
        (_BRAGINSKII,),
        ("the parallel Knudsen number is lambda/(qR), the inverse of nu_hat up to an order-one convention "
         "factor: the closure holds in the Pfirsch-Schlueter regime and fails in the banana regime of a hot core",
         "perpendicular locality is the magnetized rho/L ordering, not a collisional one")),
    _contract(
        "flr_small_fluid", "fluid model with perturbative finite-Larmor-radius corrections",
        (("rho_i_over_LTi", "small", "ion gyroradius small against the ion temperature gradient"),
         ("rho_s_over_LTe", "small", "sound gyroradius small against the electron temperature gradient"),
         ("k_perp_rho_i", "small", "FLR corrections an expansion in k_perp rho_i")),
        (_ROBERTS_TAYLOR, _BRAGINSKII)),
    _contract(
        "low_beta_reduced_mhd", "low-beta reduced MHD",
        (("inverse_aspect_ratio", "small", "the large-aspect-ratio expansion"),
         ("beta_over_inverse_aspect_ratio", "small", "beta ~ epsilon^2: the toroidal field is incompressible"),
         ("k_par_over_k_perp", "small", "field-aligned perturbations, shear-Alfven dynamics only")),
        (_STRAUSS,),
        ("the reduction's own orderings: evaluate together with the ideal or resistive MHD contract",)),
    _contract(
        "high_beta_reduced_mhd", "high-beta reduced MHD",
        (("inverse_aspect_ratio", "small", "the large-aspect-ratio expansion"),
         ("beta", "small", "beta ~ epsilon: pressure enters at first order"),
         ("k_par_over_k_perp", "small", "field-aligned perturbations")),
        (_STRAUSS,),
        ("beta/epsilon is of order one, not ordered",
         "the reduction's own orderings: evaluate together with the ideal or resistive MHD contract")),
    _contract(
        "drift_kinetic", "drift kinetics",
        (_QUASINEUTRAL, _LOW_FREQUENCY,
         ("rho_i_over_LTi", "small", "guiding-centre expansion in rho/L"),
         ("k_perp_rho_i", "small", "perturbations longer than the gyroradius")),
        (_HAZELTINE,)),
    _contract(
        "gyrokinetic_delta_f", "delta-f gyrokinetics",
        (_QUASINEUTRAL, _LOW_FREQUENCY,
         ("rho_i_over_LTi", "small", "rho* = rho_i/L small"),
         ("k_par_over_k_perp", "small", "anisotropic, field-aligned fluctuations"),
         ("fluctuation_amplitude", "small", "delta f / F_0 ~ rho*"),
         ("sonic_mach_number", "small", "low-flow ordering")),
        (_FRIEMAN, _ABEL, _PARRA),
        ("k_perp rho_i is not ordered: it may be of order one, which is what separates gyrokinetics "
         "from drift kinetics and MHD",
         "high-flow gyrokinetics allows M ~ 1",
         "a gyrofluid shares these orderings and differs by its closure")),
    _contract(
        "local_neoclassical", "local neoclassical transport",
        (("rho_i_over_LTi", "small", "rho* small"),
         ("ion_orbit_width_over_L", "small", "the banana orbit narrower than the profile gradient"),
         ("sonic_mach_number", "small", "subsonic flow")),
        (_HINTON, _HELANDER),
        ("fails in a pedestal or transport barrier, where Delta_orbit/L ~ 1",)),
    _contract(
        "banana_regime_neoclassical", "banana-regime neoclassical transport (both species)",
        (("electron_collisionality", "small", "electrons complete their banana orbits"),
         ("ion_collisionality", "small", "ions complete their banana orbits")),
        (_HELANDER,),
        ("nu* already carries epsilon^3/2: nu* << 1 is nu_hat << epsilon^3/2, up to the order-one factor "
         "between Sauter's collision frequency and Braginskii's",
         "the plateau between the banana and Pfirsch-Schlueter regimes has no ordering of its own",
         "the regime is per species: evaluate the electron and ion orderings separately when they differ")),
    _contract(
        "pfirsch_schlueter_neoclassical", "Pfirsch-Schlueter neoclassical transport (both species)",
        (("electron_parallel_knudsen_number", "small", "nu_hat_e ~ qR/lambda_e >> 1"),
         ("ion_parallel_knudsen_number", "small", "nu_hat_i ~ qR/lambda_i >> 1")),
        (_HELANDER,)),
    _contract(
        "quasi_static_equilibrium", "sequence of MHD equilibria",
        (("tau_evolution_over_tau_alfven", "large", "force balance re-established faster than the state moves"),
         ("sonic_mach_number", "small", "static force balance, no centrifugal term"),
         ("alfven_mach_number", "small", "flow absent from the magnetic force balance")),
        (_FREIDBERG,),
        ("a physical ordering, separate from whether a reconstruction converged",
         "a rotating equilibrium (M ~ 1) needs the flow-modified Grad-Shafranov equation")),
    _contract(
        "resistively_relaxed_current", "resistively relaxed current profile",
        (("tau_age_over_tau_resistive", "large", "the current has had time to diffuse resistively"),),
        (_WESSON,),
        ("tau_R = mu0 a^2 / eta carries no geometric factor: the slowest cylindrical mode decays "
         "j01^2 ~ 5.8 times faster, so an order-one ratio is ambiguous, not unrelaxed",)),
    _contract(
        "gyrokinetic_transport_separation", "gyrokinetic turbulence with transport-scale separation",
        (("rho_i_over_LTi", "small", "rho* small"),
         ("tau_transport_over_tau_turbulence", "large", "profiles evolve slowly against the turbulence")),
        (_ABEL,),
        ("the multiscale ordering of flux-tube gyrokinetics coupled to a transport solver",)),
)}


def contract(name: str) -> ApproximationContract:
    """The registered contract ``name``; ``KeyError`` lists the known ones."""
    try:
        return CONTRACTS[name]
    except KeyError:
        raise KeyError(f"no ordering contract {name!r}; known: {', '.join(CONTRACTS)}") from None
