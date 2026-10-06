"""Ordering contracts of the reduced plasma models (#1627 phase C).

Each contract names the asymptotic orderings one model is derived under, as
:class:`vaft.validation.applicability.ApproximationContract` objects, so a state
-- a summary row, a profile point, a time slice -- can be evaluated against
them with :func:`~vaft.validation.applicability.evaluate_contract` and
:func:`~vaft.validation.applicability.evaluate_population`. Nothing here
computes a quantity: the ordering parameters are :mod:`vaft.formula.ordering`
kernels applied by whoever builds the state, under the column names in
:data:`ORDERING_QUANTITIES`.

The assumptions of one model are kept separate on purpose. Ideal single-fluid
MHD needs a large Lundquist number *and* a small ion skin depth *and* a small
ion gyroradius *and* slow evolution; reducing that to ``S >> 1`` would hide
which ordering fails. Every threshold is order unity -- the point where the
expansion parameter stops separating scales -- because no other cutoff has a
derivation behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .applicability import ApproximationContract, OrderingAssumption

__all__ = [
    "CONTRACTS",
    "ORDERING_QUANTITIES",
    "OrderingQuantity",
    "contract",
]


@dataclass(frozen=True)
class OrderingQuantity:
    """What a state column means: its definition, its scale, its scope and the kernels that compute it."""

    definition: str
    scale: str
    scope: str
    kernels: Tuple[str, ...]


#: Column name -> meaning. The names are the ones the #1629 ordering atlas reads.
ORDERING_QUANTITIES: Dict[str, OrderingQuantity] = {
    "lundquist_number": OrderingQuantity(
        "S = mu0 a v_A / eta", "a", "global", ("ordering.lundquist_number",)),
    "ion_skin_depth_over_L": OrderingQuantity(
        "d_i / a", "a", "global", ("ordering.inertial_length",)),
    "rho_i_over_LTi": OrderingQuantity(
        "rho_i / L_Ti", "L_Ti", "profile", ("particle.larmor_radius", "utils.normalized_gradient_scale_length")),
    "rho_s_over_LTe": OrderingQuantity(
        "rho_s / L_Te", "L_Te", "profile", ("ordering.sound_gyroradius", "utils.normalized_gradient_scale_length")),
    "electron_knudsen_number": OrderingQuantity(
        "lambda_e / L_Te", "L_Te", "profile",
        ("ordering.thermal_speed", "ordering.braginskii_electron_collision_time", "ordering.mean_free_path",
         "ordering.knudsen_number")),
    "ion_knudsen_number": OrderingQuantity(
        "lambda_i / L_Ti", "L_Ti", "profile",
        ("ordering.thermal_speed", "ordering.braginskii_ion_collision_time", "ordering.mean_free_path",
         "ordering.knudsen_number")),
    "electron_magnetization": OrderingQuantity(
        "|Omega_ce| tau_e", "-", "profile",
        ("particle.gyrofrequency", "ordering.braginskii_electron_collision_time", "ordering.magnetization")),
    "ion_magnetization": OrderingQuantity(
        "|Omega_ci| tau_i", "-", "profile",
        ("particle.gyrofrequency", "ordering.braginskii_ion_collision_time", "ordering.magnetization")),
    "tau_evolution_over_tau_alfven": OrderingQuantity(
        "tau_evol / tau_A", "a", "time_history", ("ordering.evolution_time", "ordering.alfven_time")),
    "tau_age_over_tau_resistive": OrderingQuantity(
        "tau_age / tau_R", "a", "time_history", ("ordering.resistive_diffusion_time",)),
}


def _assume(quantity: str, ordering: str, meaning: str) -> OrderingAssumption:
    q = ORDERING_QUANTITIES[quantity]
    return OrderingAssumption(quantity, ordering, scale=q.scale, scope=q.scope, meaning=meaning)


_FREIDBERG = "J. P. Freidberg, Ideal MHD, Cambridge University Press (2014), Ch. 2"
_BRAGINSKII = "S. I. Braginskii, in Reviews of Plasma Physics, Vol. 1, Consultants Bureau (1965), p. 205"
_HINTON = "F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239"
_BISKAMP = "D. Biskamp, Magnetic Reconnection in Plasmas, Cambridge University Press (2000), Ch. 1"
_WESSON = "J. Wesson, Tokamaks, 4th ed., Oxford University Press (2011), Sec. 2.11"

#: Contract name -> contract. Global orderings use the minor radius; a layer
#: (tearing, reconnection) needs its own contract with its own scale.
CONTRACTS: Dict[str, ApproximationContract] = {c.name: c for c in (
    ApproximationContract(
        "ideal_single_fluid_mhd", "ideal single-fluid MHD",
        (_assume("lundquist_number", "large", "resistive diffusion slow against Alfvenic dynamics"),
         _assume("ion_skin_depth_over_L", "small", "Hall and two-fluid corrections small"),
         _assume("rho_i_over_LTi", "small", "finite-Larmor-radius corrections small"),
         _assume("tau_evolution_over_tau_alfven", "large", "the background evolves slowly")),
        references=(_FREIDBERG,),
        limitations=("global orderings on the minor radius: a large S says nothing about a thin resistive layer",)),
    ApproximationContract(
        "resistive_mhd", "resistive single-fluid MHD",
        (_assume("ion_skin_depth_over_L", "small", "Hall and two-fluid corrections small"),
         _assume("rho_i_over_LTi", "small", "finite-Larmor-radius corrections small"),
         _assume("tau_evolution_over_tau_alfven", "large", "the background evolves slowly")),
        references=(_BISKAMP,),
        limitations=("no requirement on S: resistivity is kept, not ordered out",
                     "a layer narrower than d_i needs two-fluid physics even where this holds globally")),
    ApproximationContract(
        "local_collisional_fluid", "local collisional (Braginskii-type) fluid closure",
        (_assume("electron_knudsen_number", "small", "electron heat flux local in the gradient"),
         _assume("ion_knudsen_number", "small", "ion heat flux and viscosity local")),
        references=(_BRAGINSKII,),
        limitations=("perpendicular gradient lengths: along the field the parallel connection length decides",)),
    ApproximationContract(
        "strongly_magnetized_fluid", "strongly magnetized transport ordering",
        (_assume("electron_magnetization", "large", "electrons gyrate many times between collisions"),
         _assume("ion_magnetization", "large", "ions gyrate many times between collisions")),
        references=(_BRAGINSKII,)),
    ApproximationContract(
        "flr_small_fluid", "fluid model with perturbative finite-Larmor-radius corrections",
        (_assume("rho_i_over_LTi", "small", "ion gyroradius small against the ion temperature gradient"),
         _assume("rho_s_over_LTe", "small", "sound gyroradius small against the electron temperature gradient")),
        references=(_HINTON,)),
    ApproximationContract(
        "quasi_static_equilibrium", "sequence of MHD equilibria",
        (_assume("tau_evolution_over_tau_alfven", "large", "force balance re-established faster than the state moves"),),
        references=(_FREIDBERG,),
        limitations=("a physical ordering, separate from whether a reconstruction converged",)),
    ApproximationContract(
        "resistively_relaxed_current", "resistively relaxed current profile",
        (_assume("tau_age_over_tau_resistive", "large", "the current has had time to diffuse resistively"),),
        references=(_WESSON,),
        limitations=("tau_R = mu0 a^2 / eta carries no geometric factor: the slowest cylindrical mode decays "
                     "j01^2 ~ 5.8 times faster, so an order-one ratio is ambiguous, not unrelaxed",)),
)}


def contract(name: str) -> ApproximationContract:
    """The registered contract ``name``; ``KeyError`` lists the known ones."""
    try:
        return CONTRACTS[name]
    except KeyError:
        raise KeyError(f"no ordering contract {name!r}; known: {', '.join(CONTRACTS)}") from None
