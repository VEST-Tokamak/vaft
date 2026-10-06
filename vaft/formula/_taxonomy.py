"""The controlled vocabulary of reduced physical representations (#1626).

A formula that turns one representation of the plasma into a more reduced one
-- a field into a profile, a profile into a scalar, a dimensional quantity
into a dimensionless one -- may say so in a ``Reduction`` docstring section::

    Reduction
    ---------
    input: profile_1d
    output: scalar_0d
    kind: quadratic_integral
    locality: global
    role: global_descriptor

Each value must come from the tuples below; ``input`` may list several
representations separated by commas. The docstring stays the single source of
truth; this module only fixes the words and parses them, so it imports nothing
and costs the numerical kernels nothing.

The four axes are independent. In particular *dimensionless is not 0-D*: a
local $\\nu_*(\\rho)$ is a ``dimensionless_normalization`` with a ``profile_1d``
output, and a global similarity variable is one with a ``scalar_0d`` output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: What space a quantity lives in.
SPATIAL_REPRESENTATIONS: Tuple[str, ...] = (
    "field_3d",
    "field_2d",
    "profile_1d",
    "scalar_0d",
    "flux_surface_quantity",
    "boundary_quantity",
    "time_series",
)

#: The mathematical operation that reduces it.
REDUCTION_KINDS: Tuple[str, ...] = (
    "integral",
    "moment",
    "quadratic_integral",
    "weighted_average",
    "projection",
    "feature_extraction",
    "extremum",
    "differential",
    "normalization",
    "dimensionless_normalization",
    "similarity_transform",
    "closure",
    "empirical_scaling",
)

#: Where the result is defined.
LOCALITIES: Tuple[str, ...] = ("point_local", "flux_surface_local", "edge", "global")

#: Why the reduced quantity is useful.
PHYSICAL_ROLES: Tuple[str, ...] = (
    "state_coordinate",
    "profile_descriptor",
    "global_descriptor",
    "regime_coordinate",
    "similarity_coordinate",
    "stability_coordinate",
    "closure_input",
    "closure_output",
)

#: Kinds whose output is dimensionless by construction.
DIMENSIONLESS_KINDS = frozenset({"dimensionless_normalization", "similarity_transform"})
#: Roles whose output must be dimensionless.
DIMENSIONLESS_ROLES = frozenset({"similarity_coordinate"})

#: The keys of a ``Reduction`` section, in the order they are written.
FIELDS: Tuple[str, ...] = ("input", "output", "kind", "locality", "role")
_VOCABULARY = {
    "output": SPATIAL_REPRESENTATIONS,
    "kind": REDUCTION_KINDS,
    "locality": LOCALITIES,
    "role": PHYSICAL_ROLES,
}


@dataclass(frozen=True)
class Reduction:
    """One formula's place in the reduced-representation taxonomy."""

    input: Tuple[str, ...]
    output: str
    kind: str
    locality: str
    role: str

    @property
    def mapping(self) -> str:
        """``"profile_1d -> scalar_0d"``: the spatial mapping as one key."""
        return f"{' + '.join(self.input)} -> {self.output}"

    @property
    def dimensionless(self) -> bool:
        """Whether the kind or the role promises a dimensionless output."""
        return self.kind in DIMENSIONLESS_KINDS or self.role in DIMENSIONLESS_ROLES

    def as_dict(self) -> dict:
        return {"input": list(self.input), "output": self.output, "kind": self.kind,
                "locality": self.locality, "role": self.role, "mapping": self.mapping}


def parse_reduction(text: Optional[str]) -> Tuple[Optional[Reduction], Tuple[str, ...]]:
    """The ``Reduction`` section's value and every problem with it.

    ``(None, ())`` when there is no section. A section with an unknown key, a
    missing key, a repeated key or a value outside the vocabulary yields
    ``(None, errors)``: a half-valid reduction is never returned.
    """
    if text is None or not text.strip():
        return None, ()
    errors = []
    values = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or not value:
            errors.append(f"Reduction line {line!r} is not 'key: value'")
            continue
        if key not in FIELDS:
            errors.append(f"Reduction key {key!r} is not one of {', '.join(FIELDS)}")
            continue
        if key in values:
            errors.append(f"Reduction key {key!r} is given twice")
            continue
        values[key] = value
    for key in FIELDS:
        if key not in values:
            errors.append(f"Reduction is missing {key!r}")
    inputs: Tuple[str, ...] = ()
    if "input" in values:
        inputs = tuple(part.strip() for part in values["input"].split(",") if part.strip())
        for item in inputs:
            if item not in SPATIAL_REPRESENTATIONS:
                errors.append(f"Reduction input {item!r} is not one of {', '.join(SPATIAL_REPRESENTATIONS)}")
        if not inputs:
            errors.append("Reduction input is empty")
    for key, allowed in _VOCABULARY.items():
        if key in values and values[key] not in allowed:
            errors.append(f"Reduction {key} {values[key]!r} is not one of {', '.join(allowed)}")
    if errors:
        return None, tuple(errors)
    return Reduction(inputs, values["output"], values["kind"], values["locality"], values["role"]), ()


# ---------------------------------------------------------------------------
# reduction graphs: which quantity is reduced to which, and by what
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """A node of a reduction graph: its symbol, representation and whether it is dimensionless."""

    symbol: str
    representation: str
    dimensionless: bool = False


@dataclass(frozen=True)
class Relation:
    """An edge: ``sources`` reduced to ``target``.

    ``formula`` names the ``vaft.formula`` function (``"category.name"``) that
    performs it; its ``Reduction`` section then supplies the kind, so the graph
    cannot disagree with the catalog. A step VAFT does elsewhere (a process,
    a flux-surface average) has ``formula=None`` and states its ``kind``.
    """

    sources: Tuple[str, ...]
    target: str
    formula: Optional[str] = None
    kind: Optional[str] = None
    note: str = ""


#: quantity key -> Quantity, shared by every family
QUANTITIES = {
    "j_phi": Quantity("$j_\\phi(\\rho)$", "profile_1d"),
    "I_p": Quantity("$I_p$", "scalar_0d"),
    "B_theta": Quantity("$B_\\theta(r)$", "profile_1d"),
    "B_p_field": Quantity("$B_p(R, Z)$", "field_2d"),
    "l_i": Quantity("$l_i$", "scalar_0d", True),
    "q": Quantity("$q(\\rho)$", "profile_1d", True),
    "s_hat": Quantity("$\\hat s(\\rho)$", "profile_1d", True),
    "q_features": Quantity("$q_0$, $q_{min}$, $q_{95}$, $r_s$", "scalar_0d", True),
    "r_mix": Quantity("$r_{mix}$", "scalar_0d"),
    "p_profile": Quantity("$p(\\rho)$", "profile_1d"),
    "nT_field": Quantity("$n(R, Z)$, $T(R, Z)$", "field_2d"),
    "p_avg": Quantity("$\\langle p\\rangle$", "scalar_0d"),
    "p_integral": Quantity("$\\int p\\,dV$", "scalar_0d"),
    "W": Quantity("$W_K = \\tfrac32\\int p\\,dV$", "scalar_0d"),
    "beta_t": Quantity("$\\beta_t$", "scalar_0d", True),
    "beta_p": Quantity("$\\beta_p$", "scalar_0d", True),
    "beta_N": Quantity("$\\beta_N$", "scalar_0d"),
    "alpha": Quantity("$\\alpha(\\rho)$", "profile_1d", True),
    "nT_profile": Quantity("$n_s(\\rho)$, $T_s(\\rho)$", "profile_1d"),
    "central_avg": Quantity("$y(0)$, $\\langle y\\rangle$", "scalar_0d"),
    "peaking": Quantity("peaking $y(0)/\\langle y\\rangle$", "scalar_0d", True),
    "a_over_L": Quantity("$a/L_n$, $a/L_T$", "profile_1d", True),
    "nu_star_local": Quantity("$\\nu^*_s(\\rho)$", "profile_1d", True),
    "n_avg": Quantity("$\\bar n_e$, $n_G$", "scalar_0d"),
    "f_G": Quantity("$f_G$", "scalar_0d", True),
    "engineering": Quantity("$I_p$, $B_t$, $n$, $T$, $P$, $R$, $\\epsilon$, $\\kappa$, $M$", "scalar_0d"),
    "rho_star": Quantity("$\\rho_*$", "scalar_0d", True),
    "nu_star": Quantity("$\\nu_*$", "scalar_0d", True),
    "omega_tau": Quantity("$\\Omega_i\\tau_E$", "scalar_0d", True),
    "tau_E": Quantity("$\\tau_E$", "scalar_0d"),
    "eng_exponents": Quantity("engineering exponents", "scalar_0d", True),
    "dimless_exponents": Quantity("dimensionless exponents", "scalar_0d", True),
}

#: family -> its relations, in reading order
REDUCTION_FAMILIES = {
    "current_q": (
        Relation(("j_phi",), "I_p", kind="integral", note="area integral; equilibrium process"),
        Relation(("j_phi",), "B_theta", "geometry.cylindrical_poloidal_field", note="Ampere, enclosed current"),
        Relation(("B_p_field",), "l_i", "virial.virial_li_from_volume"),
        Relation(("j_phi",), "q", "geometry.peaked_current_safety_factor"),
        Relation(("q",), "s_hat", "equilibrium.shear_from_r_q"),
        Relation(("q",), "q_features", kind="feature_extraction", note="axis, minimum, $\\psi_N = 0.95$, $q = m/n$"),
        Relation(("q",), "r_mix", "stability.kadomtsev_mixing_radius"),
    ),
    "pressure_energy": (
        Relation(("p_profile",), "p_integral", kind="integral", note="volume integral; equilibrium process"),
        Relation(("p_profile",), "p_avg", kind="weighted_average", note="volume average; equilibrium process"),
        Relation(("p_profile",), "alpha", "stability.ballooning_alpha_from_p_B_R"),
        Relation(("p_integral",), "beta_p", "equilibrium.beta_poloidal_from_pressure_integral"),
        Relation(("p_avg",), "beta_t", "equilibrium.beta_toroidal_from_p_B0"),
        Relation(("beta_t",), "beta_N", "equilibrium.beta_normal_from_beta_tor"),
        Relation(("beta_p",), "W", "equilibrium.kinetic_energy_from_beta_p_B_pa_V_p"),
    ),
    "kinetic_profiles": (
        Relation(("nT_profile",), "a_over_L", "utils.normalized_gradient_scale_length"),
        Relation(("nT_profile",), "central_avg", kind="weighted_average", note="axis value and volume average"),
        Relation(("central_avg",), "peaking", "equilibrium.peaking_factor"),
        Relation(("nT_profile",), "nu_star_local", "neoclassical.electron_collisionality_sauter"),
        Relation(("nT_profile",), "n_avg", kind="weighted_average", note="line average; $n_G$ from $I_p$"),
        Relation(("n_avg",), "f_G", "stability.greenwald_fraction"),
    ),
    "dimensionless_similarity": (
        Relation(("engineering",), "rho_star", "equilibrium.rho_star_from_M_T_B_R_epsilon"),
        Relation(("engineering",), "nu_star", "equilibrium.nu_star_from_n_T_B_R_epsilon_kappa_I"),
        Relation(("engineering",), "tau_E", "equilibrium.confinement_time_from_engineering_parameters"),
        Relation(("tau_E",), "omega_tau", "equilibrium.omega_i_tau_E_from_B_tau_E_M"),
        Relation(("eng_exponents",), "dimless_exponents",
                 "equilibrium.dimensionless_scaling_coeffs_from_engineering_scaling_coeffs"),
    ),
}
