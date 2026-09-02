"""Canonical physical-subject vocabulary for VAFT plotting (issue #251).

Canonical plot identity is ``subject / view / [quantity]``: the ``subject`` is
the physical or diagnostic concept a user thinks of (``flux_loop``,
``plasma_current``, ``equilibrium``), while :attr:`PlotSpec.domain` keeps
recording where the data lives (the IDS root).  This module is the single
source of truth for:

* the registered subjects and their strict aliases,
* quantity-level aliases for concise tokamak terminology,
* quantity families -- named groups of distinct, related quantities.

Alias semantics are strict: an alias is registered only when both terms
confidently denote the same concept.  A family is not a synonym; it never
appears in the alias maps.

The plotting policy this vocabulary belongs to is documented in
``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FAMILIES",
    "QUANTITY_ALIASES",
    "SUBJECTS",
    "QuantityFamily",
    "Subject",
    "resolve_family",
    "resolve_quantity",
    "resolve_subject",
    "subject_names",
]

#: Informational grouping used by documentation and discovery.
KINDS = (
    "quantity",
    "diagnostic",
    "machine",
    "reconstruction",
    "model",
    "code",
    "composite",
)


@dataclass(frozen=True)
class Subject:
    """One canonical plotting subject and its strict aliases."""

    name: str
    kind: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown subject kind {self.kind!r}; expected one of {KINDS}"
            )


@dataclass(frozen=True)
class QuantityFamily:
    """A named group of distinct but closely related canonical quantities."""

    name: str
    members: tuple[str, ...]
    aliases: tuple[str, ...] = ()


_SUBJECTS = (
    # Physical quantities
    Subject("plasma_current", "quantity", ("ip", "I_p")),
    Subject("diamagnetic_flux", "quantity"),
    Subject("electron_density", "quantity", ("ne", "n_e")),
    Subject("electron_temperature", "quantity", ("te", "T_e")),
    Subject("ion_temperature", "quantity", ("ti", "T_i")),
    Subject("thermal_pressure", "quantity", ("kinetic_pressure",)),
    Subject("limiter_current", "quantity", ("limiter_shunt",)),
    # Diagnostics
    Subject("flux_loop", "diagnostic"),
    Subject("b_field_probe", "diagnostic", ("b_pol_probe", "bpol_probe")),
    Subject("mirnov", "diagnostic", ("mirnov_coil",)),
    Subject("impa", "diagnostic", ("hall_probe_array",)),
    Subject("soft_x_rays", "diagnostic", ("soft_x_ray", "sxr")),
    Subject("interferometer", "diagnostic"),
    Subject("thomson_scattering", "diagnostic", ("thomson",)),
    Subject("charge_exchange", "diagnostic"),
    Subject("spectrometer_uv", "diagnostic"),
    Subject("barometry", "diagnostic"),
    Subject("camera_visible", "diagnostic"),
    Subject("magnetics", "diagnostic"),
    # Machine description
    Subject("wall", "machine"),
    Subject("pf_coil", "machine", ("pf_active",)),
    Subject("tf_coil", "machine", ("tf",)),
    Subject("passive_structure", "machine", ("pf_passive",)),
    Subject("coil_3d", "machine", ("coils_non_axisymmetric", "3d_coil")),
    Subject("machine", "machine"),
    # Reconstructions, models, and codes
    Subject("equilibrium", "reconstruction"),
    Subject("core_profiles", "reconstruction"),
    Subject("mhd_linear", "model"),
    Subject("chease", "code"),
    # Purpose-driven composites
    Subject("current", "composite"),
    Subject("summary", "composite"),
)

#: Canonical subject name -> :class:`Subject`.
SUBJECTS: dict[str, Subject] = {subject.name: subject for subject in _SUBJECTS}

_QUANTITY_ALIAS_PAIRS = (
    ("safety_factor", "q"),
    ("q_axis", "q0"),
    ("axis_safety_factor", "q0"),
    ("safety_factor_95", "q95"),
    ("beta_normal", "beta_n"),
    ("beta_norm", "beta_n"),
    ("beta_tor", "beta_t"),
    ("beta_toroidal", "beta_t"),
    ("toroidal_beta", "beta_t"),
    ("beta_pol", "beta_p"),
    ("beta_poloidal", "beta_p"),
    ("poloidal_beta", "beta_p"),
    ("internal_inductance", "li"),
    ("mhd_energy", "w_mhd"),
    ("energy_mhd", "w_mhd"),
    ("magnetic_energy", "w_mag"),
    ("total_energy", "w_tot"),
)


def _build_quantity_aliases() -> dict[str, str]:
    """Map quantity aliases to canonical quantities, refusing collisions."""
    aliases: dict[str, str] = {}
    canonical = {target for _, target in _QUANTITY_ALIAS_PAIRS}
    for alias, target in _QUANTITY_ALIAS_PAIRS:
        if alias in canonical:
            raise ValueError(
                f"quantity alias {alias!r} is itself a canonical quantity"
            )
        owner = aliases.setdefault(alias, target)
        if owner != target:
            raise ValueError(
                f"quantity alias {alias!r} maps to both {owner!r} and {target!r}"
            )
    return aliases


#: Concise canonical quantity names with strict aliases (issue #251 section 11).
QUANTITY_ALIASES: dict[str, str] = _build_quantity_aliases()


def resolve_quantity(term: str) -> str:
    """Return the canonical quantity name for ``term`` (name or strict alias).

    A canonical quantity resolves to itself; unknown terms raise
    :class:`KeyError`.
    """
    if term in set(QUANTITY_ALIASES.values()):
        return term
    canonical = QUANTITY_ALIASES.get(term)
    if canonical is not None:
        return canonical
    raise KeyError(f"unknown quantity {term!r}")


_FAMILIES = (
    QuantityFamily("beta", ("beta_n", "beta_p", "beta_t")),
    QuantityFamily("energy", ("w_mhd", "w_mag", "w_tot"), aliases=("w",)),
)

#: Family name -> :class:`QuantityFamily`.  A family groups distinct
#: quantities; it is not an alias and never resolves to a single member.
FAMILIES: dict[str, QuantityFamily] = {family.name: family for family in _FAMILIES}


def _build_alias_map() -> dict[str, str]:
    """Map every alias to its canonical subject, refusing collisions."""
    aliases: dict[str, str] = {}
    for subject in SUBJECTS.values():
        for alias in subject.aliases:
            if alias in SUBJECTS:
                raise ValueError(
                    f"alias {alias!r} of {subject.name!r} is already a canonical subject"
                )
            owner = aliases.setdefault(alias, subject.name)
            if owner != subject.name:
                raise ValueError(
                    f"alias {alias!r} is claimed by both {owner!r} and {subject.name!r}"
                )
    return aliases


_ALIAS_TO_SUBJECT = _build_alias_map()


def _build_family_aliases() -> dict[str, str]:
    """Map every family alias to its family, refusing collisions."""
    aliases: dict[str, str] = {}
    for family in FAMILIES.values():
        for alias in family.aliases:
            if alias in FAMILIES or alias in aliases:
                raise ValueError(f"family alias {alias!r} is already registered")
            aliases[alias] = family.name
    return aliases


_FAMILY_ALIASES = _build_family_aliases()


def subject_names() -> tuple[str, ...]:
    """Return every canonical subject name, sorted."""
    return tuple(sorted(SUBJECTS))


def resolve_subject(term: str) -> Subject:
    """Return the :class:`Subject` for a canonical name or strict alias.

    Resolution is deterministic: canonical names win, then the alias map.
    Unknown terms raise :class:`KeyError` naming the available subjects.
    """
    if term in SUBJECTS:
        return SUBJECTS[term]
    canonical = _ALIAS_TO_SUBJECT.get(term)
    if canonical is not None:
        return SUBJECTS[canonical]
    raise KeyError(
        f"unknown subject {term!r}; canonical subjects are {subject_names()}"
    )


def resolve_family(term: str) -> QuantityFamily:
    """Return the :class:`QuantityFamily` for a family name or family alias."""
    if term in FAMILIES:
        return FAMILIES[term]
    canonical = _FAMILY_ALIASES.get(term)
    if canonical is not None:
        return FAMILIES[canonical]
    raise KeyError(
        f"unknown quantity family {term!r}; families are {tuple(sorted(FAMILIES))}"
    )
