"""VEST policy for building kinetic profiles: what the pipeline hands the generic routines.

:mod:`vaft.process.profile` and :mod:`vaft.process.atomic` are
machine-independent.  They fit in whatever radial coordinate they are told,
take the Ti/Te ratio they are given for a Thomson-only slice, and take the
impurity fractions they are given for line radiation.  *Which* coordinate,
*which* ratio and *which* fractions are VEST decisions, and this module is
where a VEST pipeline gets them: the ``core_profiles`` entry under
``diagnostics`` in ``vest.yaml``, resolved per shot era exactly like every
other diagnostic policy (issue #420).

Every value carries a ``status`` -- ``assumed``, ``measured`` or
``inferred`` -- so that a stored profile can say whether the number that
shaped it was known or guessed.  The Ti/Te ratio is ``inferred``: fitted on
the shots that carry both diagnostics and applied to the shots that do not.
The carbon and oxygen fractions are ``assumed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .utils import VestConfigurationError, resolve_vest_diagnostic

__all__ = ["CoreProfilesPolicy", "POLICY_STATUSES", "vest_core_profiles_policy"]

#: How a configured value relates to the truth it stands in for.
POLICY_STATUSES = ("assumed", "measured", "inferred")

#: The radial coordinates :mod:`vaft.process.profile` can fit in; mirrored
#: here so the policy is validated without importing the process layer.
_COORDINATES = ("rho_tor_norm", "rho_pol_norm", "psi_norm")

_DIAGNOSTIC = "core_profiles"
_SOURCE = f"vest.yaml:diagnostics.{_DIAGNOSTIC}"


@dataclass(frozen=True)
class CoreProfilesPolicy:
    """The resolved VEST kinetic-profile policy for one shot.

    ``ti_te_ratio`` and ``ti_te_ratio_sigma`` are the statistical Ti/Te
    coefficient and its predictive scatter; ``impurity_fractions`` maps
    species symbol to ``n_imp / n_e``; ``*_status`` say what kind of value
    each is; ``provenance`` is the ``vest.yaml`` revision record and the
    derivation notes, verbatim.
    """

    shot: int
    coordinate: str
    ti_te_ratio: float
    ti_te_ratio_sigma: float
    ti_te_ratio_status: str
    impurity_species: tuple[str, ...]
    impurity_fractions: Mapping[str, float]
    impurity_status: Mapping[str, str]
    provenance: Mapping[str, Any]
    source: str = _SOURCE

    def ti_te_ratio_text(self) -> str:
        """The record a profile writer stores beside a ratio-derived ion temperature."""
        return (
            f"ti_te_ratio={self.ti_te_ratio:g}; sigma={self.ti_te_ratio_sigma:g}; "
            f"status={self.ti_te_ratio_status}; source={self.source}"
        )

    def impurity_text(self) -> str:
        parts = [
            f"{species}={self.impurity_fractions[species]:g}({self.impurity_status[species]})"
            for species in self.impurity_species
        ]
        return f"impurity_fractions={','.join(parts)}; source={self.source}"


def _status(block: Mapping[str, Any], context: str) -> str:
    status = block.get("status")
    if status not in POLICY_STATUSES:
        raise VestConfigurationError(
            f"{context}: status must be one of {POLICY_STATUSES}, got {status!r}"
        )
    return str(status)


def _finite_non_negative(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VestConfigurationError(f"{context}: expected a number, got {value!r}") from exc
    if not number >= 0.0 or number != number or number == float("inf"):
        raise VestConfigurationError(f"{context}: must be finite and non-negative, got {value!r}")
    return number


def vest_core_profiles_policy(shot: int, *, info_file: str | None = None) -> CoreProfilesPolicy:
    """Resolve the VEST kinetic-profile policy for *shot* from ``vest.yaml``.

    Pure configuration: no data is read.  Raises
    :class:`~vaft.machine_mapping.utils.VestConfigurationError` when the
    block is missing or malformed -- a wrong status word, an unknown
    coordinate, a negative fraction -- rather than returning a default,
    because a silently defaulted assumption is the thing this policy exists
    to prevent.
    """
    config, revision = resolve_vest_diagnostic(
        int(shot), _DIAGNOSTIC, info_file=info_file, with_provenance=True
    )
    context = f"VEST diagnostic {_DIAGNOSTIC!r}"

    coordinate = config.get("coordinate")
    if coordinate not in _COORDINATES:
        raise VestConfigurationError(
            f"{context}: coordinate must be one of {_COORDINATES}, got {coordinate!r}"
        )

    ratio_block = config.get("ti_te_ratio")
    if not isinstance(ratio_block, Mapping):
        raise VestConfigurationError(f"{context}: ti_te_ratio must be a mapping")
    ratio = _finite_non_negative(ratio_block.get("value"), f"{context} ti_te_ratio.value")
    sigma = _finite_non_negative(ratio_block.get("sigma"), f"{context} ti_te_ratio.sigma")
    ratio_status = _status(ratio_block, f"{context} ti_te_ratio")

    impurities = config.get("impurities")
    if not isinstance(impurities, Mapping):
        raise VestConfigurationError(f"{context}: impurities must be a mapping")
    species = tuple(str(item) for item in impurities.get("species", ()))
    fractions_block = impurities.get("fractions")
    if not isinstance(fractions_block, Mapping):
        raise VestConfigurationError(f"{context}: impurities.fractions must be a mapping")
    fractions: dict[str, float] = {}
    statuses: dict[str, str] = {}
    for symbol in species:
        entry = fractions_block.get(symbol)
        if not isinstance(entry, Mapping):
            raise VestConfigurationError(
                f"{context}: impurities.fractions.{symbol} must be a mapping with value and status"
            )
        fractions[symbol] = _finite_non_negative(entry.get("value"), f"{context} impurities.fractions.{symbol}.value")
        statuses[symbol] = _status(entry, f"{context} impurities.fractions.{symbol}")

    provenance = {
        "revision": revision,
        "ti_te_ratio": {
            key: value for key, value in ratio_block.items() if key not in ("value", "sigma", "status")
        },
    }
    return CoreProfilesPolicy(
        shot=int(shot),
        coordinate=str(coordinate),
        ti_te_ratio=ratio,
        ti_te_ratio_sigma=sigma,
        ti_te_ratio_status=ratio_status,
        impurity_species=species,
        impurity_fractions=fractions,
        impurity_status=statuses,
        provenance=provenance,
    )
