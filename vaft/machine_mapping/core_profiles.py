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

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional

from .utils import VestConfigurationError, resolve_vest_diagnostic

__all__ = ["CoreProfilesPolicy", "POLICY_STATUSES", "policy_for_ods", "vest_core_profiles_policy"]

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

    shot: Optional[int]
    coordinate: str
    ti_te_ratio: float
    ti_te_ratio_sigma: float
    ti_te_ratio_status: str
    impurity_species: tuple[str, ...]
    impurity_fractions: Mapping[str, float]
    impurity_status: Mapping[str, str]
    provenance: Mapping[str, Any]
    source: str = _SOURCE

    @property
    def revision_text(self) -> str:
        """``base`` or ``revision=<n>``, and whether the shot was known."""
        index = self.provenance.get("revision", {}).get("revision_index")
        era = "base" if index is None else f"revision={index}"
        return era if self.shot is not None else f"{era}; shot=unknown"

    def ti_te_ratio_text(self) -> str:
        """The record a profile writer stores beside a ratio-derived ion temperature."""
        return (
            f"ti_te_ratio={self.ti_te_ratio:g}; sigma={self.ti_te_ratio_sigma:g}; "
            f"status={self.ti_te_ratio_status}; source={self.source}; {self.revision_text}"
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
    if not (math.isfinite(number) and number >= 0.0):
        raise VestConfigurationError(f"{context}: must be finite and non-negative, got {value!r}")
    return number


@lru_cache(maxsize=64)
def vest_core_profiles_policy(shot: Optional[int], *, info_file: str | None = None) -> CoreProfilesPolicy:
    """Resolve the VEST kinetic-profile policy for *shot* from ``vest.yaml``.

    Pure configuration: no data is read.  Raises
    :class:`~vaft.machine_mapping.utils.VestConfigurationError` when the
    block is missing or malformed -- a wrong status word, an unknown
    coordinate, a negative fraction -- rather than returning a default,
    because a silently defaulted assumption is the thing this policy exists
    to prevent.  ``shot=None`` resolves the base revision and says so in the
    policy's ``revision_text``; it is for an ODS that carries no shot number,
    never a way to skip the era lookup for one that does.

    Cached: the result is a frozen record of a file that does not change
    while a pipeline runs, and a kinetic pipeline asks once per time slice.
    """
    config, revision = resolve_vest_diagnostic(
        0 if shot is None else int(shot), _DIAGNOSTIC, info_file=info_file, with_provenance=True
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
        shot=None if shot is None else int(shot),
        coordinate=str(coordinate),
        ti_te_ratio=ratio,
        ti_te_ratio_sigma=sigma,
        ti_te_ratio_status=ratio_status,
        impurity_species=species,
        impurity_fractions=fractions,
        impurity_status=statuses,
        provenance=provenance,
    )


_SHOT_PATHS = ("dataset_description.data_entry.pulse", "summary.global_quantities.pulse")


def shot_number(ods) -> Optional[int]:
    """The shot number an ODS carries, or ``None`` -- without materializing anything.

    An OMAS read of a missing leaf creates its parents, so this probes with
    ``in`` first; ``compute_power_balance`` on a shot-less ODS must not leave
    an empty ``summary`` IDS behind.
    """
    for path in _SHOT_PATHS:
        try:
            if path in ods:
                return int(ods[path])
        except Exception:  # noqa: BLE001 -- a FakeNode or a malformed leaf
            continue
    return None


def policy_for_ods(ods, shot: Optional[int] = None, *, info_file: str | None = None) -> CoreProfilesPolicy:
    """The policy for this ODS: ``shot`` if given, else the ODS's own, else the base revision.

    One rule for every pipeline.  A shot-less ODS gets the base revision with
    ``shot=unknown`` in its provenance text, which is explicit rather than
    silent; a caller that knows the shot passes it.
    """
    return vest_core_profiles_policy(shot if shot is not None else shot_number(ods), info_file=info_file)
