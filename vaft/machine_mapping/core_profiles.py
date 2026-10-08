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
shaped it was known or guessed.  The Ti/Te ratio is ``assumed`` (Ti = Te,
#1331) until enough shots carry both diagnostics to infer it; the earlier
inference stays in ``vest.yaml`` as its record.  The impurity composition is
the ``impurity_model`` preset (C6+:O8+ = 1:1 at a target Z_eff = 2, #1565),
``assumed``; the carbon and oxygen fractions line radiation uses are derived
from it rather than configured beside it, so the two cannot disagree.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping, Optional

from .utils import VestConfigurationError, resolve_vest_diagnostic

__all__ = [
    "CoreProfilesPolicy",
    "INFERRED_TI_ORIGIN",
    "POLICY_STATUSES",
    "TI_RECORD_KINDS",
    "classify_ti_record",
    "inferred_ti_text",
    "policy_for_ods",
    "ti_record_fields",
    "vest_core_profiles_policy",
    "vest_impurity_model",
]

#: How a configured value relates to the truth it stands in for.
POLICY_STATUSES = ("assumed", "measured", "inferred")

#: What an ion ``temperature_fit.parameters`` record says about the temperature
#: beside it, as :func:`classify_ti_record` reads it: a measured fit, a Ti/Te
#: ratio the writer assumed, an inference (lane K's #1426 pressure partition),
#: or a record in no grammar this module knows (a reader refuses it by name).
TI_RECORD_KINDS = ("measured", "assumed", "inferred", "unknown")

#: The ``origin`` field value that labels an inferred ion temperature.
INFERRED_TI_ORIGIN = "inferred"

_RECORD_FIELD = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([^;]*)")

#: The mixture reductions an ``impurity_model`` may name (#1565): one
#: pseudo-impurity keeping the charge, Z^2 and mass moments.
IMPURITY_REDUCTIONS = ("preserve_charge_z2_mass",)

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
    each is; ``impurity_model`` is the parsed composition preset the
    fractions were derived from (``None`` for an era that still configures
    the fractions directly); ``provenance`` is the ``vest.yaml`` revision
    record and the derivation notes, verbatim.
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
    impurity_model: Optional[Mapping[str, Any]] = None

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
        text = f"impurity_fractions={','.join(parts)}; source={self.source}"
        if self.impurity_model is not None:
            text += f"; derived_from=impurity_model(target_zeff={self.impurity_model['target_zeff']:g})"
        return text


def inferred_ti_text(method: str = "equilibrium_pressure_partition") -> str:
    """The record a writer stores beside an ion temperature it inferred, not measured.

    One spelling for producer and consumer: ``origin=inferred; method=<method>``
    is what :func:`classify_ti_record` reads back as ``"inferred"``, so a
    transport reader never takes the temperature as a measurement (#1426).
    """
    return f"origin={INFERRED_TI_ORIGIN}; method={method}"


def ti_record_fields(record: Any) -> dict[str, str]:
    """The ``key=value`` (or ``key: value``) fields of a provenance record, lower-cased keys.

    Fields are ``;``-separated; a bare word such as the policy's ``base`` carries
    no key and is skipped.  ``None`` or blank gives an empty mapping.
    """
    fields: dict[str, str] = {}
    if record is None:
        return fields
    for part in str(record).replace("\n", ";").split(";"):
        match = _RECORD_FIELD.fullmatch(part)
        if match:
            fields[match.group(1).lower()] = match.group(2).strip()
    return fields


def classify_ti_record(record: Any) -> str:
    """Which of :data:`TI_RECORD_KINDS` an ion ``temperature_fit.parameters`` record is.

    Decided positively from the grammars the repository writes, never from a
    substring: a reader that matched one literal spelling took every other
    assumed or inferred record for a measurement (cold review 0.8.0).

    * no record, or a fit record (``coordinate=<c>; method=<m>[; order=n]
      [; measured_span=a:b]``, :meth:`vaft.process.profile.FittedProfile.parameters_text`)
      -> ``"measured"``;
    * ``origin=<x>`` in any spelling -> ``"inferred"`` when ``x`` names an
      inference, ``"measured"`` when ``x`` is ``measured``, else ``"unknown"``;
    * a ratio record (``ti_te_ratio=<r>; ...`` from :meth:`CoreProfilesPolicy.ti_te_ratio_text`,
      the caller-argument record or the legacy Ti = Te fallback), or a
      ``status`` of ``assumed``/``unspecified`` -> ``"assumed"``;
    * anything else, including free text with no ``key=value`` field -> ``"unknown"``.
    """
    if record is None or not str(record).strip():
        return "measured"
    fields = ti_record_fields(record)
    origin = fields.get("origin")
    if origin is not None:
        origin = origin.lower()
        if origin == "measured":
            return "measured"
        return "inferred" if INFERRED_TI_ORIGIN in origin else "unknown"
    if "ti_te_ratio" in fields or fields.get("status", "").lower() in ("assumed", "unspecified"):
        return "assumed"
    return "measured" if "coordinate" in fields else "unknown"


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

    model_block = config.get("impurity_model")
    impurities = config.get("impurities")
    model = None
    if model_block is not None:
        if impurities is not None:
            raise VestConfigurationError(
                f"{context}: configure impurity_model or impurities, not both -- "
                "the line-radiation fractions are derived from impurity_model (#1565)"
            )
        model = _impurity_model(model_block, f"{context} impurity_model")
        species, fractions, statuses = _fractions_from_model(model)
    else:
        species, fractions, statuses = _configured_fractions(impurities, context)

    provenance = {
        "revision": revision,
        "ti_te_ratio": {
            key: value for key, value in ratio_block.items() if key not in ("value", "sigma", "status")
        },
    }
    if model is not None:
        provenance["impurities"] = {"derived_from": "impurity_model", **model["provenance"]}
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
        impurity_model=model,
    )


def _configured_fractions(impurities: Any, context: str):
    """Fractions configured directly (an era without an ``impurity_model``)."""
    if not isinstance(impurities, Mapping):
        raise VestConfigurationError(f"{context}: impurity_model (or impurities) must be a mapping")
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
    return species, fractions, statuses


def _impurity_model(block: Any, context: str) -> Mapping[str, Any]:
    """Parse and validate an ``impurity_model`` block into a read-only record.

    Element, charge state, mass and relative density stay separate: the
    charge state is the ion's, never the element's atomic number, and must
    not exceed it.  Relative densities are normalised to sum to one; the
    configured numbers are kept beside the normalised weights.
    """
    from vaft.data.synthetic_kinetic_profiles import ION_SPECIES

    if not isinstance(block, Mapping):
        raise VestConfigurationError(f"{context}: must be a mapping")
    status = _status(block, context)
    entries = block.get("species")
    if not isinstance(entries, (list, tuple)) or not entries:
        raise VestConfigurationError(f"{context}: species must be a non-empty list")
    species = []
    for index, entry in enumerate(entries):
        where = f"{context} species[{index}]"
        if not isinstance(entry, Mapping):
            raise VestConfigurationError(f"{where}: must be a mapping")
        element = str(entry.get("element"))
        if element not in ION_SPECIES:
            raise VestConfigurationError(f"{where}: unknown element {element!r}")
        z_n, standard_mass = ION_SPECIES[element]
        charge = _finite_non_negative(entry.get("charge_state"), f"{where}.charge_state")
        if not 0.0 < charge <= z_n:
            raise VestConfigurationError(
                f"{where}: charge_state must lie in (0, Z_n = {z_n}] for {element}, got {charge:g}"
            )
        mass = _finite_non_negative(entry.get("mass", standard_mass), f"{where}.mass")
        if mass <= 0.0:
            raise VestConfigurationError(f"{where}.mass: must be positive")
        relative = _finite_non_negative(entry.get("relative_density"), f"{where}.relative_density")
        species.append({"element": element, "z_n": z_n, "charge_state": charge,
                        "mass": mass, "relative_density": relative})
    if len({item["element"] for item in species}) != len(species):
        raise VestConfigurationError(f"{context}: an element appears twice")
    total = sum(item["relative_density"] for item in species)
    if total <= 0.0:
        raise VestConfigurationError(f"{context}: relative densities sum to zero")
    target = _finite_non_negative(block.get("target_zeff"), f"{context}.target_zeff")
    reduction = block.get("reduction", IMPURITY_REDUCTIONS[0])
    if reduction not in IMPURITY_REDUCTIONS:
        raise VestConfigurationError(
            f"{context}.reduction: must be one of {IMPURITY_REDUCTIONS}, got {reduction!r}"
        )
    main_ion = str(block.get("main_ion", "H"))
    if main_ion not in ION_SPECIES:
        raise VestConfigurationError(f"{context}.main_ion: unknown element {main_ion!r}")
    for item in species:
        item["weight"] = item["relative_density"] / total
    notes = {
        key: value for key, value in block.items()
        if key not in ("status", "species", "target_zeff", "reduction", "main_ion")
    }
    return MappingProxyType({
        "status": status,
        "species": tuple(MappingProxyType(item) for item in species),
        "target_zeff": target,
        "reduction": str(reduction),
        "main_ion": main_ion,
        "main_ion_charge": float(ION_SPECIES[main_ion][0]),
        "provenance": MappingProxyType({"source": f"{_SOURCE}.impurity_model", **notes}),
    })


def _fractions_from_model(model: Mapping[str, Any]):
    """``n_s / n_e`` of each species at the model's target Z_eff (quasi-neutral closure)."""
    from vaft.formula.impurity import solve_impurity_mixture_for_target_zeff

    try:
        solution = solve_impurity_mixture_for_target_zeff(
            model["target_zeff"],
            [item["weight"] for item in model["species"]],
            [item["charge_state"] for item in model["species"]],
            main_ion_charge=model["main_ion_charge"],
        )
    except ValueError as exc:
        raise VestConfigurationError(f"VEST diagnostic {_DIAGNOSTIC!r} impurity_model: {exc}") from exc
    species = tuple(item["element"] for item in model["species"])
    fractions = {symbol: float(value) for symbol, value in zip(species, solution.impurity_fractions)}
    statuses = {symbol: model["status"] for symbol in species}
    return species, fractions, statuses


def vest_impurity_model(shot: Optional[int], *, info_file: str | None = None) -> Optional[Mapping[str, Any]]:
    """The VEST impurity-composition preset for *shot*, or ``None`` for an era without one.

    An explicit modelling preset (#1565), never a global VAFT default: the
    process-layer resolver (:func:`vaft.process.impurity.resolve_impurity_composition`)
    takes it only when asked for the ``"vest"`` preset and nothing measured,
    explicit or derived outranks it.  The record keeps element, charge state,
    mass and the configured and normalised relative densities apart, with the
    target Z_eff, the reduction rule, the main ion and the ``vest.yaml``
    notes verbatim.
    """
    return vest_core_profiles_policy(shot, info_file=info_file).impurity_model



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
