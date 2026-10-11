"""Cross-backend consistency of plasma-formalism provenance (issue #1738, Phase D of #1723).

Phase C made every backend report *which* formulation a run used
(:class:`vaft.code.formalism.PlasmaFormalism`). This module checks that the
shared fields mean the same thing everywhere, so that a query or a database
grouping across solvers returns the scientifically intended set. It does not
compare solver results, rank models, or choose a solver.

Three layers of validation, kept apart (#1738 §3):

``schema``
    recognized vocabulary values, types, required fields, the schema version
    -- what any record must satisfy to be read at all;
``semantic``
    combinations that are physically contradictory whatever the solver, such as
    a hybrid without a coupling or a gyrokinetic run that follows full orbits;
``backend``
    a recorded formalism against the one the backend's own resolver derives
    from the prepared input, or against the adapter mode it claims.

The schema and semantic rules are those the record enforces on construction
(``vaft.code.formalism``), collected rather than raised, so an imported or
legacy record can be reported in full instead of failing on its first problem.
A combination that is merely *unsupported* by today's backends -- full-f or
global gyrokinetics, gyrokinetic energetic particles in an MHD code -- is valid
here (#1738 §5); :data:`FUTURE_FORMULATIONS` pins that.

Grouping (#1738 §25-26): :data:`COMPARISON_KEYS` are the fields two runs must
share before their results may be aggregated as "the same model"; a learned
surrogate is told apart from the model it emulates by its ``realization``.
``None`` (not applicable) and ``"unknown"`` (undocumented) are different keys
and never merge (#1738 §14).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "COMPARISON_KEYS",
    "ENERGETIC_POPULATIONS",
    "EVIDENCE_LEVELS",
    "FUTURE_FORMULATIONS",
    "FormalismFinding",
    "LAYERS",
    "MODE_KEYS",
    "QUERIES",
    "REFERENCE_FORMULATIONS",
    "assert_comparable",
    "backend_consistency",
    "comparison_key",
    "group_by_formalism",
    "imported_formalism_findings",
    "matches",
    "mode_consistency",
    "query",
    "realization",
    "validate_formalism",
]

#: The validation layers, in the order a record meets them.
LAYERS = ("schema", "semantic", "backend")

#: How much an imported record (a TRANSP file, a legacy product) can be trusted (#1738 §16). Local runs are
#: resolved from their own prepared input and need none of these.
EVIDENCE_LEVELS = ("verified_source", "inferred_components", "legacy_ambiguous", "unknown")

#: The fields two runs must share before their results are grouped as one model (#1738 §26), plus the
#: computational realization. Solver-specific numerical controls are deliberately absent.
COMPARISON_KEYS = (
    "scientific_operation",
    "bulk_description",
    "fluid_model",
    "kinetic_equation",
    "kinetic_coupling",
    "derived_from",
    "distribution_formulation",
    "orbit_representation",
    "spatial_domain",
    "regime",
    "field_model",
    "realization",
)

#: The fields that make a run one adapter mode rather than another (#1738 §15): what is computed, what the bulk
#: is, which equation, which coupling, which parent theory, which regime, which realization. Distribution,
#: orbit, domain and field model are configuration of a mode (an electromagnetic CGYRO run is still the
#: ``cgyro_linear`` mode), so :func:`mode_consistency` does not hold a run to the registry's default for them.
MODE_KEYS = ("scientific_operation", "bulk_description", "fluid_model", "kinetic_equation", "kinetic_coupling",
             "derived_from", "regime", "realization")

#: Population names that mean a selected energetic population rather than the thermal bulk (#1738 §8).
ENERGETIC_POPULATIONS = frozenset({"beam_ions", "fast_ions", "energetic_particles", "fusion_products", "alphas"})


@dataclass(frozen=True)
class FormalismFinding:
    """One problem of a formalism record: its layer, the field it concerns and why it is a problem."""

    layer: str
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.layer}] {self.field}: {self.message}"


def _fields() -> tuple[str, ...]:
    from dataclasses import fields

    from vaft.code.formalism import PlasmaFormalism

    return tuple(item.name for item in fields(PlasmaFormalism))


def _as_namespace(data: Any) -> Any:
    from vaft.code.formalism import PlasmaFormalism

    if isinstance(data, PlasmaFormalism):
        return data
    values = {name: None for name in _fields()}
    values.update({key: value for key, value in dict(data).items() if key in values})
    return SimpleNamespace(**values)


def validate_formalism(data: Any) -> tuple[FormalismFinding, ...]:
    """Every schema and semantic problem of a record, without raising.

    Parameters
    ----------
    data : PlasmaFormalism or mapping
        A record, or its :meth:`~vaft.code.formalism.PlasmaFormalism.as_dict`
        form (an imported or legacy product).

    Returns
    -------
    tuple of FormalismFinding
        Empty when the record is valid. Semantic rules are checked only once the
        schema is clean, because a combination of unrecognized values has no
        physical meaning to contradict.
    """
    from vaft.code.formalism import SCHEMA_VERSION, _combination_problems, _schema_problems

    findings: list[FormalismFinding] = []
    if isinstance(data, Mapping):
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            findings.append(FormalismFinding("schema", "schema_version",
                                             f"schema_version {version!r}; this VAFT reads {SCHEMA_VERSION}"))
        unknown = sorted(set(data) - set(_fields()) - {"schema_version"})
        if unknown:
            findings.append(FormalismFinding("schema", ", ".join(unknown), f"unknown formalism fields {unknown}"))
    record = _as_namespace(data)
    findings.extend(FormalismFinding("schema", name, message) for _, name, message in _schema_problems(record))
    if not findings:
        findings.extend(FormalismFinding("semantic", name, message) for name, message in _combination_problems(record))
    return tuple(findings)


def imported_formalism_findings(data: Any, evidence: str) -> tuple[FormalismFinding, ...]:
    """Schema and semantic findings of an imported record, which carries no run input to check (#1738 §16).

    ``evidence`` is one of :data:`EVIDENCE_LEVELS`. An imported record is held to
    the generic rules only: there is no prepared input for the backend layer, and
    an ``"unknown"`` field is a legitimate state, not a failure.
    """
    if evidence not in EVIDENCE_LEVELS:
        raise ValueError(f"evidence {evidence!r} is not one of {EVIDENCE_LEVELS}")
    return validate_formalism(data)


def realization(record: Any) -> str:
    """How the model is realized computationally: ``"solver"`` unless the record names a surrogate (#1738 §11)."""
    extensions = getattr(record, "extensions", None) or {}
    return str(extensions.get("realization", "solver"))


def comparison_key(record: Any) -> tuple[tuple[str, Any], ...]:
    """The grouping key of a record: :data:`COMPARISON_KEYS` with their values, ``None`` and ``"unknown"`` kept."""
    return tuple((name, realization(record) if name == "realization" else getattr(record, name))
                 for name in COMPARISON_KEYS)


def group_by_formalism(records: Iterable[Any]) -> dict[tuple, list[Any]]:
    """Records grouped by :func:`comparison_key`; two records share a group only if every key agrees."""
    groups: dict[tuple, list[Any]] = {}
    for record in records:
        groups.setdefault(comparison_key(record), []).append(record)
    return groups


def assert_comparable(records: Sequence[Any]) -> None:
    """Raise when ``records`` are not one model, naming the fields that differ (#1738 §25)."""
    keys = [dict(comparison_key(record)) for record in records]
    if not keys:
        return
    differing = sorted(name for name in COMPARISON_KEYS if len({repr(k[name]) for k in keys}) > 1)
    if differing:
        detail = "; ".join(f"{name}: {sorted({repr(k[name]) for k in keys})}" for name in differing)
        raise ValueError(f"these runs are not one formulation and must not be aggregated as one: {detail}")


def matches(record: Any, **criteria: Any) -> bool:
    """Whether ``record`` has every given field value; ``None`` matches only not-applicable, never unknown.

    A criterion value may be a tuple, meaning any of its members. ``realization``
    is accepted as a field. ``kinetic_population`` matches if the record's
    populations intersect the given ones.
    """
    for name, wanted in criteria.items():
        value = realization(record) if name == "realization" else getattr(record, name)
        if name == "kinetic_population":
            wanted_set = set(wanted if isinstance(wanted, (tuple, list, set, frozenset)) else (wanted,))
            if not value or not wanted_set.intersection(value):
                return False
            continue
        options = wanted if isinstance(wanted, tuple) else (wanted,)
        if value not in options:
            return False
    return True


#: The cross-backend questions of #1738 §20, as field criteria. A first-principles gyrokinetic calculation
#: solves the gyrokinetic equation itself; a reduced model derived from gyrokinetics does not.
QUERIES: Mapping[str, Mapping[str, Any]] = {
    "drift_kinetic": {"kinetic_equation": "drift_kinetic"},
    "first_principles_gyrokinetic": {"kinetic_equation": "gyrokinetic"},
    "hybrid_kinetic_mhd": {"bulk_description": "hybrid"},
    "guiding_center_orbits": {"bulk_description": "particle", "orbit_representation": "guiding_center"},
    "reduced_from_gyrokinetics": {"bulk_description": "reduced", "derived_from": "gyrokinetic"},
    "energetic_particles_kinetic": {"kinetic_population": tuple(sorted(ENERGETIC_POPULATIONS))},
}


def query(records: Iterable[Any], name: str) -> list[Any]:
    """The records answering one of the named cross-backend questions in :data:`QUERIES`."""
    if name not in QUERIES:
        raise KeyError(f"no formalism query {name!r}; known: {', '.join(QUERIES)}")
    return [record for record in records if matches(record, **QUERIES[name])]


def backend_consistency(recorded: Any, authoritative: Any, *, ignore: Iterable[str] = ("extensions",)
                        ) -> tuple[FormalismFinding, ...]:
    """Fields where a recorded formalism contradicts the one derived from the prepared input (#1738 §15).

    ``authoritative`` is what the backend's own resolver gives for the input the
    run was prepared with (``vaft.code.gpec._formalism.resolve_plasma_formalism``,
    ``cgyro.plasma_formalism`` ...). ``None`` there means the input could not
    settle the formalism: a recorded value is then reported, never accepted.
    """
    if authoritative is None:
        if recorded is None:
            return ()
        return (FormalismFinding("backend", "*", "the prepared input does not settle a formalism, "
                                 "but the run records one"),)
    if recorded is None:
        return (FormalismFinding("backend", "*", "the run records no formalism, but its prepared input settles one"),)
    findings = []
    for name in _fields():
        if name in set(ignore):
            continue
        mine, theirs = getattr(recorded, name), getattr(authoritative, name)
        if mine != theirs:
            findings.append(FormalismFinding("backend", name, f"recorded {mine!r}, but the prepared input gives "
                                             f"{theirs!r}"))
    return tuple(findings)


def mode_consistency(recorded: Any, adapter_mode: Any) -> tuple[FormalismFinding, ...]:
    """:data:`MODE_KEYS` fields where a run's record contradicts the adapter mode it claims to be.

    A field the mode leaves :data:`~vaft.code.formalism.UNKNOWN` (TGLF-NN's field model,
    which only the trained model knows) constrains nothing.
    """
    from vaft.code.formalism import UNKNOWN

    expected = getattr(adapter_mode, "formalism", None)
    if expected is None:
        return ()
    mine, theirs = dict(comparison_key(recorded)), dict(comparison_key(expected))
    return tuple(FormalismFinding("backend", name, f"recorded {mine[name]!r}, but mode "
                                  f"{adapter_mode.adapter}:{adapter_mode.mode} is {theirs[name]!r}")
                 for name in MODE_KEYS if theirs[name] != UNKNOWN and mine[name] != theirs[name])


def _record(**fields: Any):
    from vaft.code.formalism import PlasmaFormalism

    return PlasmaFormalism(**fields)


def _reference() -> dict[str, Any]:
    return {
        "ideal_mhd_linear_stability": _record(
            scientific_operation="ideal_stability", bulk_description="fluid", fluid_model="ideal_mhd",
            spatial_domain="whole_volume", regime="static", field_model="mhd_displacement"),
        "resistive_mhd_tearing": _record(
            scientific_operation="resistive_stability", bulk_description="fluid", fluid_model="resistive_mhd",
            spatial_domain="whole_volume", regime="linear", field_model="mhd_displacement"),
        "hybrid_mhd_drift_kinetic_stability": _record(
            scientific_operation="ideal_stability", bulk_description="hybrid", fluid_model="ideal_mhd",
            kinetic_equation="drift_kinetic", kinetic_coupling="energy", kinetic_population=("thermal_ions",),
            distribution_formulation="delta_f", orbit_representation="bounce_averaged",
            spatial_domain="whole_volume", regime="static", field_model="mhd_displacement"),
        "drift_kinetic_neoclassical": _record(
            scientific_operation="neoclassical_transport", bulk_description="kinetic",
            kinetic_equation="drift_kinetic", kinetic_population=("all",), distribution_formulation="delta_f",
            spatial_domain="local", regime="steady_state"),
        "local_deltaf_continuum_gyrokinetic_linear": _record(
            scientific_operation="microstability", bulk_description="kinetic", kinetic_equation="gyrokinetic",
            kinetic_population=("all",), distribution_formulation="delta_f", orbit_representation="gyrocenter",
            spatial_domain="local", regime="linear", field_model="electrostatic",
            extensions={"numerical_representation": "continuum"}),
        "gyrokinetic_derived_quasilinear_transport": _record(
            scientific_operation="turbulent_transport", bulk_description="reduced", derived_from="gyrokinetic",
            kinetic_population=("all",), spatial_domain="local", regime="quasilinear"),
        "fast_ion_particle_transport": _record(
            scientific_operation="fast_ion_sources", bulk_description="particle", kinetic_equation="fokker_planck",
            kinetic_coupling="sources", kinetic_population=("beam_ions",), orbit_representation="guiding_center",
            spatial_domain="whole_volume", regime="time_dependent",
            extensions={"numerical_representation": "monte_carlo"}),
        "guiding_center_test_particle": _record(
            scientific_operation="orbit_following", bulk_description="particle",
            orbit_representation="guiding_center", kinetic_population=("alphas",), spatial_domain="whole_volume"),
        "full_orbit_test_particle": _record(
            scientific_operation="orbit_following", bulk_description="particle", kinetic_equation="fokker_planck",
            orbit_representation="full_orbit", kinetic_population=("fast_ions",), spatial_domain="whole_volume",
            extensions={"numerical_representation": "monte_carlo"}),
    }


def _future() -> dict[str, Any]:
    return {
        "mars_k_kinetic_mhd": _record(
            scientific_operation="ideal_stability", bulk_description="hybrid", fluid_model="ideal_mhd",
            kinetic_equation="drift_kinetic", kinetic_coupling="pressure",
            kinetic_population=("thermal_ions", "energetic_particles"), orbit_representation="bounce_averaged",
            spatial_domain="whole_volume", regime="linear", field_model="mhd_displacement"),
        "mhd_with_gyrokinetic_energetic_particles": _record(
            scientific_operation="ideal_stability", bulk_description="hybrid", fluid_model="resistive_mhd",
            kinetic_equation="gyrokinetic", kinetic_coupling="pressure", kinetic_population=("energetic_particles",),
            orbit_representation="gyrocenter", spatial_domain="whole_volume", regime="nonlinear",
            extensions={"numerical_representation": "pic"}),
        "global_deltaf_gyrokinetic_pic": _record(
            scientific_operation="turbulent_transport", bulk_description="kinetic", kinetic_equation="gyrokinetic",
            kinetic_population=("all",), distribution_formulation="delta_f", orbit_representation="gyrocenter",
            spatial_domain="radially_global", regime="nonlinear", field_model="electromagnetic",
            extensions={"numerical_representation": "pic"}),
        "full_f_gyrokinetic_across_separatrix": _record(
            scientific_operation="turbulent_transport", bulk_description="kinetic", kinetic_equation="gyrokinetic",
            kinetic_population=("all",), distribution_formulation="full_f", orbit_representation="gyrocenter",
            spatial_domain="whole_volume", topology_domain="across_separatrix", regime="nonlinear",
            field_model="electromagnetic", extensions={"numerical_representation": "continuum"}),
        "ascot5_full_orbit": _record(
            scientific_operation="orbit_following", bulk_description="particle", kinetic_equation="fokker_planck",
            kinetic_population=("fast_ions",), orbit_representation="full_orbit", spatial_domain="whole_volume",
            regime="time_dependent", extensions={"numerical_representation": "monte_carlo"}),
        "ascot5_guiding_center": _record(
            scientific_operation="orbit_following", bulk_description="particle", kinetic_equation="fokker_planck",
            kinetic_population=("fast_ions",), orbit_representation="guiding_center", spatial_domain="whole_volume",
            regime="time_dependent", extensions={"numerical_representation": "monte_carlo"}),
        "simple_collisionless_orbits": _record(
            scientific_operation="orbit_following", bulk_description="particle", kinetic_population=("alphas",),
            orbit_representation="guiding_center", spatial_domain="whole_volume", regime="time_dependent"),
    }


class _Lazy(Mapping):
    """A mapping built on first access, so importing this module constructs no record."""

    def __init__(self, build):
        self._build, self._data = build, None

    def _get(self) -> dict:
        if self._data is None:
            self._data = self._build()
        return self._data

    def __getitem__(self, key):
        return self._get()[key]

    def __iter__(self):
        return iter(self._get())

    def __len__(self):
        return len(self._get())


#: The canonical cross-backend cases of #1738 §18: taxonomy fixtures, not solver results.
REFERENCE_FORMULATIONS: Mapping[str, Any] = _Lazy(_reference)

#: Legitimate formulations no VAFT backend runs today (#1738 §23); generic validation must accept them.
FUTURE_FORMULATIONS: Mapping[str, Any] = _Lazy(_future)
