"""The row contract of a summary preset: grain, identity, time axis and column meaning.

A summary table is only as useful as the reader's knowledge of what one row is
and what each column measures. Before this contract that knowledge lived in
column-name suffixes (``_kA``, ``_J``) and in the extractors' code. Here every
preset declares it once:

* ``grain`` -- what one row stands for: a ``shot``, an equilibrium or profile
  ``slice``, a fit ``measurement``, or a ``model_output`` of a transport code.
* ``identity`` -- the columns that name the entity apart from time.
* ``time`` -- the column that places a row in time, and the IDS time base it
  is read from, or ``None`` for a per-shot row.
* one :class:`ColumnSpec` per column -- unit, one-line definition (including
  the normalising radius and field where that matters), the quantity identity
  of :mod:`vaft.formula.boundaries` when the column *is* that quantity in that
  unit, the producing formula and the IDS path.

The contract declares; it does not compute. Declaring it changes no value and
no column name. :func:`check_preset_contract` holds the declarations to their
own rules, and :func:`contract_attrs` turns them into the ``attrs`` a summary
frame carries (``units`` in the ``{column: unit}`` form that
:mod:`vaft.plot.operational_space` reads).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Literal, Mapping, Optional

Grain = Literal["shot", "slice", "state", "measurement", "model_output"]
GRAINS = ("shot", "slice", "state", "measurement", "model_output")

#: The unit symbols a column may use. A unit is a space-separated product of
#: these symbols, each optionally raised to an integer power (``m^-3``), and an
#: optional leading decimal scale (``1e19 m^-3``); ``a/b`` stands for ``a b^-1``. ``-`` is dimensionless and
#: ``""`` marks a column that is not a physical quantity (an identifier, a label,
#: a flag, a count of points, a text or a timestamp).
UNIT_SYMBOLS = frozenset({
    "s", "ms", "us",
    "m", "T", "Wb", "rad",
    "A", "kA", "MA", "V",
    "W", "kW", "MW", "J", "kJ",
    "Pa", "eV", "keV", "%",
})
_UNIT_FACTOR = re.compile(r"^(?P<symbol>[A-Za-z%]+)(\^(?P<power>-?[1-9][0-9]*))?$")
_UNIT_SCALE = re.compile(r"^1e-?[0-9]+$")


@dataclass(frozen=True)
class TimeAxis:
    """The column that places a row in time and the IDS time base behind it."""

    column: str = "time_s"
    #: The IDS time vector the column is read from, e.g. ``"equilibrium.time"``.
    reference: str = ""
    #: Largest offset tolerated when this column is matched to another time
    #: base, or ``None`` when the preset makes no such match.
    tolerance_s: Optional[float] = None


@dataclass(frozen=True)
class ColumnSpec:
    """What one summary column measures."""

    name: str
    #: Unit in :data:`UNIT_SYMBOLS` syntax; ``"-"`` dimensionless, ``""`` not a
    #: physical quantity.
    unit: str
    #: One line. Says which radius, field or surface normalises the value when
    #: the name alone does not.
    definition: str
    #: The :mod:`vaft.formula.boundaries` quantity name this column *is*, in the
    #: same unit; ``None`` when no registered quantity matches exactly.
    quantity: Optional[str] = None
    #: Dotted path of the function that computes the value, when one does.
    formula: Optional[str] = None
    #: The IDS path the value is read from or derived on.
    ids_path: Optional[str] = None


@dataclass(frozen=True)
class PresetContract:
    """The declared meaning of one preset's rows and columns."""

    name: str
    version: int
    grain: Grain
    identity: tuple[str, ...]
    time: Optional[TimeAxis]
    columns: tuple[ColumnSpec, ...]
    #: Free text for what the contract cannot yet express (a key without its
    #: time column, a join made by index); empty when there is nothing to say.
    notes: tuple[str, ...] = ()

    def column(self, name: str) -> ColumnSpec:
        for spec in self.columns:
            if spec.name == name:
                return spec
        raise KeyError(f"preset {self.name!r} declares no column {name!r}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def unit_problem(unit: str) -> Optional[str]:
    """Why ``unit`` is not valid :data:`UNIT_SYMBOLS` syntax, or ``None``."""
    if not isinstance(unit, str):
        return f"unit must be a string, not {type(unit).__name__}"
    if unit in ("", "-"):
        return None
    factors = unit.split(" ")
    if _UNIT_SCALE.match(factors[0]):
        factors = factors[1:]
        if not factors:
            return f"unit {unit!r} is a scale without a symbol"
    # ``T/MA`` is ``T MA^-1``; the boundary registry writes beta_N's unit so.
    parts = [part for factor in factors for part in factor.split("/")]
    for part in parts:
        match = _UNIT_FACTOR.match(part)
        if match is None or match.group("symbol") not in UNIT_SYMBOLS:
            return f"unit {unit!r}: {part!r} is not a known symbol (see UNIT_SYMBOLS)"
    return None


@lru_cache(maxsize=1)
def known_quantities() -> Mapping[str, frozenset]:
    """``{quantity name: units}`` of every quantity the boundary registry uses.

    One name can appear in several units (line-averaged density is given in
    ``1e19 m^-3`` by one boundary and ``1e20 m^-3`` by another), so each name
    maps to the set of units it is registered in.

    Collected from the registered boundaries (targets and inputs) and the
    operational-space projections' axes, so a column that declares a quantity
    declares one a boundary can actually be drawn against.
    """
    from vaft.formula import boundaries as fb

    found: dict[str, set] = {}

    def add(quantity) -> None:
        if quantity is not None and getattr(quantity, "name", None):
            found.setdefault(quantity.name, set()).add(quantity.unit)

    entries = [fb.get_boundary(key) for key in fb.list_boundaries()]
    for entry in entries:
        # A window holds its two boundaries; a boundary holds its quantities.
        for boundary in (getattr(entry, "lower", None), getattr(entry, "upper", None), entry):
            if boundary is None:
                continue
            add(getattr(boundary, "target", None))
            for quantity in getattr(boundary, "inputs", ()) or ():
                add(quantity)
    from vaft.diagram import _op_space

    for key in _op_space.list_projections():
        projection = _op_space.get_projection(key)
        add(projection.x)
        add(projection.y)
    return {name: frozenset(units) for name, units in found.items()}


def check_preset_contract(contract: PresetContract, columns: tuple[str, ...],
                          key_columns: tuple[str, ...] = ()) -> list[str]:
    """Every way ``contract`` breaks its own rules for a preset with ``columns``.

    An empty list means the contract is complete and consistent: one spec per
    column and no other, identity, time and key columns present, a known grain,
    valid units, and every declared quantity known to the boundary registry in
    the column's own unit.
    """
    problems: list[str] = []
    names = [spec.name for spec in contract.columns]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        problems.append(f"columns declared more than once: {repeated}")
    missing = [column for column in columns if column not in names]
    if missing:
        problems.append(f"columns without a spec: {missing}")
    extra = [name for name in names if name not in columns]
    if extra:
        problems.append(f"specs for columns the preset does not produce: {extra}")
    if contract.grain not in GRAINS:
        problems.append(f"grain {contract.grain!r} is not one of {GRAINS}")
    for column in contract.identity:
        if column not in columns:
            problems.append(f"identity column {column!r} is not a preset column")
    if contract.time is not None and contract.time.column not in columns:
        problems.append(f"time column {contract.time.column!r} is not a preset column")
    if contract.grain == "shot" and contract.time is not None:
        problems.append("a per-shot preset has no time axis")
    if contract.grain != "shot" and contract.time is None:
        problems.append(f"a {contract.grain!r} preset needs a time axis")
    for column in key_columns:
        if column not in columns:
            problems.append(f"key column {column!r} is not a preset column")
    quantities = known_quantities()
    for spec in contract.columns:
        problem = unit_problem(spec.unit)
        if problem:
            problems.append(f"{spec.name}: {problem}")
        if not spec.definition.strip():
            problems.append(f"{spec.name}: empty definition")
        if spec.quantity is not None:
            units = quantities.get(spec.quantity)
            if units is None:
                problems.append(f"{spec.name}: quantity {spec.quantity!r} is not in the boundary registry")
            elif spec.unit not in units:
                problems.append(
                    f"{spec.name}: quantity {spec.quantity!r} is in {sorted(units)}, the column is in {spec.unit!r}")
    return problems


def contract_attrs(contract: PresetContract) -> dict:
    """The ``attrs`` a summary frame of this preset carries (``skipped`` aside).

    Only the small maps a consumer reads per operation: pandas deep-copies
    ``attrs`` into every derived frame and Series, so the definitions, formulas
    and IDS paths stay in the contract (``vaft.database.summary_contract(attrs["preset"])``).
    """
    time = None
    if contract.time is not None:
        time = {"column": contract.time.column, "reference": contract.time.reference,
                "tolerance_s": contract.time.tolerance_s}
    return {
        "preset": contract.name,
        "version": contract.version,
        "grain": contract.grain,
        "identity": list(contract.identity),
        "time_axis": time,
        "units": {spec.name: spec.unit for spec in contract.columns if spec.unit},
        "quantities": {spec.name: spec.quantity for spec in contract.columns if spec.quantity},
    }


def contract_table(contract: PresetContract):
    """One row per column: name, unit, quantity, definition, formula, IDS path."""
    import pandas as pd

    return pd.DataFrame(
        [{"column": spec.name, "unit": spec.unit, "quantity": spec.quantity or "",
          "definition": spec.definition, "formula": spec.formula or "", "ids_path": spec.ids_path or ""}
         for spec in contract.columns]
    )


__all__ = [
    "ColumnSpec",
    "GRAINS",
    "PresetContract",
    "TimeAxis",
    "UNIT_SYMBOLS",
    "check_preset_contract",
    "contract_attrs",
    "contract_table",
    "known_quantities",
    "unit_problem",
]
