"""What EFIT is told the machine's PF circuits are (issue #708).

VEST energises ten PF circuits.  EFIT's Green table describes **current
groups** over them: the solenoid split into axial segments, and the shaping
coils split at the midplane.  Splitting a circuit is EFIT's bookkeeping, not a
different machine -- every group of a circuit carries that circuit's current,
which is what :attr:`EFITCoilsetPolicy.ties` says in the k-file's ``&INWANT``
matrix.

That relationship used to be three literals in
:mod:`vaft.machine_mapping.efund_geometry`: a sixteen-name tuple, a tuple of
solenoid edges, and an ``if coil_name == "PF1" ... elif coil_name in (...)``
that returned ``None`` -- silently -- for every circuit EFIT had no group for.
It is now :data:`vest.yaml`'s ``efit_coilset`` block, resolved here and handed
to the EFUND projection and the k-file writer by the code that calls them.

The group names and their order are derived from the splits, not written down:
a circuit split ``axial`` into *n* segments becomes ``PF1-1 .. PF1-n`` ordered
upper half first by ascending |z| then the lower half the same way, and a
circuit split ``midplane`` becomes ``<name>U``, ``<name>L``.  That ordering is
the table's ``fcid`` order and the k-file's ``BRSP`` order at once, so the two
cannot disagree about which slot is which coil.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .utils import VestConfigurationError, _policy_document, _resolve_info_file_path

__all__ = [
    "EFITCoilsetPolicy",
    "vest_efit_coilset_policy",
]

#: Recognised split rules. ``axial`` needs ``edges``; ``midplane`` is the
#: two-segment case and needs nothing.
_SPLITS = ("axial", "midplane")

_STATUSES = ("assumed", "measured", "inferred")

_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class EFITCoilsetPolicy:
    """The resolved coilset: the groups, their circuits, and the series ties."""

    group_names: tuple[str, ...]
    source_circuit: Mapping[str, str]
    segment_edges: Mapping[str, tuple[float, ...]]
    ties: tuple[tuple[str, ...], ...]
    status: Mapping[str, str]
    provenance: Mapping[str, str]
    source: str = "vest.yaml:efit_coilset"

    @property
    def nfsum(self) -> int:
        """The number of current groups, which is the table's ``nfsum``."""
        return len(self.group_names)

    @property
    def circuits(self) -> tuple[str, ...]:
        """The circuits that carry a group, in the order their groups appear."""
        seen: list[str] = []
        for name in self.group_names:
            circuit = self.source_circuit[name]
            if circuit not in seen:
                seen.append(circuit)
        return tuple(seen)

    def constraint_matrix(self) -> tuple[tuple[float, ...], ...]:
        """The k-file's ``&INWANT`` coil-constraint matrix, derived.

        One row per group, one column per equality. With all-zero targets each
        column ``c`` asserts ``sum_r C[r][c] * I_r = 0``, so an equality is a
        ``+1`` and a ``-1``.

        Two kinds, and both come from the machine rather than from a hand-made
        table. Splitting a circuit into groups does not split its current, so
        each circuit contributes one column per extra group tying it back to
        its first. Then each configured series tie contributes one more.

        This was sixteen rows by twelve columns written out by index, with the
        last two lines -- the PF9/PF10 tie -- carrying no explanation at all.
        """
        index = {name: position for position, name in enumerate(self.group_names)}
        columns: list[list[float]] = []

        for circuit in self.circuits:
            members = [name for name in self.group_names if self.source_circuit[name] == circuit]
            first = members[0]
            for other in members[1:]:
                column = [0.0] * len(self.group_names)
                column[index[first]] = 1.0
                column[index[other]] = -1.0
                columns.append(column)

        for tied in self.ties:
            head = tied[0]
            for other in tied[1:]:
                column = [0.0] * len(self.group_names)
                column[index[head]] = 1.0
                column[index[other]] = -1.0
                columns.append(column)

        return tuple(
            tuple(column[row] for column in columns) for row in range(len(self.group_names))
        )

    def constraint_targets(self) -> tuple[float, ...]:
        """One zero per column: every constraint here is an equality."""
        return (0.0,) * len(self.constraint_matrix()[0])

    def group_for_element(self, coil_name: str, z: float) -> str | None:
        """The group an element belongs to, from its own position.

        ``None`` when the circuit has no group -- it is absent from the table
        exactly as it is absent from the k-file. That is a real answer, not a
        failure, so the caller decides whether to skip it or refuse.
        """
        edges = self.segment_edges.get(coil_name)
        if edges is not None:
            return self.group_names[self._segment_index(coil_name, edges, float(z))]
        if f"{coil_name}U" in self.source_circuit:
            if float(z) == 0.0:
                raise ValueError(
                    f"{coil_name} has an element on the midplane and is split there; "
                    "it cannot be assigned to the upper or the lower group"
                )
            return f"{coil_name}{'U' if float(z) > 0.0 else 'L'}"
        return None

    def _segment_index(self, coil_name: str, edges: Sequence[float], z: float) -> int:
        magnitude = abs(z)
        if magnitude > edges[-1] + _TOLERANCE:
            raise ValueError(
                f"{coil_name} element at |z| = {magnitude} lies outside its segment span "
                f"(0 to {edges[-1]})"
            )
        half = len(edges) - 1
        offset = self.group_names.index(f"{coil_name}-1")
        for index in range(half):
            if magnitude < edges[index + 1] or index == half - 1:
                return offset + (index if z >= 0.0 else index + half)
        raise AssertionError("unreachable")


def _require(block: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise VestConfigurationError(f"{where} has no '{key}'")
    return block[key]


def _status(block: Mapping[str, Any], where: str) -> str:
    value = str(_require(block, "status", where))
    if value not in _STATUSES:
        raise VestConfigurationError(
            f"{where} status must be one of {', '.join(_STATUSES)}, got '{value}'"
        )
    return value


def _build(document: Mapping[str, Any]) -> EFITCoilsetPolicy:
    block = document.get("efit_coilset")
    if not isinstance(block, Mapping):
        raise VestConfigurationError("vest.yaml has no 'efit_coilset' block")
    groups = _require(block, "groups", "efit_coilset")
    if not isinstance(groups, Mapping) or not groups:
        raise VestConfigurationError("efit_coilset.groups must be a non-empty mapping")

    names: list[str] = []
    circuits: dict[str, str] = {}
    edges_by_coil: dict[str, tuple[float, ...]] = {}
    status: dict[str, str] = {}
    provenance: dict[str, str] = {}

    for coil, spec in groups.items():
        where = f"efit_coilset.groups.{coil}"
        if not isinstance(spec, Mapping):
            raise VestConfigurationError(f"{where} must be a mapping")
        split = str(_require(spec, "split", where))
        if split not in _SPLITS:
            raise VestConfigurationError(
                f"{where} split must be one of {', '.join(_SPLITS)}, got '{split}'"
            )
        status[str(coil)] = _status(spec, where)
        provenance[str(coil)] = str(_require(spec, "provenance", where))

        if split == "midplane":
            for suffix in ("U", "L"):
                names.append(f"{coil}{suffix}")
                circuits[f"{coil}{suffix}"] = str(coil)
            continue

        edges = [float(value) for value in _require(spec, "edges", where)]
        if len(edges) < 2:
            raise VestConfigurationError(f"{where} edges needs at least two boundaries")
        if edges[0] != 0.0:
            raise VestConfigurationError(f"{where} edges must start at 0, got {edges[0]}")
        if any(later <= earlier for earlier, later in zip(edges, edges[1:])):
            raise VestConfigurationError(f"{where} edges must increase: {edges}")
        half = len(edges) - 1
        # Upper half by ascending |z|, then the lower half the same way: the
        # table's fcid order and the k-file's BRSP order at once.
        for index in range(2 * half):
            names.append(f"{coil}-{index + 1}")
            circuits[f"{coil}-{index + 1}"] = str(coil)
        edges_by_coil[str(coil)] = tuple(edges)

    ties: list[tuple[str, ...]] = []
    for entry in block.get("ties", ()) or ():
        where = "efit_coilset.ties"
        if not isinstance(entry, Mapping):
            raise VestConfigurationError(f"{where} entries must be mappings")
        tied = tuple(str(name) for name in _require(entry, "groups", where))
        unknown = [name for name in tied if name not in circuits]
        if unknown:
            raise VestConfigurationError(
                f"{where} names {', '.join(unknown)}, which are not groups of any listed circuit"
            )
        if len(tied) < 2:
            raise VestConfigurationError(f"{where} entries must tie at least two groups")
        status[f"tie:{'='.join(tied)}"] = _status(entry, where)
        provenance[f"tie:{'='.join(tied)}"] = str(_require(entry, "provenance", where))
        ties.append(tied)

    return EFITCoilsetPolicy(
        group_names=tuple(names),
        source_circuit=dict(circuits),
        segment_edges=dict(edges_by_coil),
        ties=tuple(ties),
        status=dict(status),
        provenance=dict(provenance),
    )


@lru_cache(maxsize=8)
def _cached(info_file: str | None) -> EFITCoilsetPolicy:
    return _build(_policy_document(_resolve_info_file_path(info_file)))


def vest_efit_coilset_policy(*, info_file: str | None = None) -> EFITCoilsetPolicy:
    """The coilset EFIT's table and k-file describe.

    Not keyed by shot: the table is an artifact shared by many shots, and which
    circuits it describes is a property of the table, not of a discharge. Which
    circuits a *discharge* energised is a separate question and a later stage
    of #708.
    """
    return _cached(info_file)
