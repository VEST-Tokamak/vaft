"""Deterministic comparison and reporting for OMAS reference products.

The comparator is intentionally independent of HSDS.  Both inputs are already
materialized ODS objects (or flat path mappings), which keeps legacy reference
access out of production and CI execution.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from fnmatch import fnmatchcase
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


class ParityClassification(str, Enum):
    """Scientific interpretation of one comparison outcome."""

    EXACT = "exact_parity"
    ACCEPTABLE = "acceptable_numerical_parity"
    INTENTIONAL = "intentional_improvement"
    REGRESSION = "unintended_regression"
    UNAVAILABLE = "unavailable_reference"


class DifferenceKind(str, Enum):
    """Machine-readable reason for a non-exact comparison."""

    EXACT = "exact"
    MISSING_REFERENCE = "structure.missing_reference"
    MISSING_CANDIDATE = "structure.missing_candidate"
    SHAPE = "structure.shape"
    TYPE = "structure.type"
    METADATA = "metadata"
    TIME = "time"
    NUMERICAL = "numerical"
    VALUE = "value"


@dataclass(frozen=True)
class Tolerance:
    """Resolved comparison policy for a single ODS path."""

    rtol: float = 0.0
    atol: float = 0.0
    equal_nan: bool = True
    mismatch_classification: ParityClassification = ParityClassification.REGRESSION
    note: str | None = None
    rule: str | None = None


@dataclass(frozen=True)
class ToleranceRule:
    """Glob-selected overrides applied in declaration order."""

    path: str
    rtol: float | None = None
    atol: float | None = None
    equal_nan: bool | None = None
    mismatch_classification: ParityClassification | None = None
    note: str | None = None


@dataclass(frozen=True)
class TolerancePolicy:
    """Versioned default, time, and path-specific tolerance policy."""

    schema_version: int = 1
    default: Tolerance = field(default_factory=Tolerance)
    time: Tolerance = field(default_factory=lambda: Tolerance(atol=1.0e-9))
    rules: tuple[ToleranceRule, ...] = ()

    def resolve(self, path: str, *, is_time: bool = False) -> Tolerance:
        """Resolve a path using last-matching-rule-wins semantics."""

        selected = self.time if is_time else self.default
        values = {
            "rtol": selected.rtol,
            "atol": selected.atol,
            "equal_nan": selected.equal_nan,
            "mismatch_classification": selected.mismatch_classification,
            "note": selected.note,
            "rule": selected.rule,
        }
        for rule in self.rules:
            if not fnmatchcase(path, rule.path):
                continue
            for name in (
                "rtol",
                "atol",
                "equal_nan",
                "mismatch_classification",
                "note",
            ):
                value = getattr(rule, name)
                if value is not None:
                    values[name] = value
            values["rule"] = rule.path
        return Tolerance(**values)


@dataclass(frozen=True)
class ComparisonEntry:
    """Comparison result for one ODS leaf path."""

    path: str
    classification: ParityClassification
    kind: DifferenceKind
    message: str
    reference_shape: tuple[int, ...] | None = None
    candidate_shape: tuple[int, ...] | None = None
    reference_type: str | None = None
    candidate_type: str | None = None
    rtol: float | None = None
    atol: float | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    policy_rule: str | None = None
    policy_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classification"] = self.classification.value
        payload["kind"] = self.kind.value
        return payload


@dataclass(frozen=True)
class ODSComparison:
    """Complete comparison with deterministic summaries and report renderers."""

    reference_label: str
    candidate_label: str
    entries: tuple[ComparisonEntry, ...]
    scope: str = "union"
    compared_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = 1

    @property
    def passed(self) -> bool:
        return all(
            entry.classification is not ParityClassification.REGRESSION
            for entry in self.entries
        )

    @property
    def classification(self) -> ParityClassification:
        present = {entry.classification for entry in self.entries}
        for classification in (
            ParityClassification.REGRESSION,
            ParityClassification.INTENTIONAL,
            ParityClassification.ACCEPTABLE,
            ParityClassification.UNAVAILABLE,
        ):
            if classification in present:
                return classification
        return ParityClassification.EXACT

    def summary(self) -> dict[str, Any]:
        classifications = {item.value: 0 for item in ParityClassification}
        kinds = {item.value: 0 for item in DifferenceKind}
        for entry in self.entries:
            classifications[entry.classification.value] += 1
            kinds[entry.kind.value] += 1
        return {
            "passed": self.passed,
            "classification": self.classification.value,
            "path_count": len(self.entries),
            "classifications": classifications,
            "kinds": kinds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "compared_at": self.compared_at,
            "reference": self.reference_label,
            "candidate": self.candidate_label,
            "scope": self.scope,
            "summary": self.summary(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_markdown(self) -> str:
        summary = self.summary()
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "# ODS reference comparison",
            "",
            f"- Status: **{status}**",
            f"- Classification: `{self.classification.value}`",
            f"- Reference: `{self.reference_label}`",
            f"- Candidate: `{self.candidate_label}`",
            f"- Path scope: `{self.scope}`",
            f"- Compared paths: {summary['path_count']}",
            "",
            "## Classification summary",
            "",
            "| Classification | Count |",
            "| --- | ---: |",
        ]
        lines.extend(
            f"| `{name}` | {count} |"
            for name, count in summary["classifications"].items()
        )
        non_exact = [
            entry for entry in self.entries if entry.kind is not DifferenceKind.EXACT
        ]
        lines.extend(
            [
                "",
                "## Differences",
                "",
                "| Path | Classification | Kind | Detail |",
                "| --- | --- | --- | --- |",
            ]
        )
        if not non_exact:
            lines.append("| _none_ | exact | exact | All selected paths match exactly. |")
        else:
            for entry in non_exact:
                detail = entry.message.replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{entry.path}` | `{entry.classification.value}` | "
                    f"`{entry.kind.value}` | {detail} |"
                )
        lines.append("")
        return "\n".join(lines)


def _classification(value: Any) -> ParityClassification:
    try:
        return ParityClassification(str(value))
    except ValueError as exc:
        choices = ", ".join(item.value for item in ParityClassification)
        raise ValueError(
            f"Unsupported mismatch_classification {value!r}; expected one of {choices}"
        ) from exc


def _tolerance(payload: Mapping[str, Any], *, fallback_atol: float = 0.0) -> Tolerance:
    rtol = float(payload.get("rtol", 0.0))
    atol = float(payload.get("atol", fallback_atol))
    if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
        raise ValueError("rtol and atol must be finite and non-negative")
    return Tolerance(
        rtol=rtol,
        atol=atol,
        equal_nan=bool(payload.get("equal_nan", True)),
        mismatch_classification=_classification(
            payload.get(
                "mismatch_classification", ParityClassification.REGRESSION.value
            )
        ),
        note=payload.get("note"),
    )


def load_tolerance_policy(source: str | Path | Mapping[str, Any]) -> TolerancePolicy:
    """Load and validate a versioned YAML tolerance policy."""

    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        with Path(source).expanduser().open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Tolerance policy schema_version must be 1")
    default = _tolerance(payload.get("defaults", {}).get("numeric", {}))
    time = _tolerance(payload.get("defaults", {}).get("time", {}), fallback_atol=1.0e-9)
    rules: list[ToleranceRule] = []
    for index, item in enumerate(payload.get("rules", [])):
        if not isinstance(item, Mapping) or not item.get("path"):
            raise ValueError(f"Tolerance rule {index} must define a non-empty path")
        classification = item.get("mismatch_classification")
        rtol = item.get("rtol")
        atol = item.get("atol")
        if rtol is not None and (not np.isfinite(float(rtol)) or float(rtol) < 0.0):
            raise ValueError(f"Tolerance rule {index} has invalid rtol")
        if atol is not None and (not np.isfinite(float(atol)) or float(atol) < 0.0):
            raise ValueError(f"Tolerance rule {index} has invalid atol")
        rules.append(
            ToleranceRule(
                path=str(item["path"]),
                rtol=None if rtol is None else float(rtol),
                atol=None if atol is None else float(atol),
                equal_nan=item.get("equal_nan"),
                mismatch_classification=(
                    None if classification is None else _classification(classification)
                ),
                note=item.get("note"),
            )
        )
    return TolerancePolicy(
        schema_version=1,
        default=default,
        time=time,
        rules=tuple(rules),
    )


def _flatten(value: Any) -> dict[str, Any]:
    if hasattr(value, "flat") and callable(value.flat):
        return {str(path): leaf for path, leaf in value.flat().items()}
    if not isinstance(value, Mapping):
        raise TypeError("compare_ods inputs must be OMAS ODS objects or path mappings")
    if all(isinstance(key, str) and "." in key for key in value):
        return {str(path): leaf for path, leaf in value.items()}

    result: dict[str, Any] = {}

    def visit(node: Any, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                visit(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, (list, tuple)) and node and all(
            isinstance(child, Mapping) for child in node
        ):
            for index, child in enumerate(node):
                visit(child, f"{prefix}.{index}")
        else:
            result[prefix] = node

    visit(value, "")
    return result


def _is_metadata(path: str) -> bool:
    return path.startswith("dataset_description.") or ".ids_properties." in path


def _is_time(path: str) -> bool:
    return path.split(".")[-1] == "time"


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(np.asarray(value).shape)


def _type_name(value: Any) -> str:
    array = np.asarray(value)
    return str(array.dtype) if array.shape else type(value).__name__


def _numeric(value: Any) -> bool:
    return np.issubdtype(np.asarray(value).dtype, np.number)


def _errors(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[float | None, float | None]:
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    finite = np.isfinite(ref) & np.isfinite(got)
    if not np.array_equal(ref[~finite], got[~finite], equal_nan=True):
        return None, None
    if not np.any(finite):
        return 0.0, 0.0
    absolute = np.asarray(np.abs(got[finite] - ref[finite]), dtype=float)
    denominator = np.asarray(np.abs(ref[finite]), dtype=float)
    relative = np.divide(
        absolute,
        denominator,
        out=np.full_like(absolute, np.inf),
        where=denominator != 0.0,
    )
    relative[(denominator == 0.0) & (absolute == 0.0)] = 0.0
    return float(np.max(absolute)), float(np.max(relative))


def _mismatch_classification(
    tolerance: Tolerance, default: ParityClassification = ParityClassification.REGRESSION
) -> ParityClassification:
    return tolerance.mismatch_classification or default


def _entry(
    *,
    path: str,
    classification: ParityClassification,
    kind: DifferenceKind,
    message: str,
    reference: Any = None,
    candidate: Any = None,
    tolerance: Tolerance | None = None,
    max_abs_error: float | None = None,
    max_rel_error: float | None = None,
) -> ComparisonEntry:
    return ComparisonEntry(
        path=path,
        classification=classification,
        kind=kind,
        message=message,
        reference_shape=None if reference is None else _shape(reference),
        candidate_shape=None if candidate is None else _shape(candidate),
        reference_type=None if reference is None else _type_name(reference),
        candidate_type=None if candidate is None else _type_name(candidate),
        rtol=None if tolerance is None else tolerance.rtol,
        atol=None if tolerance is None else tolerance.atol,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        policy_rule=None if tolerance is None else tolerance.rule,
        policy_note=None if tolerance is None else tolerance.note,
    )


def compare_ods(
    reference: Any,
    candidate: Any,
    *,
    policy: TolerancePolicy | str | Path | Mapping[str, Any] | None = None,
    paths: Sequence[str] | None = None,
    scope: str = "union",
    reference_label: str = "reference",
    candidate_label: str = "candidate",
) -> ODSComparison:
    """Compare two ODS products leaf-by-leaf.

    ``paths`` contains optional glob patterns. ``scope="union"`` compares all
    paths and therefore reports unexpected additions and removals.  A compact
    reference fixture should use ``scope="reference"`` so unselected candidate
    paths are outside its declared comparison surface.
    """

    resolved_policy = (
        TolerancePolicy()
        if policy is None
        else policy
        if isinstance(policy, TolerancePolicy)
        else load_tolerance_policy(policy)
    )
    reference_flat = _flatten(reference)
    candidate_flat = _flatten(candidate)
    if scope not in {"union", "reference", "intersection"}:
        raise ValueError("scope must be 'union', 'reference', or 'intersection'")
    if scope == "reference":
        selected = sorted(reference_flat)
    elif scope == "intersection":
        selected = sorted(set(reference_flat) & set(candidate_flat))
    else:
        selected = sorted(set(reference_flat) | set(candidate_flat))
    if paths:
        selected = [
            path
            for path in selected
            if any(fnmatchcase(path, pattern) for pattern in paths)
        ]
        if not selected:
            raise ValueError("No ODS paths matched the requested path patterns")

    entries: list[ComparisonEntry] = []
    for path in selected:
        tolerance = resolved_policy.resolve(path, is_time=_is_time(path))
        if path not in reference_flat:
            entries.append(
                _entry(
                    path=path,
                    classification=_mismatch_classification(tolerance),
                    kind=DifferenceKind.MISSING_REFERENCE,
                    message="Path exists only in the candidate product.",
                    candidate=candidate_flat[path],
                    tolerance=tolerance,
                )
            )
            continue
        if path not in candidate_flat:
            entries.append(
                _entry(
                    path=path,
                    classification=_mismatch_classification(tolerance),
                    kind=DifferenceKind.MISSING_CANDIDATE,
                    message="Path exists only in the reference product.",
                    reference=reference_flat[path],
                    tolerance=tolerance,
                )
            )
            continue

        expected = reference_flat[path]
        actual = candidate_flat[path]
        if _shape(expected) != _shape(actual):
            entries.append(
                _entry(
                    path=path,
                    classification=_mismatch_classification(tolerance),
                    kind=(DifferenceKind.TIME if _is_time(path) else DifferenceKind.SHAPE),
                    message=(
                        f"Time coordinate shape differs: {_shape(expected)} != {_shape(actual)}."
                        if _is_time(path)
                        else f"Shape differs: {_shape(expected)} != {_shape(actual)}."
                    ),
                    reference=expected,
                    candidate=actual,
                    tolerance=tolerance,
                )
            )
            continue

        if _numeric(expected) != _numeric(actual):
            entries.append(
                _entry(
                    path=path,
                    classification=_mismatch_classification(tolerance),
                    kind=DifferenceKind.TYPE,
                    message=(
                        f"Numeric/non-numeric type differs: {_type_name(expected)} != "
                        f"{_type_name(actual)}."
                    ),
                    reference=expected,
                    candidate=actual,
                    tolerance=tolerance,
                )
            )
            continue

        if _numeric(expected):
            expected_array = np.asarray(expected)
            actual_array = np.asarray(actual)
            exact = np.array_equal(expected_array, actual_array, equal_nan=True)
            if exact:
                entries.append(
                    _entry(
                        path=path,
                        classification=ParityClassification.EXACT,
                        kind=DifferenceKind.EXACT,
                        message="Values match exactly.",
                        reference=expected,
                        candidate=actual,
                        tolerance=tolerance,
                        max_abs_error=0.0,
                        max_rel_error=0.0,
                    )
                )
                continue
            close = bool(
                np.allclose(
                    expected_array,
                    actual_array,
                    rtol=tolerance.rtol,
                    atol=tolerance.atol,
                    equal_nan=tolerance.equal_nan,
                )
            )
            max_abs, max_rel = _errors(expected_array, actual_array)
            kind = (
                DifferenceKind.METADATA
                if _is_metadata(path)
                else DifferenceKind.TIME
                if _is_time(path)
                else DifferenceKind.NUMERICAL
            )
            entries.append(
                _entry(
                    path=path,
                    classification=(
                        ParityClassification.ACCEPTABLE
                        if close
                        else _mismatch_classification(tolerance)
                    ),
                    kind=kind,
                    message=(
                        "Values differ within tolerance."
                        if close
                        else "Values differ beyond tolerance."
                    ),
                    reference=expected,
                    candidate=actual,
                    tolerance=tolerance,
                    max_abs_error=max_abs,
                    max_rel_error=max_rel,
                )
            )
            continue

        expected_array = np.asarray(expected)
        actual_array = np.asarray(actual)
        if np.array_equal(expected_array, actual_array):
            entries.append(
                _entry(
                    path=path,
                    classification=ParityClassification.EXACT,
                    kind=DifferenceKind.EXACT,
                    message="Values match exactly.",
                    reference=expected,
                    candidate=actual,
                    tolerance=tolerance,
                )
            )
        else:
            entries.append(
                _entry(
                    path=path,
                    classification=_mismatch_classification(tolerance),
                    kind=(
                        DifferenceKind.METADATA
                        if _is_metadata(path)
                        else DifferenceKind.VALUE
                    ),
                    message="Non-numeric values differ.",
                    reference=expected,
                    candidate=actual,
                    tolerance=tolerance,
                )
            )

    return ODSComparison(
        reference_label=reference_label,
        candidate_label=candidate_label,
        entries=tuple(entries),
        scope=scope,
    )


def write_comparison_reports(
    comparison: ODSComparison,
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Write deterministic machine-readable and human-readable reports."""

    written_json = None if json_path is None else Path(json_path).expanduser()
    written_markdown = (
        None if markdown_path is None else Path(markdown_path).expanduser()
    )
    if written_json is not None:
        written_json.parent.mkdir(parents=True, exist_ok=True)
        written_json.write_text(
            json.dumps(
                comparison.to_dict(), indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    if written_markdown is not None:
        written_markdown.parent.mkdir(parents=True, exist_ok=True)
        written_markdown.write_text(comparison.to_markdown(), encoding="utf-8")
    return written_json, written_markdown


def _load_local_ods(path: str | Path):
    from .._local_io import load_ods

    ods, _ = load_ods(path)
    return ods


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerances", type=Path)
    parser.add_argument("--path", action="append", dest="paths")
    parser.add_argument(
        "--scope",
        choices=("union", "reference", "intersection"),
        default="union",
    )
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    comparison = compare_ods(
        _load_local_ods(args.reference),
        _load_local_ods(args.candidate),
        policy=args.tolerances,
        paths=args.paths,
        scope=args.scope,
        reference_label=str(args.reference),
        candidate_label=str(args.candidate),
    )
    write_comparison_reports(
        comparison,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    print(comparison.to_markdown())
    return 0 if comparison.passed else 1


__all__ = [
    "ComparisonEntry",
    "DifferenceKind",
    "ODSComparison",
    "ParityClassification",
    "Tolerance",
    "TolerancePolicy",
    "ToleranceRule",
    "compare_ods",
    "load_tolerance_policy",
    "write_comparison_reports",
]


if __name__ == "__main__":
    raise SystemExit(main())
