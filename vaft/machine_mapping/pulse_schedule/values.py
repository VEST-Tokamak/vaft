"""Conservative parsing of hand-typed ShotLog cells.

Every parser here either returns a value it can defend or says it could not.
Nothing is guessed: a cell that is not a number, a range or a plain arithmetic
expression keeps its raw text, and the caller decides whether to flag it.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

#: ``a-b``, ``a~b``, ``a – b`` with an optional ``ms`` on either end.
RANGE_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(?:ms)?\s*[-~–]\s*(-?\d+(?:\.\d+)?)\s*(?:ms)?\s*$",
    re.IGNORECASE,
)
#: A single time, optionally suffixed ``ms``.
SCALAR_TIME_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:ms)?\s*$", re.IGNORECASE)
EXPRESSION_RE = re.compile(r"^[\d\s.+*/()\-]+$")
#: A gas valve setting such as ``90V(H2)``, ``90 V (He)``, ``90(H2)`` or ``90V``.
VALVE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*V?\s*(?:\(\s*([A-Za-z][A-Za-z0-9]*)\s*\))?\s*$",
    re.IGNORECASE,
)
#: One or two probe positions in metres: ``0.7``, ``0.7/0.55``, ``0.7, 0.55 m``.
POSITION_RE = re.compile(
    r"^(-?\d+(?:\.\d+)?)(?:\s*(?:/|,)\s*(-?\d+(?:\.\d+)?))?\s*m?$", re.IGNORECASE
)


def number(value: str | int | float) -> int | float:
    """Parse a number, keeping integers integral so YAML stays readable."""
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def _evaluate_expression(expression: str) -> int | float:
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("unsupported expression")

    return visit(ast.parse(expression, mode="eval"))


def parse_value(raw: Any, unit: str | None = None, trigger: bool = False) -> dict[str, Any]:
    """Return a value record: raw text plus whatever can be derived from it.

    ``derived_value`` is a number, an evaluated expression (``12*3`` is how the
    capacitor banks are logged), or ``{onset, offset}`` for a range. Units are
    never inferred: ``unit`` is whatever the header stated, or ``None``.
    """
    text = str(raw).strip()
    result: dict[str, Any] = {
        "raw": text,
        "expression": None,
        "derived_value": None,
        "unit": unit,
        "validation_flags": [],
    }
    range_match = RANGE_RE.match(text)
    if range_match:
        onset, offset = number(range_match.group(1)), number(range_match.group(2))
        result["derived_value"] = {"onset": onset, "offset": offset}
        if trigger:
            result["trigger_onset"] = onset
            result["trigger_offset"] = offset
        return result
    if EXPRESSION_RE.match(text) and any(char in text for char in "*/+"):
        result["expression"] = text
        try:
            result["derived_value"] = _evaluate_expression(text)
        except (SyntaxError, ValueError, ZeroDivisionError):
            result["validation_flags"].append("expression_not_evaluable")
        return result
    try:
        result["derived_value"] = number(text)
    except ValueError:
        result["validation_flags"].append("empty_value" if not text else "not_numeric")
    return result


def parse_time_window(value: Any) -> tuple[int | float, int | float] | None:
    """Parse a trigger time or an inclusive ``start-end`` window, in ms.

    A scalar is an instantaneous trigger, so its start and end coincide. The
    result is on the ShotLog's own clock; the DAQ offset is applied by the
    caller, once, so the stored record never mixes the two clocks.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = number(value)
        return parsed, parsed
    text = str(value).strip()
    match = RANGE_RE.fullmatch(text)
    if match:
        start, end = number(match.group(1)), number(match.group(2))
        return (start, end) if start <= end else None
    match = SCALAR_TIME_RE.fullmatch(text)
    if match:
        parsed = number(match.group(1))
        return parsed, parsed
    return None


def is_range(value: Any) -> bool:
    """True for an explicit ``start-end`` window rather than a lone number."""
    return value is not None and RANGE_RE.fullmatch(str(value).strip()) is not None


def parse_valve(value: Any) -> dict[str, Any] | None:
    """Parse a gas valve setting into ``{voltage_V, species}``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return {"voltage_V": number(value), "species": None}
    match = VALVE_RE.fullmatch(str(value).strip())
    if not match:
        return None
    return {"voltage_V": number(match.group(1)), "species": match.group(2)}


def parse_position(value: Any) -> list[int | float] | None:
    """Parse one or two probe locations expressed in metres."""
    if value is None or isinstance(value, bool):
        return None
    match = POSITION_RE.fullmatch(str(value).strip())
    if not match:
        return None
    return [number(group) for group in match.groups() if group is not None]


def normalize_status(strings: list[str]) -> dict[str, Any]:
    """Classify a shot's outcome from its remark text.

    Only failure is ever *stated* in the ShotLog; a good shot has no remark
    saying so. The absence of a failure keyword is therefore ``unknown``, not
    ``success`` -- promoting it would claim something nobody wrote down.
    """
    # Repeated Excel headers such as ``Fail`` and ``Remarks`` are not evidence
    # that every following shot failed.
    header_only = {"fail", "remark", "remarks"}
    evidence = [value for value in strings if value.strip().lower() not in header_only]
    joined = "\n".join(evidence).lower()
    if "data not saved" in joined:
        normalized, basis = "not_recorded", "remark:data_not_saved"
    elif any(term in joined for term in ("fail", "error", "안나감", "power x", "나감")):
        normalized, basis = "failure", "remark:failure_keyword"
    elif "?" in joined:
        normalized, basis = "unknown", "remark:question_mark"
    else:
        normalized, basis = "unknown", "no_explicit_outcome"
    raw = [
        value
        for value in evidence
        if any(term in value.lower() for term in ("fail", "error", "saved", "안나감", "?", "나감"))
    ]
    return {"normalized": normalized, "raw": raw, "basis": basis}
