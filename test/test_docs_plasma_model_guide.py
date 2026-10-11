"""The plasma-model guide speaks the code's vocabulary (#1739 §33).

``docs/_guide/Plasma_model_guide.md`` carries two tables generated from the
formalism record and the adapter-mode registry, between ``<!-- generated:... -->``
markers, and a glossary whose record values are hand-written. The generated
tables must equal what this module renders from the code, and every record
value written by hand -- in a table column or row headed by a record field, in
the glossary, or as an ``adapter:mode`` name -- must exist in the code, so the
page cannot drift into a second vocabulary. After a vocabulary change, rewrite the generated tables with::

    PYTHONPATH=. python -P test/test_docs_plasma_model_guide.py --write
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "docs" / "_guide" / "Plasma_model_guide.md"

AXIS_ROWS = (
    ("what is computed", "scientific_operation", "SCIENTIFIC_OPERATIONS"),
    ("bulk description", "bulk_description", "BULK_DESCRIPTIONS"),
    ("fluid model", "fluid_model", "FLUID_MODELS"),
    ("kinetic equation", "kinetic_equation", "KINETIC_EQUATIONS"),
    ("kinetic coupling", "kinetic_coupling", "KINETIC_COUPLINGS"),
    ("distribution formulation", "distribution_formulation", "DISTRIBUTION_FORMULATIONS"),
    ("orbit representation", "orbit_representation", "ORBIT_REPRESENTATIONS"),
    ("spatial domain", "spatial_domain", "SPATIAL_DOMAINS"),
    ("topology domain", "topology_domain", "TOPOLOGY_DOMAINS"),
    ("regime", "regime", "REGIMES"),
    ("field model", "field_model", "FIELD_MODELS"),
)


def _code(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return ", ".join(f"`{item}`" for item in value)
    return f"`{value}`"


def render_axes() -> str:
    from vaft.code import formalism

    rows = ["| axis | field | controlled values |", "| --- | --- | --- |"]
    for label, field, constant in AXIS_ROWS:
        rows.append(f"| {label} | `{field}` | {_code(getattr(formalism, constant))} |")
    return "\n".join(rows)


def render_matrix() -> str:
    from vaft.validation.model_contracts import ADAPTER_MODES, contracts_for

    rows = ["| mode | operation | bulk | fluid model | kinetic equation | coupling | orbit | domain | regime | "
            "contracts (role) | not covered | audit |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode in ADAPTER_MODES:
        record = mode.formalism
        if record is None:
            rows.append(f"| `{mode.adapter}:{mode.mode}` | *unclassified* | — | — | — | — | — | — | — | — | "
                        f"{mode.note or '—'} | {mode.audit} |")
            continue
        uses, unmapped = contracts_for(record)
        contracts = ", ".join(f"`{use.contract}` ({use.role})" for use in uses) or "—"
        cells = (record.scientific_operation, record.bulk_description, record.fluid_model, record.kinetic_equation,
                 record.kinetic_coupling, record.orbit_representation, record.spatial_domain, record.regime)
        rows.append(f"| `{mode.adapter}:{mode.mode}` | " + " | ".join(_code(c) for c in cells)
                    + f" | {contracts} | {'; '.join(unmapped) or '—'} | {mode.audit} |")
    return "\n".join(rows)


def render_external_codes() -> str:
    from vaft.diagram import EXTERNAL_CODES

    rows = ["| code | bulk | kinetic equation / fluid model | domain | numerical realization |",
            "| --- | --- | --- | --- | --- |"]
    rows += [f"| {c.code} | {c.bulk} | {c.model} | {c.domain} | {c.realization} |" for c in EXTERNAL_CODES]
    return "\n".join(rows)


GENERATED = {"formalism-axes": render_axes, "formalism-matrix": render_matrix, "external-codes": render_external_codes}
TOKEN = r"`([A-Za-z0-9_.-]+)`"


def _vocabulary() -> dict[str, set[str]]:
    from vaft.code import formalism

    vocabulary = {field: set(values) for field, values in formalism.AXES.items()}
    vocabulary["derived_from"] = set(formalism.KINETIC_EQUATIONS) | set(formalism.FLUID_MODELS)
    return vocabulary


def _block(text: str, name: str) -> re.Match:
    match = re.search(rf"<!-- generated:{name} -->\n(.*?)\n<!-- /generated -->", text, re.S)
    assert match, f"the guide has no generated:{name} block"
    return match


def _tables(text: str):
    """Every markdown table outside the generated blocks, as lists of cell lists."""
    text = re.sub(r"<!-- generated:.*?<!-- /generated -->", "", text, flags=re.S)
    table: list[list[str]] = []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            if not all(set(cell) <= set("- ") for cell in cells):
                table.append(cells)
        elif table:
            yield table
            table = []


def test_generated_tables_match_the_code():
    text = PAGE.read_text(encoding="utf-8")
    for name, render in GENERATED.items():
        assert _block(text, name).group(1).strip() == render(), (
            f"generated:{name} is stale; run `PYTHONPATH=. python -P test/test_docs_plasma_model_guide.py --write`"
        )


def test_hand_written_record_values_exist_in_the_vocabulary():
    """A column or a row headed by a record field may only hold that field's values."""
    vocabulary = _vocabulary()
    checked = 0
    for table in _tables(PAGE.read_text(encoding="utf-8")):
        header = table[0]
        for column, title in enumerate(header):
            field = re.fullmatch(r"`([a-z_]+)`(?: of the model run)?", title)
            if field and field.group(1) in vocabulary:
                for row in table[1:]:
                    for value in re.findall(TOKEN, row[column]):
                        assert value in vocabulary[field.group(1)], f"{field.group(1)}={value!r} is not in the record"
                        checked += 1
        for row in table[1:]:
            field = re.fullmatch(r"`([a-z_]+)`", row[0])
            if field and field.group(1) in vocabulary:
                for cell in row[1:]:
                    for value in re.findall(TOKEN, cell):
                        assert value in vocabulary[field.group(1)], f"{field.group(1)}={value!r} is not in the record"
                        checked += 1
    assert checked >= 8


def test_glossary_values_exist_in_the_vocabulary():
    vocabulary = _vocabulary()
    text = PAGE.read_text(encoding="utf-8")
    glossary = re.search(r"<!-- glossary -->\n(.*?)\n<!-- /glossary -->", text, re.S)
    assert glossary, "the guide has no glossary block"
    checked = 0
    for row in glossary.group(1).splitlines()[2:]:
        record = row.rstrip(" |").split(" | ")[-1]
        for field, values in re.findall(r"`([a-z_]+)`: ((?:`[A-Za-z0-9_.-]+`(?:, )?)+)", record):
            assert field in vocabulary, f"glossary names an unknown record field {field!r}"
            for value in re.findall(TOKEN, values):
                assert value in vocabulary[field], f"glossary names {field}={value!r}, not in the record"
                checked += 1
    assert checked >= 15


def test_every_named_mode_is_a_registered_adapter_mode():
    from vaft.validation.model_contracts import ADAPTER_MODES

    known = {f"{m.adapter}:{m.mode}" for m in ADAPTER_MODES}
    named = set(re.findall(r"`([a-z0-9_]+:[a-z0-9_]+)`", PAGE.read_text(encoding="utf-8")))
    assert named and named <= known, sorted(named - known)


def _write() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for name, render in GENERATED.items():
        block = _block(text, name)
        text = text[:block.start(1)] + render() + text[block.end(1):]
    PAGE.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        _write()
    else:
        sys.exit("usage: python test/test_docs_plasma_model_guide.py --write")
