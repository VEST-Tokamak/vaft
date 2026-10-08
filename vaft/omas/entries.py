"""Turn OMAS inputs into the ``(label, ods)`` entries the recipes read.

Accepts a single ``ODS``, an ``ODC``, or a list/tuple of either; ordering is
the caller's ordering (ODC key order, list order) and labels follow
:func:`vaft.plot.backend.entries.label_entries`.  Native IMAS objects are
another namespace's business and are refused by name.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from vaft.plot.backend.entries import LABEL_OPTIONS as _LABEL_OPTIONS
from vaft.plot.backend.entries import label_entries

__all__ = ["extract_labels_from_odc", "normalize_entries"]


def extract_labels_from_odc(odc: Any, opt: str = "shot") -> list[str]:
    """Return one label per ODC entry.

    ``opt`` selects ``shot``/``pulse`` (the data-entry pulse number), ``run``, or
    ``key`` (the ODC key).  Entries missing the requested metadata fall back to
    their key, so the returned order always matches ``odc.keys()``.
    """
    if opt not in _LABEL_OPTIONS:
        opt = "key"
    labels: list[str] = []
    for key in odc.keys():
        if opt == "key":
            labels.append(str(key))
            continue
        field_name = "run" if opt == "run" else "pulse"
        try:
            data_entry = odc[key].get("dataset_description.data_entry", {})
            value = data_entry.get(field_name)
        except Exception:
            value = None
        labels.append(str(key) if value is None else str(value))
    return labels


def normalize_entries(
    source: Any, *, label: str | Sequence[str] = "shot"
) -> tuple[tuple[str, Any], ...]:
    """Return deterministic ``(label, ods)`` pairs for any supported input.

    Accepts a single ``ODS``, an ``ODC``, a list/tuple of either, or a plain
    ``{key: ods}`` mapping.  Ordering is the caller's ordering: ODC key order,
    list order, or mapping order.  ``label`` may be one of
    ``shot``/``pulse``/``run``/``key`` or an explicit sequence of labels.  A
    plain mapping's keys are the caller's own names for its entries -- the
    factors of a scan, ``{0.1: ods, 1.0: ods}`` -- so with the default
    ``label="shot"`` it is labelled by its keys; pass ``label="pulse"`` or
    ``"run"`` to label it from the data entries instead.
    """
    from omas import ODC, ODS

    # ODC subclasses ODS in OMAS, so the collection check must come first.
    if isinstance(source, ODC):
        entries = [(str(key), source[key]) for key in source.keys()]
    elif isinstance(source, ODS):
        entries = [("0", source)]
    elif isinstance(source, Mapping):
        entries = [(str(key), value) for key, value in source.items()]
        for key, value in entries:
            if not isinstance(value, ODS):
                raise TypeError(f"mapping entry {key!r} is {type(value).__name__}, not an omas ODS")
        if label == "shot":
            label = "key"
    elif isinstance(source, (list, tuple)):
        entries = []
        for position, item in enumerate(source):
            for key, ods in normalize_entries(item, label="key"):
                suffix = f"{position}" if key == "0" else f"{position}.{key}"
                entries.append((suffix, ods))
    else:
        hint = (
            "; native IMAS objects are plotted by vaft.imas.plot_*"
            if type(source).__module__.partition(".")[0] == "imas"
            else ""
        )
        raise TypeError(
            "expected an omas ODS, an ODC, or a list of them; got "
            f"{type(source).__name__}{hint}"
        )
    return label_entries(entries, label)
