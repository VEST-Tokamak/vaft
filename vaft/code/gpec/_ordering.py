"""The advisory ordering state of a GPEC-suite case, from the equilibrium it runs on (#1987).

:attr:`GPECSuiteConfig.ordering_state` feeds the advisory ordering check each
run records (#1823).  A workflow stage holds a g-file, and sometimes a fuller
stage ODS; this turns either into that state through
:func:`vaft.process.ordering_state.ordering_inputs_from_ods`, and never
raises: the check is advisory, so a failure is recorded in the provenance and
the run goes ahead with an empty state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

__all__ = ["ordering_state_for_case"]


def ordering_state_for_case(
    geqdsk: Path | str,
    *,
    time_s: float | None = None,
    ods: Any = None,
    z_eff: float | None = None,
    plasma_onset: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ordering quantities of one case, and how they were obtained.

    Parameters
    ----------
    geqdsk : path
        The g-file the case runs on [-].
    time_s : float, optional
        The case time; required with ``ods``, which is matched to it by time
        (never by index) [s].
    ods : ODS, optional
        A fuller stage ODS (equilibrium and ``core_profiles``); without it the
        g-file's own single slice is used, which yields the geometric
        orderings and leaves the rest unassessed [-].
    z_eff : float, optional
        The stage's policy ``Z_eff``; the builder assumes none [-].
    plasma_onset : float, optional
        Plasma onset time, for the plasma-age ordering [s].

    Returns
    -------
    tuple
        ``(state, provenance)``: the state for ``GPECSuiteConfig.ordering_state``
        (empty on failure) and a JSON-ready record of the source, the policy
        inputs and either the builder's provenance or the error [-].
    """
    record: dict[str, Any] = {
        "source": "stage ods" if ods is not None else "g-file",
        "geqdsk": str(geqdsk),
        "z_eff": z_eff,
        "z_eff_source": "stage policy" if z_eff is not None else "not given",
        "plasma_onset": plasma_onset,
    }
    try:
        from vaft.process.ordering_state import ordering_inputs_from_ods

        if ods is None:
            import numpy as np

            from vaft.data.eqdsk import read_geqdsk

            ods = read_geqdsk(geqdsk).to_omas()
            # One slice: the g-file's own state, so its own time is the match.
            time_s = float(np.atleast_1d(ods["equilibrium.time"])[0])
        elif time_s is None:
            raise ValueError("a stage ODS needs the case time to be matched by time")
        inputs = ordering_inputs_from_ods(ods, float(time_s), z_eff=z_eff, plasma_onset=plasma_onset)
    except Exception as exc:  # noqa: BLE001 - advisory: recorded, never raised
        record["error"] = f"{type(exc).__name__}: {exc}"
        return {}, record
    record["time_s"] = float(time_s)
    record["builder"] = _json_ready(inputs.provenance)
    return dict(inputs.state), record


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
