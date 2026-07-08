"""Parameter scans over TES forward solves.

Generalises the legacy ``ip_scan.py`` workflow: vary one ``TESConfig`` field
(``ip0_kA`` by default) across a list of values, running a fresh TES solve in a
per-value sub-directory and collecting each ``TESResult``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .config import TESConfig, TESResult
from .inputs import prepare_tes_inputs
from .runner import run_tes


def _default_subdir(param: str, value: Any) -> str:
    if param == "ip0_kA":
        return f"ip_{int(round(float(value))):03d}kA"
    sval = f"{value:g}" if isinstance(value, float) else str(value)
    return f"{param}_{sval}"


def scan_tes(
    ods: Any,
    base_config: TESConfig,
    values: Sequence[Any],
    param: str = "ip0_kA",
    subdir: Optional[Callable[[str, Any], str]] = None,
    on_result: Optional[Callable[[Any, TESResult], None]] = None,
) -> list[tuple[Any, TESResult]]:
    """Run a one-parameter TES scan.

    Parameters
    ----------
    ods : ODS
        Source geometry/coils/targets. Reused for every point.
    base_config : TESConfig
        Template configuration; ``param`` and ``workdir`` are overridden per point.
    values : sequence
        Values of ``param`` to scan.
    param : str
        Name of the ``TESConfig`` field to vary (default ``"ip0_kA"``).
    subdir : callable, optional
        ``(param, value) -> str`` naming the per-point sub-directory under
        ``base_config.workdir``.
    on_result : callable, optional
        ``(value, result)`` callback invoked after each solve (e.g. for logging).

    Returns
    -------
    list of (value, TESResult)
    """
    subdir = subdir or _default_subdir
    root = Path(base_config.workdir).expanduser()
    out: list[tuple[Any, TESResult]] = []

    for value in values:
        workdir = root / subdir(param, value)
        config = replace(base_config, workdir=workdir, **{param: value})
        inputs = prepare_tes_inputs(ods, config)
        result = run_tes(inputs, config)
        out.append((value, result))
        if on_result is not None:
            on_result(value, result)

    return out
