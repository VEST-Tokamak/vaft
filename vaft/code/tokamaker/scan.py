"""Parameter scans over TokaMaker forward solves.

Varies one ``TokaMakerConfig`` field (``ip`` by default) across a list of
values, running a fresh solve in a per-value sub-directory and collecting each
``TokaMakerResult`` — the same shape as ``scan_tes``.

Same-process scans are supported: the runner reuses the single per-kernel
``OFT_env`` and releases the solver with ``reset()`` after every solve. Pin
``base_config.mesh_file`` so all points share one cached mesh build; without
it each per-point workdir resolves its own hash-named mesh file and re-meshes.
Scanning a geometry or resolution field (``limiter``, ``dx_*``) changes the
geometry hash and correctly triggers a rebuild per point.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .config import TokaMakerConfig, TokaMakerResult
from .inputs import prepare_tokamaker_inputs
from .runner import run_tokamaker


def _default_subdir(param: str, value: Any) -> str:
    if param == "ip":
        # %g keeps sub-kA precision so nearby scan points (e.g. 40.0 and
        # 40.4 kA) never share a directory and overwrite each other's outputs
        return f"ip_{float(value) / 1000.0:g}kA"
    sval = f"{value:g}" if isinstance(value, float) else str(value)
    return f"{param}_{sval}"


def scan_tokamaker(
    ods: Any,
    base_config: TokaMakerConfig,
    values: Sequence[Any],
    param: str = "ip",
    subdir: Optional[Callable[[str, Any], str]] = None,
    on_result: Optional[Callable[[Any, TokaMakerResult], None]] = None,
) -> list[tuple[Any, TokaMakerResult]]:
    """Run a one-parameter TokaMaker scan.

    Parameters
    ----------
    ods : ODS
        Source geometry/coils/targets. Reused for every point.
    base_config : TokaMakerConfig
        Template configuration; ``param`` and ``workdir`` are overridden per
        point. Set ``mesh_file`` here to share one mesh across all points.
    values : sequence
        Values of ``param`` to scan.
    param : str
        Name of the ``TokaMakerConfig`` field to vary (default ``"ip"`` [A]).
    subdir : callable, optional
        ``(param, value) -> str`` naming the per-point sub-directory under
        ``base_config.workdir``.
    on_result : callable, optional
        ``(value, result)`` callback invoked after each solve (e.g. for logging).

    Returns
    -------
    list of (value, TokaMakerResult)

    Raises
    ------
    ValueError
        If ``param`` is not a field of ``TokaMakerConfig``.
    """
    valid_fields = {entry.name for entry in fields(TokaMakerConfig)}
    if param not in valid_fields:
        raise ValueError(
            f"Unknown TokaMakerConfig field {param!r}. "
            f"Valid fields: {', '.join(sorted(valid_fields))}"
        )

    subdir = subdir or _default_subdir
    root = Path(base_config.workdir).expanduser()
    out: list[tuple[Any, TokaMakerResult]] = []

    for value in values:
        workdir = root / subdir(param, value)
        config = replace(base_config, workdir=workdir, **{param: value})
        inputs = prepare_tokamaker_inputs(ods, config)
        result = run_tokamaker(inputs, config)
        out.append((value, result))
        if on_result is not None:
            on_result(value, result)

    return out
