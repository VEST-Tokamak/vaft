"""Discovering what a TRANSP run directory holds.

VAFT does not run TRANSP, so there is no configuration to write and no
executable to find; this module exists for the one half of the usual code
adapter that still applies -- saying which output files a directory contains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from vaft.code.base import CodeResult

__all__ = ["TRANSPResult", "collect_transp_outputs"]

#: TRANSP's output file, ``<runid>.CDF``.  The ``PH`` sibling
#: (``<runid>PH.CDF``) holds the particle histories and is a separate product.
OUTPUT_SUFFIX = ".CDF"


@dataclass
class TRANSPResult(CodeResult):
    """What a TRANSP run directory holds.

    ``returncode`` is ``None``: nothing was run.  ``outputs["cdf"]`` lists the
    output files found, most recent first.
    """

    runid: str = ""
    native: Mapping[str, str] = field(default_factory=dict)


def collect_transp_outputs(directory: str | Path) -> TRANSPResult:
    """List the TRANSP output files in ``directory``.

    Best-effort about content -- it does not open the files -- but a missing
    directory is an error, matching the other ``collect_*`` adapters.
    """
    workdir = Path(directory).expanduser()
    if not workdir.is_dir():
        raise FileNotFoundError(f"TRANSP output directory does not exist: {workdir}")

    # A run directory is routinely shared with other codes' products, so match
    # TRANSP's own naming rather than every file that ends in .CDF: a runid is
    # a shot number followed by a letter-and-digits sequence.
    candidates = sorted(
        (path for path in workdir.glob(f"*{OUTPUT_SUFFIX}") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    runid = candidates[0].stem if candidates else ""
    return TRANSPResult(
        returncode=None,
        workdir=workdir,
        outputs={"cdf": tuple(candidates)},
        runid=runid,
    )
