"""Discovering what a TRANSP run directory holds.

VAFT does not run TRANSP, so there is no configuration to write and no
executable to find; this module exists for the one half of the usual code
adapter that still applies -- saying which output files a directory contains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vaft.code.base import CodeResult

__all__ = ["OUTPUT_SUFFIX", "RUNID_PATTERN", "TRANSPResult", "collect_transp_outputs"]

#: TRANSP's output file, ``<runid>.CDF``.  The ``PH`` sibling
#: (``<runid>PH.CDF``) holds the particle histories and is a separate product.
OUTPUT_SUFFIX = ".CDF"

#: A TRANSP runid: a shot number, a run letter, and a run number -- ``45453X01``.
#: Matching this rather than every ``.CDF`` is what keeps the ``PH`` sibling
#: (``45453X01PH``) from being taken for the run itself.
RUNID_PATTERN = re.compile(r"[0-9]+[A-Za-z][0-9]+")


@dataclass
class TRANSPResult(CodeResult):
    """What a TRANSP run directory holds.

    ``returncode`` is ``None`` because nothing was run -- this is a listing of
    files that already exist.  That also makes the inherited ``ok`` ``False``
    for a perfectly good directory, as it does for the other read-only
    ``collect_*`` adapters: ask whether ``outputs["cdf"]`` is empty instead.

    ``outputs["cdf"]`` lists the runid-shaped output files in name order and
    ``outputs["other_cdf"]`` the remaining ``.CDF`` files, so a run directory
    shared with another code's products loses nothing but does not have its
    runid decided by one.
    """

    runid: str = ""


def collect_transp_outputs(directory: str | Path) -> TRANSPResult:
    """List the TRANSP output files in ``directory``.

    Best-effort about content -- it does not open the files -- but a missing
    directory is an error, matching the other ``collect_*`` adapters.
    """
    workdir = Path(directory).expanduser()
    if not workdir.is_dir():
        raise FileNotFoundError(f"TRANSP output directory does not exist: {workdir}")

    # A run directory holds the PH sibling as well, and is routinely shared
    # with other codes' products, so match TRANSP's own naming rather than
    # every file that ends in .CDF.  Ordering is by name and not by mtime: a
    # copy or a checkout normalizes mtimes, and picking the runid by the newest
    # timestamp would then hand back "45453X01PH" for a directory whose run is
    # 45453X01.
    every = sorted(path for path in workdir.glob(f"*{OUTPUT_SUFFIX}") if path.is_file())
    runs = tuple(path for path in every if RUNID_PATTERN.fullmatch(path.stem))
    others = tuple(path for path in every if path not in runs)
    return TRANSPResult(
        returncode=None,
        workdir=workdir,
        outputs={"cdf": runs, "other_cdf": others},
        runid=runs[0].stem if runs else "",
    )
