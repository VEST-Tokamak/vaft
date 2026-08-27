"""Standalone parser for EFIT a-files.

The a-file is EFIT's own summary of a completed reconstruction. VAFT collected
and hashed a-files but never read them, so EFIT's convergence verdict -- the
only place it is stated -- never reached the pipeline.

Scope, deliberately narrow
--------------------------
Only fields whose position in the record has been **verified against a real
VEST a-file and its partner g-file** are parsed into named attributes:

* the identification and status header, which is whitespace-delimited and
  therefore unambiguous;
* the first scalar record, ``tsaisq, rcencm, bcentr, pasmat, cpasma``, confirmed
  against ``g039915.00319`` -- ``bcentr`` matches ``BCENTR`` to nine digits,
  ``cpasma`` matches ``CURRENT`` to 8 ppm, and ``rcencm`` is ``RCENTR`` in cm.

The remainder of the scalar block does **not** match the ordering documented for
mainline EFIT in this build: the packaged file holds 78 scalars where that
ordering predicts 82, and its trailing arrays are 151 values where
``nsilop + magpri + nfcoil + nesum`` predicts 91. Reading ``terror`` or the
reconstructed-signal arrays from a guessed index would produce confidently wrong
validation numbers, which is worse than reporting none, so everything past the
first record is preserved verbatim as :attr:`AEQDSK.raw_scalars` and
:attr:`AEQDSK.raw_trailing` instead. Extending this parser needs the ``write_a``
source for the EFIT build in use.

Records are fixed-width ``(1x, 4e16.9)``: whitespace splitting is wrong, because
a negative value runs together with the value before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: Values EFIT writes when a quantity is unavailable.
SENTINELS = (1.0e11, -999.0)

#: Fortran ``e16.9`` field width, and the leading carriage-control column.
FIELD_WIDTH = 16
RECORD_SKIP = 1


class AEQDSKError(ValueError):
    """Raised when an a-file cannot be read as one."""


def _clean(value: float) -> float:
    return float("nan") if any(np.isclose(value, s) for s in SENTINELS) else value


def _record(line: str) -> list[float]:
    body = line[RECORD_SKIP:]
    values = []
    for start in range(0, len(body), FIELD_WIDTH):
        chunk = body[start : start + FIELD_WIDTH].strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:
            raise AEQDSKError(f"cannot read {chunk!r} as a Fortran real") from exc
    return values


@dataclass(frozen=True)
class AEQDSK:
    """Parsed EFIT a-file: verified named fields plus the untouched remainder."""

    shot: int
    time_ms: float
    #: 1 when EFIT considers the slice converged.
    jflag: int
    #: 0 when EFIT reports no error condition.
    lflag: int
    limloc: str
    mco2v: int
    mco2r: int
    qmflag: str
    nlold: int
    nlnew: int
    #: Total chi-square of the fit, EFIT's own value.
    tsaisq: float
    #: Reference major radius, centimetres.
    rcencm: float
    #: Vacuum toroidal field at ``rcencm``.
    bcentr: float
    #: Measured plasma current.
    pasmat: float
    #: Reconstructed plasma current.
    cpasma: float
    raw_scalars: tuple[float, ...] = ()
    raw_counts: tuple[int, ...] = ()
    raw_trailing: tuple[float, ...] = ()
    source: Path | None = None

    @property
    def converged(self) -> bool:
        """EFIT's own verdict: converged and free of a reported error."""
        return self.jflag == 1 and self.lflag == 0

    @property
    def time_seconds(self) -> float:
        return self.time_ms / 1000.0

    def to_omas(self, ods: Any = None, *, time_index: int = 0) -> Any:
        """Write the verified fields, preserving everything else as parameters."""
        from omas import ODS

        if ods is None:
            ods = ODS()
        root = f"equilibrium.code.parameters.time_slice.{time_index}.aeqdsk"
        for name in (
            "shot", "time_ms", "jflag", "lflag", "limloc", "mco2v", "mco2r",
            "qmflag", "nlold", "nlnew", "tsaisq", "rcencm", "bcentr",
            "pasmat", "cpasma",
        ):
            ods[f"{root}.{name}"] = getattr(self, name)
        ods[f"{root}.converged"] = int(self.converged)
        # Not parsed into named fields on purpose; see the module docstring.
        ods[f"{root}.raw.scalars"] = np.asarray(self.raw_scalars, dtype=float)
        ods[f"{root}.raw.counts"] = np.asarray(self.raw_counts, dtype=int)
        ods[f"{root}.raw.trailing"] = np.asarray(self.raw_trailing, dtype=float)
        ods[f"{root}.raw.layout_verified_through"] = "cpasma"
        return ods


def read_aeqdsk(path: str | Path) -> AEQDSK:
    """Read an EFIT a-file, parsing only the positions verified for this format."""
    source = Path(path).expanduser()
    lines = source.read_text(errors="replace").splitlines()
    if len(lines) < 5:
        raise AEQDSKError(f"{source} is too short to be an EFIT a-file")

    try:
        shot = int(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise AEQDSKError(f"{source} has no shot number on line 2") from exc

    header = lines[3]
    if not header.lstrip().startswith("*"):
        raise AEQDSKError(f"{source} has no '*' status header on line 4")
    tokens = header.lstrip()[1:].split()
    if len(tokens) < 9:
        raise AEQDSKError(
            f"{source} status header has {len(tokens)} fields, expected 9: {header!r}"
        )
    try:
        time_ms = float(tokens[0])
        jflag, lflag = int(tokens[1]), int(tokens[2])
        limloc = tokens[3]
        mco2v, mco2r = int(tokens[4]), int(tokens[5])
        qmflag = tokens[6]
        nlold, nlnew = int(tokens[7]), int(tokens[8])
    except ValueError as exc:
        raise AEQDSKError(f"{source} status header is malformed: {header!r}") from exc

    scalars: list[float] = []
    counts: tuple[int, ...] = ()
    trailing: list[float] = []
    seen_counts = False
    for line in lines[4:]:
        stripped = line.strip()
        if not stripped:
            continue
        # The integer counts record (nsilop, magpri, nfcoil, nesum) separates
        # the scalar block from the trailing arrays.
        if not seen_counts and stripped.replace("-", "").replace(" ", "").isdigit():
            counts = tuple(int(token) for token in stripped.split())
            seen_counts = True
            continue
        if not stripped[0].isdigit() and stripped[0] not in "+-.":
            continue  # trailing label records such as "MAG"
        values = _record(line)
        (trailing if seen_counts else scalars).extend(values)

    if len(scalars) < 5:
        raise AEQDSKError(
            f"{source} has {len(scalars)} scalars before the counts record, "
            "expected at least the first 5"
        )
    tsaisq, rcencm, bcentr, pasmat, cpasma = (_clean(v) for v in scalars[:5])

    return AEQDSK(
        shot=shot,
        time_ms=time_ms,
        jflag=jflag,
        lflag=lflag,
        limloc=limloc,
        mco2v=mco2v,
        mco2r=mco2r,
        qmflag=qmflag,
        nlold=nlold,
        nlnew=nlnew,
        tsaisq=tsaisq,
        rcencm=rcencm,
        bcentr=bcentr,
        pasmat=pasmat,
        cpasma=cpasma,
        raw_scalars=tuple(scalars),
        raw_counts=counts,
        raw_trailing=tuple(trailing),
        source=source,
    )


__all__ = ["AEQDSK", "AEQDSKError", "SENTINELS", "read_aeqdsk"]
