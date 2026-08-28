"""Parser for EFIT a-files, mapped from ``EFIT/efit/write_a.f90``.

The a-file is EFIT's own summary of a completed reconstruction. VAFT collected
and hashed a-files but never read them, so EFIT's convergence verdict and its
final Grad-Shafranov error -- both stated nowhere else -- never reached the
pipeline.

Layout
------
Every field name and position below is taken from the formatted branch of
``write_a.f90`` (``keqdsk >= 1``), whose records are ``(1x, 4e16.9)``. Fixed
width matters: a negative value runs together with the value before it, so
splitting on whitespace loses values.

The record sequence is a fixed 24-value head, then the interferometer arrays
sized by ``nco2v``/``nco2r`` from the status header, then 44 more values, then
the ``nsilop, magpri, nfsum, nesum`` counts record, then ``csilop``/``cmpr2`` as
one stream, ``ccbrsp``, ``eccurt``, and finally 60 trailing values. For VEST's
``nco2v=3, nco2r=2`` and counts ``11 64 16 0`` that is 78 scalars and 151
trailing values, which is exactly what the packaged ``a039915.00319`` contains.

Convergence semantics
---------------------
``terror`` is the final value of ``errorm`` from ``residu`` (``fit.f90``)::

    errorm = max over the (nw, nh) grid of |psi - psi_previous|
    errorm = errorm / |sidif| / relax

so it is the largest inter-iteration change in poloidal flux, normalized by the
flux span and by the relaxation factor -- dimensionless, and the *same*
quantity that ``cerror`` records per iteration and that the iteration exit test
compares against the ``error`` namelist value.

``jflag`` and ``lflag`` do **not** report that comparison. ``jflag`` starts at 1
and only drops to 0 when ``chkerr`` sets ``lflag > 0`` (and ``|ierchk| <= 1``);
``chkerr`` judges ``terror`` against ``errmin`` when ``iconvr == 2`` and against
``error`` otherwise. Those two thresholds differ by default (``errmin = 1e-2``,
``error = 1e-2``, both from ``set_defaults.f90``) and can differ by orders of
magnitude once a k-file sets one of them. ``jflag == 1`` therefore means "passed
EFIT's acceptance checks", not "reached the requested iteration tolerance".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

#: Values EFIT writes when a quantity is unavailable.
SENTINELS = (1.0e11, -999.0)

#: Fortran ``e16.9`` field width, and the leading carriage-control column.
FIELD_WIDTH = 16
RECORD_SKIP = 1

#: Scalars written before the interferometer arrays (``write_a.f90`` records for
#: ``chisq`` through ``vertn``).
HEAD_SCALARS = (
    "chisq", "rcencm", "bcentr", "ipmeas",
    "ipmhd", "rcntr", "zcntr", "aminor",
    "elong", "utri", "ltri", "volume",
    "rcurrt", "zcurrt", "qstar", "betat",
    "betap", "li", "gapin", "gapout",
    "gaptop", "gapbot", "q95", "vertn",
)

#: Scalars written after them, ``shear`` through ``tavem``.
TAIL_SCALARS = (
    "shear", "bpolav", "s1", "s2",
    "s3", "qout", "sepin", "sepout",
    "septop", "sibdry", "area", "wmhd",
    "terror", "elongm", "qm", "cdflux",
    "alpha", "rttt", "psiref", "indent",
    "rseps1", "zseps1", "rseps2", "zseps2",
    "sepexp", "sepbot", "btaxp", "btaxv",
    "aq1", "aq2", "aq3", "dsep",
    "rm", "zm", "psim", "taumhd",
    "betapd", "betatd", "wdia", "diamagnetic_flux_vs",
    "vloopt", "taudia", "qmerci", "tavem",
)

#: The 60 values after the reconstructed-signal arrays.  ``write_a.f90`` writes
#: five literal zeros among them; they are named so the positions stay explicit.
TRAILING_SCALARS = (
    "pbinj", "rvsin", "zvsin", "rvsout",
    "zvsout", "unused_1", "unused_2", "unused_3",
    "unused_4", "unused_5", "zuperts", "chipre",
    "cjor95", "pp95", "drsep", "yyy2",
    "xnnc", "cprof", "oring", "cjor0",
    "fexpan", "qmin", "chimse", "ssi01",
    "fexpvs", "sepnose", "ssi95", "rhoqmin",
    "cjor99", "cj1ave", "rmidin", "rmidout",
    "psurfa", "peak", "dminux", "dminlx",
    "dolubaf", "dolubafm", "diludom", "diludomm",
    "ratsol", "rvsiu", "zvsiu", "rvsid",
    "zvsid", "rvsou", "zvsou", "rvsod",
    "zvsod", "condno", "psin32", "psin21",
    "rq32in", "rq21top", "chilibt", "li3",
    "xbetapr", "tflux", "tchimls", "twagap",
)

#: Reconstructed-signal arrays, in the order the counts record sizes them.
SIGNAL_ARRAYS = ("csilop", "cmpr2", "ccbrsp", "eccurt")
COUNT_NAMES = ("nsilop", "magpri", "nfsum", "nesum")


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
    """One EFIT a-file, named per ``write_a.f90``."""

    shot: int
    time_ms: float
    #: 1 unless ``chkerr`` raised an error condition; see the module docstring --
    #: this is EFIT's acceptance verdict, not its iteration-tolerance test.
    jflag: int
    #: 0 when ``chkerr`` reported nothing; otherwise the last error code it set.
    lflag: int
    limloc: str
    nco2v: int
    nco2r: int
    qmflag: str
    nlold: int
    nlnew: int
    scalars: Mapping[str, float]
    arrays: Mapping[str, np.ndarray]
    counts: Mapping[str, int]
    fit_type: str = ""
    source: Path | None = None

    def __getitem__(self, key: str) -> Any:
        if key in self.scalars:
            return self.scalars[key]
        return self.arrays[key]

    def __contains__(self, key: object) -> bool:
        return key in self.scalars or key in self.arrays

    @property
    def chisq(self) -> float:
        """EFIT's own total chi-square for the slice."""
        return self.scalars["chisq"]

    @property
    def terror(self) -> float:
        """Final normalized Grad-Shafranov iteration error."""
        return self.scalars["terror"]

    @property
    def accepted(self) -> bool:
        """EFIT's acceptance verdict.

        ``chkerr`` found nothing to complain about, or was never called. This is
        deliberately not called ``converged``: whether the iteration reached its
        requested tolerance is a separate question, answered by comparing
        :attr:`terror` with the applicable namelist threshold.
        """
        return self.jflag == 1 and self.lflag == 0

    @property
    def time_seconds(self) -> float:
        return self.time_ms / 1000.0

    def to_omas(self, ods: Any = None, *, time_index: int = 0) -> Any:
        """Store the a-file as equilibrium code parameters for one slice."""
        from omas import ODS

        if ods is None:
            ods = ODS()
        root = f"equilibrium.code.parameters.time_slice.{time_index}.aeqdsk"
        for name, value in (
            ("shot", self.shot), ("time_ms", self.time_ms),
            ("jflag", self.jflag), ("lflag", self.lflag),
            ("limloc", self.limloc), ("qmflag", self.qmflag),
            ("nco2v", self.nco2v), ("nco2r", self.nco2r),
            ("nlold", self.nlold), ("nlnew", self.nlnew),
            ("fit_type", self.fit_type), ("accepted", int(self.accepted)),
        ):
            ods[f"{root}.{name}"] = value
        for name, count in self.counts.items():
            ods[f"{root}.counts.{name}"] = int(count)
        for name, value in self.scalars.items():
            ods[f"{root}.{name}"] = float(value)
        for name, values in self.arrays.items():
            if values.size:
                ods[f"{root}.arrays.{name}"] = values
        ods[f"{root}.mapping_source"] = "EFIT/efit/write_a.f90"
        return ods


def _parse_header(source: Path, header: str) -> tuple:
    tokens = header.lstrip()[1:].split()
    if len(tokens) < 9:
        raise AEQDSKError(
            f"{source} status header has {len(tokens)} fields, expected 9: {header!r}"
        )
    try:
        return (
            float(tokens[0]), int(tokens[1]), int(tokens[2]), tokens[3],
            int(tokens[4]), int(tokens[5]), tokens[6], int(tokens[7]), int(tokens[8]),
        )
    except ValueError as exc:
        raise AEQDSKError(f"{source} status header is malformed: {header!r}") from exc


def read_aeqdsk(path: str | Path) -> AEQDSK:
    """Read a formatted EFIT a-file into the fields ``write_a.f90`` writes."""
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
    time_ms, jflag, lflag, limloc, nco2v, nco2r, qmflag, nlold, nlnew = _parse_header(
        source, header
    )

    before: list[float] = []
    after: list[float] = []
    counts: tuple[int, ...] = ()
    fit_type = ""
    seen_counts = False
    for line in lines[4:]:
        stripped = line.strip()
        if not stripped:
            continue
        if not seen_counts and stripped.replace(" ", "").lstrip("-").isdigit():
            counts = tuple(int(token) for token in stripped.split())
            seen_counts = True
            continue
        if stripped[0] not in "+-." and not stripped[0].isdigit():
            fit_type = stripped.split()[-1]  # the trailing "header fit_type" record
            continue
        (after if seen_counts else before).extend(_record(line))

    expected_scalars = len(HEAD_SCALARS) + 2 * nco2v + 2 * nco2r + len(TAIL_SCALARS)
    if len(before) != expected_scalars:
        raise AEQDSKError(
            f"{source} holds {len(before)} scalars before the counts record; "
            f"write_a.f90 with nco2v={nco2v}, nco2r={nco2r} writes "
            f"{expected_scalars}. This build's a-file layout differs from the "
            "mapped source, so nothing is read positionally."
        )
    if len(counts) != len(COUNT_NAMES):
        raise AEQDSKError(
            f"{source} has no {'/'.join(COUNT_NAMES)} counts record"
        )

    scalars: dict[str, float] = {}
    for index, name in enumerate(HEAD_SCALARS):
        scalars[name] = _clean(before[index])
    arrays: dict[str, np.ndarray] = {}
    cursor = len(HEAD_SCALARS)
    for name, size in (
        ("rco2v", nco2v), ("dco2v", nco2v), ("rco2r", nco2r), ("dco2r", nco2r)
    ):
        arrays[name] = np.array(
            [_clean(value) for value in before[cursor : cursor + size]], dtype=float
        )
        cursor += size
    for offset, name in enumerate(TAIL_SCALARS):
        scalars[name] = _clean(before[cursor + offset])

    count_map = dict(zip(COUNT_NAMES, counts))
    sizes = (count_map["nsilop"], count_map["magpri"], count_map["nfsum"], count_map["nesum"])
    expected_trailing = sum(sizes) + len(TRAILING_SCALARS)
    if len(after) != expected_trailing:
        raise AEQDSKError(
            f"{source} holds {len(after)} values after the counts record; "
            f"write_a.f90 with counts {counts} writes {expected_trailing}"
        )
    cursor = 0
    for name, size in zip(SIGNAL_ARRAYS, sizes):
        arrays[name] = np.array(
            [_clean(value) for value in after[cursor : cursor + size]], dtype=float
        )
        cursor += size
    for offset, name in enumerate(TRAILING_SCALARS):
        scalars[name] = _clean(after[cursor + offset])

    return AEQDSK(
        shot=shot,
        time_ms=time_ms,
        jflag=jflag,
        lflag=lflag,
        limloc=limloc,
        nco2v=nco2v,
        nco2r=nco2r,
        qmflag=qmflag,
        nlold=nlold,
        nlnew=nlnew,
        scalars=MappingProxyType(scalars),
        arrays=MappingProxyType(arrays),
        counts=MappingProxyType(count_map),
        fit_type=fit_type,
        source=source,
    )


__all__ = [
    "AEQDSK",
    "AEQDSKError",
    "COUNT_NAMES",
    "HEAD_SCALARS",
    "SENTINELS",
    "SIGNAL_ARRAYS",
    "TAIL_SCALARS",
    "TRAILING_SCALARS",
    "read_aeqdsk",
]
