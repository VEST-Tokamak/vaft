"""The one encoder/decoder for the time part of EFIT file names.

EFIT names every per-slice file ``<kind>0<shot>.<ms>`` and, when the slice
time has a non-zero microsecond part (``ITIMEU``), appends ``_<us>``::

    k039915.00306        0.306 s
    g039915.00306_320    0.30632 s
    m039915.00306_320.nc 0.30632 s

The k-file writer produces these names and EFIT reproduces them for its g-,
a- and m-files (``set_filename.F90``), so the writer and every reader must
agree on the convention. They do so by going through this module: a name is
always reduced to an integer number of microseconds, which is exact, sorts
by time, and keeps two slices of the same millisecond apart.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

__all__ = [
    "time_to_microseconds",
    "encode_time_suffix",
    "decode_time_suffix",
    "slice_file_name",
    "split_slice_file_name",
    "file_name_microseconds",
    "whole_millisecond_tstep",
]

_SUFFIX = re.compile(r"^(\d+)(?:_(\d{1,3}))?$")
_KINDS = frozenset("kgam")


def time_to_microseconds(time_seconds: float) -> int:
    """Slice time in seconds -> the integer microseconds the file name carries."""
    return int(round(float(time_seconds) * 1.0e6))


def encode_time_suffix(microseconds: int) -> str:
    """``306000 -> "00306"``, ``306320 -> "00306_320"``."""
    microseconds = int(microseconds)
    if microseconds < 0:
        raise ValueError(f"EFIT file names cannot carry a negative time: {microseconds} us")
    ms, us = divmod(microseconds, 1000)
    return f"{ms:05d}_{us:03d}" if us else f"{ms:05d}"


def decode_time_suffix(suffix: str) -> int:
    """``"00306" -> 306000``, ``"00306_320" -> 306320``; ``ValueError`` otherwise."""
    match = _SUFFIX.match(str(suffix))
    if match is None:
        raise ValueError(f"not an EFIT time suffix: {suffix!r}")
    return int(match.group(1)) * 1000 + int(match.group(2) or 0)


def slice_file_name(kind: str, shot: int | str, time_seconds: float) -> str:
    """Name of the ``kind`` (``k``/``g``/``a``/``m``) file of one slice, no extension."""
    return f"{kind}0{shot}.{encode_time_suffix(time_to_microseconds(time_seconds))}"


def split_slice_file_name(name: str | Path) -> tuple[str, int]:
    """``"g039915.00306_320" -> ("039915", 306320)``.

    The leading kind letter and a trailing ``.nc`` are optional. Raises
    ``ValueError`` when the name does not follow the convention.
    """
    text = Path(name).name
    if text[:1].lower() in _KINDS:
        text = text[1:]
    if text.lower().endswith(".nc"):
        text = text[:-3]
    shot, dot, suffix = text.rpartition(".")
    if not dot or not shot:
        raise ValueError(f"not an EFIT slice file name: {Path(name).name!r}")
    return shot, decode_time_suffix(suffix)


def file_name_microseconds(name: str | Path) -> int | None:
    """Microseconds encoded in an EFIT slice file name, ``None`` when there are none."""
    try:
        return split_slice_file_name(name)[1]
    except ValueError:
        return None


def whole_millisecond_tstep(text: str) -> float:
    """An argparse ``type`` for ``--tstep``: seconds, a whole number of milliseconds.

    The EFIT studies key their slices by whole millisecond
    (``int(round(t * 1000))`` on the request side, EFIT's printed ``t=`` --
    the floor -- on the log and file side). A step that is not a multiple of
    1 ms puts slices between the keys, where ``round`` and ``floor`` disagree
    and every other slice is paired with the wrong output, and a step below
    1 ms folds several slices into one key. Both are refused with the reason
    (cold review 0.7.0 efit-workflows F9; PR #1920 review F1).
    """
    value = float(text)
    milliseconds = value * 1000.0
    if value < 1.0e-3 or abs(milliseconds - round(milliseconds)) > 1.0e-9:
        raise argparse.ArgumentTypeError(
            f"{text} s is not a whole number of milliseconds >= 0.001: the study keys "
            "slices by whole millisecond, so --tstep must be 0.001, 0.002, ..."
        )
    return value
