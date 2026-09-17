"""Packaged sample shots, parsed once and handed out as private copies.

Not a test module (the leading underscore keeps pytest from collecting it),
and the same shape as `_plasma_timing_fixtures.py`: a module-level
``lru_cache`` holds the pristine product, and every caller receives
``.copy()``.

The gate is what motivates it (#827). `pytest -m "core and not perf"` is the
required check on every pull request into `develop` (#515), and thirteen of its
modules each parsed ``samples/39915/omas.json.gz`` for themselves -- most once
under a module fixture, one of them ten times from a helper called per test.
Counted by wrapping `vaft.omas.load` for a gate run on `develop`, it was
entered 49 times to produce the same handful of objects over and over; routing
those thirteen through here takes the gate to 26, and the modules themselves
from 27 parses to 3.

Parsing once is only safe because the copy is real. `ODS.copy()` is
``copy.deepcopy``, and omas's ``ODS.__deepcopy__`` rebuilds through
``same_init_ods()`` -- carrying ``imas_version``, ``consistency_check`` and the
COCOS settings -- then re-parents the children, so a copy reports the same
locations as the original. Measured against the load it replaces:

    load (cold)  1.78 s
    load (warm)  0.81 s
    copy         0.08 s

A copy per caller is not politeness, and handing out the cached object would be
a defect rather than an optimization: an OMAS read of an absent path
*materializes* it, so a module that only looks at the shot would still leave the
next module a different one. The tests that mutate what they are given already
deep-copy first; this keeps that assumption true for the ones that do not.
"""

from __future__ import annotations

import contextlib
import functools
import io
import warnings
from pathlib import Path

from omas import ODS

import vaft
import vaft.omas


def sample_ods(shot: int = 39915, representation: str = "omas") -> ODS:
    """A private copy of the packaged sample of ``shot``.

    The ``vaft.data.sample`` route: what a caller asks for by shot and
    representation rather than by stored filename.
    """
    source = vaft.data.sample(int(shot), representation=representation)
    if not _cacheable(source):
        return _load(source)
    return _cached(source).copy()


def packaged_ods(relative_path: str) -> ODS:
    """A private copy of the packaged artefact at ``relative_path``.

    The ``vaft.data.data_path`` route, for the callers that name the file --
    ``"samples/41524/imas.nc"`` and friends -- rather than the shot.
    """
    source = vaft.data.data_path(relative_path)
    if not _cacheable(source):
        return _load(source)
    return _cached(source).copy()


#: Suffixes whose load is parsed wholly into memory, leaving nothing open
#: behind it. Only these are cached.
_CACHEABLE_SUFFIXES = (".json", ".json.gz")


def _cacheable(source: Path) -> bool:
    """Whether this product is worth retaining for the whole session.

    Only the JSON sample is, and the reason is that only it repeats. The
    duplication this helper exists to remove is
    ``samples/39915/omas.json.gz``, parsed 27 times across the modules served
    here; the netCDF samples are read two or three times in total, so caching
    them would buy nothing measurable.

    Given nothing to gain, the conservative side is the right one: a netCDF- or
    HDF5-backed product may hold a live file handle, and an ``lru_cache`` keeps
    whatever it holds until the process ends rather than releasing it when the
    module that asked for it finishes. Retaining an open handle for a whole
    session to save no time is not a trade worth making.

    (This predicate was first written to explain a segfault in the netCDF
    backend seen while reordering these modules. It did not: an interleaved
    A/B, two reps per arm, showed patched and unpatched trees both passing that
    order, and both earlier crashes coincided with another process holding the
    machine near its memory limit. The guard is kept on the argument above, not
    on that one.)
    """
    return str(source).endswith(_CACHEABLE_SUFFIXES)


@functools.lru_cache(maxsize=None)
def _cached(source: Path) -> ODS:
    """The pristine product, keyed by the file rather than by how it was asked for.

    The two public routes reach the same bytes by different spellings --
    ``sample(39915, "omas")`` and ``data_path("samples/39915/omas.json.gz")`` are
    one file, and both ``vaft.data`` accessors return the same absolute path for
    it. Keying on the caller's phrasing instead would parse that file twice.

    Never returned to a caller -- only copied.
    """
    return _load(source)


def _load(source: Path) -> ODS:
    """One parse, cached or not."""
    # The sample carries a RuntimeWarning about an un-inferrable DD version and
    # writes to stderr while parsing. Both were already suppressed by the
    # callers that used a `_load` helper; doing it here keeps the noise from
    # landing on whichever module happens to warm the cache first.
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(source)
