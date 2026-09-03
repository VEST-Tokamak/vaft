"""Compatibility shim: the recipes moved to :mod:`vaft.plot.backend.recipes`.

The extraction layer is backend-neutral since issue #63 and lives under
``vaft.plot.backend``; the OMAS-specific entry normalisation lives in
:mod:`vaft.omas.entries`.  Every name this module used to expose -- public
and private, since tests and ``vaft.omas.discovery`` import the helpers by
name -- resolves to the same objects, so ``RECIPES`` here *is* the backend
table.  Removed two minor releases after 0.7.0.
"""

from __future__ import annotations

from vaft.plot.backend import recipes as _recipes
from vaft.omas.entries import extract_labels_from_odc, normalize_entries

# Order is load-bearing: the backend's names are copied in below, and the two
# OMAS-specific names above must survive it -- the backend defines neither.
globals().update({
    name: value for name, value in vars(_recipes).items()
    if not (name.startswith("__") and name.endswith("__"))
})
__all__ = list(_recipes.__all__) + ["extract_labels_from_odc", "normalize_entries"]
