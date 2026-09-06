"""Deprecated alias of :mod:`vaft.machine_mapping.coils_non_axisymmetric_geometry`.

The geometry/domain representation of the IMAS ``coils_non_axisymmetric``
concept moved to ``coils_non_axisymmetric_geometry`` so that the module name
carries the IDS vocabulary; the IMAS mapper is
:mod:`vaft.machine_mapping.coils_non_axisymmetric`.  Importing this module
warns and re-exports the canonical module's public names plus the two
support names callers historically imported from here
(``VestConfigurationError``, ``data_path``).  Scheduled for removal two minor
releases after the rename ships (current version 0.6.2; remove in 0.9.0).
"""

from __future__ import annotations

import warnings

from .coils_non_axisymmetric_geometry import *  # noqa: F401,F403
from .coils_non_axisymmetric_geometry import __all__ as _canonical_all
from .coils_non_axisymmetric_geometry import data_path  # noqa: F401
from .utils import VestConfigurationError  # noqa: F401

__all__ = list(_canonical_all)

warnings.warn(
    "vaft.machine_mapping.coil_geometry_3d is deprecated and will be removed in "
    "vaft 0.9.0; import vaft.machine_mapping.coils_non_axisymmetric_geometry instead",
    DeprecationWarning,
    stacklevel=2,
)
