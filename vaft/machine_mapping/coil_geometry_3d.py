"""Deprecated alias of :mod:`vaft.machine_mapping.coils_non_axisymmetric_geometry`.

The geometry/domain representation of the IMAS ``coils_non_axisymmetric``
concept moved to ``coils_non_axisymmetric_geometry`` so that the module name
carries the IDS vocabulary; the IMAS mapper is
:mod:`vaft.machine_mapping.coils_non_axisymmetric`.  Importing this module
warns and re-exports the public names unchanged.
"""

from __future__ import annotations

import warnings

from .coils_non_axisymmetric_geometry import (  # noqa: F401
    VEST_3D_COIL_SETS,
    CoilExcitation,
    CoilFilament,
    CoilSet3D,
    CoilSetSpec,
    Vest3DCoilConfig,
    load_vest_3d_coil_config,
    parse_gpec_coil_dat,
)
from .coils_non_axisymmetric_geometry import __all__  # noqa: F401

warnings.warn(
    "vaft.machine_mapping.coil_geometry_3d is deprecated; import "
    "vaft.machine_mapping.coils_non_axisymmetric_geometry instead",
    DeprecationWarning,
    stacklevel=2,
)
