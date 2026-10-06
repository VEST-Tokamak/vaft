"""MITIM-fusion integration layer: installation, availability, execution, provenance (#1588).

MITIM (https://github.com/pabloprf/MITIM-fusion) is an optional external code
that runs in its own isolated interpreter (``install/install_mitim.sh``); VAFT
never imports ``mitim_tools``. Importing this package needs neither MITIM nor
GACODE. Stage A1 provides the availability probe, the per-run MITIM
configuration, the driver runner and a NEO smoke capability; PORTALS flux
matching follows in later stages of #1588.
"""

from .availability import STATUSES, MITIMAvailability, mitim_availability
from .config import (
    MITIM_PYTHON_ENV,
    SUPPORTED_MITIM_VERSIONS,
    VAFT_MACHINE,
    MITIMConfig,
    MITIMResult,
    mitim_user_config,
)
from .runner import run_mitim_driver, run_neo_smoke

__all__ = [
    "MITIMAvailability",
    "MITIMConfig",
    "MITIMResult",
    "MITIM_PYTHON_ENV",
    "STATUSES",
    "SUPPORTED_MITIM_VERSIONS",
    "VAFT_MACHINE",
    "mitim_availability",
    "mitim_user_config",
    "run_mitim_driver",
    "run_neo_smoke",
]
