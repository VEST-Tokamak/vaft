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
from .compare import compare_tglf_inputs, read_input_tglf
from .coordinates import r_over_a_at, rho_tor_norm_at
from .runner import mitim_tglf_local_inputs, run_mitim_driver, run_neo_smoke

__all__ = [
    "MITIMAvailability",
    "MITIMConfig",
    "MITIMResult",
    "MITIM_PYTHON_ENV",
    "STATUSES",
    "SUPPORTED_MITIM_VERSIONS",
    "VAFT_MACHINE",
    "compare_tglf_inputs",
    "mitim_availability",
    "mitim_tglf_local_inputs",
    "r_over_a_at",
    "read_input_tglf",
    "rho_tor_norm_at",
    "mitim_user_config",
    "run_mitim_driver",
    "run_neo_smoke",
]
