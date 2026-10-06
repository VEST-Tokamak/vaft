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
from .compare import (
    compare_neo_fluxes,
    compare_tglf_inputs,
    effective_tglf_controls,
    neo_input_charges,
    neo_local_to_profile_normalisation,
    read_input_neo,
    read_input_tglf,
)
from .coordinates import r_over_a_at, rho_tor_norm_at
from .runner import (
    mitim_tglf_local_inputs,
    run_mitim_driver,
    run_mitim_neo,
    run_mitim_tglf,
    run_neo_smoke,
)

__all__ = [
    "neo_input_charges",
    "run_mitim_neo",
    "read_input_neo",
    "neo_local_to_profile_normalisation",
    "compare_neo_fluxes",
    "MITIMAvailability",
    "MITIMConfig",
    "MITIMResult",
    "MITIM_PYTHON_ENV",
    "STATUSES",
    "SUPPORTED_MITIM_VERSIONS",
    "VAFT_MACHINE",
    "compare_tglf_inputs",
    "effective_tglf_controls",
    "mitim_availability",
    "mitim_tglf_local_inputs",
    "r_over_a_at",
    "read_input_tglf",
    "rho_tor_norm_at",
    "mitim_user_config",
    "run_mitim_driver",
    "run_mitim_tglf",
    "run_neo_smoke",
]
