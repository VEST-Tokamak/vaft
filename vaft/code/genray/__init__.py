"""GENRAY electron-cyclotron ray tracing: ``ec_launchers + equilibrium + core_profiles -> waves`` (#264).

GENRAY is an optional external executable (``$GENRAYHOME/bin/xgenray``, built
by ``install/install_genray.sh``). Importing this package never needs it.
"""

from .config import GENRAY_HOME_ENV, GENRAY_HOME_EXECUTABLE, GENRAYConfig, GENRAYInputs, GENRAYResult
from .inputs import eccone_angles, genray_namelists, prepare_genray_inputs, render_namelists
from .outputs import EC_WAVE_TYPE, collect_genray_outputs, genray_to_waves, read_genray_netcdf
from .runner import find_genray_executable, run, run_genray

__all__ = [
    "EC_WAVE_TYPE",
    "GENRAYConfig",
    "GENRAYInputs",
    "GENRAYResult",
    "GENRAY_HOME_ENV",
    "GENRAY_HOME_EXECUTABLE",
    "collect_genray_outputs",
    "eccone_angles",
    "find_genray_executable",
    "genray_namelists",
    "genray_to_waves",
    "prepare_genray_inputs",
    "read_genray_netcdf",
    "render_namelists",
    "run",
    "run_genray",
]
