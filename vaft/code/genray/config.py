"""Configuration, input and result objects for the GENRAY EC ray-tracing adapter (#264).

GENRAY (CompX, https://github.com/compxco/genray) is an external executable.
VAFT owns the conversion from IMAS ``ec_launchers`` + ``equilibrium`` +
``core_profiles`` into a GENRAY run, the execution contract, and the mapping
of the result into IMAS ``waves``; it never builds or bundles GENRAY itself
(``install/install_genray.sh`` builds a tree the operator supplies).

Nothing physical is defaulted silently. The wave mode is required, because the
O/X choice is a statement about the launcher and the plasma, not a numerical
setting; the launched power and Zeff are taken from the ODS or must be passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from ..execution import ExecutionBackend

#: Environment variable naming the GENRAY installation root.
GENRAY_HOME_ENV = "GENRAYHOME"
#: Executable beneath ``$GENRAYHOME``; upstream's make target is ``xgenray``.
GENRAY_HOME_EXECUTABLE = Path("bin/xgenray")

#: ``ioxm`` for each accepted mode: GENRAY's +1 is O, -1 is X.
_IOXM = {"O": 1, "X": -1}


@dataclass(frozen=True)
class GENRAYConfig:
    """Runtime configuration for one GENRAY EC run.

    Parameters
    ----------
    mode : {"O", "X"}
        Launched cold-plasma mode [-]. Required: VAFT does not infer it.
    time : float
        Requested physical time [s]. Equilibrium, profiles and launcher are
        each taken at the sample nearest this time, and each must lie within
        ``time_tolerance`` of it.
    time_tolerance : float
        Largest accepted distance between ``time`` and a used sample [s].
    beam_index : int
        ``ec_launchers.beam`` entry to launch [-].
    harmonic : int
        EC harmonic handed to GENRAY's current-drive efficiency (``jwave``) [-].
        It does not affect propagation or absorption, and the driven current
        is not mapped yet, so it is recorded but has no effect on ``waves``.
    power_w : float or None
        Launched power [W]; ``None`` takes ``ec_launchers.beam.power_launched``
        at ``time`` and refuses a missing or non-positive value.
    zeff : float or None
        Uniform Zeff [-] when ``core_profiles`` has none; ``None`` then refuses.
    minimum_temperature_ev : float or None
        Floor applied to the electron temperature [eV]. Fitted profiles often
        pin Te = 0 at the separatrix, which GENRAY's absorption cannot take;
        ``None`` refuses such a profile instead of editing it. The number of
        floored points is recorded in the provenance.
    n_rho : int
        Points of the uniform sqrt(psi_N) grid the profiles are tabulated on [-].
    max_steps : int
        Ray-integration step cap (``maxsteps_rk``) [-].
    max_reflections : int
        Reflections allowed per ray (``ireflm``) [-].
    namelist_overrides : mapping
        ``{group: {name: value}}`` applied last, for settings this adapter
        does not expose. Recorded in the provenance.
    """

    mode: str
    time: float
    time_tolerance: float = 5.0e-4
    beam_index: int = 0
    harmonic: int = 1
    power_w: Optional[float] = None
    zeff: Optional[float] = None
    minimum_temperature_ev: Optional[float] = None
    n_rho: int = 41
    max_steps: int = 10000
    max_reflections: int = 1
    namelist_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    # --- runtime ---
    executable: Optional[str] = None  # explicit path; else $GENRAYHOME/bin/xgenray
    workdir: Path | str = Path(".")
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    backend: Optional["ExecutionBackend"] = None

    def __post_init__(self) -> None:
        if self.mode not in _IOXM:
            raise ValueError(f"mode must be 'O' or 'X', got {self.mode!r}")
        for name in ("time", "time_tolerance"):
            value = float(getattr(self, name))
            if value != value or (name == "time_tolerance" and value < 0.0):
                raise ValueError(f"{name} must be a finite{' non-negative' if name == 'time_tolerance' else ''} number")
        if int(self.n_rho) < 3:
            raise ValueError("n_rho must be at least 3")
        if int(self.harmonic) < 1:
            raise ValueError("harmonic must be a positive integer")
        if self.power_w is not None and not float(self.power_w) > 0.0:
            raise ValueError("power_w must be positive")
        if self.zeff is not None and not float(self.zeff) >= 1.0:
            raise ValueError("zeff must be at least 1")
        if self.minimum_temperature_ev is not None and not float(self.minimum_temperature_ev) > 0.0:
            raise ValueError("minimum_temperature_ev must be positive")

    @property
    def ioxm(self) -> int:
        return _IOXM[self.mode]


@dataclass
class GENRAYInputs:
    """A materialised GENRAY case: the working directory and what VAFT put in it."""

    workdir: Path
    genray_in: Path
    eqdsk: Path
    provenance: dict[str, Any]
    files: tuple[Path, ...] = ()


@dataclass
class GENRAYResult:
    """Outcome of one GENRAY run."""

    returncode: Optional[int]
    workdir: Path
    stdout: str = ""
    stderr: str = ""
    netcdf: Optional[Path] = None
    parsed: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Exit 0, a ``genray.nc``, every total present and at least one ray in the plasma."""
        return (
            self.returncode == 0
            and self.netcdf is not None
            and isinstance(self.parsed, dict)
            and bool(self.parsed.get("complete"))
        )
