"""Native PENTRC output containers: a transcript of one ``pentrc_output_n*.nc``.

PENTRC computes the neoclassical toroidal viscous torque a non-axisymmetric
field exerts on a tokamak plasma, given the perturbed field GPEC produced and
a kinetic profile.  This layer reads what it wrote and stops there, following
the split :mod:`vaft.code.gpec` and :mod:`vaft.code.transp` already keep: the
layer above owns the interpretation, and it cannot own it if this one has
already quietly reinterpreted the file.

Five facts the file forces:

- **The torque is complex, and its imaginary part is not a torque.**
  ``T_<method>`` carries a real/imaginary ``i`` dimension.  The real part is
  the toroidal torque; the imaginary part is the perturbed potential energy,
  and PENTRC divides it by ``2 n`` before reporting it as the global
  ``dW_total_<method>`` attribute (``pentrc/torque.F90:2026-2029``).  So ``dW``
  is ``imag(T) / (2 n)``, not ``imag(T) / 2`` -- the two agree only at
  ``n = 1``.  :func:`energy_profile` applies the run's own ``n``.

- **``ell`` is the integer bounce harmonic, and the sum over it is the
  torque.**  ``torque.F90:1983`` writes it as ``(/(i,i=-nl,nl)/)`` and
  ``energy.f90:124`` names it "Bounce harmonic".  It is *not* the effective
  bounce harmonic: that is ``leff = ell - sigma n q``
  (``energy.f90:125-126``), a real number PENTRC uses internally and does not
  write to this file.  One ``ell`` alone is a resonance, not a torque.

- **The same variable names carry a different quantity when the run asks for a
  heat moment.**  With ``moment='heat'`` (``pentrc.F90``), ``T_<method>`` is
  written with identical dimensions and identical ``Nm`` units but
  ``long_name = "Integrated A*e*psi'*Gamma/2pi"`` instead of
  ``"Integrated Toroidal Torque"`` (``torque.F90:2053-2061``).  Nothing in the
  name, shape or unit distinguishes them, so :func:`torque_profile` checks the
  ``long_name`` and refuses the heat moment rather than returning it as a
  torque.

- **There are eighteen methods on three grids, and they are a decomposition,
  not a set of alternatives.**  ``params.f90:49-52`` lists them:
  ``fgar`` is the full general-aspect-ratio calculation and ``tgar`` and
  ``pgar`` are its **trapped** and **passing** parts, so the three are related
  rather than competing.  :data:`TORQUE_METHODS` carries all eighteen with
  PENTRC's own one-line description of each, and nothing here defaults to one.
  Each method may also be written on any of three radial grids
  (:data:`TORQUE_GRIDS`): ``lsode`` takes the suffix ``_<method>`` and the
  other two ``_<method>_<grid>`` (``torque.F90:2012-2016``).  The grids differ
  in length because ``lsode`` steps adaptively, so a method's grid is its own
  and nothing here interpolates between them.

- **The equilibrium and kinetic profiles are on their own grid.**  ``psi_n``
  carries the frequencies and collisionality, which is none of the torque
  grids.  :func:`PentrcOutput.profile` serves only the variables that live
  there, so asking it for a torque is an error rather than an array of the
  wrong rank.

The file is netCDF classic with 64-bit offsets (``NF90_64BIT_OFFSET``,
``torque.F90:1906``), read through xarray.  Reading is lazy: a production run
is modest but a campaign is a dozen of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

__all__ = [
    "PROFILE_VARIABLES",
    "PentrcFormatError",
    "PentrcOutput",
    "TORQUE_GRIDS",
    "TORQUE_LONG_NAME",
    "TORQUE_METHODS",
    "TORQUE_QUANTITIES",
    "energy_profile",
    "read_pentrc_output",
    "torque_profile",
]


class PentrcFormatError(ValueError):
    """The file is not a PENTRC output, or does not carry what was asked of it."""


#: What ``long_name`` a genuine toroidal-torque variable carries.
#:
#: A ``moment='heat'`` run writes the same variable names, dimensions and
#: ``Nm`` units with a different ``long_name``; this is the only thing in the
#: file that tells the two apart.
TORQUE_LONG_NAME = "Integrated Toroidal Torque"

#: PENTRC's eighteen calculations, with its own description of each
#: (``pentrc/params.f90:49-75``).
#:
#: They are a decomposition, not alternatives: ``fgar`` is the full
#: general-aspect-ratio calculation and ``tgar``/``pgar`` are its trapped and
#: passing parts.  Nothing in this module defaults to one.
TORQUE_METHODS: Mapping[str, str] = MappingProxyType({
    "fgar": "Full general-aspect-ratio calculation",
    "tgar": "Trapped particle general-aspect-ratio calculation",
    "pgar": "Passing particle general-aspect-ratio calculation",
    "rlar": "Trapped particle large-aspect-ratio calculation",
    "clar": "Trapped particle cylindrical large-aspect-ratio calculation",
    "fcgl": "Fluid Chew-Goldberger-Low calculation",
    "fwmm": "Full energy calculation using MXM euler lagrange matrix",
    "twmm": "Trapped energy calculation using MXM euler lagrange matrix",
    "pwmm": "Passing energy calculation using MXM euler lagrange matrix",
    "ftmm": "Full torque calculation using MXM euler lagrange matrix",
    "ttmm": "Trapped torque calculation using MXM euler lagrange matrix",
    "ptmm": "Passing torque calculation using MXM euler lagrange matrix",
    "fkmm": "Full MXM euler lagrange energy matrix norm calculation",
    "tkmm": "Trapped MXM euler lagrange energy matrix norm calculation",
    "pkmm": "Passing MXM euler lagrange energy matrix norm calculation",
    "frmm": "Full MXM euler lagrange torque matrix norm calculation",
    "trmm": "Trapped MXM euler lagrange torque matrix norm calculation",
    "prmm": "Passing MXM euler lagrange torque matrix norm calculation",
})

#: The three radial grids a method may be written on
#: (``pentrc/params.f90:53``), and how each appears in a variable name.
TORQUE_GRIDS: Mapping[str, str] = MappingProxyType({
    "lsode": "the solver's own adaptive grid; suffix '_<method>'",
    "equil": "the equilibrium grid; suffix '_<method>_equil'",
    "input": "the grid the kinetic input was given on; suffix '_<method>_input'",
})

#: The per-method quantities, each carrying ``(psi, ell, i)``.
TORQUE_QUANTITIES: Mapping[str, str] = MappingProxyType({
    "Gamma": "nonambipolar particle (or heat) flux [1/s m^2]",
    "chi": "nonambipolar particle (or heat) diffusivity [m^2/s]",
    "dTdpsi": "torque density, per unit normalized flux [N m]",
    "T": (
        "radially integrated toroidal torque [N m]; complex, with the real "
        "part the torque and the imaginary part 2*n times the perturbed energy"
    ),
})

#: Equilibrium and kinetic profiles, all on the ``psi_n`` grid -- which is none
#: of the torque grids.
PROFILE_VARIABLES: Mapping[str, str] = MappingProxyType({
    "q": "safety factor [-]",
    "eps_r": "inverse aspect ratio [-]",
    "dvdpsi": "differential volume [m^3]",
    "mu0P": "equilibrium pressure, times mu0 [T^2]",
    "n_i": "ion density [m^-3]",
    "n_e": "electron density [m^-3]",
    "T_i": "ion temperature [eV]",
    "T_e": "electron temperature [eV]",
    "zeff": "effective charge [-]",
    "logLambda": "Coulomb logarithm [-]",
    "nu_i": "ion collision rate [1/s]",
    "nu_e": "electron collision rate [1/s]",
    "omega_E": "electric precession frequency [rad/s]",
    "omega_N": "density-gradient diamagnetic frequency [rad/s]",
    "omega_T": "temperature-gradient diamagnetic frequency [rad/s]",
    "omega_trans": "transit frequency [rad/s]",
    "omega_gyro": "gyro-frequency [rad/s]",
    "omega_b_rlar": "reduced bounce frequency [rad/s]",
    "omega_d_rlar": "reduced magnetic precession frequency [rad/s]",
})


def _scalar(value: Any) -> Any:
    """An attribute that may surface as a scalar or a one-element array."""
    if isinstance(value, (list, tuple)):
        return _scalar(value[0]) if value else None
    array = np.asarray(value)
    return array.reshape(-1)[0] if array.ndim else array.item()


def _suffix(method: str, grid: str) -> str:
    """PENTRC's own variable suffix for one method on one grid."""
    return f"_{method}" if grid == "lsode" else f"_{method}_{grid}"


@dataclass
class PentrcOutput:
    """One ``pentrc_output_n*.nc``, read lazily and converted nowhere.

    Attributes
    ----------
    path : Path
        The file this reads [-].
    n_tor : int
        Toroidal mode number, from the file's own ``n`` attribute [-].
    attrs : dict
        The file's global attributes, as read [any].
    """

    path: Path
    n_tor: int
    attrs: dict
    _dataset: Any = None

    def __enter__(self) -> "PentrcOutput":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying dataset."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    @property
    def dataset(self):
        """The open xarray dataset."""
        if self._dataset is None:
            raise PentrcFormatError(f"{self.path} is closed")
        return self._dataset

    def available(self) -> tuple[tuple[str, str], ...]:
        """Every ``(method, grid)`` pair this run actually wrote, sorted.

        A run may write one method on several grids, or several methods on
        one; both are reported, because "the run did not compute fgar" and
        "the run computed fgar on a grid you did not ask for" are different
        answers.
        """
        return tuple(
            sorted(
                (method, grid)
                for method in TORQUE_METHODS
                for grid in TORQUE_GRIDS
                if f"T{_suffix(method, grid)}" in self.dataset.variables
            )
        )

    def psi_norm(self, method: str, grid: str = "lsode") -> np.ndarray:
        """The radial grid one calculation was written on [-].

        Each ``(method, grid)`` pair has its own; they are different lengths
        because ``lsode`` steps adaptively, and this module does not
        interpolate between them.
        """
        self._check(method, grid)
        return np.asarray(
            self.dataset[f"psi{_suffix(method, grid)}"].values, dtype=float
        )

    def profile_psi_norm(self) -> np.ndarray:
        """The ``psi_n`` grid the equilibrium and kinetic profiles sit on [-]."""
        return np.asarray(self.dataset["psi_n"].values, dtype=float)

    def profile(self, name: str) -> np.ndarray:
        """One equilibrium or kinetic profile, in PENTRC's own units.

        Restricted to :data:`PROFILE_VARIABLES`, which are the variables on the
        ``psi_n`` grid.  Asking for a torque quantity here is an error rather
        than an array of the wrong rank on the wrong grid.

        Raises
        ------
        PentrcFormatError
            The name is not a profile variable, or the run did not write it.
        """
        if name not in PROFILE_VARIABLES:
            raise PentrcFormatError(
                f"{name!r} is not a profile variable. Profiles live on psi_n and "
                f"are {sorted(PROFILE_VARIABLES)}; a torque quantity is read with "
                "complex_quantity, which knows its method and grid"
            )
        if name not in self.dataset.variables:
            raise PentrcFormatError(
                f"{self.path.name} carries no {name!r}; it has "
                f"{sorted(n for n in PROFILE_VARIABLES if n in self.dataset.variables)}"
            )
        return np.asarray(self.dataset[name].values, dtype=float)

    def complex_quantity(
        self, quantity: str, method: str, grid: str = "lsode"
    ) -> np.ndarray:
        """One ``(psi, ell)`` complex array, unsummed.

        Returned with ``ell`` intact, because summing it is a decision: a
        single bounce harmonic is a resonance, and the torque is the sum.

        Parameters
        ----------
        quantity : str
            One of :data:`TORQUE_QUANTITIES` [n/a].
        method : str
            One of :data:`TORQUE_METHODS` [n/a].
        grid : str, optional
            One of :data:`TORQUE_GRIDS` [n/a].
        """
        if quantity not in TORQUE_QUANTITIES:
            raise PentrcFormatError(
                f"quantity must be one of {sorted(TORQUE_QUANTITIES)}, not {quantity!r}"
            )
        self._check(method, grid)
        name = f"{quantity}{_suffix(method, grid)}"
        if name not in self.dataset.variables:
            raise PentrcFormatError(f"{self.path.name} carries no {name!r}")
        values = np.asarray(self.dataset[name].values, dtype=float)
        if values.shape[-1] != 2:
            raise PentrcFormatError(
                f"{name} has trailing dimension {values.shape[-1]}, not the "
                "real/imaginary pair PENTRC writes"
            )
        return values[..., 0] + 1j * values[..., 1]

    def long_name(self, quantity: str, method: str, grid: str = "lsode") -> str:
        """The ``long_name`` PENTRC gave one variable, or ``""``."""
        name = f"{quantity}{_suffix(method, grid)}"
        if name not in self.dataset.variables:
            return ""
        return str(self.dataset[name].attrs.get("long_name", "")).strip()

    def ell(self) -> np.ndarray:
        """The integer bounce harmonics the run resolved [-].

        ``ell``, not ``leff``: the effective bounce harmonic
        ``leff = ell - sigma n q`` is real, is used inside PENTRC and is not
        written to this file.
        """
        return np.asarray(self.dataset["ell"].values, dtype=int)

    def _check(self, method: str, grid: str) -> None:
        if method not in TORQUE_METHODS:
            raise PentrcFormatError(
                f"method must be one of PENTRC's {len(TORQUE_METHODS)} methods, not "
                f"{method!r}; see TORQUE_METHODS"
            )
        if grid not in TORQUE_GRIDS:
            raise PentrcFormatError(
                f"grid must be one of {sorted(TORQUE_GRIDS)}, not {grid!r}"
            )
        if f"psi{_suffix(method, grid)}" not in self.dataset.variables:
            present = self.available()
            same_method = [g for m, g in present if m == method]
            if same_method:
                raise PentrcFormatError(
                    f"{self.path.name} computed {method!r} on {sorted(same_method)}, "
                    f"not on {grid!r}. Each grid is its own; nothing here "
                    "interpolates between them"
                )
            raise PentrcFormatError(
                f"{self.path.name} did not compute {method!r} on any grid; it has "
                f"{list(present)}"
            )


def read_pentrc_output(path: str | Path) -> PentrcOutput:
    """Open one PENTRC output file.

    Parameters
    ----------
    path : path-like
        A ``pentrc_output_n*.nc`` [-].

    Returns
    -------
    PentrcOutput
        Open; close it, or use it as a context manager.

    Raises
    ------
    PentrcFormatError
        The file carries no toroidal mode number, which every PENTRC output
        does -- so its absence means this is a different file.
    """
    import xarray as xr

    location = Path(path)
    dataset = xr.open_dataset(location)
    n_raw = dataset.attrs.get("n")
    if n_raw is None:
        dataset.close()
        raise PentrcFormatError(
            f"{location} carries no 'n' global attribute; every PENTRC output does, "
            "so this is not one"
        )
    return PentrcOutput(
        path=location,
        n_tor=int(_scalar(n_raw)),
        attrs={key: _scalar(value) for key, value in dataset.attrs.items()},
        _dataset=dataset,
    )


def _torque_variable(output: PentrcOutput, method: str, grid: str) -> np.ndarray:
    """``T`` for one calculation, refusing a variable that is not a torque."""
    declared = output.long_name("T", method, grid)
    if declared and declared != TORQUE_LONG_NAME:
        raise PentrcFormatError(
            f"T{_suffix(method, grid)} declares long_name {declared!r}, not "
            f"{TORQUE_LONG_NAME!r}. A moment='heat' run writes this variable with "
            "the same name, shape and Nm units carrying "
            "A*e*psi'*Gamma/2pi instead, so reading it as a torque would return a "
            "plausible number for a different quantity"
        )
    return output.complex_quantity("T", method, grid)


def torque_profile(
    output: PentrcOutput, method: str, grid: str = "lsode"
) -> tuple[np.ndarray, np.ndarray]:
    """Radially accumulated NTV torque, summed over the bounce harmonics.

    Parameters
    ----------
    output : PentrcOutput
        An open run [-].
    method : str
        One of :data:`TORQUE_METHODS`.  Required: ``fgar`` and its ``tgar`` and
        ``pgar`` parts are different quantities on different grids [n/a].
    grid : str, optional
        One of :data:`TORQUE_GRIDS` [n/a].

    Returns
    -------
    (ndarray, ndarray)
        The calculation's own ``psi_norm`` grid [-] and the torque enclosed by
        each point [N m].

    Raises
    ------
    PentrcFormatError
        The run did not write this calculation, or wrote a heat moment under
        the same variable name.

    Notes
    -----
    The real part alone.  The imaginary part of ``T`` is not a torque; see
    :func:`energy_profile`.
    """
    psi_norm = output.psi_norm(method, grid)
    return psi_norm, np.real(_torque_variable(output, method, grid).sum(axis=1))


def energy_profile(
    output: PentrcOutput, method: str, grid: str = "lsode"
) -> tuple[np.ndarray, np.ndarray]:
    """Radially accumulated perturbed potential energy.

    ``imag(T) / (2 n)`` -- PENTRC's own reduction to the ``dW_total`` global
    attribute (``torque.F90:2029``).  The ``n`` is the run's, which is why this
    is a function rather than a factor a caller applies: at ``n = 1`` the
    divisor is 2 and at ``n = 3`` it is 6, so a study that compares modes would
    otherwise be wrong by ``n``.

    Returns
    -------
    (ndarray, ndarray)
        The calculation's ``psi_norm`` grid [-] and the enclosed energy [J].
    """
    psi_norm = output.psi_norm(method, grid)
    summed = _torque_variable(output, method, grid).sum(axis=1)
    return psi_norm, np.imag(summed) / (2.0 * output.n_tor)
