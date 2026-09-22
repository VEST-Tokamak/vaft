"""Native PENTRC output containers: a transcript of one ``pentrc_output_n*.nc``.

PENTRC computes the neoclassical toroidal viscous torque a non-axisymmetric
field exerts on a tokamak plasma, given the perturbed field GPEC produced and
a kinetic profile.  This layer reads what it wrote and stops there, following
the split :mod:`vaft.code.gpec` and :mod:`vaft.code.transp` already keep: the
layer above owns the interpretation, and it cannot own it if this one has
already quietly reinterpreted the file.

Four facts the file forces:

- **The torque is complex, and its imaginary part is not a torque.**
  ``T_fgar`` carries a real/imaginary ``i`` dimension.  The real part is the
  toroidal torque in newton metres; the imaginary part is the perturbed
  potential energy, and PENTRC divides it by ``2 n`` before reporting it as
  the global ``dW_total_fgar`` attribute (``pentrc/torque.F90:2026-2029``).
  So ``dW`` is ``imag(T) / (2 n)``, not ``imag(T) / 2`` -- the two agree only
  at ``n = 1``, which is the mode most runs use.  :func:`energy_profile`
  applies the run's own ``n``.

- **A torque profile has to be summed over ``ell`` first.**  ``ell`` is the
  bounce harmonic of the resonance -- ``pentrc/energy.f90:126`` calls it "the
  effective bounce harmonic ``ell - sigma*n*q`` where sigma=0(1) for
  trapped(passing)" -- and runs from -4 to 4 in the files checked.  One
  ``ell`` alone is a resonance, not a torque.

- **There are two methods and they do not agree.**  ``fgar`` and ``tgar`` are
  written side by side on *different radial grids* (55 and 34 points in the
  reference), and their totals differ by a factor of order one -- 0.794 against
  0.351 N m for one run here, 6.85 against 9.05 for another.  Neither is a
  refinement of the other, so nothing here picks one:
  :data:`TORQUE_METHODS` names them and the caller says which.

- **The equilibrium and kinetic profiles are on their own grid.**  ``psi_n``
  carries the frequencies and collisionality (129 points), which is neither of
  the torque grids.  Interpolating between them is the caller's decision, so
  the grids come back as they are.

The file is netCDF-4, read through xarray.  Reading is lazy: a production run
is modest but a campaign is a dozen of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "PROFILE_VARIABLES",
    "PentrcFormatError",
    "PentrcOutput",
    "TORQUE_METHODS",
    "TORQUE_QUANTITIES",
    "energy_profile",
    "read_pentrc_output",
    "torque_profile",
]


class PentrcFormatError(ValueError):
    """The file is not a PENTRC output, or does not carry what was asked of it."""


#: The two torque calculations PENTRC writes, and what each assumes.
#:
#: They sit side by side in one file on **different radial grids** and their
#: totals differ by order unity.  Neither is a correction of the other, and
#: nothing in this module defaults to one.
TORQUE_METHODS: Mapping[str, str] = {
    "fgar": "full general aspect ratio -- the method the MAST-U study reports",
    "tgar": "trapped-particle general aspect ratio",
}

#: The per-method quantities, each carrying ``(psi, ell, i)``.
TORQUE_QUANTITIES: Mapping[str, str] = {
    "Gamma": "nonambipolar particle flux [1/s m^2]",
    "chi": "nonambipolar particle diffusivity [m^2/s]",
    "dTdpsi": "toroidal torque density, per unit normalized flux [N m]",
    "T": (
        "radially integrated toroidal torque [N m]; complex, with the real "
        "part the torque and the imaginary part 2*n times the perturbed energy"
    ),
}

#: Equilibrium and kinetic profiles, all on the ``psi_n`` grid -- which is
#: neither torque grid.
PROFILE_VARIABLES: Mapping[str, str] = {
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
}


def _scalar(value: Any) -> Any:
    """An attribute that may surface as a scalar or a one-element array."""
    if isinstance(value, (list, tuple)):
        return _scalar(value[0]) if value else None
    array = np.asarray(value)
    return array.reshape(-1)[0] if array.ndim else array.item()


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

    def methods(self) -> tuple[str, ...]:
        """Which of :data:`TORQUE_METHODS` this run actually wrote."""
        return tuple(
            name for name in TORQUE_METHODS if f"T_{name}" in self.dataset.variables
        )

    def psi_norm(self, method: str) -> np.ndarray:
        """The radial grid one method was computed on [-].

        Each method has its own; they are not interchangeable and this module
        does not interpolate between them.
        """
        self._check_method(method)
        return np.asarray(self.dataset[f"psi_{method}"].values, dtype=float)

    def profile_psi_norm(self) -> np.ndarray:
        """The ``psi_n`` grid the equilibrium and kinetic profiles sit on [-]."""
        return np.asarray(self.dataset["psi_n"].values, dtype=float)

    def profile(self, name: str) -> np.ndarray:
        """One equilibrium or kinetic profile, in PENTRC's own units.

        Raises
        ------
        PentrcFormatError
            The run did not write it.
        """
        if name not in self.dataset.variables:
            raise PentrcFormatError(
                f"{self.path.name} carries no {name!r}; it has "
                f"{sorted(n for n in PROFILE_VARIABLES if n in self.dataset.variables)}"
            )
        return np.asarray(self.dataset[name].values, dtype=float)

    def complex_quantity(self, quantity: str, method: str) -> np.ndarray:
        """One ``(psi, ell)`` complex array, unsummed.

        Returned with ``ell`` intact, because summing it is a decision: a
        single harmonic is a resonance, and the torque is the sum.

        Parameters
        ----------
        quantity : str
            One of :data:`TORQUE_QUANTITIES` [n/a].
        method : str
            One of :data:`TORQUE_METHODS` [n/a].
        """
        if quantity not in TORQUE_QUANTITIES:
            raise PentrcFormatError(
                f"quantity must be one of {sorted(TORQUE_QUANTITIES)}, not {quantity!r}"
            )
        self._check_method(method)
        name = f"{quantity}_{method}"
        if name not in self.dataset.variables:
            raise PentrcFormatError(f"{self.path.name} carries no {name!r}")
        values = np.asarray(self.dataset[name].values, dtype=float)
        if values.shape[-1] != 2:
            raise PentrcFormatError(
                f"{name} has trailing dimension {values.shape[-1]}, not the "
                "real/imaginary pair PENTRC writes"
            )
        return values[..., 0] + 1j * values[..., 1]

    def ell(self) -> np.ndarray:
        """The bounce harmonics the run resolved [-]."""
        return np.asarray(self.dataset["ell"].values, dtype=int)

    def _check_method(self, method: str) -> None:
        if method not in TORQUE_METHODS:
            raise PentrcFormatError(
                f"method must be one of {sorted(TORQUE_METHODS)}, not {method!r}"
            )
        if f"psi_{method}" not in self.dataset.variables:
            raise PentrcFormatError(
                f"{self.path.name} did not compute {method!r}; it has "
                f"{list(self.methods())}"
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


def torque_profile(output: PentrcOutput, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Radially accumulated NTV torque, summed over the bounce harmonics.

    Parameters
    ----------
    output : PentrcOutput
        An open run [-].
    method : str
        One of :data:`TORQUE_METHODS`.  Required: the two are computed on
        different grids and differ by order unity [n/a].

    Returns
    -------
    (ndarray, ndarray)
        The method's own ``psi_norm`` grid [-] and the torque enclosed by each
        point [N m].

    Notes
    -----
    The real part alone.  The imaginary part of ``T`` is not a torque; see
    :func:`energy_profile`.
    """
    psi_norm = output.psi_norm(method)
    return psi_norm, np.real(output.complex_quantity("T", method).sum(axis=1))


def energy_profile(output: PentrcOutput, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Radially accumulated perturbed potential energy.

    ``imag(T) / (2 n)`` -- PENTRC's own reduction to the ``dW_total`` global
    attribute (``pentrc/torque.F90:2029``).  The ``n`` is the run's, which is
    why this is a function rather than a factor a caller applies: at ``n = 1``
    the divisor is 2 and at ``n = 3`` it is 6, and a study that compares modes
    would otherwise be wrong by ``n``.

    Returns
    -------
    (ndarray, ndarray)
        The method's ``psi_norm`` grid [-] and the enclosed energy [J].
    """
    psi_norm = output.psi_norm(method)
    summed = output.complex_quantity("T", method).sum(axis=1)
    return psi_norm, np.imag(summed) / (2.0 * output.n_tor)
