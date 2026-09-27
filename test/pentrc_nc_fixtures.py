"""Synthetic PENTRC output files, shaped like a ``pentrc_output_n*.nc``.

The layout follows what PENTRC writes (``pentrc/torque.F90:1900-2070``,
``pentrc/params.f90:47-53``): equilibrium and kinetic profiles on ``psi_n``,
per-method torque quantities on their own radial grids, an integer
bounce-harmonic dimension ``ell``, and a real/imaginary ``i`` dimension on
every torque quantity.

The awkward details are deliberate, each read off the *format* rather than off
any particular run:

* ``psi_n`` and the per-method grids are **three different lengths**, so
  nothing can resolve a grid by counting and nothing can silently share one;
* a method's suffix is ``_<method>`` on the ``lsode`` grid and
  ``_<method>_<grid>`` on the others, so a fixture can carry both;
* ``T`` is complex, and its imaginary part is **2 n** times the perturbed
  energy, not 2 times it (``torque.F90:2029``).  The builder takes ``n`` so a
  test can tell the two apart, which needs ``n != 1``;
* ``ell`` spans several integers and a torque is the sum over it, so a fixture
  whose harmonics all carried the same value could not catch a reader that took
  one;
* ``long_name`` is settable, because a ``moment='heat'`` run writes the same
  variable names, shapes and units under a different one;
* the global ``T_total_*`` and ``dW_total_*`` attributes are computed here the
  way PENTRC computes them, so a reader can be checked against the file's own
  arithmetic rather than against a number written twice.

Every value is arbitrary: the numbers are drawn from a seeded generator and
describe no machine and no discharge.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

#: PENTRC declares units per variable; nothing infers them.
UNITS = {
    "dvdpsi": "m^3",
    "n_i": "m^-3",
    "n_e": "m^-3",
    "T_i": "eV",
    "T_e": "eV",
    "nu_i": "1/s",
    "nu_e": "1/s",
    "omega_E": "rad/s",
    "omega_b_rlar": "rad/s",
    "omega_d_rlar": "rad/s",
    "Gamma": "1/sm^2",
    "chi": "m^2/s",
    "dTdpsi": "Nm per psi",
    "T": "Nm",
}

#: The ``long_name`` a genuine torque run writes, and the heat-moment one.
TORQUE_LONG_NAMES = {
    "Gamma": "Nonambipolar Particle Flux",
    "chi": "Nonambipolar Particle Diffusivity",
    "dTdpsi": "Toroidal Torque Profile",
    "T": "Integrated Toroidal Torque",
}
HEAT_LONG_NAMES = {
    "Gamma": "Nonambipolar Heat Flux",
    "chi": "Nonambipolar Heat Diffusivity",
    "dTdpsi": " A*e*psi'*Gamma/2pi profile",
    "T": "Integrated A*e*psi'*Gamma/2pi",
}

ELL = np.arange(-4, 5)

PROFILE_NAMES = (
    "q", "eps_r", "dvdpsi", "n_i", "n_e", "T_i", "T_e", "zeff",
    "nu_i", "nu_e", "omega_E", "omega_b_rlar", "omega_d_rlar",
)


def _suffix(method: str, grid: str) -> str:
    return f"_{method}" if grid == "lsode" else f"_{method}_{grid}"


def _torque_block(n_psi: int, n_ell: int, scale: float, seed: int):
    """A ``(psi, ell, i)`` block whose ell-sum accumulates radially."""
    rng = np.random.default_rng(seed)
    # Per-harmonic increments, so the ell-sum is a genuine sum and the radial
    # profile is genuinely cumulative -- a flat block would hide both.
    return np.cumsum(rng.normal(size=(n_psi, n_ell, 2)) * scale, axis=0)


def pentrc_dataset(
    *,
    n_tor: int = 1,
    n_psi_n: int = 21,
    calculations: dict[tuple[str, str], int] | None = None,
    moment: str = "torque",
    seed: int = 7,
) -> xr.Dataset:
    """Build one synthetic PENTRC output.

    Parameters
    ----------
    n_tor : int
        The toroidal mode number.  It matters: the energy relation divides by
        ``2 n``, so a fixture built at ``n = 1`` cannot distinguish that from
        dividing by 2.
    calculations : dict, optional
        ``{(method, grid): grid_length}``.  Defaults to ``fgar`` and ``tgar``
        on ``lsode`` at different lengths.
    moment : str
        ``"torque"`` or ``"heat"``.  A heat run writes identical names, shapes
        and units under a different ``long_name``.
    """
    if calculations is None:
        calculations = {("fgar", "lsode"): 13, ("tgar", "lsode"): 9}
    long_names = TORQUE_LONG_NAMES if moment == "torque" else HEAT_LONG_NAMES

    psi_n = np.linspace(0.02, 0.99, n_psi_n)
    variables: dict = {}
    coords: dict = {
        "psi_n": psi_n,
        "ell": ELL,
        "i": np.array([0, 1], dtype=np.int32),
    }
    attrs: dict = {
        "title": "PENTRC fundamental outputs",
        "version": "synthetic",
        "machine": "",
        "n": np.int32(n_tor),
    }

    rng = np.random.default_rng(seed)
    for name in PROFILE_NAMES:
        variables[name] = xr.DataArray(
            1.0 + rng.random(n_psi_n),
            dims=("psi_n",),
            attrs={"units": UNITS.get(name, "")},
        )

    for offset, ((method, grid), size) in enumerate(sorted(calculations.items())):
        suffix = _suffix(method, grid)
        coords[f"psi{suffix}"] = np.linspace(0.05, 0.98, size)
        # Different scales per calculation, so two of them never coincide.
        scale = 0.10 / (1.0 + offset)
        for quantity in ("Gamma", "chi", "dTdpsi", "T"):
            variables[f"{quantity}{suffix}"] = xr.DataArray(
                _torque_block(size, ELL.size, scale, seed + offset * 17),
                dims=(f"psi{suffix}", "ell", "i"),
                attrs={
                    "units": UNITS.get(quantity, ""),
                    "long_name": long_names[quantity],
                },
            )
        summed = variables[f"T{suffix}"].values.sum(axis=1)
        # Exactly PENTRC's own reduction (torque.F90:2026-2029).
        attrs[f"T_total{suffix}"] = float(summed[-1, 0])
        attrs[f"dW_total{suffix}"] = float(summed[-1, 1] / (2 * n_tor))

    return xr.Dataset(variables, coords=coords, attrs=attrs)


def write_pentrc_output(path, **kwargs) -> str:
    """Write a synthetic PENTRC output and return its path."""
    dataset = pentrc_dataset(**kwargs)
    dataset.to_netcdf(path)
    dataset.close()
    return str(path)
