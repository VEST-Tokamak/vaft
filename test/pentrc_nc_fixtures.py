"""Synthetic PENTRC output files, shaped like a real ``pentrc_output_n*.nc``.

The layout mirrors what PENTRC v1.5.4 writes: equilibrium and kinetic
profiles on ``psi_n``, two torque methods on **their own separate grids**, a
bounce-harmonic dimension ``ell``, and a real/imaginary ``i`` dimension on
every torque quantity.

The awkward details are deliberate, and each was read off a real file rather
than imagined:

* ``psi_fgar``, ``psi_tgar`` and ``psi_n`` are **three different lengths**
  (55, 34 and 129 in the reference), so nothing can resolve a grid by
  counting and nothing can silently share one;
* the two methods' totals differ by order unity -- neither is a refinement of
  the other;
* ``T`` is complex, and its imaginary part is **2 n** times the perturbed
  energy, not 2 times it (``pentrc/torque.F90:2029``).  The fixture builder
  takes ``n`` so a test can tell the two apart, which needs ``n != 1``;
* ``ell`` runs -4..4 and a torque is the sum over it, so a fixture whose
  harmonics all carried the same value could not catch a reader that took
  one;
* the global ``T_total_*`` and ``dW_total_*`` attributes are computed here
  the way PENTRC computes them, so a reader can be checked against the file's
  own arithmetic rather than against a number written twice.

Nothing here is copied from a real run: the numbers are synthetic.
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
    "dTdpsi": "Nm per unit normalized flux",
    "T": "Nm",
}

ELL = np.arange(-4, 5)


def _torque_block(n_psi: int, n_ell: int, scale: float, seed: int):
    """A ``(psi, ell, i)`` block whose ell-sum accumulates radially."""
    rng = np.random.default_rng(seed)
    # Per-harmonic increments, so the ell-sum is a genuine sum and the radial
    # profile is genuinely cumulative -- a flat block would hide both.
    steps = rng.normal(size=(n_psi, n_ell, 2)) * scale
    return np.cumsum(steps, axis=0)


def pentrc_dataset(
    *,
    n_tor: int = 1,
    n_psi_n: int = 21,
    n_fgar: int = 13,
    n_tgar: int = 9,
    methods: tuple[str, ...] = ("fgar", "tgar"),
    seed: int = 7,
) -> xr.Dataset:
    """Build one synthetic PENTRC output.

    ``n_tor`` matters: the energy relation divides by ``2 n``, so a fixture
    built at ``n = 1`` cannot distinguish that from dividing by 2.
    """
    psi_n = np.linspace(0.02, 0.99, n_psi_n)
    variables: dict = {}
    coords = {"psi_n": psi_n, "ell": ELL, "i": np.array([0, 1], dtype=np.int32)}
    attrs: dict = {
        "title": "PENTRC fundamental outputs",
        "version": "synthetic",
        "machine": "",
        "n": np.int32(n_tor),
        "R0": 1.75,
        "B0": 1.69,
    }

    rng = np.random.default_rng(seed)
    for name in ("q", "eps_r", "dvdpsi", "n_i", "n_e", "T_i", "T_e", "zeff",
                 "nu_i", "nu_e", "omega_E", "omega_b_rlar", "omega_d_rlar"):
        values = 1.0 + rng.random(n_psi_n)
        variables[name] = xr.DataArray(
            values, dims=("psi_n",), attrs={"units": UNITS.get(name, "")}
        )

    sizes = {"fgar": n_fgar, "tgar": n_tgar}
    # Different scales, so the two methods' totals differ by order unity as
    # they do in a real run.
    scales = {"fgar": 0.10, "tgar": 0.04}
    for offset, method in enumerate(methods):
        size = sizes[method]
        grid = np.linspace(0.05, 0.98, size)
        coords[f"psi_{method}"] = grid
        for quantity in ("Gamma", "chi", "dTdpsi", "T"):
            block = _torque_block(size, ELL.size, scales[method], seed + offset * 17)
            variables[f"{quantity}_{method}"] = xr.DataArray(
                block,
                dims=(f"psi_{method}", "ell", "i"),
                attrs={"units": UNITS.get(quantity, "")},
            )
        summed = variables[f"T_{method}"].values.sum(axis=1)
        # Exactly PENTRC's own reduction (torque.F90:2026-2029).
        attrs[f"T_total_{method}"] = float(summed[-1, 0])
        attrs[f"dW_total_{method}"] = float(summed[-1, 1] / (2 * n_tor))

    return xr.Dataset(variables, coords=coords, attrs=attrs)


def write_pentrc_output(path, **kwargs) -> str:
    """Write a synthetic PENTRC output and return its path."""
    dataset = pentrc_dataset(**kwargs)
    dataset.to_netcdf(path)
    dataset.close()
    return str(path)
