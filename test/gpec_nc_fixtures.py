"""Synthetic ideal-GPEC netCDF fixtures shaped like the real output files.

The layouts mirror the shot-48226 @ 300 ms reference run
(``gpec_control_output_n1.nc`` / ``gpec_cylindrical_output_n1.nc``, GPEC
v1.5.5): complex values as a *leading* length-2 ``i`` dimension, global
attributes ``n``/``machine``/``shot``/``time``/``version``/energies, and the
``(z, R)``-ordered cylindrical grids.  Kept small so the parser and IMAS
mapping are exercised hermetically without committed ``.nc`` files.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _complex_pair(values: np.ndarray) -> np.ndarray:
    """Stack a complex array into GPEC's leading-``i`` real/imaginary layout."""
    values = np.asarray(values, dtype=complex)
    return np.stack([values.real, values.imag])


def write_control_nc(
    path,
    *,
    n=1,
    m_count=5,
    theta_count=9,
    coil_names=("MID",),
    energy_vacuum=0.25,
    energy_surface=0.5,
    energy_plasma=0.75,
    filename_mode=None,
):
    """Write a miniature ``gpec_control_output_n<mode>.nc``; returns its data."""
    m = np.arange(-2, -2 + m_count)
    theta = np.linspace(0.0, 1.0, theta_count)
    b_n = (m + 1.0) * 1e-4 + 1j * m * 1e-5
    xi_n = (m - 0.5) * 1e-3 - 1j * m * 1e-4
    b_n_fun = np.cos(2 * np.pi * theta) * 1e-4 + 1j * np.sin(2 * np.pi * theta) * 1e-4
    phi_coil = np.stack([b_n * (index + 1.0) for index in range(len(coil_names))])
    strlen = 24
    name_chars = np.stack(
        [
            np.frombuffer(name.ljust(strlen)[:strlen].encode(), dtype="S1")
            for name in coil_names
        ]
    )

    ds = xr.Dataset(
        {
            "b_n": (("i", "m"), _complex_pair(b_n), {"units": "Tesla"}),
            "xi_n": (("i", "m"), _complex_pair(xi_n), {"units": "m"}),
            "b_n_fun": (("i", "theta"), _complex_pair(b_n_fun), {"units": "Tesla"}),
            "Phi": (("i", "m"), _complex_pair(b_n * 2.0), {"units": "Wb"}),
            "Phi_x": (("i", "m"), _complex_pair(b_n * 0.5), {"units": "Wb"}),
            "Phi_coil": (
                ("i", "coil_index", "m"),
                np.stack([phi_coil.real, phi_coil.imag]),
                {"units": "Wb"},
            ),
            "coil_name": (("coil_index", "coil_strlen"), name_chars),
            "W_e_eigenvalue": (("i", "mode"), _complex_pair(np.linspace(1.0, 2.0, m_count))),
            "R": (("theta",), 0.4 + 0.2 * np.cos(2 * np.pi * theta), {"units": "m"}),
            "z": (("theta",), 0.3 * np.sin(2 * np.pi * theta), {"units": "m"}),
            "q_rational": (("psi_n_rational",), np.array([2.0, 3.0])),
        },
        coords={
            "i": [0, 1],
            "m": m,
            "mode": np.arange(1, m_count + 1),
            "theta": theta,
            "psi_n_rational": np.array([0.5, 0.8]),
            "coil_index": np.arange(len(coil_names)),
        },
        attrs={
            "title": "GPEC outputs",
            "jacobian": "hamada",
            "helicity": -1,
            "machine": "VEST",
            "shot": 0,
            "time": 0,
            "n": n,
            "version": "v1.5.5-test",
            "energy_vacuum": energy_vacuum,
            "energy_surface": energy_surface,
            "energy_plasma": energy_plasma,
        },
    )
    mode_label = filename_mode if filename_mode is not None else n
    target = path / f"gpec_control_output_n{mode_label}.nc"
    ds.to_netcdf(target)
    return {"b_n": b_n, "xi_n": xi_n, "b_n_fun": b_n_fun, "phi_coil": phi_coil}


def write_cylindrical_nc(path, *, n=1, nr=7, nz=5):
    """Write a miniature ``gpec_cylindrical_output_n<mode>.nc``; returns its data."""
    R = np.linspace(0.1, 0.9, nr)
    z = np.linspace(-1.0, 1.0, nz)
    zz, rr = np.meshgrid(z, R, indexing="ij")
    b_plasma = (rr + 1j * zz) * 1e-3
    b_total = b_plasma + (0.5 - 0.25j) * 1e-3
    ds = xr.Dataset(
        {
            "l": (("z", "R"), np.ones((nz, nr))),
            "b_r_equil": (("z", "R"), rr * 0.1, {"units": "Tesla"}),
            "b_z_equil": (("z", "R"), zz * 0.1, {"units": "Tesla"}),
            "b_t_equil": (("z", "R"), 0.3 / rr, {"units": "Tesla"}),
            "b_r": (("i", "z", "R"), _complex_pair(b_total), {"units": "Tesla"}),
            "b_z": (("i", "z", "R"), _complex_pair(b_total * 2.0), {"units": "Tesla"}),
            "b_t": (("i", "z", "R"), _complex_pair(b_total * 3.0), {"units": "Tesla"}),
            "b_r_plasma": (("i", "z", "R"), _complex_pair(b_plasma), {"units": "Tesla"}),
            "b_z_plasma": (("i", "z", "R"), _complex_pair(b_plasma * 2.0), {"units": "Tesla"}),
            "b_t_plasma": (("i", "z", "R"), _complex_pair(b_plasma * 3.0), {"units": "Tesla"}),
            "xi_r": (("i", "z", "R"), _complex_pair(b_plasma * 10.0), {"units": "m"}),
        },
        coords={"i": [0, 1], "R": ("R", R, {"units": "m"}), "z": ("z", z, {"units": "m"})},
        attrs={
            "title": "GPEC outputs",
            "machine": "VEST",
            "shot": 0,
            "time": 0,
            "n": n,
            "version": "v1.5.5-test",
        },
    )
    ds.to_netcdf(path / f"gpec_cylindrical_output_n{n}.nc")
    return {"R": R, "z": z, "b_plasma": b_plasma, "b_total": b_total}
