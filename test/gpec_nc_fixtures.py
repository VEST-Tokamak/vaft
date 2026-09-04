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


#: DCON's own equilibrium-summary globals, in the shape `equil/equil_out.f`
#: computes them.  Values are physically plausible for VEST rather than
#: arbitrary, so a test that reads one back is legible.
DEFAULT_DCON_EQUILIBRIUM = {
    "amean": 0.25, "rmean": 0.45, "aratio": 1.8, "kappa": 1.6,
    "delta1": 0.3, "delta2": 0.25,
    "ro": 0.44, "zo": 0.01, "psio": 0.012, "psilow": 0.01,
    "q0": 1.05, "qmin": 1.02, "qmax": 8.4, "qa": 8.6, "q95": 6.2,
    "crnt": 0.09, "bt0": 0.18,
    "betat": 0.02, "betan": 1.4,
    "betap1": 0.5, "betap2": 0.55, "betap3": 0.6,
    "li1": 0.9, "li2": 0.95, "li3": 1.0,
}

DEFAULT_DCON_COORDINATES = {
    "jacobian": "hamada",
    "power_bp": 0.0, "power_b": 0.0, "power_r": 0.0,
    "mpsi": 128, "mtheta": 256,
}


def write_dcon_output_nc(
    path,
    *,
    n=1,
    mlow=-2,
    mhigh=0,
    w_t=-0.3,
    mode_labels=None,
    equilibrium=None,
    coordinates=None,
    profiles=False,
    edge_scan=False,
):
    """Write a miniature ``dcon_output_n<mode>.nc``; returns its data.

    Mirrors `dcon/dcon_netcdf.f`: complex values as a *trailing* length-2 ``i``
    dimension, the mode-range provenance and the equilibrium summary as global
    attributes, and `psi_n_edge`/`q_edge`/`dW_edge` present only when the run
    performed an edge scan (`size_edge > 0`).

    ``equilibrium``/``coordinates`` accept ``True`` for the default block,
    ``None``/``False`` for none at all, or a dict to override individual
    values -- so a test can assert on the "this file carries nothing" path
    without hand-rolling a second fixture.  ``w_t`` accepts a scalar or one
    value per mode.
    """
    mpert = mhigh - mlow + 1
    mode = np.asarray(mode_labels if mode_labels is not None else range(1, mpert + 1), dtype=int)
    w_t_values = np.atleast_1d(np.asarray(w_t, dtype=float))
    if w_t_values.size == 1:
        w_t_values = np.repeat(w_t_values, mode.size)
    psi_n = np.linspace(0.01, 0.99, 9)

    data = {
        "W_t_eigenvalue": (("mode", "i"), np.stack([w_t_values, np.zeros_like(w_t_values)], axis=-1)),
    }
    if profiles:
        data.update(
            {
                "f": (("psi_n",), 0.1 + 0.01 * psi_n),
                "mu0p": (("psi_n",), 0.02 * (1.0 - psi_n)),
                "dvdpsi": (("psi_n",), 1.0 + psi_n),
                "q": (("psi_n",), 1.05 + 7.0 * psi_n**2, {"long_name": "Safety Factor"}),
            }
        )
    coords = {"i": [0, 1], "mode": mode, "m": np.arange(mlow, mhigh + 1), "psi_n": psi_n}

    if edge_scan:
        q_edge = np.linspace(6.0, 8.0, 5)
        dw_edge = np.linspace(0.4, -0.6, 5)
        data["dW_edge"] = (
            ("psi_n_edge", "i"),
            np.stack([dw_edge, np.zeros_like(dw_edge)], axis=-1),
            {"long_name": "Least Stable Total Energy Eigenvalues"},
        )
        data["q_edge"] = (("psi_n_edge",), q_edge, {"long_name": "Safety Factor"})
        coords["psi_n_edge"] = np.linspace(0.95, 0.994, 5)

    attrs = {"mlow": mlow, "mhigh": mhigh, "mpert": mpert, "mband": 0, "n": n}
    if equilibrium:
        attrs.update(DEFAULT_DCON_EQUILIBRIUM)
        if isinstance(equilibrium, dict):
            attrs.update(equilibrium)
        attrs.setdefault("qlim", 8.0)
        attrs.setdefault("psilim", 0.994)
    if coordinates:
        attrs.update(DEFAULT_DCON_COORDINATES)
        if isinstance(coordinates, dict):
            attrs.update(coordinates)

    ds = xr.Dataset(data, coords=coords, attrs=attrs)
    target = path / f"dcon_output_n{n}.nc"
    ds.to_netcdf(target)
    return {"path": target, "psi_n": psi_n, "mode": mode, "attrs": attrs}
