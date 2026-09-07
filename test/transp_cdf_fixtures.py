"""Synthetic TRANSP output files, shaped like a real ``<runid>.CDF``.

The layout mirrors a MAST run (45453X01, TRANSP output, 1884 variables): a
``TIME`` axis for the scalars and a ``TIME3`` axis for the profiles carrying
the same values, zone centres on ``X`` and zone outer boundaries on ``XB``,
and CGS units declared on each variable.

The awkward details are deliberate, and each was found in the real file rather
than imagined:

* ``X`` and ``XB`` are **the same length**, so a reader that resolves a grid
  by counting cannot tell them apart -- and they interleave, ``X`` inside
  ``XB`` zone by zone;
* the grids are declared ``(TIME3, X)``, i.e. time-varying, though their
  values do not change;
* units are CGS and are declared per variable (``N/CM**3``, ``Nt-M/CM3``,
  ``CM**3``), which is the only place they are stated;
* ``PLFLX`` is in ``Wb/rad`` and measured from the axis, with ``PLFLXA``
  carrying the enclosed flux separately, so normalized flux is their ratio;
* ``PLFLX2PI`` is the same flux in webers, exactly ``2*pi`` times ``PLFLX``;
* ``TQTOTNB`` is present, dimensionless and zero -- an empty placeholder.

Nothing here is copied from a real run: the numbers are synthetic.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

#: TRANSP declares CGS on every variable it writes; nothing infers them.
UNITS = {
    "TIME": "SECONDS",
    "TIME3": "SECONDS",
    "NE": "N/CM**3",
    "NI": "N/CM**3",
    "TE": "EV",
    "TI": "EV",
    "OMEGA": "RAD/SEC",
    "DVOL": "CM**3",
    "VRPOT": "VOLTS",
    "PLFLX": "Wb/rad",
    "PLFLX2PI": "WEBERS",
    "PLFLXA": "Wb/rad",
    "TQIN": "Nt-M/CM3",
}


def write_transp_cdf(
    path,
    *,
    runid="90001A01",
    times=(0.10, 0.20, 0.30),
    zones=6,
    torque_density=-2.0e-8,
    zone_volume=1.0e4,
    flux_edge=0.05,
    plflx2pi_factor=2.0 * np.pi,
    include_placeholder=True,
):
    """Write a miniature ``<runid>.CDF``; returns the ground-truth arrays."""
    time = np.asarray(times, dtype="float32")
    # Zone centres inside their own outer boundaries, same length -- as the
    # real file has them.
    edges = np.linspace(0.0, 1.0, zones + 1)[1:]
    centres = edges - 0.5 / zones
    x = np.tile(centres, (time.size, 1)).astype("float32")
    xb = np.tile(edges, (time.size, 1)).astype("float32")

    flux = np.tile(edges * flux_edge, (time.size, 1)).astype("float32")
    edge_flux = np.full(time.size, flux_edge, dtype="float32")
    density = np.tile(np.linspace(4.0e13, 4.0e12, zones), (time.size, 1)).astype("float32")
    temperature = np.tile(np.linspace(1.2e3, 1.0e2, zones), (time.size, 1)).astype("float32")
    omega = np.tile(np.linspace(-1.2e4, -2.0e3, zones), (time.size, 1)).astype("float32")
    potential = np.tile(np.linspace(0.0, -80.0, zones), (time.size, 1)).astype("float32")
    torque = np.full((time.size, zones), torque_density, dtype="float32")
    volume = np.full((time.size, zones), zone_volume, dtype="float32")

    data = {
        "X": (("TIME3", "X"), x, {"units": ""}),
        "XB": (("TIME3", "XB"), xb, {"units": ""}),
        "NE": (("TIME3", "X"), density, {"units": UNITS["NE"], "long_name": "ELECTRON DENSITY"}),
        "TE": (("TIME3", "X"), temperature, {"units": UNITS["TE"]}),
        "TI": (("TIME3", "X"), temperature * 1.1, {"units": UNITS["TI"]}),
        "OMEGA": (("TIME3", "X"), omega, {"units": UNITS["OMEGA"]}),
        "DVOL": (("TIME3", "X"), volume, {"units": UNITS["DVOL"]}),
        "TQIN": (("TIME3", "X"), torque, {"units": UNITS["TQIN"], "long_name": "TOTAL INPUT TORQUE"}),
        # Present in the file, deliberately not in VARIABLE_DESCRIPTIONS: a
        # reader must be able to describe it from what the file says.
        "Q": (("TIME3", "X"), np.tile(np.linspace(1.0, 5.0, zones), (time.size, 1)).astype("float32"),
              {"units": "", "long_name": "SAFETY FACTOR"}),
        "VRPOT": (("TIME3", "XB"), potential, {"units": UNITS["VRPOT"]}),
        "PLFLX": (("TIME3", "XB"), flux, {"units": UNITS["PLFLX"]}),
        "PLFLX2PI": (("TIME3", "XB"), (flux * plflx2pi_factor).astype("float32"), {"units": UNITS["PLFLX2PI"]}),
        "PLFLXA": (("TIME3",), edge_flux, {"units": UNITS["PLFLXA"]}),
    }
    if include_placeholder:
        # Dimensionless and zero, exactly as the real file writes it.
        data["TQTOTNB"] = ((), np.float32(0.0), {"units": UNITS["TQIN"]})

    dataset = xr.Dataset(
        data,
        coords={
            "TIME": ("TIME", time, {"units": UNITS["TIME"]}),
            "TIME3": ("TIME3", time, {"units": UNITS["TIME3"]}),
        },
        attrs={"title": "synthetic TRANSP output"},
    )
    target = path / f"{runid}.CDF"
    dataset.to_netcdf(target, format="NETCDF3_CLASSIC")
    return {
        "path": target,
        "runid": runid,
        "time": time,
        "x": centres,
        "xb": edges,
        "n_e": density,
        "T_e": temperature,
        "omega": omega,
        "plflx": flux,
        "plflxa": edge_flux,
        "torque_density": torque,
        "zone_volume": volume,
        "zones": zones,
    }
