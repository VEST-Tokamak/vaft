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
  values do not change -- as they are, and do not, in the reference run;
* units are CGS and are declared per variable (``N/CM**3``, ``Nt-M/CM3``,
  ``CM**3``), which is the only place they are stated;
* ``PLFLX`` is in ``Wb/rad`` and measured from the axis, with ``PLFLXA``
  carrying the enclosed flux separately on the ``TIME`` axis -- so normalized
  flux is their ratio, and reading ``PLFLXA`` at the wrong sample is a
  mistake a fixture with a constant edge flux could not catch.  It rises
  through the run, as it does in the reference file (0.0272 to 0.0765);
* the flux profile is **not** proportional to ``XB``, so ``psi_norm`` and the
  radial coordinate are distinguishable -- in the reference run they are
  0.0049 against 0.05 at the innermost boundary;
* ``TQTOTNB`` is present and zero, carrying no radial or time axis, but
  declaring torque-density units all the same -- an empty placeholder, not a
  dimensionless flag;
* variables live on a third radial dimension (``RMAJM``) as well, on the
  scalar ``TIME`` axis alone, and as bare dimensionless scalars: 79, 474 and
  423 of the reference file's variables respectively.

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
    "BPHXB": "NT-M",
    "BDENS": "N/CM**3",
}


def write_transp_cdf(
    path,
    *,
    runid="90001A01",
    times=(0.10, 0.20, 0.30),
    times3=None,
    zones=6,
    torque_density=-2.0e-8,
    zone_volume=1.0e4,
    flux_edge=(0.030, 0.045, 0.050),
    plflx2pi_factor=2.0 * np.pi,
    include_placeholder=True,
    static_grids=False,
    boundary_points=None,
    reverse_xb_at=None,
    centres_outside=False,
    no_units_variable=False,
):
    """Write a miniature ``<runid>.CDF``; returns the ground-truth arrays.

    The keyword arguments exist to break one thing at a time: ``times3``
    writes a profile axis that disagrees with the scalar one, ``static_grids``
    writes ``X``/``XB`` without a time dimension, ``boundary_points`` gives
    ``XB`` a length of its own, ``reverse_xb_at`` makes one sample of the
    boundary grid run backwards, and ``centres_outside`` puts every zone
    centre outside its own boundary while leaving both grids increasing.
    """
    time = np.asarray(times, dtype="float32")
    time3 = np.asarray(times if times3 is None else times3, dtype="float32")
    # Zone centres inside their own outer boundaries, same length -- as the
    # real file has them.
    edges = np.linspace(0.0, 1.0, zones + 1)[1:]
    centres = edges + 0.5 / zones if centres_outside else edges - 0.5 / zones
    boundaries = edges if boundary_points is None else np.linspace(0.0, 1.0, boundary_points + 1)[1:]

    def profile(row, axis_size=None, dtype="float32"):
        return np.tile(row, ((time3.size if axis_size is None else axis_size), 1)).astype(dtype)

    x = centres.astype("float32") if static_grids else profile(centres)
    xb = boundaries.astype("float32") if static_grids else profile(boundaries)
    if reverse_xb_at is not None:
        assert not static_grids, "reverse_xb_at needs a time-varying XB"
        xb = xb.copy()
        xb[reverse_xb_at] = xb[reverse_xb_at][::-1]

    # Flux rises through the run and is not proportional to XB, so neither the
    # sample nor the coordinate can be mistaken for the other.
    scale = np.asarray(flux_edge, dtype="float64").reshape(-1)
    edge_flux = np.resize(scale, time.size).astype("float32")
    flux_shape = 0.3 * boundaries + 0.7 * boundaries**2
    flux = (np.resize(scale, time3.size)[:, None] * flux_shape[None, :]).astype("float32")

    density = profile(np.linspace(4.0e13, 4.0e12, zones))
    temperature = profile(np.linspace(1.2e3, 1.0e2, zones))
    omega = profile(np.linspace(-1.2e4, -2.0e3, zones))
    potential = profile(np.linspace(0.0, -80.0, boundaries.size))
    torque = np.full((time3.size, zones), torque_density, dtype="float32")
    volume = np.full((time3.size, zones), zone_volume, dtype="float32")
    # A third radial dimension, as 79 of the reference file's variables have.
    major_radius_points = 2 * zones + 1
    beam_density = profile(np.linspace(1.0e11, 0.0, major_radius_points))

    x_dims = ("X",) if static_grids else ("TIME3", "X")
    xb_dims = ("XB",) if static_grids else ("TIME3", "XB")
    data = {
        "X": (x_dims, x, {"units": ""}),
        "XB": (xb_dims, xb, {"units": ""}),
        "NE": (("TIME3", "X"), density, {"units": UNITS["NE"], "long_name": "ELECTRON DENSITY"}),
        "TE": (("TIME3", "X"), temperature, {"units": UNITS["TE"]}),
        "TI": (("TIME3", "X"), temperature * 1.1, {"units": UNITS["TI"]}),
        "OMEGA": (("TIME3", "X"), omega, {"units": UNITS["OMEGA"]}),
        "DVOL": (("TIME3", "X"), volume, {"units": UNITS["DVOL"]}),
        "TQIN": (("TIME3", "X"), torque, {"units": UNITS["TQIN"], "long_name": "TOTAL INPUT TORQUE"}),
        # Present in the file, deliberately not in VARIABLE_DESCRIPTIONS: a
        # reader must be able to describe it from what the file says.
        "Q": (("TIME3", "X"), profile(np.linspace(1.0, 5.0, zones)),
              {"units": "", "long_name": "SAFETY FACTOR"}),
        "VRPOT": (("TIME3", "XB"), potential, {"units": UNITS["VRPOT"]}),
        "PLFLX": (("TIME3", "XB"), flux, {"units": UNITS["PLFLX"]}),
        "PLFLX2PI": (("TIME3", "XB"), (flux * plflx2pi_factor).astype("float32"),
                     {"units": UNITS["PLFLX2PI"]}),
        # On TIME, as the reference run writes it -- not on the profile axis.
        "PLFLXA": (("TIME",), edge_flux, {"units": UNITS["PLFLXA"]}),
        # A scalar time series, and a variable on a third radial dimension.
        "BPHXB": (("TIME",), np.linspace(-0.5, -0.8, time.size).astype("float32"),
                  {"units": UNITS["BPHXB"], "long_name": "TOTAL PLASMA TORQUE"}),
        "BDENS": (("TIME3", "RMAJM"), beam_density,
                  {"units": UNITS["BDENS"], "long_name": "BEAM ION DENSITY VS MAJOR RADIUS"}),
        # A dimensionless scalar that is not a placeholder: 423 of the
        # reference file's variables are shaped like this.
        "NLTAUP": ((), np.float32(1.0), {"units": "", "long_name": "PARTICLE CONFINEMENT FLAG"}),
    }
    if no_units_variable:
        # Every variable in the reference run declares units, so this one is a
        # deliberate divergence: the reader's "no units attribute" default has
        # to be reachable from somewhere.
        data["NOUNITS"] = (("TIME3", "X"), profile(np.ones(zones)),
                           {"long_name": "A QUANTITY WITH NO DECLARED UNITS"})
    if include_placeholder:
        # Zero and without an axis, but declaring torque-density units, exactly
        # as the real file writes it.
        data["TQTOTNB"] = ((), np.float32(0.0), {"units": UNITS["TQIN"]})

    dataset = xr.Dataset(
        data,
        coords={
            "TIME": ("TIME", time, {"units": UNITS["TIME"]}),
            "TIME3": ("TIME3", time3, {"units": UNITS["TIME3"]}),
        },
        attrs={"title": "synthetic TRANSP output"},
    )
    target = path / f"{runid}.CDF"
    dataset.to_netcdf(target, format="NETCDF3_CLASSIC")
    return {
        "path": target,
        "runid": runid,
        "time": time,
        "time3": time3,
        "x": centres,
        "xb": boundaries,
        "n_e": density,
        "T_e": temperature,
        "omega": omega,
        "plflx": flux,
        "plflxa": edge_flux,
        "psi_norm": flux_shape,
        "beam_density": beam_density,
        "torque_density": torque,
        "zone_volume": volume,
        "zones": zones,
    }
