"""Prescribed active-field correction for the pinned NICE filament model.

NICE caps coil count at 100 and discretizes small rectangles at 5 cm even
when finer spacing is requested. Correct the observation operator with the
exact ODS winding response: y_native = y - (B_exact - B_native). This does
not fit currents or change diagnostic uncertainties.
"""
from dataclasses import replace
import numpy as np

from .geometry import _element_polygon


def active_response(ods, geometry, diagnostics, time):
    from vaft.process.electromagnetics import compute_point_response_matrices
    from vaft.formula.magnetics import project_poloidal_field

    currents = np.array(
        [
            np.interp(
                time, ods["pf_active.time"], ods[f"pf_active.coil.{i}.current.data"]
            )
            for i in range(len(ods["pf_active.coil"]))
        ]
    )
    exact, native = [], []
    for i in range(len(currents)):
        for j in range(len(ods[f"pf_active.coil.{i}.element"])):
            b = f"pf_active.coil.{i}.element.{j}"
            r, z = np.mean(_element_polygon(ods, b), axis=0)
            exact.append((r, z, float(ods[b + ".turns_with_sign"]) * currents[i]))
    for coil in geometry["pf_active"]:
        p = np.asarray(coil["outline"])
        lo, hi = p.min(axis=0), p.max(axis=0)
        span = hi - lo
        spacing = 0.05 if np.prod(span) < 0.01 else 0.1
        nr, nz = np.floor(span / spacing).astype(int) + 1
        for ir in range(nr):
            for iz in range(nz):
                r, z = lo + span * [(ir + 0.5) / nr, (iz + 0.5) / nz]
                native.append(
                    (r, z, coil["turns"] * currents[coil["coil_index"]] / (nr * nz))
                )

    def evaluate(sources, d, regularized=False):
        sources = np.asarray(sources)
        positions = (
            [[d.geometry["r"], d.geometry["z"]]]
            if d.family == "bpol_probe"
            else [[p["r"], p["z"]] for p in d.geometry["positions"]]
        )
        vals = []
        for r, z in positions:
            # NICE Filament clamps observation points within d_filament=.01.
            if regularized and np.any(
                np.hypot(r - sources[:, 0], z - sources[:, 1]) < 0.01
            ):
                raise ValueError(
                    "Active response correction requires sensors at least d_filament=.01 m from native filaments"
                )
            psi, bz, br = compute_point_response_matrices(
                [r],
                [z],
                sources[:, 0],
                sources[:, 1],
                turns=np.ones(len(sources)),
                components=("psi", "bz", "br"),
            )
            response = (
                project_poloidal_field(br[0], bz[0], d.geometry["poloidal_angle"])
                if d.family == "bpol_probe"
                else psi[0]
            )
            vals.append(float(response @ sources[:, 2]))
        if len(vals) != 1:
            raise ValueError(
                "Active response correction currently supports single-position flux loops"
            )
        return vals[0]

    corrected, audit = [], []
    for d in diagnostics:
        if d.enabled and d.family in ("bpol_probe", "flux_loop"):
            e, n = evaluate(exact, d), evaluate(native, d, True)
            delta = e - n
            corrected.append(
                replace(d, value=d.value - delta, active_response_correction=delta)
            )
            audit.append(
                {
                    "ods_path": d.ods_path,
                    "family": d.family,
                    "exact": e,
                    "native": n,
                    "correction": delta,
                    "uncertainty": d.uncertainty,
                }
            )
        else:
            corrected.append(d)
    return tuple(corrected), audit
