"""Synthetic #1178 operating-space results, for the plots (#1886) and their tests.

Every value here is synthetic and the provenance says so: these exercise the
schema and the drawing, never physics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaft.process.coil_operating_space import COIL_EXCITATION_CONVENTION, CoilOperatingSpace

GROUPS = ("upper", "middle", "lower")


def synthetic_operating_space(*, kind: str = "regular", points: int = 12, seed: int = 1886) -> CoilOperatingSpace:
    """A three-group result with a resonant metric and a signed NTV torque.

    ``kind="regular"`` is a ``points x points`` grid over the relative phases
    of groups 1 and 2 at unit amplitudes; ``"irregular"`` is random amplitudes
    and phases.  One sample is at ``A_0 = 0`` (relative phase undefined) and
    the torque carries one ``failed`` and one ``not_computed`` row.
    """
    rng = np.random.default_rng(seed)
    if kind == "regular":
        axis = np.linspace(0.0, 2 * np.pi, points, endpoint=False)
        p1, p2 = (grid.ravel() for grid in np.meshgrid(axis, axis, indexing="ij"))
        amplitudes = np.ones((p1.size, 3))
        phases = np.column_stack([np.zeros(p1.size), p1, p2])
        grid = {"kind": "regular", "axes": {"phase_rel_middle": axis, "phase_rel_lower": axis}}
    else:
        amplitudes = rng.uniform(0.2, 1.5, size=(points * points, 3))
        phases = rng.uniform(0.0, 2 * np.pi, size=(points * points, 3))
        grid = {"kind": "irregular"}
    amplitudes[-1, 0] = 0.0
    ids = [f"s{i:04d}" for i in range(amplitudes.shape[0])]
    samples = pd.DataFrame({"sample_id": ids, "role": "scan"})
    for index, group in enumerate(GROUPS):
        samples[f"amplitude_{group}"] = amplitudes[:, index]
        samples[f"phase_{group}"] = phases[:, index]
    samples.loc[0, "role"] = "reference"

    c = amplitudes * np.exp(1j * phases)
    q = np.array([[1.0, 0.4 + 0.3j, -0.2j], [0.4 - 0.3j, -0.5, 0.1], [0.2j, 0.1, 0.8]])
    torque = np.real(np.einsum("ni,ij,nj->n", c.conj(), q, c))
    resonant = np.abs(c @ np.array([1.0, 0.6 * np.exp(0.7j), 0.3])) * 1e-4
    rows = []
    for i, sample in enumerate(ids):
        rows.append({"sample_id": sample, "metric": "phi_res_rms_edge", "value": resonant[i], "unit": "T",
                     "status": "valid"})
        rows.append({"sample_id": sample, "metric": "ntv_torque", "value": torque[i], "unit": "N m",
                     "status": "valid"})
    values = pd.DataFrame(rows)
    for sample, status in ((ids[1], "failed"), (ids[2], "not_computed")):
        mask = (values["sample_id"] == sample) & (values["metric"] == "ntv_torque")
        values.loc[mask, ["value", "status"]] = [np.nan, status]
    return CoilOperatingSpace(
        n=1,
        groups=GROUPS,
        samples=samples,
        values=values,
        convention={**COIL_EXCITATION_CONVENTION, "frame": "synthetic: no machine frame"},
        grid=grid,
        provenance={
            "phi_res_rms_edge": {"synthetic": True, "field": "Phi_res", "window": "edge", "reduction": "rms"},
            "ntv_torque": {"synthetic": True, "method": "none", "psi_n": 1.0, "kinetic": "none"},
        },
    )
