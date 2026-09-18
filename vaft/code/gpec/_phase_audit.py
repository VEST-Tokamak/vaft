"""The phase audit of a GPEC coil field: is the stored harmonic conjugated?

GPEC writes its cylindrical field harmonics as complex numbers and says
nowhere which reconstruction they belong to.  ``Re(C e^{-i n phi})`` and
``Re(C e^{+i n phi})`` are different fields, and a consumer that picks the
wrong one traces a perturbation with its toroidal phase mirrored -- islands in
the wrong place rather than islands of the wrong size.  Decision D-06 makes
resolving that an audit rather than an assumption: the coil field is a vacuum
field, so it can be recomputed from the coil geometry by Biot-Savart and the
two hypotheses measured against it.

The measurement itself is machine- and code-independent and lives in
:func:`vaft.process.perturbation.toroidal_phase_audit`; this module is the
part that knows where GPEC keeps the two files.  The verdict -- how much
separation counts as settled -- is in :mod:`vaft.validation.perturbation_phase`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # the package's modules reach vaft.process lazily
    from vaft.process.perturbation import ToroidalPhaseAudit

__all__ = [
    "BrzphiHarmonics",
    "CoilPhaseAudit",
    "audit_coil_field_phase",
    "read_brzphi_harmonics",
]

#: Column positions of the three complex components in a ``*BRZPHI`` data row,
#: whose nine entries are ``l r z re(b_r) im(b_r) re(b_z) im(b_z) re(b_phi)
#: im(b_phi)``.
_COMPONENT_COLUMNS = ((3, 4), (5, 6), (7, 8))

#: What the three columns are, in file order.
BRZPHI_COMPONENTS: tuple[str, ...] = ("b_r", "b_z", "b_phi")

#: How GPEC's multi-mode writer declares which modes a file superposes.
_MODES_NOTE = re.compile(r"^\s*modes\s*=\s*([0-9,\s]+)$", re.IGNORECASE)


@dataclass(frozen=True)
class BrzphiHarmonics:
    """One ``*BRZPHI`` file's complex field on its (R, Z) grid."""

    points_rz: np.ndarray
    """``(P, 2)`` of major radius and height [m]."""
    values: np.ndarray
    """``(P, 3)`` complex ``(b_r, b_z, b_phi)`` [T]."""
    region: np.ndarray
    """The ``l`` column of each kept row [-]."""
    n_tor: int | None
    """The single toroidal mode this file is for, or ``None`` when it is not one [-]."""
    modes: tuple[int, ...]
    """Every mode the file declares; one entry for an ordinary single-mode file [-]."""
    path: str


def read_brzphi_harmonics(path: str | Path, *, region: int | None = None) -> BrzphiHarmonics:
    """Read a ``gpec_*brzphi_n*.out`` as points and complex field values.

    ``l`` marks where each grid point sits relative to the plasma: GPEC sets it
    to 1 inside the boundary, 2 inside the innermost computed surface, and
    leaves it 0 in the vacuum region (``gpeq.f::gpeq_rzgrid``).  It is a label,
    not a validity flag -- a coil field is a vacuum field on the whole grid --
    so the default keeps every row.  ``region`` restricts to one label for a
    caller that wants the legacy's sampling, which took ``l == 1``.

    A multi-mode file's ``n`` header is **not a mode number**: GPEC's multi-mode
    writer prints ``n = 123`` for a file superposing modes 1, 2 and 3, and
    declares the real list in a ``modes = 1,2,3`` note above it.  The note is
    what ``modes`` reports; ``n_tor`` is filled in only when the file is for one
    mode, so nothing downstream can read 123 as a toroidal mode number.

    Parameters
    ----------
    path : str or Path
        The ``.out`` file.
    region : int, optional
        Keep only rows whose ``l`` equals this.  ``None``, the default, keeps
        all of them [-].

    Raises
    ------
    ValueError
        The file has no data rows, or none in the requested region.
    """
    from ._ascii_output import read_gpec_ascii

    output = read_gpec_ascii(path)
    tables = [table for section in output.sections for table in section.tables]
    rows = [table.data for table in tables if table.data.shape[1] == 9]
    if not rows:
        raise ValueError(
            f"{path} carries no nine-column BRZPHI table; its columns are "
            f"{[table.columns for table in tables]}"
        )
    data = np.vstack(rows)
    if region is not None:
        data = data[np.isclose(data[:, 0], float(region))]
        if not data.size:
            raise ValueError(f"{path} has no rows with l = {region}")
    values = np.stack(
        [data[:, real] + 1j * data[:, imaginary] for real, imaginary in _COMPONENT_COLUMNS],
        axis=1,
    )
    modes: tuple[int, ...] = ()
    for note in output.notes:
        match = _MODES_NOTE.match(note)
        if match:
            modes = tuple(int(token) for token in match.group(1).replace(",", " ").split())
            break
    declared = output.attrs.get("n")
    if not modes and declared is not None:
        modes = (int(declared),)
    return BrzphiHarmonics(
        points_rz=data[:, 1:3].copy(),
        values=values,
        region=data[:, 0].copy(),
        n_tor=int(modes[0]) if len(modes) == 1 else None,
        modes=modes,
        path=str(path),
    )


@dataclass(frozen=True)
class CoilPhaseAudit:
    """A :class:`ToroidalPhaseAudit` with the GPEC files it was measured on."""

    audit: "ToroidalPhaseAudit"
    n_tor: int
    coil_in: str
    coil_brzphi: str
    machine: str
    coil_names: tuple[str, ...]
    source_files: tuple[str, ...]


def audit_coil_field_phase(
    coil_in: str | Path,
    coil_brzphi: str | Path,
    *,
    n_tor: int | None = None,
    coil_data_dir: str | Path | None = None,
    sample_points: int = 12,
    phi_samples: int = 48,
    region: int | None = None,
) -> CoilPhaseAudit:
    """Audit a GPEC coil-field harmonic against Biot-Savart on its own coils.

    Parameters
    ----------
    coil_in : str or Path
        The run's ``coil.in``, which names the coil sets and their currents.
    coil_brzphi : str or Path
        The matching ``gpec_cbrzphi_n*.out``.
    n_tor : int, optional
        The toroidal mode number.  Defaults to the one the ``.out`` header
        declares; when both are given and disagree this raises rather than
        picking one [-].
    coil_data_dir : str or Path, optional
        Where the ``<machine>_<set>.dat`` geometry files are, when the
        ``data_dir`` written into ``coil.in`` does not exist here.
    sample_points : int, optional
        How many grid points to evaluate, spread evenly through the file's rows.
        The cost is ``sample_points * phi_samples`` Biot-Savart evaluations
        against every filament segment [-].
    phi_samples : int, optional
        How many toroidal angles to evaluate at.  Raised to ``8*|n|`` and to 16
        when either is larger, so the projection resolves the mode [-].
    region : int, optional
        Restrict the sampled rows to one value of the file's ``l`` column; see
        :func:`read_brzphi_harmonics` [-].

    Returns
    -------
    CoilPhaseAudit
        The measurement and the files it came from.

    Raises
    ------
    ValueError
        The mode number is neither given nor declared, the two disagree, or the
        ``coil.in`` carries no energised coil.
    FileNotFoundError
        A ``.dat`` file the ``coil.in`` names is not there.

    Notes
    -----
    The sampled points are spread through the file's rows by index, which
    follows GPEC's ``R`` then ``Z`` loop order, so they walk the grid in
    columns rather than clustering.  Which points are used barely matters: on
    the DIII-D ideal example the stored-hypothesis relative norm is 1.1e-8 over
    points inside the plasma and 1.3e-8 over points in the vacuum region,
    because the coil field is a vacuum field on the whole grid.

    ``phi_samples`` does matter.  A discrete coil set has content at every
    toroidal harmonic and a sampled projection folds the ones above
    ``phi_samples/2`` into the one being measured, so the residual has a floor
    set by the sampling rather than by the phase convention.  On a real run
    with six sectors that floor is far below anything a verdict cares about --
    1e-8 at 48 samples -- but on a coarse set it is visible, and two
    projections taken over different ``phi_samples`` are not the same number.

    ``coil.in``'s ``ip_direction`` and ``bt_direction`` words are **not**
    applied: the sampling angle here is the plain counter-clockwise machine
    angle, and the agreement measured on real runs says that is the frame GPEC
    writes the coil field in.  Both runs it was measured on carry
    ``ip_direction="positive"`` with ``bt_direction="negative"``, so a run
    with a different pair has not been checked and would be the thing to look
    at first if an audit of one came back conjugated.
    """
    from vaft.process.coils_non_axisymmetric import biot_savart_filaments
    from vaft.process.perturbation import toroidal_phase_audit

    from ._coil_input import coil_filaments_from_coil_in

    harmonics = read_brzphi_harmonics(coil_brzphi, region=region)
    if len(harmonics.modes) > 1:
        # Its columns hold a superposition, so no single-mode projection of the
        # coil field is the thing they should be compared against.  Its "n"
        # header is the mode list run together, not a mode.
        raise ValueError(
            f"{coil_brzphi} superposes modes {harmonics.modes}; audit the "
            "per-mode files instead, there is no one mode for this one to be in"
        )
    if n_tor is None:
        n_tor = harmonics.n_tor
    elif harmonics.n_tor is not None and int(n_tor) != harmonics.n_tor:
        raise ValueError(
            f"{coil_brzphi} declares n = {harmonics.n_tor} but n_tor = {n_tor} was passed"
        )
    if n_tor is None:
        raise ValueError(
            f"{coil_brzphi} declares no toroidal mode number; pass n_tor explicitly"
        )
    n_tor = int(n_tor)

    total = harmonics.points_rz.shape[0]
    count = min(max(1, int(sample_points)), total)
    indices = np.unique(np.linspace(0, total - 1, count, dtype=int))
    points = harmonics.points_rz[indices]
    stored = harmonics.values[indices]

    samples = max(int(phi_samples), 8 * abs(n_tor), 16)
    phi = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    r = np.repeat(points[:, 0], samples)
    z = np.repeat(points[:, 1], samples)
    phi_flat = np.tile(phi, points.shape[0])
    probes = np.column_stack((r * np.cos(phi_flat), r * np.sin(phi_flat), z))

    filaments = coil_filaments_from_coil_in(coil_in, coil_data_dir=coil_data_dir)
    field_xyz = biot_savart_filaments(
        filaments.loops_xyz, filaments.currents_a, probes
    )
    # Cartesian to cylindrical, in the file's own (b_r, b_z, b_phi) order.
    cylindrical = np.column_stack(
        (
            field_xyz[:, 0] * np.cos(phi_flat) + field_xyz[:, 1] * np.sin(phi_flat),
            field_xyz[:, 2],
            -field_xyz[:, 0] * np.sin(phi_flat) + field_xyz[:, 1] * np.cos(phi_flat),
        )
    ).reshape(points.shape[0], samples, 3)

    audit = toroidal_phase_audit(stored, cylindrical, phi, n_tor=n_tor)
    return CoilPhaseAudit(
        audit=audit,
        n_tor=n_tor,
        coil_in=str(coil_in),
        coil_brzphi=str(coil_brzphi),
        machine=filaments.machine,
        coil_names=filaments.coil_names,
        source_files=tuple(str(item) for item in filaments.source_files),
    )
