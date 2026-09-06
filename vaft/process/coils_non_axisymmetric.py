"""Kernels on non-axisymmetric coil sets: toroidal mode content and vacuum field.

The two operations every 3-D coil workflow repeats, on plain arrays, with the
sign conventions stated once.  Geometry objects live in
:mod:`vaft.machine_mapping.coils_non_axisymmetric_geometry`; the IMAS mapper in
:mod:`vaft.machine_mapping.coils_non_axisymmetric`; the GPEC ``coil.in``
serialization in :mod:`vaft.code.gpec`.  Nothing here reads an ODS or knows
which machine the coils belong to.

* :func:`toroidal_mode_decomposition` -- complex Fourier coefficient of a
  per-sector quantity (typically coil currents) for each toroidal mode
  number, ``C_n = <v_k exp(-i n phi_k)>``.
* :func:`biot_savart_filaments` -- vacuum magnetic field of closed polyline
  filaments carrying given currents, at arbitrary probe points.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "biot_savart_filaments",
    "toroidal_mode_decomposition",
]

MU0_OVER_4PI = 1.0e-7  # [T m / A], exact in the pre-2019 SI and within 2e-10 relative afterwards


def toroidal_mode_decomposition(
    phi_rad, values, modes: Iterable[int]
) -> Mapping[int, complex]:
    """Complex toroidal Fourier coefficient of a per-sector quantity.

    Parameters
    ----------
    phi_rad : array_like
        Toroidal angle of each sector sample, any order, shape ``(K,)`` [rad].
    values : array_like
        The quantity at each sector, real or complex, shape ``(K,)``; coil
        currents are the usual case [any].
    modes : iterable of int
        Toroidal mode numbers ``n`` to evaluate [-].

    Returns
    -------
    dict[int, complex]
        ``C_n`` for each requested ``n``, in the units of ``values`` [any].

    Raises
    ------
    ValueError
        ``phi_rad`` and ``values`` differ in length or are not one-dimensional.

    Convention
    ----------
    ``C_n = (1/K) sum_k v_k exp(-i n phi_k)``, the plain sample mean, so a
    pattern ``v_k = A cos(n phi_k + delta)`` returns ``C_n = (A/2) exp(+i delta)``
    when the sectors resolve ``n`` without aliasing: the peak amplitude is
    ``2 |C_n|`` and the phase of ``C_n`` is the pattern's phase ``delta`` at
    ``phi = 0``.  The ``exp(-i n phi)`` kernel is the one GPEC's ``coil.in``
    documentation and the legacy hsyun_GPEC ``fourier_mode_coefficient``
    use; a ``exp(+i n phi)`` convention conjugates every coefficient.  No
    normalisation by ``2/K`` is applied, so ``C_0`` is the mean.

    Assumptions
    -----------
    Uniform weighting of the sectors.  With ``K`` equally spaced sectors the
    coefficients are exact for ``|n| < K/2``; with fewer or unevenly spaced
    sectors they are the least-squares-free sample projection and alias.

    Applicability
    -------------
    Machine-independent.  Any toroidal array of discrete elements: RMP coil
    rows, error-field correction coils, saddle loops.

    Provenance
    ----------
    .. [GPEC] GPEC ``coil.in`` documentation: per-coil currents are
       combined per toroidal harmonic with ``exp(-i n phi)``.
    .. [legacy] hsyun_GPEC ``library/gpec_input_processor.py``
       ``fourier_mode_coefficient`` (same kernel), ported unchanged.
    """
    phi = np.asarray(phi_rad, dtype=float)
    v = np.asarray(values)
    if phi.ndim != 1 or v.ndim != 1 or phi.shape != v.shape:
        raise ValueError(
            f"phi_rad and values must be one-dimensional and equal in length, got {phi.shape} and {v.shape}"
        )
    return {int(n): complex(np.mean(v * np.exp(-1j * int(n) * phi))) for n in modes}


def biot_savart_filaments(points_xyz, currents_a, probes_xyz) -> np.ndarray:
    """Vacuum magnetic field of closed polyline filaments at probe points.

    Parameters
    ----------
    points_xyz : array_like
        Filament vertices, shape ``(F, P, 3)`` or a sequence of ``F`` arrays
        ``(P_f, 3)``; each filament is traversed in vertex order and must be
        closed (first vertex equals last) [m].
    currents_a : array_like
        Current in each filament, positive along the vertex order, shape
        ``(F,)``; multiply by the winding turns first when a filament stands
        for a multi-turn coil [A].
    probes_xyz : array_like
        Observation points, shape ``(N, 3)`` [m].

    Returns
    -------
    np.ndarray
        Cartesian field ``(B_x, B_y, B_z)`` at each probe, shape ``(N, 3)`` [T].

    Raises
    ------
    ValueError
        A filament is not closed, or the number of currents differs from the
        number of filaments.

    Convention
    ----------
    Right-handed Cartesian axes.  Each straight segment contributes
    ``mu0/(4 pi) I (dl x r) / |r|^3`` with ``dl`` along the vertex order and
    ``r`` from the segment midpoint to the probe (midpoint rule), so the
    field is the one the right-hand rule assigns to a current flowing in the
    vertex order.  A coil traversed the other way, or a negative current,
    flips the sign.

    Assumptions
    -----------
    Filaments are thin (no conductor cross-section); the midpoint rule is
    second-order accurate in segment length and undefined at a probe lying
    on a segment midpoint.

    Applicability
    -------------
    Machine-independent.  Any set of polyline coils: RMP, error-field
    correction, PF coils sampled as polygons.

    Limitations
    -----------
    ``O(F P N)`` memory when ``points_xyz`` is passed as one array; loop over
    probe blocks for very large arrays.

    Provenance
    ----------
    .. [Jackson] J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Sec. 5.2:
       the Biot-Savart law for a line current.
    .. [legacy] hsyun_GPEC ``library/gpec_input_processor.py`` and
       ``library/mast_u_input_convertor.py`` carried three copies of this
       midpoint-rule kernel; this is the single replacement.
    """
    if isinstance(points_xyz, np.ndarray) and points_xyz.ndim == 3:
        filaments: Sequence[np.ndarray] = [points_xyz[k] for k in range(points_xyz.shape[0])]
    else:
        filaments = [np.asarray(f, dtype=float) for f in points_xyz]
    currents = np.asarray(currents_a, dtype=float).reshape(-1)
    probes = np.asarray(probes_xyz, dtype=float)
    if probes.ndim != 2 or probes.shape[1] != 3:
        raise ValueError(f"probes_xyz must be (N, 3), got {probes.shape}")
    if currents.shape[0] != len(filaments):
        raise ValueError(f"{currents.shape[0]} currents for {len(filaments)} filaments")
    field = np.zeros((probes.shape[0], 3), dtype=float)
    for k, pts in enumerate(filaments):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"filament {k} must be (P, 3), got {pts.shape}")
        if not np.allclose(pts[0], pts[-1]):
            raise ValueError(f"filament {k} is not closed (first vertex != last vertex)")
        if currents[k] == 0.0:
            continue
        dl = pts[1:] - pts[:-1]  # (S, 3)
        mid = 0.5 * (pts[1:] + pts[:-1])  # (S, 3)
        r = probes[:, None, :] - mid[None, :, :]  # (N, S, 3)
        r3 = np.linalg.norm(r, axis=2) ** 3  # (N, S)
        cross = np.cross(dl[None, :, :], r)  # (N, S, 3)
        field += MU0_OVER_4PI * currents[k] * np.sum(cross / r3[:, :, None], axis=1)
    return field
