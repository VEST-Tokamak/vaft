"""Axisymmetric electromagnetic response: what a current does elsewhere, and how the vessel answers.

Two halves.  The first builds free-space Green's-function response matrices --
the poloidal flux and field per unit current -- for active coils, passive
vessel loops and plasma filaments.  The second integrates the passive circuit
those matrices describe, giving the eddy currents a coil programme drives in
the vessel.

Physics only.  Nothing here reads a machine description: every geometry,
resistance and mutual-inductance matrix arrives as an argument, so the same
code serves any axisymmetric machine.  The VEST geometry and the packaged
coupling matrices live in :mod:`vaft.machine_mapping`.

Notation
--------
psi      : poloidal flux per unit source current            [Wb/A]
Br, Bz   : field components per unit source current          [T/A]
R        : loop resistance matrix, diagonal                  [ohm]
M        : passive-passive mutual inductance matrix            [H]
L        : passive-active mutual inductance matrix             [H]
dt_sub   : substep of the circuit integration                  [s]

Conventions
-----------
**Response matrices are per unit current**, so a field is recovered by
contracting one with a current history; nothing here carries a current of its
own.  Sources are ideal filaments unless a turn count says otherwise, and a
turn count multiplies the response rather than changing the geometry.

**The passive circuit is driven by the rate of change of the active current**,
not by the current itself, which is why a coil programme flat in time drives
no eddy current however large it is.

The singularity where an observation point coincides with its source is
handled by evaluating either side of it and averaging, rather than by
excluding the point; see :func:`compute_br_bz_phi`.

Provenance
----------
.. [1] :mod:`vaft.formula.green`, which supplies every Green's function used
   here, including the exact elliptic-integral forms.
.. [2] The legacy ``vfit_eddy`` workflow, whose matrix setup
   :func:`solve_eddy_currents` retains structural similarity to.
"""

from vaft.formula.green import (
    calculate_distance,
    green_br_bz,
    green_br_bz_exact,
    green_psi_exact,
    green_r,
)
from typing import List, Dict, Any, Tuple
import numpy as np
from numpy import ndarray


__all__ = [
    "calc_grid",
    "compute_br_bz_phi",
    "compute_impedance_matrices",
    "compute_mutual_passive_active",
    "compute_point_response_matrices",
    "compute_response_matrix",
    "compute_response_vector",
    "compute_vacuum_fields_1d",
    "solve_eddy_currents",
    "wall_propagator",
]
# from scipy.linalg import expm # 행렬 지수 함수 - EVD 방법으로 대체

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not found. Falling back to slower Python execution for solve_eddy_currents. Install Numba for performance.")


# Description of the axisymmetric mutual electromagnetics calculations.
def compute_br_bz_phi(
    r_obs: np.ndarray,
    z_obs: np.ndarray,
    r_src: float,
    z_src: float,
    shift: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Field and flux of one unit-current ring at a set of observation points.

    Parameters
    ----------
    r_obs : array_like
        Major radius of the observation points [m].
    z_obs : array_like
        Height of the observation points [m].
    r_src : float
        Major radius of the source ring [m].
    z_src : float
        Height of the source ring [m].
    shift : float, optional
        Offset used to step around the source singularity [m].

    Returns
    -------
    tuple of np.ndarray
        The radial and vertical field per unit current in tesla per ampere, and
        the poloidal flux per unit current in weber per ampere [-].

    Convention
    ----------
    Per unit current, so the caller multiplies by an actual current. An
    observation point sitting on the source has no finite response; rather than
    excluding it, the response is evaluated at two points either side and
    averaged, which keeps the returned array the same shape as the input and gives
    a finite value whose error is second order in the offset.

    Defaults
    --------
    ``shift = 0.01`` is a numerical convenience. The averaging is applied only
    within a third of it from the source, so a point comfortably away from the
    ring is untouched.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Free space: no iron, no shielding, no image currents. The source is an ideal
    filament, so the response very close to a real conductor of finite section is
    not physical, which is what the offset is papering over rather than solving.

    Provenance
    ----------
    .. [1] The axisymmetric Green's functions in :mod:`vaft.formula.green`.
    """
    distances = calculate_distance(r_obs, r_src, z_obs, z_src)
    
    condition = distances < (shift / 3.0)

    br1_shifted, bz1_shifted = green_br_bz(r_obs + shift, z_obs, r_src, z_src)
    br2_shifted, bz2_shifted = green_br_bz(r_obs - shift, z_obs, r_src, z_src)
    phi1_shifted = green_r(r_obs + shift, z_obs, r_src, z_src)
    phi2_shifted = green_r(r_obs - shift, z_obs, r_src, z_src)

    br_direct, bz_direct = green_br_bz(r_obs, z_obs, r_src, z_src)
    phi_direct = green_r(r_obs, z_obs, r_src, z_src)

    br_final = np.where(condition, (br1_shifted + br2_shifted) / 2.0, br_direct)
    bz_final = np.where(condition, (bz1_shifted + bz2_shifted) / 2.0, bz_direct)
    phi_final = np.where(condition, (phi1_shifted + phi2_shifted) / 2.0, phi_direct)
    
    return br_final, bz_final, phi_final

def calc_grid(
    xvar: List[float],
    zvar: List[float],
    coil_turns: List[List[float]],
    coil_r: List[List[float]],
    coil_z: List[List[float]],
    loop_geometry_type: List[int],
    loop_outline_r: List[List[float]],
    loop_outline_z: List[List[float]],
    loop_rectangle_r: List[float],
    loop_rectangle_z: List[float]
    ) -> Tuple[ndarray, ndarray, ndarray]:
    """Assemble the response of every coil and passive loop on a rectangular grid.

    Parameters
    ----------
    xvar : array_like
        Major-radius grid axis [m].
    zvar : array_like
        Height grid axis [m].
    coil_turns : array_like
        Turns of each active coil [-].
    coil_r : array_like
        Major radius of each coil [m].
    coil_z : array_like
        Height of each coil [m].
    loop_geometry_type : sequence
        Which shape each passive loop is described by [-].
    loop_outline_r : sequence
        Major radius of each outline-described loop's vertices [m].
    loop_outline_z : sequence
        Height of those vertices [m].
    loop_rectangle_r : sequence
        Major radius of each rectangle-described loop [m].
    loop_rectangle_z : sequence
        Height of those rectangles [m].

    Returns
    -------
    np.ndarray
        Response per unit current with one row per grid point and one column per
        source, coils first and passive loops after [-].

    Convention
    ----------
    Rows run over the flattened grid and columns over the sources, coils before
    loops, which is the ordering every consumer of this matrix assumes. A loop
    described by an outline is reduced to its centroid before its response is
    taken, so an extended loop is treated as a filament at that point.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The centroid reduction means a loop large compared with its distance to the
    grid is poorly represented. Builds the whole grid at once, so memory grows as
    the product of grid points and sources.

    Provenance
    ----------
    .. [1] :func:`compute_br_bz_phi`, evaluated per source and stacked.
    """
    nbcoil = len(coil_turns)
    nbloop = len(loop_geometry_type)
    total_points = len(xvar) * len(zvar)

    br_array = np.zeros((total_points, nbcoil + nbloop))
    bz_array = np.zeros((total_points, nbcoil + nbloop))
    phi_array = np.zeros((total_points, nbcoil + nbloop))

    count = 0
    for i, xr in enumerate(xvar):
        for j, zr in enumerate(zvar):
            if count % 100 == 0:
                percent = (count * 100.0) / (total_points - 1)
                print(f"{percent:.2f}%")

            # Active coils
            for ii in range(nbcoil):
                sum_br, sum_bz, sum_phi = 0.0, 0.0, 0.0

                for jj in range(len(coil_turns[ii])):
                    nbturns = coil_turns[ii][jj]
                    r2 = coil_r[ii][jj]
                    z2 = coil_z[ii][jj]
                    br_val, bz_val, phi_val = compute_br_bz_phi(xr, zr, r2, z2)
                    sum_br += br_val * nbturns
                    sum_bz += bz_val * nbturns
                    sum_phi += phi_val * nbturns

                br_array[count][ii] = sum_br
                bz_array[count][ii] = sum_bz
                phi_array[count][ii] = sum_phi

            # Passive loops
            for ii in range(nbloop):
                if loop_geometry_type[ii] == 1:
                    nbelti = len(loop_outline_r[ii])
                    r2 = sum(loop_outline_r[ii]) / (nbelti - 1)
                    z2 = sum(loop_outline_z[ii]) / (nbelti - 1)
                else:
                    r2 = loop_rectangle_r[ii]
                    z2 = loop_rectangle_z[ii]

                br_val, bz_val, phi_val = compute_br_bz_phi(xr, zr, r2, z2)
                br_array[count][nbcoil + ii] = br_val
                bz_array[count][nbcoil + ii] = bz_val
                phi_array[count][nbcoil + ii] = phi_val

            count += 1

    return br_array, bz_array, phi_array

def compute_response_matrix(
    observation_points: List[List[float]],
    coil_data: List[Dict[str, Any]],
    passive_loop_data: List[Dict[str, Any]],
    plasma_points: List[List[float]] = None
    ) -> Tuple[ndarray, ndarray, ndarray]:
    """Flux and field response at arbitrary observation points, per source.

    Parameters
    ----------
    observation_points : array_like
        Points at which to evaluate, as major radius and height pairs [m].
    coil_data : sequence
        Active coil geometry and turns [-].
    passive_loop_data : sequence
        Passive loop geometry [-].
    plasma_points : array_like, optional
        Plasma filament positions [m].

    Returns
    -------
    tuple of np.ndarray
        The flux, vertical field and radial field response, each with one row per
        observation point and one column per source [-].

    Convention
    ----------
    Unlike :func:`calc_grid`, the observation points are arbitrary rather than a
    rectangular mesh, which is what a diagnostic set of sensor positions needs.
    Column order is coils, then passive loops, then plasma filaments when given.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Same filament idealization as the rest of the module. Plasma filaments are
    positions only: this returns their response, not their currents.

    Provenance
    ----------
    .. [1] :func:`compute_br_bz_phi`, evaluated per source.
    """
    nb_obs = len(observation_points)
    nb_coil = len(coil_data)
    nb_loop = len(passive_loop_data)
    
    # Handle plasma argument robustly (now also accepts empty list as 'no plasma')
    if plasma_points is None or (isinstance(plasma_points, (list, tuple)) and len(plasma_points) == 0):
        nb_plas = 0
        actual_plasma_points = []
    elif isinstance(plasma_points, (list, tuple)) and len(plasma_points) == 2 and all(isinstance(x, (float, int)) for x in plasma_points):
        nb_plas = 1
        actual_plasma_points = [plasma_points]
    elif isinstance(plasma_points, (list, tuple)) and len(plasma_points) > 0 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in plasma_points):
        nb_plas = len(plasma_points)
        actual_plasma_points = plasma_points
    else:
        raise ValueError("plasma_points must be None, a single [r, z] point, or a list of [r, z] points")

    total_sources = nb_coil + nb_loop + nb_plas

    Psi_matrix = np.zeros((nb_obs, total_sources))
    Bz_matrix = np.zeros((nb_obs, total_sources))
    Br_matrix = np.zeros((nb_obs, total_sources))

    for i_obs, (r1, z1) in enumerate(observation_points):
        # Active Coils contribution
        for i_c, coil in enumerate(coil_data):
            sum_psi_coil = 0.0
            sum_bz_coil = 0.0
            sum_br_coil = 0.0
            for element in coil['elements']:
                r2_c, z2_c, turns_c = element['r'], element['z'], element['turns']
                br, bz, psi = compute_br_bz_phi(r1, z1, r2_c, z2_c) # Uses the corrected version
                sum_psi_coil += psi * turns_c
                sum_bz_coil += bz * turns_c
                sum_br_coil += br * turns_c
            Psi_matrix[i_obs, i_c] = sum_psi_coil
            Bz_matrix[i_obs, i_c] = sum_bz_coil
            Br_matrix[i_obs, i_c] = sum_br_coil

        # Passive Loops contribution
        for i_l, loop in enumerate(passive_loop_data):
            if loop['geometry_type'] == 1: # Polygon (Outline)
                r2_l = np.mean(loop['outline_r'])
                z2_l = np.mean(loop['outline_z'])
            else: # Rectangle
                r2_l = loop['rectangle_r']
                z2_l = loop['rectangle_z']
            
            br, bz, psi = compute_br_bz_phi(r1, z1, r2_l, z2_l)
            Psi_matrix[i_obs, nb_coil + i_l] = psi
            Bz_matrix[i_obs, nb_coil + i_l] = bz
            Br_matrix[i_obs, nb_coil + i_l] = br

        # Plasma Elements contribution
        for i_p, (r2_p, z2_p) in enumerate(actual_plasma_points):
            br, bz, psi = compute_br_bz_phi(r1, z1, r2_p, z2_p)
            Psi_matrix[i_obs, nb_coil + nb_loop + i_p] = psi
            Bz_matrix[i_obs, nb_coil + nb_loop + i_p] = bz
            Br_matrix[i_obs, nb_coil + nb_loop + i_p] = br
            
    return Psi_matrix, Bz_matrix, Br_matrix

def compute_response_vector(
    coil_data: List[Dict[str, Any]],
    passive_loop_data: List[Dict[str, Any]],
    plasma_points: List[List[float]],
    observation_points: List[List[float]]
    ) -> Tuple[ndarray, ndarray, ndarray]:
    """The same response as the matrix form, with the arguments in the legacy order.

    Parameters
    ----------
    coil_data : sequence
        Active coil geometry and turns [-].
    passive_loop_data : sequence
        Passive loop geometry [-].
    plasma_points : array_like
        Plasma filament positions [m].
    observation_points : array_like
        Points at which to evaluate [m].

    Returns
    -------
    tuple of np.ndarray
        Exactly what :func:`compute_response_matrix` returns [-].

    Convention
    ----------
    A thin wrapper that reorders its arguments and delegates. It exists because
    the legacy call order put the observation points last; new code should call
    the matrix form directly.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] :func:`compute_response_matrix`, which does the work.
    """
    return compute_response_matrix(
        observation_points=observation_points,
        coil_data=coil_data,
        passive_loop_data=passive_loop_data,
        plasma_points=plasma_points
    )

def compute_point_response_matrices(
    obs_r: np.ndarray,
    obs_z: np.ndarray,
    src_r: np.ndarray,
    src_z: np.ndarray,
    turns: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    n_groups: int | None = None,
    components: Tuple[str, ...] = ("psi", "bz", "br"),
) -> Tuple[ndarray, ...]:
    """Broadcast response of point sources at observation points, with exact kernels.

    Parameters
    ----------
    obs_r : array_like
        Major radius of the observation points [m].
    obs_z : array_like
        Height of the observation points [m].
    src_r : array_like
        Major radius of the sources [m].
    src_z : array_like
        Height of the sources [m].
    turns : array_like, optional
        Turn count weighting each source [-].
    groups : array_like, optional
        Which output column each source contributes to [-].
    n_groups : int, optional
        How many columns the grouping produces [-].
    components : sequence of str, optional
        Which of the flux and the two field components to return [-].

    Returns
    -------
    dict of str to np.ndarray
        One matrix per requested component, per unit current [-].

    Convention
    ----------
    Exact elliptic-integral kernels rather than the approximations elsewhere in
    this module, and fully broadcast rather than looped, which is what makes it
    the right choice for many sources at many points.

    Grouping sums columns as it goes, so a set of filaments belonging to one
    circuit becomes a single column instead of being summed afterwards; on a real
    machine that turns several hundred filament columns into a handful of circuit
    ones. Turns weight each source before that sum.

    Defaults
    --------
    Returning all three components is a numerical convenience, not a physical
    choice, and narrowing it is worth doing: each field component costs roughly
    twice the flux pass.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A source coincident with an observation point has no finite response here;
    unlike :func:`compute_br_bz_phi`, nothing steps around the singularity. On
    axis the kernels take their analytic limits.

    Provenance
    ----------
    .. [1] The exact elliptic-integral Green's functions in
       :mod:`vaft.formula.green`.
    """
    obs_r = np.asarray(obs_r, dtype=float).ravel()
    obs_z = np.asarray(obs_z, dtype=float).ravel()
    src_r = np.asarray(src_r, dtype=float).ravel()
    src_z = np.asarray(src_z, dtype=float).ravel()
    if obs_r.shape != obs_z.shape or src_r.shape != src_z.shape:
        raise ValueError("observation/source r and z arrays must have equal shapes")

    ro = obs_r[:, None]
    zo = obs_z[:, None]
    rs = src_r[None, :]
    zs = src_z[None, :]

    if not components or any(c not in ("psi", "bz", "br") for c in components):
        raise ValueError(
            f'components must be a non-empty subset of ("psi", "bz", "br"); '
            f"got {components!r}"
        )
    matrices = {}
    if "psi" in components:
        matrices["psi"] = green_psi_exact(ro, zo, rs, zs)
    if "bz" in components or "br" in components:
        matrices["br"], matrices["bz"] = green_br_bz_exact(ro, zo, rs, zs)

    if turns is not None:
        w = np.asarray(turns, dtype=float).ravel()[None, :]
        matrices = {k: v * w for k, v in matrices.items()}

    if groups is not None:
        groups = np.asarray(groups, dtype=int).ravel()
        if groups.shape != src_r.shape:
            raise ValueError("groups must have one entry per source")
        ng = int(n_groups if n_groups is not None else groups.max() + 1)
        if groups.min() < 0 or groups.max() >= ng:
            raise ValueError(
                f"group indices must lie in [0, {ng}); "
                f"got range [{groups.min()}, {groups.max()}]"
            )
        onehot = np.zeros((src_r.size, ng))
        onehot[np.arange(src_r.size), groups] = 1.0
        matrices = {k: v @ onehot for k, v in matrices.items()}

    return tuple(matrices[name] for name in components)


def compute_mutual_passive_active(
    passive_loop_geometry: List[Tuple[str, float, float, float]],
    coil_geometry: List[List[Tuple[float, float, int]]],
) -> np.ndarray:
    """Mutual inductance between passive loops and active coils, derived from geometry.

    Parameters
    ----------
    passive_loop_geometry : sequence
        Geometry of each passive loop [-].
    coil_geometry : sequence
        Geometry and turns of each active coil [-].

    Returns
    -------
    np.ndarray
        Mutual inductance with one row per loop and one column per coil [H].

    Convention
    ----------
    Derived from the geometry given, which is what makes it a usable fallback when
    a packaged coupling matrix does not apply. The packaged matrix represents a
    historical coil geometry; a machine whose coils have since moved needs this
    instead, and the two must not be mixed within one calculation.

    Applicability
    -------------
    Machine-independent.  The geometry is the argument; the packaged matrix it
    substitutes for is VEST's.

    Limitations
    -----------
    Filament idealization, so it is less accurate than a matrix computed from the
    real conductor sections. Use it when the packaged matrix does not match the
    geometry, not in preference to one that does.

    Provenance
    ----------
    .. [1] The Green's functions in :mod:`vaft.formula.green`; the packaged
       alternative is the machine layer's ``em_coupling`` asset.
    """
    coupling = np.zeros((len(passive_loop_geometry), len(coil_geometry)))

    for i_loop, (_, r_loop, z_loop, coefficient) in enumerate(
        passive_loop_geometry
    ):
        for i_coil, elements in enumerate(coil_geometry):
            if not elements:
                raise ValueError(f"active coil {i_coil} has no geometry elements")
            total_turns = sum(element[2] for element in elements)
            mean_response = sum(
                green_r(r_loop, z_loop, r_coil, z_coil)
                for r_coil, z_coil, _ in elements
            ) / len(elements)
            coupling[i_loop, i_coil] = coefficient * total_turns * mean_response

    return coupling


def compute_impedance_matrices(
    loop_resistances: np.ndarray,
    passive_loop_geometry: List[Tuple[str, float, float, float]],  
    # e.g. [(loop_name, average_r, average_z, geometry_coef), ...]
    coil_geometry: List[List[Tuple[float, float, int]]] | None,
    # Retained for API compatibility; canonical coupling now comes from mutual_pa.
    mutual_pp: np.ndarray,       # mutual_passive_passive from ODS
    mutual_pa: np.ndarray,       # mutual_passive_active from ODS
    plasma_rz: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the resistance and inductance matrices of the passive circuit.

    Parameters
    ----------
    loop_resistances : array_like
        Resistance of each passive loop [ohm].
    passive_loop_geometry : sequence
        Geometry of each passive loop [-].
    coil_geometry : sequence
        Geometry of the active coils; superseded by the mutual matrix below [-].
    mutual_pp : array_like
        Passive-to-passive mutual inductance [H].
    mutual_pa : array_like
        Passive-to-active mutual inductance [H].
    plasma_rz : array_like
        Plasma filament positions, coupled in when given [m].

    Returns
    -------
    tuple of np.ndarray
        The diagonal resistance matrix in ohms, and the two inductance matrices in
        henries, the second widened by the plasma coupling when supplied [-].

    Convention
    ----------
    Resistance is diagonal: loops are resistively independent and coupled only
    inductively. The passive-to-active matrix is widened to the right by the
    plasma coupling when filaments are given, so the plasma enters the circuit as
    another driving current rather than as a separate term.

    The coil geometry argument is kept for compatibility and is superseded by the
    supplied mutual matrix; passing geometry alone no longer determines the
    coupling.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Takes the mutual matrices as given and cannot check they were computed for
    this geometry, which is exactly the mismatch
    :func:`compute_mutual_passive_active` exists to resolve.

    Provenance
    ----------
    .. [1] The circuit form the eddy-current solve integrates; see
       :func:`solve_eddy_currents`.
    """
    nbloop = len(passive_loop_geometry)
    nbplas = len(plasma_rz)
    loop_resistances = np.asarray(loop_resistances)
    mutual_pp = np.asarray(mutual_pp)
    mutual_pa = np.asarray(mutual_pa)

    if loop_resistances.shape != (nbloop,):
        raise ValueError(
            f"loop_resistances must have shape ({nbloop},), "
            f"got {loop_resistances.shape}"
        )
    if mutual_pp.shape != (nbloop, nbloop):
        raise ValueError(
            f"mutual_pp must have shape ({nbloop}, {nbloop}), "
            f"got {mutual_pp.shape}"
        )
    if mutual_pa.ndim != 2 or mutual_pa.shape[0] != nbloop:
        raise ValueError(
            f"mutual_pa must have {nbloop} passive-loop rows, "
            f"got shape {mutual_pa.shape}"
        )
    if coil_geometry is not None and len(coil_geometry) != mutual_pa.shape[1]:
        raise ValueError(
            "coil_geometry and mutual_pa describe different numbers of "
            f"active coils ({len(coil_geometry)} and {mutual_pa.shape[1]})"
        )
    # Build R (nbloop x nbloop)
    R_mat = np.diag(loop_resistances)

    # M is the standard passive-to-passive coupling from em_coupling.
    M_mat = mutual_pp

    # Plasma filaments are transient VAFT solver inputs; until they are
    # represented by pf_plasma.element URIs, compute only that portion here.
    if nbplas == 0:
        L_mat = mutual_pa
    else:
        plasma_coupling = np.zeros((nbloop, nbplas))
        for i_loop, (loop_name, r1, z1, coef) in enumerate(passive_loop_geometry):
            for j_plasma, (rp, zp) in enumerate(plasma_rz):
                plasma_coupling[i_loop, j_plasma] = (
                    coef * green_r(r1, z1, rp, zp)
                )
        L_mat = np.hstack((mutual_pa, plasma_coupling))

    return R_mat, L_mat, M_mat

# _solve_eddy_currents_original = solve_eddy_currents # Keep a reference to the original, just in case

#: Relative tolerance for treating M_mat as symmetric. A matrix that came
#: through `em_coupling()` is symmetric to float64 round-off; one written by
#: hand (test fixtures, legacy ODSs) may not be, and must keep the general path.
_SYMMETRIC_RTOL = 1.0e-10


def wall_propagator(
    R_mat: np.ndarray,
    M_mat: np.ndarray,
    dt_sub: float,
    *,
    method: str = "auto",
) -> np.ndarray:
    """The one-substep propagator of the passive circuit.

    Parameters
    ----------
    R_mat : array_like
        Loop resistance matrix, diagonal and positive [ohm].
    M_mat : array_like
        Passive-to-passive mutual inductance matrix [H].
    dt_sub : float
        Length of the substep [s].
    method : str, optional
        Which decomposition to use: automatic, general, or symmetric [-].

    Returns
    -------
    np.ndarray
        The propagator advancing the loop currents by one substep [-].

    Raises
    ------
    ValueError
        The method is not one of the three accepted names.

    Convention
    ----------
    The matrix exponential of the negative circuit operator times the substep.
    :func:`solve_eddy_currents` consumes nothing from the decomposition except
    this, so it is the whole seam for a faster one.

    The symmetric path uses the symmetric-definite pencil: with resistance
    diagonal and positive and inductance symmetric, a change of variable makes the
    operator symmetric, so the symmetric eigensolver applies and the eigenvector
    inverse is a transpose. **On the real 950-loop machine that is about eleven
    times faster than the general solver and agrees to about one part in 1e14**
    (#308). It is valid only when the inductance is symmetric, which the loader
    guarantees for the packaged asset (#347) but nothing guarantees for a
    caller-built matrix, so the automatic mode checks and otherwise leaves the
    general path unchanged.

    Defaults
    --------
    Automatic selection is a numerical convenience: it takes the fast path when
    the check passes and is otherwise identical to the general one.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Assumes the circuit is constant over the substep, which is what makes a single
    propagator reusable across every step.

    Provenance
    ----------
    .. [1] Issue #308 for the measured speedup and agreement; issue #347 for the
       loader guarantee that makes the symmetric path safe on the packaged asset.
    """
    if method not in ("auto", "eig", "eigh"):
        raise ValueError(f"method must be 'auto', 'eig' or 'eigh', got {method!r}")
    if method == "auto":
        symmetric = np.allclose(M_mat, M_mat.T, rtol=_SYMMETRIC_RTOL, atol=0.0)
        # The pencil form takes R's diagonal; a coupled R must keep the general path.
        diagonal = np.count_nonzero(R_mat - np.diag(np.diag(R_mat))) == 0
        method = "eigh" if (symmetric and diagonal) else "eig"

    if method == "eigh":
        if np.count_nonzero(R_mat - np.diag(np.diag(R_mat))):
            raise ValueError("eigh propagator requires a diagonal R_mat")
        r = np.diag(R_mat)
        if not np.all(r > 0.0):
            raise ValueError("eigh propagator requires a diagonal positive R_mat")
        s = 1.0 / np.sqrt(r)
        S = (M_mat * s[:, None]) * s[None, :]
        S = (S + S.T) / 2.0  # remove residual round-off asymmetry before eigh
        lam, Q = np.linalg.eigh(S)
        # dy/dt = -S^-1 y  =>  expm over one substep in y-space, then map back.
        E = Q @ np.diag(np.exp(-dt_sub / lam)) @ Q.T
        # y = R^1/2 I evolves by E, so I(t+dt) = R^-1/2 E R^1/2 I(t):
        # row-scale by s = R^-1/2, column-scale by 1/s = R^1/2.
        return (E * s[:, None]) * (1.0 / s)[None, :]
    # general path -- what solve_eddy_currents did before, pinv fallback included
    try:
        B_inv_M = np.linalg.inv(M_mat)
    except np.linalg.LinAlgError:
        B_inv_M = np.linalg.pinv(M_mat)
    A_sys = -B_inv_M @ R_mat
    w, E_vec = np.linalg.eig(A_sys)
    E_inv = np.linalg.inv(E_vec)
    RLR = E_vec @ np.diag(np.exp(w * dt_sub)) @ E_inv
    return np.real(RLR) if np.isrealobj(A_sys) and not np.isrealobj(RLR) else RLR


def solve_eddy_currents(
    R_mat: np.ndarray,    # (nbloop, nbloop)
    L_mat: np.ndarray,    # (nbloop, nbcoil+nbplas)
    M_mat: np.ndarray,    # (nbloop, nbloop)
    coil_plasma_currents: np.ndarray,  # (n_times, nbcoil+nbplas)
    time: np.ndarray,     # (n_times,)
    dt_sub: float = 5e-5,
    *,
    method: str = "auto",
    ) -> np.ndarray:
    """Integrate the vessel eddy currents driven by a coil and plasma programme.

    Parameters
    ----------
    R_mat : array_like
        Loop resistance matrix [ohm].
    L_mat : array_like
        Passive-to-active mutual inductance [H].
    M_mat : array_like
        Passive-to-passive mutual inductance [H].
    coil_plasma_currents : array_like
        Driving current history, one row per time and one column per source [A].
    time : array_like
        Time base of that history [s].
    dt_sub : float, optional
        Substep of the integration [s].
    method : str, optional
        Decomposition passed to the propagator [-].

    Returns
    -------
    np.ndarray
        Eddy current in each passive loop over time [A].

    Convention
    ----------
    **Driven by the rate of change of the driving current**, not by its value, so
    a flat coil programme drives nothing however large. The integration is
    substepped and advanced by a single reused propagator, which is what makes it
    cheap over a long shot.

    Defaults
    --------
    The substep is a hard-coded value. It is meant to be finer than the input
    grid, and the shipped value is coarser than the 40 microsecond diagnostics
    grid, making this a mild rate reduction rather than the refinement intended;
    tracked in #425.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The driving derivative is taken from the supplied history, so a coarsely
    sampled programme understates a fast ramp. Assumes the circuit is constant in
    time, so no geometry or resistance change during the shot is represented.

    Provenance
    ----------
    .. [1] The legacy ``vfit_eddy`` workflow, whose matrix setup this retains
       structural similarity to.
    .. [2] Issue #425, tracking the substep default.
    """
    nbloop = R_mat.shape[0]
    n_times_original = len(time)
    n_active_sources = coil_plasma_currents.shape[1]

    if nbloop == 0:
        return np.zeros((n_times_original, 0))
    if n_times_original == 0:
        return np.zeros((0, nbloop))

    # M_mat is inverted (or decomposed) inside wall_propagator; only R_mat's
    # inverse is needed here, for the particular solution.
    try:
        C_R_inv = np.linalg.inv(R_mat)
    except np.linalg.LinAlgError:
        # print(f"Error inverting R_mat: {e}. Using pseudo-inverse as fallback.")
        try:
            C_R_inv = np.linalg.pinv(R_mat)
        except np.linalg.LinAlgError as e_pinv:
            print(f"Pseudo-inverse of R_mat also failed: {e_pinv}. Aborting.")
            return np.full((n_times_original, nbloop), np.nan)
            
    if np.any(np.isnan(C_R_inv)) or np.any(np.isinf(C_R_inv)):
        print("Error: Inverse of R_mat contains NaN/Inf after fallback. Aborting.")
        return np.full((n_times_original, nbloop), np.nan)

    # One-substep propagator expm(-M^-1 R dt). The symmetric-pencil path is used
    # automatically when M_mat is symmetric (the loader guarantees that for the
    # packaged asset, #347); a hand-built asymmetric M keeps the general path.
    try:
        RLR_mat = wall_propagator(R_mat, M_mat, dt_sub, method=method)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error building the wall propagator: {e}. Aborting.")
        return np.full((n_times_original, nbloop), np.nan)
    if np.any(np.isnan(RLR_mat)) or np.any(np.isinf(RLR_mat)):
        print("Error: RLR_mat contains NaN/Inf. Aborting.")
        return np.full((n_times_original, nbloop), np.nan)

    # Fine time grid
    if n_times_original == 0: 
        return np.zeros((0, nbloop))
    if time.size == 1 or np.isclose(time[0], time[-1]):
        t_fine = np.array([time[0]])
    elif time[0] > time[-1]: 
        t_fine = np.array([]) 
    else:
        t_fine = np.arange(time[0], time[-1], dt_sub) 
        if t_fine.size == 0 and not np.isclose(time[0],time[-1]): 
             t_fine = np.array([time[0]])

    n_fine_steps = len(t_fine)
    if n_fine_steps == 0:
        return np.zeros((n_times_original, nbloop)) 

    # --- Pre-calculate interpolated active currents and their derivatives ---
    coil_plasma_currents_fine = np.zeros((n_fine_steps, n_active_sources))
    if n_active_sources > 0:
        if n_times_original > 1:
            for i_src in range(n_active_sources):
                # anti-alias: substepping grid for the integrator.  dt_sub is
                # meant to be finer than the input grid; issue #425 tracks the
                # shipped default (5e-5) being coarser than the 4e-5 diagnostics
                # grid, which makes this a mild rate reduction.
                coil_plasma_currents_fine[:, i_src] = np.interp(t_fine, time, coil_plasma_currents[:, i_src])
        elif n_times_original == 1 and n_fine_steps > 0: 
            for i_src in range(n_active_sources):
                coil_plasma_currents_fine[:, i_src] = coil_plasma_currents[0, i_src]

    d_coil_plasma_dt_fine = np.zeros_like(coil_plasma_currents_fine)
    if n_fine_steps > 1 and n_active_sources > 0:
        diff_currents = np.diff(coil_plasma_currents_fine, axis=0)
        d_coil_plasma_dt_fine[:-1, :] = diff_currents / dt_sub
        d_coil_plasma_dt_fine[-1, :] = d_coil_plasma_dt_fine[-2, :] 
    elif n_fine_steps == 1 and n_active_sources > 0: 
        d_coil_plasma_dt_fine[:, :] = 0.0 
    # --- End of pre-calculation ---

    i_loop_old = np.zeros(nbloop) 
    i_loop_fine_out = np.zeros((n_fine_steps, nbloop))
    if n_fine_steps > 0:
        i_loop_fine_out[0, :] = i_loop_old 

    # print("Starting EVD time integration loop (Optimized - pre-calculated derivatives)...")
    
    for i_sub in range(n_fine_steps - 1): 
        # if i_sub % 10000 == 0: # Progress indicator can be re-enabled if needed for long runs
        #     print(f"EVD loop: iteration {i_sub} / {n_fine_steps - 1}")

        current_dIc_dt = d_coil_plasma_dt_fine[i_sub, :]
        
        Vw_source_term = -L_mat @ current_dIc_dt
        I_particular = C_R_inv @ Vw_source_term
        
        i_loop_new = I_particular + RLR_mat @ (i_loop_old - I_particular)
            
        i_loop_old = i_loop_new.copy()
        i_loop_fine_out[i_sub+1, :] = i_loop_new 

    # Interpolate results back to original time grid
    I_loop_final = np.zeros((n_times_original, nbloop))
    if n_times_original > 0: 
        if n_fine_steps == 0: 
            pass 
        elif n_fine_steps == 1: 
            for i_l in range(nbloop):
                I_loop_final[:, i_l] = i_loop_fine_out[0, i_l] 
        else: 
            for i_l in range(nbloop):
                if time.size > 1 and np.isclose(t_fine[0], t_fine[-1]) and t_fine.size > 1:
                    I_loop_final[:,i_l] = i_loop_fine_out[0,i_l]
                else:
                    # anti-alias: return leg of the substepping above; the
                    # solved loop current is smooth on t_fine by construction.
                    I_loop_final[:, i_l] = np.interp(time, t_fine, i_loop_fine_out[:, i_l])
    
    # print("EVD method (Optimized) finished.")
    return I_loop_final

def compute_vacuum_fields_1d(
    coil_plus_loop_currents: np.ndarray,  # shape (n_times, nb_coil+nb_loop)
    coil_plus_loop_psi_resp: np.ndarray,  # shape (n_points, nb_coil+nb_loop)
    coil_plus_loop_br_resp: np.ndarray,   # shape (n_points, nb_coil+nb_loop)
    coil_plus_loop_bz_resp: np.ndarray,   # shape (n_points, nb_coil+nb_loop)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contract current histories with response matrices into field histories.

    Parameters
    ----------
    coil_plus_loop_currents : array_like
        Current history, one row per time and one column per source [A].
    coil_plus_loop_psi_resp : array_like
        Flux response per unit current [Wb/A].
    coil_plus_loop_br_resp : array_like
        Radial field response per unit current [T/A].
    coil_plus_loop_bz_resp : array_like
        Vertical field response per unit current [T/A].

    Returns
    -------
    tuple of np.ndarray
        Flux in weber and the two field components in tesla, each with one row per
        time and one column per observation point [-].

    Convention
    ----------
    The linear step the response matrices exist for: each output is the current
    history contracted against its own response. **Vacuum fields only** -- the
    plasma's own contribution is not here unless it entered as a filament current
    in the input.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Column order in the currents must match the response matrices' own, and
    nothing checks it; a mismatched ordering gives plausible numbers that are
    wrong.

    Provenance
    ----------
    .. [1] The response matrices built by :func:`compute_response_matrix` or
       :func:`calc_grid`.
    """
    n_times = coil_plus_loop_currents.shape[0]
    n_points = coil_plus_loop_psi_resp.shape[0]

    psi_out = np.zeros((n_times, n_points))
    br_out = np.zeros((n_times, n_points))
    bz_out = np.zeros((n_times, n_points))

    for i_time in range(n_times):
        ix = coil_plus_loop_currents[i_time]  # shape (nb_coil+nb_loop,)
        psi_out[i_time] = coil_plus_loop_psi_resp @ ix
        br_out[i_time]  = coil_plus_loop_br_resp  @ ix
        bz_out[i_time]  = coil_plus_loop_bz_resp  @ ix

    return psi_out, br_out, bz_out
