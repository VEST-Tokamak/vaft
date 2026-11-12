"""Issue #2270 P2: equilibrium volume weighting and the GPEC resonant table now live
in ``vaft.process``.

The volume weights had two copies -- the impurity views' ``_volume_weights_at``
and ``vaft.omas.vest_upstream._volume_weights_on_grid`` (the composition stage
and the kinetic-state inference) -- that disagreed on degenerate input and
both read ``V`` against a ``sqrt(psi_N)`` proxy radius.  On an authentic
radius the result is bit-identical to both; the corrections are pinned here.
The resonant-table loop moved verbatim and is held to a frozen copy.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from omas import ODS

from vaft.formula.stability import island_width_from_resonant_flux, resonant_flux_from_delta
from vaft.omas import vest_upstream as vu
from vaft.plot.backend import recipes
from vaft.process.perturbation import RESONANT_SURFACE_FIELDS, resonant_delta, resonant_table
from vaft.process.profile import equilibrium_volume_weights, volume_weights_on_grid

PSI_N = np.linspace(0.0, 1.0, 41)
RHO = np.linspace(0.0, 1.0, 21)


# --- frozen pre-move algorithms (develop 7edf8e026) -------------------------


def _frozen_view_kernel(rho_eq, volume, rho):
    ok = np.isfinite(rho_eq) & np.isfinite(volume)
    order = np.argsort(rho_eq[ok])
    edges = np.concatenate(([0.0], 0.5 * (rho[1:] + rho[:-1]), [rho[-1]]))
    dv = np.clip(np.diff(np.interp(edges, rho_eq[ok][order], volume[ok][order])), 0.0, None)
    return dv if np.sum(dv) > 0.0 else None


def _frozen_upstream_kernel(rho_eq, volume, rho):
    ok = np.isfinite(rho_eq) & np.isfinite(volume)
    order = np.argsort(rho_eq[ok])
    edges = np.concatenate(([0.0], 0.5 * (rho[1:] + rho[:-1]), [rho[-1]]))
    return np.clip(np.diff(np.interp(edges, rho_eq[ok][order], volume[ok][order])), 0.0, None)


def _frozen_resonant_rows(psi_n, harmonics, field, surfaces, chi1, n_tor):
    rows = []
    for surface in surfaces:
        m_pol = int(round(n_tor * surface["q"]))
        column = int(np.argmin(np.abs(harmonics - m_pol)))
        if int(harmonics[column]) != m_pol:
            continue
        jump = resonant_delta(
            psi_n, field[:, column], psi_rational=surface["psi_n"], area=surface["area"],
            dq_dpsi_norm=surface["dq_dpsi_n"], chi1=chi1, m_pol=m_pol, n_tor=n_tor,
        )
        flux = resonant_flux_from_delta(jump.delta, surface["geometric_factor"], n_tor)
        rows.append({
            "psi_n": surface["psi_n"], "q": surface["q"], "m_pol": m_pol, "delta": jump.delta,
            "Phi_res": complex(flux), "phase": float(np.angle(flux)),
            "w_isl": float(island_width_from_resonant_flux(
                flux, surface["area"], surface["q"], surface["dq_dpsi_n"], m_pol, chi1,
            )),
        })
    return rows


# --- fixtures ---------------------------------------------------------------


def _slice(rho_tor_norm, *, volume=None, q=None, with_psi=True, time=0.3) -> ODS:
    ods = ODS(consistency_check=False)
    base = "equilibrium.time_slice.0"
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    ods["equilibrium.time"] = np.asarray([time])
    ods[f"{base}.time"] = time
    if with_psi:
        ods[f"{base}.profiles_1d.psi"] = 0.1 * PSI_N
        ods[f"{base}.global_quantities.psi_axis"] = 0.0
        ods[f"{base}.global_quantities.psi_boundary"] = 0.1
    ods[f"{base}.profiles_1d.rho_tor_norm"] = rho_tor_norm
    ods[f"{base}.profiles_1d.volume"] = 0.2 * PSI_N if volume is None else volume
    if q is not None:
        ods[f"{base}.profiles_1d.q"] = q
    return ods


AUTHENTIC = PSI_N**0.6


# --- the kernel and its adapter ---------------------------------------------


def test_on_an_authentic_radius_both_callers_are_bit_identical():
    ods = _slice(AUTHENTIC)
    volume = 0.2 * PSI_N
    expected = _frozen_upstream_kernel(AUTHENTIC, volume, RHO)
    np.testing.assert_array_equal(vu._volume_weights_on_grid(ods, 0, RHO), expected)
    np.testing.assert_array_equal(recipes._volume_weights_at(ods, 0.3, RHO), expected)
    np.testing.assert_array_equal(_frozen_view_kernel(AUTHENTIC, volume, RHO), expected)


def test_the_weights_sum_to_the_enclosed_volume():
    dv = volume_weights_on_grid(AUTHENTIC, 0.2 * PSI_N, RHO)
    assert np.sum(dv) == pytest.approx(0.2)
    assert np.all(dv >= 0.0)


def test_a_proxy_radius_is_re_derived_from_q_not_read_as_toroidal():
    q = 1.0 + 3.0 * PSI_N
    ods = _slice(np.sqrt(PSI_N), q=q)
    from vaft.process.profile import _equilibrium_grid_at_index

    derived = _equilibrium_grid_at_index(ods, 0).rho_tor_norm
    got = equilibrium_volume_weights(ods, 0, RHO)
    np.testing.assert_array_equal(got, volume_weights_on_grid(derived, 0.2 * PSI_N, RHO))
    # the proxy's answer is a different set of surfaces
    old = _frozen_upstream_kernel(np.sqrt(PSI_N), 0.2 * PSI_N, RHO)
    assert np.max(np.abs(got - old)) > 1e-3


@pytest.mark.parametrize("with_psi", [True, False])
def test_a_proxy_radius_without_q_is_declined(with_psi):
    assert equilibrium_volume_weights(_slice(np.sqrt(PSI_N), with_psi=with_psi), 0, RHO) is None


def test_a_constant_volume_is_declined_by_both_callers():
    """Upstream used to return zeros here, which the composition turned into 0/0."""
    ods = _slice(AUTHENTIC, volume=np.full(PSI_N.size, 0.2))
    assert np.all(_frozen_upstream_kernel(AUTHENTIC, np.full(PSI_N.size, 0.2), RHO) == 0.0)
    assert vu._volume_weights_on_grid(ods, 0, RHO) is None
    assert recipes._volume_weights_at(ods, 0.3, RHO) is None


@pytest.mark.parametrize(
    "rho",
    [np.array([0.5]), np.array([]), np.array([0.0, 0.5, 0.4, 1.0]), np.array([0.0, np.nan, 1.0])],
    ids=["one-point", "empty", "non-increasing", "non-finite"],
)
def test_a_grid_that_cannot_be_weighted_is_declined(rho):
    assert volume_weights_on_grid(AUTHENTIC, 0.2 * PSI_N, rho) is None


def test_mismatched_equilibrium_arrays_are_declined():
    assert volume_weights_on_grid(AUTHENTIC, 0.2 * PSI_N[:-1], RHO) is None


def test_the_view_still_matches_by_time():
    ods = _slice(AUTHENTIC)
    assert recipes._volume_weights_at(ods, 0.3 + 1e-3, RHO) is None
    assert recipes._volume_weights_at(ods, 0.3 + 1e-4, RHO) is not None


def test_the_packaged_48224_sample_is_unchanged_and_left_unwritten():
    """The one packaged sample with V(rho) stores an authentic radius: bit-identical."""
    from _sample_fixtures import sample_ods

    ods = sample_ods(48224)
    before = sorted(copy.deepcopy(ods).flat().keys())
    base = "equilibrium.time_slice.0.profiles_1d"
    rho_eq = np.asarray(ods[f"{base}.rho_tor_norm"], dtype=float)
    volume = np.asarray(ods[f"{base}.volume"], dtype=float)
    got = equilibrium_volume_weights(ods, 0, RHO)
    np.testing.assert_array_equal(got, _frozen_upstream_kernel(rho_eq, volume, RHO))
    assert np.sum(got) == pytest.approx(volume[-1], rel=1e-12)
    assert sorted(ods.flat().keys()) == before


# --- resonant_table ---------------------------------------------------------


GRID = np.linspace(0.2, 0.9, 40001)


def _kink(psi_res, left, right):
    d = GRID - psi_res
    return np.where(d < 0, left * d, right * d).astype(complex)


def _spectrum():
    harmonics = np.array([1, 2, 3, 4])
    field = np.zeros((GRID.size, harmonics.size), dtype=complex)
    field[:, 1] = _kink(0.45, 1.0, 4.0) * (1 + 0.5j)
    field[:, 2] = _kink(0.7, -2.0, 1.0)
    surfaces = [
        {"psi_n": 0.45, "q": 2.0, "area": 1.3, "dq_dpsi_n": 2.0, "geometric_factor": 0.8},
        {"psi_n": 0.7, "q": 3.0, "area": 2.1, "dq_dpsi_n": 3.5, "geometric_factor": 1.1},
        {"psi_n": 0.85, "q": 5.0, "area": 2.9, "dq_dpsi_n": 5.0, "geometric_factor": 1.4},
    ]
    return harmonics, field, surfaces


def test_the_resonant_table_is_the_moved_loop():
    harmonics, field, surfaces = _spectrum()
    got = resonant_table(GRID, harmonics, field, surfaces, chi1=0.3, n_tor=1)
    expected = _frozen_resonant_rows(GRID, harmonics, field, surfaces, 0.3, 1)
    assert [row["m_pol"] for row in got] == [2, 3]  # m = 5 is outside the band
    assert got == expected
    assert set(RESONANT_SURFACE_FIELDS) <= set(surfaces[0])


def test_a_misshapen_field_is_refused():
    harmonics, field, surfaces = _spectrum()
    with pytest.raises(ValueError, match=r"declared \(psi, m\)"):
        resonant_table(GRID, harmonics[:-1], field, surfaces, chi1=0.3, n_tor=1)


# --- cold review: the proxy branch writes nothing, and declines without psi ---


def _key_tree(node, prefix=""):
    """Every key path, empty nodes included -- ``flat()`` hides those."""
    keys = set()
    if hasattr(node, "keys") and not isinstance(node, np.ndarray):
        for key in node.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(path)
            keys |= _key_tree(node[key], path)
    return keys


@pytest.mark.parametrize("q", [1.0 + 3.0 * PSI_N, None], ids=["derivable", "no-q"])
def test_the_proxy_branch_never_writes_the_callers_ods(q):
    ods = _slice(np.sqrt(PSI_N), q=q)
    before = _key_tree(ods)
    equilibrium_volume_weights(ods, 0, RHO)
    recipes._volume_weights_at(ods, 0.3, RHO)
    assert _key_tree(ods) == before


def test_the_private_copy_derives_what_the_full_slice_derives():
    """On the real 39915 EFIT slices (proxy radius, full ODS) the radius is the same."""
    from _sample_fixtures import sample_ods

    from vaft.process.profile import _equilibrium_grid_at_index, _profiles_1d_copy

    ods = sample_ods(39915)
    compared = 0
    for index in range(len(ods["equilibrium.time_slice"])):
        full = _equilibrium_grid_at_index(ods, index)
        private = _equilibrium_grid_at_index(_profiles_1d_copy(ods, index), 0)
        if full is None:
            assert private is None
            continue
        assert private.rho_tor_norm_trusted == full.rho_tor_norm_trusted
        np.testing.assert_array_equal(private.rho_tor_norm, full.rho_tor_norm)
        compared += 1
    assert compared


@pytest.mark.parametrize("psi", [np.zeros(PSI_N.size), np.where(PSI_N > 0.5, np.nan, 0.1 * PSI_N)],
                         ids=["flat", "non-finite"])
def test_a_proxy_with_unusable_psi_is_declined(psi):
    ods = _slice(np.sqrt(PSI_N), q=1.0 + 3.0 * PSI_N)
    ods["equilibrium.time_slice.0.profiles_1d.psi"] = psi
    assert equilibrium_volume_weights(ods, 0, RHO) is None
