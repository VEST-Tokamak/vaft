"""Fixed-current closure contracts and independently defined geometry metrics."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vaft.code.tokamaker import TokaMakerConfig, compare_equilibria, fixed_to_free
from vaft.code.tokamaker import closure
from vaft.code.tokamaker.runner import _native_active_x_points
from vaft.code.tokamaker.config import TokaMakerResult
from vaft.data.equilibrium import Contour
from vaft.process.equilibrium import convert_cocos, solovev_example


def test_comparison_is_cocos_invariant_and_preserves_missing_q():
    target = solovev_example(resolution=33)
    result = compare_equilibria(target, convert_cocos(target, 2))
    assert result['lcfs']['max_distance_m'] == 0
    assert result['axis_displacement_m'] == 0
    assert result['globals']['thermal_energy']['difference'] == pytest.approx(0)
    assert result['globals']['q95']['target'] is None
    field_q = result['globals']['q95_field_integral']
    assert field_q['target'] is not None
    assert field_q['target'] == pytest.approx(field_q['solved'])
    assert field_q['difference'] == pytest.approx(0)


def test_segment_distance_and_axis_displacement_are_independent_of_vertex_order():
    eq = solovev_example(resolution=33)
    square = Contour([.2, .6, .6, .2], [-.2, -.2, .2, .2])
    target = replace(eq, lcfs=square)
    shifted = replace(target, lcfs=Contour(square.r + .01, square.z),
                      magnetic_axis=(eq.magnetic_axis[0] + .01, eq.magnetic_axis[1]))
    result = compare_equilibria(target, shifted)
    assert result['lcfs']['max_distance_m'] == pytest.approx(.01)
    assert result['axis_displacement_m'] == pytest.approx(.01)
    assert closure._distances(np.array([[.4, -.2]]), square.points)[0] == pytest.approx(0)


@pytest.mark.parametrize('backend', ['direct', 'tokamaker_vfixed'])
def test_routes_feed_fitted_currents_to_forward_without_constraints(tmp_path, monkeypatch, backend):
    eq = solovev_example(resolution=33)
    fit = SimpleNamespace(currents_A={'PF1': 123.}, status='bounds_unverified', accepted=False)
    calls = []
    monkeypatch.setattr(closure, 'fit_free_boundary_coils', lambda *a, **kw: calls.append('direct') or fit)
    monkeypatch.setattr(closure, 'fit_free_boundary_coils_vfixed', lambda *a, **kw: calls.append('vfixed') or fit)
    def prepare(machine, cfg):
        assert cfg.coil_currents == {'PF1': 123.}
        assert cfg.profile_equilibrium is not None
        assert cfg.profile_mode == 'equilibrium'
        assert cfg.vsc_coil is None
        assert cfg.eqdsk_cocos == 7
        assert cfg.init_r0 == pytest.approx((eq.lcfs.r.max() + eq.lcfs.r.min())/2)
        return 'inputs'
    monkeypatch.setattr(closure, 'prepare_tokamaker_inputs', prepare)
    monkeypatch.setattr(closure, 'run_tokamaker', lambda *a: TokaMakerResult(
        0, tmp_path, gfile=tmp_path/'g000000.00000', scalars={'coil_currents_A': {'PF1': 123.}}))
    monkeypatch.setattr(closure, 'as_equilibrium', lambda *a, **kw: eq)
    result = fixed_to_free(eq, {}, tmp_path/'run', fit_backend=backend)
    assert calls == ['direct' if backend == 'direct' else 'vfixed']
    assert result.status == 'converged'
    assert not result.fit.accepted  # convergence never upgrades inverse hardware feasibility
    assert result.current_max_change_A == 0
    assert result.comparison['lcfs']['max_distance_m'] == 0
    assert (tmp_path/'run'/'closure.json').is_file()


def test_failed_forward_and_vsc_rejection(tmp_path, monkeypatch):
    eq = solovev_example(resolution=33)
    with pytest.raises(ValueError, match='requires refine_shape'):
        fixed_to_free(eq, {}, tmp_path/'no-refinement', verify_refinement=True)
    assert not (tmp_path/'no-refinement').exists()
    with pytest.raises(ValueError, match='unchanged PF'):
        fixed_to_free(eq, {}, tmp_path/'bad', config=TokaMakerConfig(vsc_coil='PF1'))
    assert not (tmp_path/'bad').exists()
    fit = SimpleNamespace(currents_A={'PF1': 1.}, status='residual_or_conditioning', accepted=False)
    monkeypatch.setattr(closure, 'fit_free_boundary_coils', lambda *a, **kw: fit)
    monkeypatch.setattr(closure, 'prepare_tokamaker_inputs', lambda *a: 'inputs')
    monkeypatch.setattr(closure, 'run_tokamaker', lambda *a: TokaMakerResult(1, tmp_path, error='no convergence'))
    result = fixed_to_free(eq, {}, tmp_path/'fail')
    assert result.status == 'solver_failed'
    assert result.realized_currents_A == {}
    assert result.comparison == {}


def test_native_density_seed_has_physical_units_and_no_vacuum_current():
    from vaft.code.tokamaker.runner import _initial_current_density
    eq = solovev_example(resolution=33)
    points = np.array([eq.magnetic_axis, [.9, .8]])
    actual = _initial_current_density(eq, points)
    expected = -2*np.pi*(points[0, 0]*eq.pprime[0] + eq.ffprime[0]/(4e-7*np.pi*points[0, 0]))
    assert actual[0] == pytest.approx(expected)
    assert actual[1] == 0


def test_native_saddle_diagnostics_restore_double_null_when_gridded_export_misses_xpoints():
    target = solovev_example('double_null', resolution=65)
    active = closure.derive_boundary_representation(target).x_points
    saddles = [[x.r, x.z] for x in active if x.active]
    gridded = replace(target, limiter=None)
    result = closure._native_boundary_comparison(target, gridded, {
        'diverted': True, 'native_active_x_points_m': saddles,
        'native_x_flux_tolerance_fraction': 1e-4,
    })
    assert result['native_boundary']['topology'] == 'double_null'
    assert result['native_boundary']['unmatched_target'] == 0
    assert result['native_boundary']['matched_displacements_m'] == pytest.approx([0., 0.])


def test_native_topology_rejects_flux_offset_or_unmatched_extra_saddle():
    target = solovev_example('single_null', resolution=65)
    active = [x for x in closure.derive_boundary_representation(target).x_points if x.active]
    lower = [active[0].r, active[0].z]
    upper = [lower[0], -lower[1]]
    flux = SimpleNamespace(eval=lambda point: np.array([0.005 if point[1] > 0 else 0.]))
    gs = SimpleNamespace(get_xpoints=lambda: (np.array([lower, upper]), True),
                         lim_point=np.array(lower), get_field_eval=lambda _: flux,
                         get_stats=lambda: {'dflux': 1.})
    sidecar = _native_active_x_points(gs)
    assert sidecar['native_active_x_points_m'] == [lower]
    result = closure._native_boundary_comparison(target, target, {
        **sidecar, 'diverted': True})
    assert result['native_boundary']['topology'] == 'lower_single_null'
    extra = closure._native_boundary_comparison(target, target, {
        **sidecar, 'diverted': True, 'native_active_x_points_m': [lower, upper]})
    assert extra['native_boundary']['topology'] == 'ambiguous'
    assert extra['native_boundary']['unmatched_solved'] == 1
