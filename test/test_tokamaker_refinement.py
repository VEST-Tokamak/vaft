"""Native shape-refinement wiring and preservation of the initial closure."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from tokamaker_fakes import make_fake_oft, make_inputs
from vaft.code.tokamaker import TokaMakerConfig, prepare_shape_refinement, run_tokamaker, fixed_to_free
from vaft.code.tokamaker import closure
from vaft.code.tokamaker._oft import import_oft
from vaft.code.tokamaker.config import TokaMakerResult
from vaft.process.equilibrium import solovev_example


def test_arclength_and_saddle_constraints_use_physical_bounds(tmp_path, monkeypatch):
    eq = solovev_example(resolution=33)
    refine = prepare_shape_refinement(eq, {'PF1': 123.}, current_bounds={'PF1': (-1000., 1000.)},
                                      boundary_samples=16, x_points=[(.3, -.2)],
                                      regularization=.02, current_scale_A=100.)
    assert refine.isoflux_points_m.shape == (17, 2)
    assert refine.isoflux_points_m[-1] == pytest.approx([.3, -.2])
    calls, _ = make_fake_oft(monkeypatch)
    cls = import_oft().TokaMaker
    for name in ('set_isoflux', 'set_saddles', 'set_coil_bounds'):
        monkeypatch.setattr(cls, name, lambda self, *a, _name=name, **kw: calls.append((_name, a, kw)), raising=False)
    inputs = make_inputs(tmp_path)
    inputs.coil_currents = {'PF1': 123.}
    result = run_tokamaker(inputs, TokaMakerConfig(workdir=tmp_path), refinement=refine)
    assert result.ok
    names = [c[0] for c in calls]
    assert names.index('init_psi') < names.index('set_isoflux') < names.index('solve')
    isoflux_call = calls[names.index('set_isoflux')]
    assert isoflux_call[2]['weights'][:-1] == pytest.approx(np.ones(16))
    assert isoflux_call[2]['weights'][-1] == pytest.approx(100.)
    assert calls[names.index('set_saddles')][1][0] == pytest.approx(np.array([[.3, -.2]]))
    assert calls[names.index('set_coil_bounds')][1][0] == {'PF1': (-1000., 1000.)}
    term = calls[names.index('coil_reg_term')]
    assert term[1:] == ({'PF1': .01}, pytest.approx(1.23), .02)
    assert result.scalars['shape_refinement']['reference_currents_A'] == {'PF1': 123.}


@pytest.mark.parametrize('bounds', [{}, {'PF1': (-100., 100.)}, {'PF1': (-np.inf, 1000.)}])
def test_refinement_requires_complete_finite_bounds_containing_start(bounds):
    with pytest.raises(ValueError):
        prepare_shape_refinement(solovev_example(resolution=33), {'PF1': 123.}, current_bounds=bounds)


def test_initial_and_refined_results_remain_separate(tmp_path, monkeypatch):
    eq = solovev_example(resolution=33)
    fit = SimpleNamespace(currents_A={'PF1': 123.}, status='accepted', accepted=True)
    monkeypatch.setattr(closure, 'fit_free_boundary_coils', lambda *a, **kw: fit)
    monkeypatch.setattr(closure, 'prepare_tokamaker_inputs', lambda m, c: SimpleNamespace(mesh_file=tmp_path/'mesh'))
    monkeypatch.setattr(closure, 'as_equilibrium', lambda *a, **kw: eq)
    calls = []
    def run(inputs, cfg, **kw):
        calls.append((cfg, kw))
        current = 140. if kw else 123.
        return TokaMakerResult(0, cfg.workdir, gfile=tmp_path/'gfile',
                              scalars={'coil_currents_A': {'PF1': current}})
    monkeypatch.setattr(closure, 'run_tokamaker', run)
    result = fixed_to_free(eq, {}, tmp_path/'run', refine_shape=True,
                           fit_options={'current_bounds': {'PF1': (-200., 200.)}})
    assert result.status == result.refinement_status == 'converged'
    assert result.initial_currents_A == result.realized_currents_A == {'PF1': 123.}
    assert result.refined_currents_A == {'PF1': 140.}
    assert calls[0][1] == {}
    assert calls[1][1]['refinement'].reference_currents_A == {'PF1': 123.}
    assert calls[1][0].mesh_file == tmp_path/'mesh'
    assert calls[1][0].init_equilibrium is eq
    assert result.forward is not result.refined


def test_saddles_can_be_explicitly_disabled_for_limited_case():
    eq = solovev_example(resolution=33)
    refine = prepare_shape_refinement(eq, {'PF1': 0.}, current_bounds={'PF1': (-1., 1.)}, x_points=[])
    assert refine.saddle_points_m.shape == (0, 2)
    shifted = replace(eq, lcfs=None)
    with pytest.raises(ValueError, match='closed'):
        prepare_shape_refinement(shifted, {'PF1': 0.}, current_bounds={'PF1': (-1., 1.)})


@pytest.mark.parametrize('verification_fails', [False, True])
def test_frozen_verification_uses_same_native_state_and_preserves_refinement(
        tmp_path, monkeypatch, verification_fails):
    calls, _ = make_fake_oft(monkeypatch, solve_error='frozen failure' if verification_fails else None,
                             solve_error_at=2)
    cls = import_oft().TokaMaker
    monkeypatch.setattr(cls, 'set_isoflux',
                        lambda self, points, **kw: calls.append(('set_isoflux', points)), raising=False)
    monkeypatch.setattr(cls, 'set_saddles',
                        lambda self, points, **kw: calls.append(('set_saddles', points)), raising=False)
    monkeypatch.setattr(cls, 'set_coil_bounds',
                        lambda self, bounds: calls.append(('set_coil_bounds', bounds)), raising=False)
    refine = prepare_shape_refinement(solovev_example(resolution=33),
                                      {'PF1': -640., 'PF2': 320.},
                                      current_bounds={'PF1': (-1000., 1000.),
                                                      'PF2': (-1000., 1000.)}, x_points=[])
    result = run_tokamaker(make_inputs(tmp_path), TokaMakerConfig(workdir=tmp_path),
                           refinement=refine, verify_refinement=True)
    names = [call[0] for call in calls]
    assert result.ok and result.gfile.is_file()
    assert names.index('save_eqdsk') < names.index('set_isoflux', names.index('solve')+1)
    assert names.count('solve') == 2 and names.count('init') == 1
    assert calls[names.index('set_isoflux', names.index('solve')+1)][1] is None
    assert calls[names.index('set_saddles', names.index('solve')+1)][1] is None
    assert result.verified_free is not None
    assert result.verified_free.ok is not verification_fails
    assert result.verified_free.scalars['shape_constraints_cleared'] is True
    if verification_fails:
        assert result.verified_free.gfile is None
        assert 'frozen failure' in result.verified_free.error
    else:
        assert result.verified_free.gfile.is_file()
        assert result.verified_free.scalars['coil_currents_A'] == {'PF1': -640., 'PF2': 320.}


def test_every_saddle_shares_lcfs_flux_without_duplicate_constraints():
    eq = solovev_example(resolution=33)
    options = {'current_bounds': {'PF1': (-1., 1.)}, 'boundary_samples': 16}
    boundary = prepare_shape_refinement(eq, {'PF1': 0.}, x_points=[], **options)
    existing = boundary.isoflux_points_m[3]
    extra = np.array([.3, -.44])
    refined = prepare_shape_refinement(eq, {'PF1': 0.},
                                       x_points=[existing, extra, extra], **options)
    assert len(refined.isoflux_points_m) == 17
    for saddle in refined.saddle_points_m:
        assert np.count_nonzero(np.all(refined.isoflux_points_m == saddle, axis=1)) == 1
