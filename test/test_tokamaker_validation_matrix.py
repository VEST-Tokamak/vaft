"""Small analytic topology and representability checks for the #1608 matrix."""
from dataclasses import replace
import numpy as np
import pytest
from vaft.code.tokamaker.profiles import equilibrium_to_tokamaker_profiles
from vaft.process.equilibrium import derive_boundary_representation, fit_free_boundary_coils
from validation.fixed_free_1608.matrix import machine_case, target_case


@pytest.mark.parametrize('family', ['solovev', 'guazzotto_freidberg', 'guazzotto_pedestal'])
@pytest.mark.parametrize('topology', ['limited', 'lower_single_null', 'double_null'])
def test_analytic_matrix_has_canonical_sources_and_requested_topology(family, topology):
    eq = target_case(family, topology, pedestal=.1 if family == 'guazzotto_pedestal' else 0.)
    eq, _, _ = machine_case(eq, topology)
    assert eq.convention.cocos == 11
    assert derive_boundary_representation(eq).topology.value == topology
    profiles = equilibrium_to_tokamaker_profiles(eq)
    assert profiles.ip_A == pytest.approx(eq.ip)
    assert np.isfinite(profiles.pprime).all() and np.isfinite(profiles.ffprime).all()


def test_current_pedestal_changes_representable_volume_source():
    base = target_case('guazzotto_freidberg', 'limited', resolution=33)
    pedestal = target_case('guazzotto_pedestal', 'limited', pedestal=.1, resolution=33)
    assert not np.allclose(base.pprime, pedestal.pprime)
    assert not np.allclose(base.ffprime, pedestal.ffprime)
    eq, machine, _ = machine_case(pedestal, 'limited')
    fit = fit_free_boundary_coils(eq, machine, boundary_samples=24, regularization=1e-5)
    assert fit.optimizer_success and np.isfinite(list(fit.currents_A.values())).all()


@pytest.mark.parametrize('term', ['pressure_pedestal', 'bootstrap_fraction', 'mach_number'])
def test_surface_current_or_flow_contribution_is_explicitly_rejected(term):
    eq = target_case('guazzotto_freidberg', 'limited', resolution=33)
    model = replace(eq.metadata['model'], **{term: .1})
    unsupported = replace(eq, metadata={**eq.metadata, 'model': model})
    with pytest.raises(ValueError, match='surface-current or flow'):
        equilibrium_to_tokamaker_profiles(unsupported)
    _, machine, _ = machine_case(eq, 'limited')
    with pytest.raises(ValueError, match='surface-current or flow'):
        fit_free_boundary_coils(unsupported, machine)
