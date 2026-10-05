"""Canonical source shape, convention and OFT normalization contracts."""

from dataclasses import replace

import numpy as np
import pytest

from vaft.code.tokamaker import TokaMakerConfig, equilibrium_to_tokamaker_profiles
from vaft.code.tokamaker.profiles import profiles_for_config, profile_targets
from vaft.process.equilibrium import convert_cocos, solovev_example


@pytest.mark.parametrize("cocos", [1, 2, 7, 11, 12, 17])
def test_cocos_full_weber_per_radian_and_public_orientation(cocos):
    eq = solovev_example(resolution=33)
    actual = equilibrium_to_tokamaker_profiles(convert_cocos(eq, cocos))
    np.testing.assert_allclose(actual.psi_n, np.linspace(0, 1, len(eq.psi_1d)), atol=1e-14)
    np.testing.assert_allclose(actual.pprime, -2 * np.pi * eq.pprime)
    np.testing.assert_allclose(actual.ffprime, -2 * np.pi * eq.ffprime)
    assert actual.axis_pressure_Pa == pytest.approx(eq.pressure[0])
    assert actual.ip_A == pytest.approx(eq.ip)


def test_profile_order_and_toroidal_field_sign_do_not_flip_normalized_coordinate():
    eq = solovev_example(resolution=33)
    reversed_tables = replace(eq, psi_1d=eq.psi_1d[::-1], pprime=eq.pprime[::-1],
                              ffprime=eq.ffprime[::-1], pressure=eq.pressure[::-1], bt0=-eq.bt0)
    original = equilibrium_to_tokamaker_profiles(eq)
    reversed_result = equilibrium_to_tokamaker_profiles(reversed_tables)
    np.testing.assert_allclose(reversed_result.psi_n, original.psi_n)
    np.testing.assert_allclose(reversed_result.ffprime, original.ffprime)
    assert reversed_result.axis_pressure_Pa == pytest.approx(original.axis_pressure_Pa)


def test_negative_canonical_current_fails_explicitly():
    eq = solovev_example(resolution=33)
    eq = replace(eq, ip=-eq.ip, convention=replace(eq.convention, ip_sign=None))
    with pytest.raises(ValueError, match="positive canonical Ip"):
        equilibrium_to_tokamaker_profiles(eq)


def test_explicit_tables_and_pure_pressure_normalization():
    cfg = TokaMakerConfig(profile_mode="explicit", profile_tables={
        "psi_n": [0., .5, 1.], "pprime": [1., .5, 0.], "ffprime": [0., 0., 0.],
        "axis_pressure_Pa": 100.})
    profiles = profiles_for_config(cfg)
    assert profile_targets(profiles, {"Ip": 1e5}) == {"pax": 100.}
    assert profiles.solver_tables()["pp_prof"]["type"] == "linterp"
    with pytest.raises(ValueError, match="conflicts"):
        profile_targets(profiles, {"Ip": 1e5, "R0": .4})
    with pytest.raises(ValueError, match="spanning"):
        profiles_for_config(replace(cfg, profile_tables={**cfg.profile_tables, "psi_n": [0., .5, .8]}))


def test_zero_pressure_does_not_enable_axis_pressure_constraint():
    eq = solovev_example(resolution=33)
    eq = replace(eq, pprime=np.zeros_like(eq.pprime), pressure=np.zeros_like(eq.pressure))
    profiles = equilibrium_to_tokamaker_profiles(eq)
    assert profile_targets(profiles, {"Ip": 1e5}) == {"Ip": 1e5}
    assert profiles_for_config(TokaMakerConfig()) is None


def test_prepared_inputs_bind_canonical_scales_and_respect_explicit_bt(tmp_path):
    from test_tokamaker_inputs import _build_ods
    from vaft.code.tokamaker import prepare_tokamaker_inputs

    eq = solovev_example(resolution=33)
    cfg = TokaMakerConfig(workdir=tmp_path, time=.4, profile_mode="equilibrium", profile_equilibrium=eq)
    prepared = prepare_tokamaker_inputs(_build_ods(), cfg)
    assert prepared.targets["Ip"] == pytest.approx(eq.ip)
    assert prepared.targets["pax"] == pytest.approx(eq.pressure[0])
    assert prepared.f0 == pytest.approx(eq.r0 * eq.bt0)
    overridden = prepare_tokamaker_inputs(_build_ods(), replace(cfg, bt0=.2))
    assert overridden.f0 == pytest.approx(.2 * cfg.major_r)
