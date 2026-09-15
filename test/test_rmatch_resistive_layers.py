"""Per-rational-surface `eta`/`massden` for rmatch (#716).

The packaged `rmatch.in` supplies one scalar where RMATCH reads an array of
one value per rational surface (`match.f:99`, guarded at `:183`), so it stops
before writing `globalsol.bin`. These cover the composition that fills it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.code.gpec import RDCONOptions
from vaft.code.gpec._solvers import (
    _namelist_array,
    _replace_namelist_scalar,
    enable_rdcon_matching_output,
)
from vaft.data.resources import data_path
from vaft.process.equilibrium import resistive_layer_at, resistive_layer_parameters


def _profiles(points: int = 64):
    """A monotonic, physically ordered VEST-like pair of kinetic profiles."""
    psi_norm = np.linspace(0.0, 1.0, points)
    t_e = 100.0 * (1.0 - psi_norm**2) + 1.0  # eV, hot core, cool edge
    n_e = 1.0e19 * (1.0 - psi_norm**2) + 1.0e17  # m^-3
    return psi_norm, t_e, n_e


# ---------------------------------------------------------------------------
# the physics composition
# ---------------------------------------------------------------------------

def test_resistivity_rises_outward_as_the_plasma_cools():
    """Spitzer goes as T_e^-3/2, so a cooling profile must give rising eta."""
    psi_norm, t_e, n_e = _profiles()
    got = resistive_layer_at([0.1, 0.5, 0.9], psi_norm=psi_norm, t_e=t_e, n_e=n_e)
    assert np.all(np.diff(got["eta"]) > 0.0)
    assert np.all(np.diff(got["mass_density"]) < 0.0)


def test_resistivity_follows_the_spitzer_scaling_exactly():
    """Not merely monotonic: the T_e^-3/2 law itself."""
    psi_norm, t_e, n_e = _profiles()
    got = resistive_layer_at([0.2, 0.7], psi_norm=psi_norm, t_e=t_e, n_e=n_e)
    ratio = got["eta"][1] / got["eta"][0]
    expected = (got["t_e"][0] / got["t_e"][1]) ** 1.5
    assert ratio == pytest.approx(expected, rel=1e-9)


def test_mass_density_is_the_ion_mass_times_the_density():
    from vaft.formula.constants import MI_P

    psi_norm, t_e, n_e = _profiles()
    got = resistive_layer_at(
        [0.3], psi_norm=psi_norm, t_e=t_e, n_e=n_e, ion_mass_amu=2.0
    )
    assert got["mass_density"][0] == pytest.approx(got["n_e"][0] * 2.0 * MI_P)


def test_a_hydrogen_default_matches_vest():
    """VEST runs hydrogen, so the default must not be deuterium."""
    psi_norm, t_e, n_e = _profiles()
    hydrogen = resistive_layer_at([0.3], psi_norm=psi_norm, t_e=t_e, n_e=n_e)
    deuterium = resistive_layer_at(
        [0.3], psi_norm=psi_norm, t_e=t_e, n_e=n_e, ion_mass_amu=2.0
    )
    assert deuterium["mass_density"][0] == pytest.approx(
        2.0 * hydrogen["mass_density"][0]
    )


def test_a_profile_that_reaches_zero_temperature_is_reported_not_clipped():
    """The divergence is real; hiding it would hide an unusable input."""
    psi_norm = np.linspace(0.0, 1.0, 16)
    t_e = 100.0 * (1.0 - psi_norm)  # exactly zero at the separatrix
    n_e = np.full_like(psi_norm, 1.0e19)
    got = resistive_layer_at([1.0], psi_norm=psi_norm, t_e=t_e, n_e=n_e)
    assert not np.isfinite(got["eta"][0])


def test_mismatched_profile_length_is_refused():
    psi_norm, t_e, n_e = _profiles()
    with pytest.raises(ValueError, match="coordinate points"):
        resistive_layer_at([0.5], psi_norm=psi_norm, t_e=t_e[:-1], n_e=n_e)


def test_a_non_positive_ion_mass_is_refused():
    psi_norm, t_e, n_e = _profiles()
    with pytest.raises(ValueError, match="ion_mass_amu"):
        resistive_layer_at(
            [0.5], psi_norm=psi_norm, t_e=t_e, n_e=n_e, ion_mass_amu=0.0
        )


def test_the_finding_variant_returns_one_value_per_surface_it_found():
    psi_norm, t_e, n_e = _profiles()
    q = 2.0 + 10.0 * psi_norm**2
    got = resistive_layer_parameters(psi_norm, q, 1, t_e=t_e, n_e=n_e)
    assert got["eta"].size == got["m"].size > 0
    assert got["mass_density"].size == got["m"].size


# ---------------------------------------------------------------------------
# the namelist rewrite
# ---------------------------------------------------------------------------

def test_the_rewrite_replaces_the_whole_line_including_its_comment():
    """A dangling comment would be read as a continuation of the list."""
    source = data_path("gpec/rmatch.in").read_text(encoding="utf-8")
    assert "eta=8e-8" in source  # the scalar this exists to replace

    out = _replace_namelist_scalar(
        source, "eta", _namelist_array("eta", [1e-6, 2e-6], note="Ohm m")
    )
    eta_lines = [l for l in out.splitlines() if l.strip().startswith("eta=")]
    assert len(eta_lines) == 1
    assert "1.000000e-06, 2.000000e-06" in eta_lines[0]
    assert "Resistivity at each rational surface" not in out
    assert len(out.splitlines()) == len(source.splitlines())


def test_the_rewrite_leaves_every_other_setting_alone():
    source = data_path("gpec/rmatch.in").read_text(encoding="utf-8")
    out = _replace_namelist_scalar(source, "eta", _namelist_array("eta", [1e-6]))
    for untouched in ("match_flag=t", "nroot=5", "msing=20", "model=\"deltac\""):
        assert untouched in out


def test_massden_is_matched_on_its_own_and_not_by_a_prefix():
    """`massden` must not be found by a loose search for `den` or `eta`."""
    source = data_path("gpec/rmatch.in").read_text(encoding="utf-8")
    out = _replace_namelist_scalar(source, "eta", _namelist_array("eta", [1e-6]))
    assert "massden=3.3e-07" in out  # untouched by the eta replacement


def test_a_template_without_the_assignment_is_refused():
    with pytest.raises(ValueError, match="has no `eta` assignment"):
        _replace_namelist_scalar("&RMATCH_INPUT\n/\n", "eta", "    eta=1.0\n")


# ---------------------------------------------------------------------------
# the opt-in
# ---------------------------------------------------------------------------

def test_the_feature_is_off_until_all_three_profiles_are_given():
    psi_norm, t_e, n_e = _profiles()
    assert RDCONOptions().has_kinetic_profiles is False
    assert RDCONOptions(t_e=t_e).has_kinetic_profiles is False
    assert RDCONOptions(t_e=t_e, n_e=n_e).has_kinetic_profiles is False
    assert RDCONOptions(
        t_e=t_e, n_e=n_e, psi_norm=psi_norm
    ).has_kinetic_profiles is True


# ---------------------------------------------------------------------------
# rdcon has to write what rmatch reads
# ---------------------------------------------------------------------------

def test_enabling_the_matching_output_flips_only_bin_delmatch(tmp_path):
    """`rdcon/gal.f:2007` writes both files rmatch opens under this one flag.

    `out_galsol`/`bin_galsol` add a dozen `galsol_left_*`/`galsol_right_*`
    files that rmatch never reads, so they stay off.
    """
    path = tmp_path / "rdcon.in"
    path.write_text(data_path("gpec/rdcon.in").read_text(encoding="utf-8"), encoding="utf-8")

    assert enable_rdcon_matching_output(path) is True
    text = path.read_text(encoding="utf-8")
    assert "bin_delmatch=t" in text
    assert "out_galsol=f" in text
    assert "bin_galsol=f" in text
    # the explanatory comment survives, unlike the eta/massden lines
    assert "Output solution for rmatch" in text


def test_enabling_is_idempotent_and_reports_when_nothing_changed(tmp_path):
    path = tmp_path / "rdcon.in"
    path.write_text("&GAL_OUTPUT\n    bin_delmatch=t\n/\n", encoding="utf-8")
    assert enable_rdcon_matching_output(path) is False
    assert path.read_text(encoding="utf-8").count("bin_delmatch=t") == 1


def test_a_template_without_the_flag_is_left_alone(tmp_path):
    """Not an error: a template that never had it needs no rewrite."""
    path = tmp_path / "rdcon.in"
    path.write_text("&GAL_OUTPUT\n/\n", encoding="utf-8")
    assert enable_rdcon_matching_output(path) is False
