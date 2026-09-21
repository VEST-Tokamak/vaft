"""The GPEC mode spectrum and island-overlap figures (migration slice V5).

Everything here is derived from what the mapper wrote: the perturbed flux on
its ``(psi_N, m)`` grid, and the rational-surface geometry beside it.  The
fixtures go through the mapper for the same reason the resonant-plot tests do
-- what is under test is whether the IDS carries enough.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from gpec_nc_fixtures import write_control_nc, write_cylindrical_nc, write_profile_nc
from vaft.machine_mapping.gpec_ideal import gpec_ideal
from vaft.plot.backend.recipes import RECIPES


def _run(path, *, n=1, rational_q=(2.0, 3.0), **control):
    write_control_nc(path, n=n, **control)
    write_cylindrical_nc(path, n=n)
    return write_profile_nc(path, n=n, rational_q=rational_q)


@pytest.fixture
def mapped(tmp_path):
    """An ODS built the way a real ideal-GPEC run reaches one."""
    native = _run(tmp_path, n=1)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})
    return ods, native


def test_the_map_is_the_mapped_array_itself(mapped):
    """The acceptance gate is numeric: same arrays in, same arrays drawn."""
    ods, native = mapped
    model = RECIPES["mhd_linear_field_spectrum"].builder(ods, unit="T")

    np.testing.assert_array_equal(model.r, native["psi_n"])
    np.testing.assert_array_equal(model.z, native["m_out"])
    # `Jbgradpsi` reaches the container as (m, psi) and the IDS as (psi, m);
    # Field2D wants (vertical, horizontal), which is (m, psi) again.
    np.testing.assert_allclose(model.values, np.abs(native["Jbgradpsi"]))


def test_the_map_names_both_axes_and_is_not_drawn_to_an_equal_aspect(mapped):
    ods, _ = mapped
    model = RECIPES["mhd_linear_field_spectrum"].builder(ods)

    assert r"\psi_N" in model.x_label
    assert "m" in model.y_label
    assert model.aspect_equal is False


def test_the_unit_is_shown_and_the_values_carry_it(mapped):
    """Gauss and tesla differ by 1e4; a map that implies one is unreadable."""
    ods, _ = mapped
    tesla = RECIPES["mhd_linear_field_spectrum"].builder(ods, unit="T")
    gauss = RECIPES["mhd_linear_field_spectrum"].builder(ods, unit="G")

    assert "[T]" in tesla.value_label and "[G]" in gauss.value_label
    np.testing.assert_allclose(gauss.values, tesla.values * 1e4)


def test_the_spectrum_defaults_to_the_outermost_mapped_surface(mapped):
    ods, native = mapped
    model = RECIPES["mhd_linear_spectrum_b_field_perturbed"].builder(ods, unit="T")

    outermost = float(np.max(native["psi_n"]))
    assert f"{outermost:.4f}" in model.title
    assert "outermost mapped surface" in model.title
    series = model.series[0]
    np.testing.assert_array_equal(series.x, native["m_out"])
    np.testing.assert_allclose(
        series.y, np.abs(native["Jbgradpsi"][:, int(np.argmax(native["psi_n"]))])
    )


def test_the_spectrum_reports_the_surface_it_drew_not_the_one_requested(mapped):
    """GPEC's radial grid clusters around the singular surfaces, so a request
    almost never lands on a sample; a title naming the request would put a
    number on the figure that is not where the data came from."""
    ods, native = mapped
    target = float(np.median(native["psi_n"]))
    model = RECIPES["mhd_linear_spectrum_b_field_perturbed"].builder(ods, psi_n=target)

    drawn = native["psi_n"][int(np.argmin(np.abs(native["psi_n"] - target)))]
    assert f"{float(drawn):.4f}" in model.title
    assert f"nearest to {target:g}" in model.title


def test_a_surface_outside_the_mapped_grid_is_refused(mapped):
    """psi_N = 1 is outside a GPEC run's grid, and snapping to the boundary
    would answer a different question under the caller's label."""
    ods, _ = mapped
    with pytest.raises(ValueError, match="outside the mapped radial grid"):
        RECIPES["mhd_linear_spectrum_b_field_perturbed"].builder(ods, psi_n=1.0)


def test_chirikov_is_the_process_layer_reduction_of_the_derived_widths(mapped):
    from vaft.process.perturbation import chirikov

    ods, _ = mapped
    widths = RECIPES["mhd_linear_profile_island_width"].builder(ods)
    model = RECIPES["mhd_linear_profile_chirikov"].builder(ods)

    np.testing.assert_array_equal(model.series[0].x, widths.series[0].x)
    np.testing.assert_allclose(
        model.series[0].y,
        chirikov(widths.series[0].x, widths.series[0].y, definition="surface"),
    )


def test_the_overlap_criterion_is_drawn_as_data(mapped):
    """K = 1 is the physics of the figure, so it reaches to_xarray and the
    plotly backend with the rest of it rather than being an axhline."""
    ods, _ = mapped
    model = RECIPES["mhd_linear_profile_chirikov"].builder(ods)

    criterion = model.series[1]
    assert "K = 1" in criterion.label
    np.testing.assert_array_equal(criterion.y, [1.0, 1.0])
    np.testing.assert_array_equal(
        criterion.x, [model.series[0].x.min(), model.series[0].x.max()]
    )


def test_overlap_is_refused_when_there_is_no_neighbour_to_overlap_with(tmp_path):
    """One island cannot overlap anything, and `chirikov` would report the
    distance to a surface that is not there."""
    _run(tmp_path, n=1, rational_q=(2.0,))
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})

    with pytest.raises(ValueError, match="measured against a neighbour"):
        RECIPES["mhd_linear_profile_chirikov"].builder(ods)


# --- D-05: the pedestal boundary a result carries, never a hard-coded one ----

def test_no_pedestal_boundary_is_drawn_unless_the_caller_carries_one(mapped):
    ods, _ = mapped
    assert RECIPES["mhd_linear_profile_chirikov"].builder(ods).reference_lines == ()


def test_a_pedestal_top_is_drawn_with_the_method_that_produced_it(mapped):
    from vaft.process.profile import PedestalTop

    ods, _ = mapped
    pedestal = PedestalTop(
        position=0.93, method="eped_fit", quantity="p_total", coordinate="psi_norm",
    )
    model = RECIPES["mhd_linear_profile_chirikov"].builder(ods, pedestal=pedestal)

    (line,) = model.reference_lines
    assert line.x == pytest.approx(0.93)
    assert "eped_fit" in line.label and "p_total" in line.label


def test_a_fallback_pedestal_says_so_rather_than_passing_as_a_fit(mapped):
    from vaft.process.profile import PedestalTop

    ods, _ = mapped
    pedestal = PedestalTop(
        position=0.85, method="fallback", quantity="p_total", coordinate="psi_norm",
        reason="no profile available",
    )
    (line,) = RECIPES["mhd_linear_profile_chirikov"].builder(
        ods, pedestal=pedestal
    ).reference_lines

    assert "fallback" in line.label
    assert "no profile available" in line.label


def test_a_pedestal_in_another_coordinate_is_refused(mapped):
    from vaft.process.profile import PedestalTop

    ods, _ = mapped
    pedestal = PedestalTop(
        position=0.93, method="eped_fit", quantity="p_total",
        coordinate="rho_tor_norm",
    )
    with pytest.raises(ValueError, match="rho_tor_norm"):
        RECIPES["mhd_linear_profile_chirikov"].builder(ods, pedestal=pedestal)


# --- which cell a multi-mode product draws by default ------------------------

def test_a_multi_mode_gpec_product_defaults_to_the_strongly_driven_mode(tmp_path):
    """GPEC's `energy_perturbed` is the positive energy of the driven
    response, not a signed potential energy, so ranking it the way DCON's is
    ranked picks the mode the coils barely excite.
    """
    ods = ODS(consistency_check=False)
    for n, energy in ((1, 10.0), (2, 1e-9)):
        run = tmp_path / f"n{n}"
        run.mkdir()
        _run(run, n=n, energy_vacuum=energy, energy_surface=0.0, energy_plasma=0.0)
        gpec_ideal(ods, str(run), {"modes": [1, 2], "mode": n})

    for name in (
        "mhd_linear_field_spectrum",
        "mhd_linear_spectrum_b_field_perturbed",
        "mhd_linear_profile_chirikov",
        "mhd_linear_profile_resonant_flux",
    ):
        assert "n=1" in RECIPES[name].builder(ods).title, name


def test_a_driven_energy_is_not_labelled_as_a_potential_energy(tmp_path):
    """`delta W` is DCON's symbol and its sign is a stability verdict; GPEC's
    number is a magnitude, and borrowing the symbol invites a reader to read a
    verdict off it."""
    _run(tmp_path, n=1, energy_vacuum=1.0, energy_surface=2.0, energy_plasma=3.0)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})

    title = RECIPES["mhd_linear_field_spectrum"].builder(ods).title
    assert r"\delta W" not in title
    assert "6" in title  # 1 + 2 + 3 J, reported with its own sign
