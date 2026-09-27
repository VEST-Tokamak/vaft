"""Reading a PENTRC output: what is a torque, what is not, and on which grid."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.code import pentrc
from vaft.code.pentrc import PentrcFormatError

from pentrc_nc_fixtures import ELL, pentrc_dataset, write_pentrc_output


@pytest.fixture
def run_n3(tmp_path):
    """A run at n = 3, so the 2n energy divisor is distinguishable from 2."""
    path = write_pentrc_output(tmp_path / "pentrc_output_n3.nc", n_tor=3)
    with pentrc.read_pentrc_output(path) as output:
        yield output


def test_the_mode_number_comes_from_the_file(run_n3):
    assert run_n3.n_tor == 3


def test_a_file_without_a_mode_number_is_refused(tmp_path):
    dataset = pentrc_dataset()
    del dataset.attrs["n"]
    path = tmp_path / "not_pentrc.nc"
    dataset.to_netcdf(path)
    dataset.close()
    with pytest.raises(PentrcFormatError) as excinfo:
        pentrc.read_pentrc_output(path)
    assert "'n' global attribute" in str(excinfo.value)


def test_the_torque_reproduces_the_files_own_total(run_n3):
    """The file states the total; summing the profile must agree with it."""
    for method, grid in run_n3.available():
        _, torque = pentrc.torque_profile(run_n3, method, grid)
        suffix = f"_{method}" if grid == "lsode" else f"_{method}_{grid}"
        assert torque[-1] == pytest.approx(run_n3.attrs[f"T_total{suffix}"], rel=1e-12)


def test_the_energy_divides_by_two_n_not_by_two(run_n3):
    """PENTRC's own reduction (torque.F90:2029). At n = 1 the two agree."""
    for method, grid in run_n3.available():
        _, energy = pentrc.energy_profile(run_n3, method, grid)
        suffix = f"_{method}" if grid == "lsode" else f"_{method}_{grid}"
        assert energy[-1] == pytest.approx(run_n3.attrs[f"dW_total{suffix}"], rel=1e-12)

        summed = run_n3.complex_quantity("T", method, grid).sum(axis=1)
        naive = np.imag(summed)[-1] / 2.0
        # The discriminating check: /2 is wrong by exactly n.
        assert naive == pytest.approx(3.0 * energy[-1], rel=1e-12)


def test_at_n_equals_one_the_two_divisors_coincide(tmp_path):
    """Which is why a fixture at n = 1 could not have caught it."""
    path = write_pentrc_output(tmp_path / "n1.nc", n_tor=1)
    with pentrc.read_pentrc_output(path) as run:
        _, energy = pentrc.energy_profile(run, "fgar")
        summed = run.complex_quantity("T", "fgar").sum(axis=1)
        assert np.imag(summed)[-1] / 2.0 == pytest.approx(energy[-1], rel=1e-12)


def test_the_imaginary_part_is_not_returned_as_a_torque(run_n3):
    _, torque = pentrc.torque_profile(run_n3, "fgar")
    summed = run_n3.complex_quantity("T", "fgar").sum(axis=1)
    assert torque == pytest.approx(np.real(summed))
    assert not np.allclose(torque, np.abs(summed))


# ------------------------------------------------- the heat-moment trap


def test_a_heat_moment_run_is_refused_rather_than_read_as_a_torque(tmp_path):
    """Identical names, identical shapes, identical Nm units, different quantity."""
    path = write_pentrc_output(tmp_path / "heat.nc", n_tor=3, moment="heat")
    with pentrc.read_pentrc_output(path) as run:
        # Nothing about the variable's shape or unit gives it away.
        assert run.complex_quantity("T", "fgar").shape[1] == ELL.size
        assert run.long_name("T", "fgar") == "Integrated A*e*psi'*Gamma/2pi"

        with pytest.raises(PentrcFormatError) as excinfo:
            pentrc.torque_profile(run, "fgar")
        assert pentrc.TORQUE_LONG_NAME in str(excinfo.value)

        with pytest.raises(PentrcFormatError):
            pentrc.energy_profile(run, "fgar")


def test_a_torque_run_passes_the_same_check(tmp_path):
    path = write_pentrc_output(tmp_path / "torque.nc", n_tor=3)
    with pentrc.read_pentrc_output(path) as run:
        assert run.long_name("T", "fgar") == pentrc.TORQUE_LONG_NAME
        assert pentrc.torque_profile(run, "fgar")[1].size > 0


def test_a_file_declaring_no_long_name_is_allowed_through(tmp_path):
    """An older writer may omit it; refusing then would reject valid files."""
    dataset = pentrc_dataset(n_tor=1)
    del dataset["T_fgar"].attrs["long_name"]
    path = tmp_path / "bare.nc"
    dataset.to_netcdf(path)
    dataset.close()
    with pentrc.read_pentrc_output(path) as run:
        assert run.long_name("T", "fgar") == ""
        assert pentrc.torque_profile(run, "fgar")[1].size > 0


# ----------------------------------------------------- methods and grids


def test_all_eighteen_methods_are_named_with_pentrcs_own_description():
    assert len(pentrc.TORQUE_METHODS) == 18
    assert pentrc.TORQUE_METHODS["fgar"].startswith("Full general-aspect-ratio")
    assert pentrc.TORQUE_METHODS["tgar"].startswith("Trapped particle general")
    assert pentrc.TORQUE_METHODS["pgar"].startswith("Passing particle general")
    assert set(pentrc.TORQUE_GRIDS) == {"lsode", "equil", "input"}


def test_the_registries_cannot_be_mutated():
    with pytest.raises(TypeError):
        pentrc.TORQUE_METHODS["zzzz"] = "no"
    with pytest.raises(TypeError):
        pentrc.TORQUE_GRIDS["zzzz"] = "no"


def test_the_two_calculations_sit_on_different_grids(run_n3):
    """Lengths differ because lsode steps adaptively."""
    fgar = run_n3.psi_norm("fgar")
    tgar = run_n3.psi_norm("tgar")
    assert fgar.size != tgar.size
    assert run_n3.profile_psi_norm().size not in {fgar.size, tgar.size}


def test_a_passing_only_run_reports_what_it_has(tmp_path):
    """TORQUE_METHODS covers all eighteen, so pgar alone is found."""
    path = write_pentrc_output(
        tmp_path / "pgar.nc", n_tor=1, calculations={("pgar", "lsode"): 11}
    )
    with pentrc.read_pentrc_output(path) as run:
        assert run.available() == (("pgar", "lsode"),)
        assert pentrc.torque_profile(run, "pgar")[0].size == 11


def test_a_method_on_a_grid_the_run_did_not_use_says_which_it_did(tmp_path):
    """'Did not compute fgar' is wrong when it computed fgar on another grid."""
    path = write_pentrc_output(
        tmp_path / "equil.nc", n_tor=1, calculations={("fgar", "equil"): 7}
    )
    with pentrc.read_pentrc_output(path) as run:
        assert run.available() == (("fgar", "equil"),)
        assert pentrc.torque_profile(run, "fgar", "equil")[0].size == 7

        with pytest.raises(PentrcFormatError) as excinfo:
            pentrc.torque_profile(run, "fgar", "lsode")
        message = str(excinfo.value)
        assert "computed 'fgar' on ['equil']" in message
        assert "did not compute" not in message


def test_a_method_absent_from_every_grid_says_so(tmp_path):
    path = write_pentrc_output(
        tmp_path / "one.nc", n_tor=1, calculations={("fgar", "lsode"): 7}
    )
    with pentrc.read_pentrc_output(path) as run:
        with pytest.raises(PentrcFormatError) as excinfo:
            pentrc.torque_profile(run, "clar")
        assert "did not compute 'clar' on any grid" in str(excinfo.value)


def test_one_method_on_two_grids_is_reported_as_two_calculations(tmp_path):
    path = write_pentrc_output(
        tmp_path / "both.nc", n_tor=1,
        calculations={("fgar", "lsode"): 13, ("fgar", "equil"): 7},
    )
    with pentrc.read_pentrc_output(path) as run:
        assert run.available() == (("fgar", "equil"), ("fgar", "lsode"))
        assert run.psi_norm("fgar", "equil").size == 7
        assert run.psi_norm("fgar", "lsode").size == 13


@pytest.mark.parametrize("bad", ["whichever", "FGAR", ""])
def test_an_unknown_method_is_refused(run_n3, bad):
    with pytest.raises(PentrcFormatError) as excinfo:
        pentrc.torque_profile(run_n3, bad)
    assert "methods" in str(excinfo.value)


def test_an_unknown_grid_is_refused(run_n3):
    with pytest.raises(PentrcFormatError) as excinfo:
        pentrc.torque_profile(run_n3, "fgar", "whatever")
    assert "lsode" in str(excinfo.value)


# ----------------------------------------------------------- ell and rank


def test_the_harmonics_come_back_unsummed_and_are_integers(run_n3):
    """One ell is a resonance; the torque is the sum, and summing is a step."""
    unsummed = run_n3.complex_quantity("T", "fgar")
    assert unsummed.shape == (run_n3.psi_norm("fgar").size, ELL.size)
    assert np.iscomplexobj(unsummed)

    ell = run_n3.ell()
    assert np.issubdtype(ell.dtype, np.integer)
    assert list(ell) == list(ELL)

    _, torque = pentrc.torque_profile(run_n3, "fgar")
    assert torque == pytest.approx(np.real(unsummed.sum(axis=1)))
    assert not np.allclose(np.real(unsummed[:, 0]), torque)


def test_the_docstring_distinguishes_ell_from_leff():
    """ell is the integer bounce harmonic; leff = ell - sigma n q is not here."""
    from vaft.code.pentrc import outputs

    # Collapsed, so the assertions survive line wrapping.
    prose = " ".join(outputs.__doc__.split())
    assert "``ell`` is the integer bounce harmonic" in prose
    assert "leff = ell - sigma n q" in prose
    assert "does not write to this file" in prose


def test_every_torque_quantity_is_readable(run_n3):
    for quantity in pentrc.TORQUE_QUANTITIES:
        values = run_n3.complex_quantity(quantity, "fgar")
        assert values.shape == (run_n3.psi_norm("fgar").size, ELL.size)

    with pytest.raises(PentrcFormatError):
        run_n3.complex_quantity("not_a_quantity", "fgar")


# -------------------------------------------------------------- profiles


def test_profiles_come_back_in_pentrcs_own_units(run_n3):
    """This layer converts nothing; T_e is in eV because PENTRC says so."""
    assert run_n3.profile("T_e").shape == run_n3.profile_psi_norm().shape
    assert "eV" in pentrc.PROFILE_VARIABLES["T_e"]


def test_profile_refuses_a_variable_that_is_not_on_the_profile_grid(run_n3):
    """profile('T_fgar') would otherwise return a rank-3 array on another grid."""
    with pytest.raises(PentrcFormatError) as excinfo:
        run_n3.profile("T_fgar")
    assert "not a profile variable" in str(excinfo.value)
    assert "complex_quantity" in str(excinfo.value)


def test_profile_names_what_the_run_actually_wrote(run_n3):
    with pytest.raises(PentrcFormatError) as excinfo:
        run_n3.profile("omega_N")   # a real profile variable this fixture omits
    assert "carries no" in str(excinfo.value)


# ----------------------------------------------------------------- misc


def test_a_closed_output_says_so(tmp_path):
    path = write_pentrc_output(tmp_path / "x.nc")
    output = pentrc.read_pentrc_output(path)
    output.close()
    with pytest.raises(PentrcFormatError):
        output.available()
    output.close()  # idempotent


def test_the_torque_accumulates_radially(run_n3):
    """It is an integrated torque, so the edge value is not a point value."""
    _, torque = pentrc.torque_profile(run_n3, "fgar")
    assert torque.size > 1
    assert torque[0] != pytest.approx(torque[-1])


def test_the_reducers_stay_inside_the_submodule():
    """``torque_profile`` and ``energy_profile`` are not bound on ``vaft.code``.

    Unqualified they would be ambiguous -- ``vaft.code.transp`` also produces
    a torque profile, from an entirely different quantity -- so they are
    reached as ``pentrc.torque_profile``. The container and the reader, whose
    names say PENTRC, are exported.
    """
    import vaft.code as code

    assert "pentrc" in code.__all__
    assert "read_pentrc_output" in code.__all__
    assert "torque_profile" not in code.__all__
    assert "energy_profile" not in code.__all__
    assert code.pentrc.torque_profile is pentrc.torque_profile
