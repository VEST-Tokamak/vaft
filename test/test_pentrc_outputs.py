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
    for method in run_n3.methods():
        _, torque = pentrc.torque_profile(run_n3, method)
        assert torque[-1] == pytest.approx(
            run_n3.attrs[f"T_total_{method}"], rel=1e-12
        )


def test_the_energy_divides_by_two_n_not_by_two(run_n3):
    """PENTRC's own reduction (torque.F90:2029). At n = 1 the two agree."""
    for method in run_n3.methods():
        _, energy = pentrc.energy_profile(run_n3, method)
        assert energy[-1] == pytest.approx(
            run_n3.attrs[f"dW_total_{method}"], rel=1e-12
        )

        summed = run_n3.complex_quantity("T", method).sum(axis=1)
        naive = np.imag(summed)[-1] / 2.0
        # The discriminating check: /2 is wrong by exactly n.
        assert naive == pytest.approx(3.0 * energy[-1], rel=1e-12)
        assert naive != pytest.approx(run_n3.attrs[f"dW_total_{method}"], rel=1e-6)


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


def test_the_two_methods_sit_on_different_grids(run_n3):
    """55 and 34 points in the reference; nothing may share or infer one."""
    fgar = run_n3.psi_norm("fgar")
    tgar = run_n3.psi_norm("tgar")
    assert fgar.size != tgar.size
    assert run_n3.profile_psi_norm().size not in {fgar.size, tgar.size}


def test_the_two_methods_do_not_agree(run_n3):
    """Neither is a refinement of the other, so neither can be a default."""
    _, fgar = pentrc.torque_profile(run_n3, "fgar")
    _, tgar = pentrc.torque_profile(run_n3, "tgar")
    assert fgar[-1] != pytest.approx(tgar[-1], rel=1e-3)


def test_the_method_is_required_and_checked(run_n3):
    with pytest.raises(PentrcFormatError) as excinfo:
        pentrc.torque_profile(run_n3, "whichever")
    assert "fgar" in str(excinfo.value)


def test_a_method_the_run_did_not_compute_is_named(tmp_path):
    path = write_pentrc_output(tmp_path / "fgar_only.nc", methods=("fgar",))
    with pentrc.read_pentrc_output(path) as run:
        assert run.methods() == ("fgar",)
        with pytest.raises(PentrcFormatError) as excinfo:
            pentrc.torque_profile(run, "tgar")
        assert "did not compute" in str(excinfo.value)


def test_the_harmonics_come_back_unsummed(run_n3):
    """One ell is a resonance; the torque is the sum, and summing is a step."""
    unsummed = run_n3.complex_quantity("T", "fgar")
    assert unsummed.shape == (run_n3.psi_norm("fgar").size, ELL.size)
    assert np.iscomplexobj(unsummed)
    assert list(run_n3.ell()) == list(ELL)

    _, torque = pentrc.torque_profile(run_n3, "fgar")
    assert torque == pytest.approx(np.real(unsummed.sum(axis=1)))
    # A single harmonic is not the torque.
    assert not np.allclose(np.real(unsummed[:, 0]), torque)


def test_every_torque_quantity_is_readable(run_n3):
    for quantity in pentrc.TORQUE_QUANTITIES:
        values = run_n3.complex_quantity(quantity, "fgar")
        assert values.shape == (run_n3.psi_norm("fgar").size, ELL.size)

    with pytest.raises(PentrcFormatError):
        run_n3.complex_quantity("not_a_quantity", "fgar")


def test_profiles_come_back_in_pentrcs_own_units(run_n3):
    """This layer converts nothing; T_e is in eV because PENTRC says so."""
    assert run_n3.profile("T_e").shape == run_n3.profile_psi_norm().shape
    assert "eV" in pentrc.PROFILE_VARIABLES["T_e"]

    with pytest.raises(PentrcFormatError) as excinfo:
        run_n3.profile("temperature")
    assert "carries no" in str(excinfo.value)


def test_a_closed_output_says_so(tmp_path):
    path = write_pentrc_output(tmp_path / "x.nc")
    output = pentrc.read_pentrc_output(path)
    output.close()
    with pytest.raises(PentrcFormatError):
        output.methods()
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
