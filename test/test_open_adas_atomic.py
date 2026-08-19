from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from omas import ODS

from vaft.data.open_adas import (
    ADF11FormatError,
    ADASDownloadError,
    default_adf11_files,
    get_adf11_path,
    read_adf11,
)
from vaft.formula.atomic import (
    fractional_abundances,
    interpolate_adf11,
    line_cooling_coefficient,
)
from vaft.process.atomic import (
    compute_line_radiation_power_series,
    compute_time_match_atol,
    normalize_atomic_symbol,
)


def _adf11_text(blocks: list[list[float]]) -> str:
    lines = ["2 2 2 / synthetic", "", "10.0 14.0", "0.0 2.0"]
    for index, values in enumerate(blocks, start=1):
        lines.append(f"---------------- /IPRT=1/IGRD=1/TYPE=TEST/Z1={index}/")
        lines.append(" ".join(str(value) for value in values))
    return "\n".join(lines) + "\n"


def _write_tables(tmp_path: Path) -> tuple[Path, Path, Path]:
    acd = tmp_path / "acd96_c.dat"
    scd = tmp_path / "scd96_c.dat"
    plt = tmp_path / "plt96_c.dat"
    acd.write_text(_adf11_text([[-8.0] * 4, [-8.0] * 4]), encoding="ascii")
    scd.write_text(_adf11_text([[-8.0] * 4, [-8.0] * 4]), encoding="ascii")
    plt.write_text(_adf11_text([[-20.0] * 4, [-19.0] * 4]), encoding="ascii")
    read_adf11.cache_clear()
    return acd, scd, plt


def test_read_adf11_multiline_grids_and_blocks(tmp_path):
    source = tmp_path / "acd96_c.dat"
    source.write_text(
        "2 3 3 / synthetic\n\n10.0 12.0\n14.0\n0.0 1.0\n2.0\n"
        "--- /IPRT=1/IGRD=1/TYPE=TEST/Z=1/\n"
        "-8 -8 -8 -8\n-8 -8 -8 -8 -8\n"
        "--- /IPRT=1/IGRD=1/TYPE=TEST/Z=2/\n"
        "-7 -7 -7 -7 -7\n-7 -7 -7 -7\n",
        encoding="ascii",
    )
    read_adf11.cache_clear()

    table = read_adf11(source)

    assert table.file_type == "acd"
    assert table.log_coefficients.shape == (2, 3, 3)
    np.testing.assert_array_equal(table.log_density_cm3, [10.0, 12.0, 14.0])
    np.testing.assert_array_equal(table.log_temperature_eV, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(table.z1, [1, 2])
    for array in (
        table.log_density_cm3,
        table.log_temperature_eV,
        table.log_coefficients,
        table.z1,
        table.metastables,
    ):
        assert not array.flags.writeable


def test_read_adf11_rejects_malformed_file(tmp_path):
    source = tmp_path / "acd96_c.dat"
    source.write_text("2 2 nope\n", encoding="ascii")
    read_adf11.cache_clear()
    with pytest.raises(ADF11FormatError, match="too short|Invalid ADF11 header"):
        read_adf11(source)


def test_cache_hit_uses_explicit_cache_directory(tmp_path):
    cached = tmp_path / "acd96_c.dat"
    cached.write_text("cached", encoding="ascii")

    assert get_adf11_path("acd96_c.dat", cache_dir=tmp_path) == cached


def test_cache_miss_downloads_atomically(tmp_path):
    payload = (_adf11_text([[-8.0] * 4, [-8.0] * 4]) + "C" * 1200).encode("ascii")

    class Response:
        status = 200

        def read(self):
            return payload

    with patch("vaft.data.open_adas.urlopen", return_value=nullcontext(Response())):
        result = get_adf11_path("acd96_c.dat", cache_dir=tmp_path)

    assert result.read_bytes() == payload
    assert not list(tmp_path.glob(".acd96_c.dat.*"))


def test_download_failure_is_explicit(tmp_path):
    with patch("vaft.data.open_adas.urlopen", side_effect=OSError("offline")):
        with pytest.raises(ADASDownloadError, match="offline"):
            get_adf11_path("acd96_c.dat", cache_dir=tmp_path)


def test_default_files_reject_unknown_species():
    assert default_adf11_files("C")["plt"] == "plt96_c.dat"
    assert default_adf11_files("T") == default_adf11_files("D")
    with pytest.raises(KeyError, match="No default ADF11 files"):
        default_adf11_files("U")


def test_interpolation_fractional_abundance_and_cooling(tmp_path):
    acd, scd, plt = _write_tables(tmp_path)
    ne = np.asarray([[1.0e16], [1.0e20]])
    te = np.asarray([1.0, 100.0])

    rates = interpolate_adf11(read_adf11(acd), ne, te, multiply_density=True)
    assert rates.shape == (2, 2, 2)
    fractions = fractional_abundances(ne, te, acd, scd)
    assert fractions.shape == (2, 2, 3)
    np.testing.assert_allclose(fractions.sum(axis=-1), 1.0)
    np.testing.assert_allclose(fractions, 1.0 / 3.0)

    cooling = line_cooling_coefficient("C", ne, te, acd=acd, scd=scd, plt=plt)
    np.testing.assert_allclose(cooling, (11.0 / 3.0) * 1.0e-26, rtol=1.0e-13)


def test_line_cooling_interpolates_plt_density(tmp_path):
    acd, scd, plt = _write_tables(tmp_path)
    plt.write_text(
        _adf11_text([
            [-20.0, -18.0, -20.0, -18.0],
            [-19.0, -17.0, -19.0, -17.0],
        ]),
        encoding="ascii",
    )
    read_adf11.cache_clear()

    cooling = line_cooling_coefficient(
        "C",
        np.asarray([1.0e16, 1.0e20]),
        np.asarray([10.0, 10.0]),
        acd=acd,
        scd=scd,
        plt=plt,
    )

    assert cooling[1] > 50.0 * cooling[0]


def _minimal_ods() -> ODS:
    ods = ODS()
    ods["equilibrium.time_slice.0.time"] = 0.0
    ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = [0.0, 1.0]
    ods["core_profiles.profiles_1d.0.time"] = 0.0
    ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = [0.0, 1.0]
    ods["core_profiles.profiles_1d.0.electrons.density"] = [1.0e19, 1.0e19]
    ods["core_profiles.profiles_1d.0.electrons.temperature"] = [100.0, 100.0]
    return ods


def test_process_line_radiation_fallback_volume_and_offline_degradation(caplog):
    ods = _minimal_ods()
    with patch("vaft.process.atomic.line_cooling_coefficient", return_value=np.asarray([2.0e-31, 2.0e-31])):
        power = compute_line_radiation_power_series(
            ods,
            eq_indices=[0],
            eq_times=np.asarray([0.0]),
            volume_series=np.asarray([2.0]),
            line_radiation_species=["C"],
            impurity_fractions={"C": 0.01},
        )
    np.testing.assert_allclose(power, [4.0e5])

    with patch(
        "vaft.process.atomic.line_cooling_coefficient",
        side_effect=ADASDownloadError("offline"),
    ):
        power = compute_line_radiation_power_series(
            ods,
            eq_indices=[0],
            eq_times=np.asarray([0.0]),
            volume_series=np.asarray([2.0]),
            line_radiation_species=["C"],
            impurity_fractions={"C": 0.01},
        )
    np.testing.assert_array_equal(power, [0.0])
    assert "line radiation is zero" in caplog.text


def test_atomic_inputs_must_be_finite_and_positive(tmp_path):
    acd, scd, _ = _write_tables(tmp_path)
    with pytest.raises(ValueError, match="finite positive"):
        fractional_abundances([0.0], [100.0], acd, scd)


def test_atomic_symbol_normalization_handles_full_element_names():
    assert normalize_atomic_symbol("carbon6+") == "C"
    assert normalize_atomic_symbol("neon") == "Ne"
    assert normalize_atomic_symbol("nickel2+") == "Ni"
    assert normalize_atomic_symbol("not-an-element") is None


def test_atomic_process_validates_time_and_volume_axes():
    with pytest.raises(ValueError, match="base_atol"):
        compute_time_match_atol([0.0, 1.0], base_atol=-1.0)

    ods = _minimal_ods()
    with pytest.raises(ValueError, match="eq_times"):
        compute_line_radiation_power_series(
            ods,
            eq_indices=[0],
            eq_times=np.asarray([]),
            volume_series=np.asarray([2.0]),
        )
    with pytest.raises(ValueError, match="volume_series"):
        compute_line_radiation_power_series(
            ods,
            eq_indices=[0],
            eq_times=np.asarray([0.0]),
            volume_series=np.asarray([]),
        )
