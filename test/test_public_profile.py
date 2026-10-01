"""ITPA PR08 profile database: UFILE/0D readers and the ODS mapping (#1205 stage 3).

Offline: the discharge is synthetic, written in the release's formats (UFILE
blocks with abutting fixed-format numbers, CSV and fixed-width 0D files, the
missing markers).  Only the pinned real discharge is fetched, and only with
``VAFT_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from vaft.data.public import fetch_pr08, pr08_mapping_coverage, pr08_to_omas, read_pr08
from vaft.data.public import itpa_profile
from vaft.data.public.itpa_profile import read_pr08_0d, read_ufiles

omas = pytest.importorskip("omas")

RHO_C = [0.25, 0.5, 0.75]  # zone centres
RHO_B = [0.5, 0.75, 1.0]  # zone boundaries
TIMES = [0.19, 0.20]


def _numbers(values):
    # 13-character fixed format, six per line; negatives abut their neighbour.
    text = "".join(f"{v:13.6E}" for v in values)
    return "\n".join(text[i:i + 78] for i in range(0, len(text), 78))


def _block_2d(name, unit, rho, times, values):
    return (
        f"  9999 test 2                  ;-SHOT #- IDENTIFICATION\n"
        f" UNKNOWN                       ;-SHOT DATE-\n"
        f" 0                             ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-\n"
        f"                               ;-INDEPENDENT VARIABLE (X) LABEL-\n"
        f"                     Seconds   ;-INDEPENDENT VARIABLE (Y) LABEL-\n"
        f" {name:<19} {unit:<9} ;-DEPENDENT VARIABLE LABEL-\n"
        f" 0                             ;-STATUS FLAG\n"
        f"         {len(rho):>2}                    ;-# OF X PTS-\n"
        f"         {len(times):>2}                    ;-# OF Y PTS-  X, Y, F(X,Y) DATA FOLLOW:\n"
        f"{_numbers(list(rho) + list(times) + list(np.ravel(values)))}\n"
        ";----END-OF-DATA-----------------COMMENTS:----------\n"
        f"  {name} synthetic\n"
        + "*" * 80 + "\n" + "*" * 80 + "\n"
    )


def _block_1d(name, unit, times, values):
    return (
        f"  9999 test 1                  ;-SHOT #- IDENTIFICATION\n"
        f" UNKNOWN                       ;-SHOT DATE-\n"
        f" 0                             ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-\n"
        f"                     SECONDS   ;-INDEPENDENT VARIABLE LABEL-\n"
        f" {name:<19} {unit:<9} ;-DEPENDENT VARIABLE LABEL-\n"
        f" 0                             ;-STATUS FLAG\n"
        f"          {len(times)}                    ;-# OF PTS-  X, F(X) DATA FOLLOW:\n"
        f"{_numbers(list(times) + list(values))}\n"
        ";----END-OF-DATA-----------------COMMENTS:----------\n"
        + "*" * 80 + "\n"
    )


def _profile(scale, rho=RHO_C):
    # rows = times, columns = rho; decreasing in rho
    return np.array([[scale * (1.0 - 0.5 * r) * (1 + 0.1 * k) for r in rho] for k in range(len(TIMES))])


@pytest.fixture()
def discharge_dir(tmp_path):
    two_d = "".join([
        _block_2d("TE", "eV", RHO_C, TIMES, _profile(1000.0)),
        _block_2d("TEEB", "eV", RHO_C, TIMES, _profile(50.0)),
        _block_2d("NE", "m-3", RHO_C, TIMES, _profile(4.0e19)),
        _block_2d("TI", "eV", RHO_C, TIMES, _profile(800.0)),
        _block_2d("QEI", "W/m3", RHO_C, TIMES, _profile(1.0e4)),
        _block_2d("QRAD", "W/m3", RHO_C, TIMES, _profile(2.0e3)),
        _block_2d("QOHM", "W/m3", RHO_C, TIMES, _profile(5.0e4)),
        # boundary grid: its own equilibrium block
        _block_2d("Q", "0", RHO_B, TIMES, _profile(2.0, RHO_B)),  # PR08 gives |q|
        _block_2d("CURTOT", "A/m2", RHO_C, TIMES, _profile(-3.0e5)),  # abutting negatives
        _block_2d("VOLUME", "m3", RHO_B, TIMES, _profile(3.0, RHO_B)),
        _block_2d("CHIE", "m2/s", RHO_B, TIMES, _profile(1.5, RHO_B)),
        # centre grid, but the equilibrium block is on the boundary grid
        _block_2d("PRES", "Pa", RHO_C, TIMES, _profile(1.0e4)),
        # measured profile on its own grid, with a missing point
        _block_2d("TEXP", "eV", [0.1, 0.6], TIMES[:1], [[900.0, -9.999e-09]]),
        _block_2d("NEEB", "m-3", [0.1, 0.6], TIMES[:1], [[1.0e18, 1.0e18]]),  # off-grid error bar
        _block_2d("VROT", "rad/s", RHO_C, TIMES, _profile(1.0e4)),
    ])
    one_d = _block_1d("IP", "Amps", [0.19, 0.25], [8.0e5, 8.1e5]) + _block_1d(
        "BT", "Tesla", [0.19, 0.20], [-0.5, -0.5]
    )
    (tmp_path / "pr08_test_9999_2d.dat").write_text(two_d)
    (tmp_path / "pr08_test_9999_1d.dat").write_text(one_d)
    (tmp_path / "pr08_test_9999_0d.dat").write_text(
        "TOK,SHOT,TIME,PGASA,PGASZ,EVAP,BT,IP,RGEO\n"
        "TEST,9999,1.900E-01,2.000E+00,1.000E+00,????????,-9.999E-09,8.000E+05,1.000E+00\n"
    )
    (tmp_path / "pr08_test_9999_com.dat").write_text("  synthetic discharge for tests\n")
    return tmp_path


@pytest.fixture()
def discharge(discharge_dir):
    return read_pr08(discharge_dir, "test", 9999)


# ---------------------------------------------------------------- readers


def test_ufile_2d_is_time_by_rho_with_x_fastest(discharge):
    te = discharge.two_d["TE"]
    assert te.values.shape == (2, 3)
    np.testing.assert_allclose(te.rho, RHO_C)
    np.testing.assert_allclose(te.time, TIMES)
    np.testing.assert_allclose(te.values, _profile(1000.0))
    assert te.unit == "eV"


def test_ufile_negative_numbers_that_abut_are_split(discharge):
    np.testing.assert_allclose(discharge.two_d["CURTOT"].values, _profile(-3.0e5))


def test_ufile_missing_marker_becomes_nan(discharge):
    texp = discharge.two_d["TEXP"]
    assert texp.values[0, 0] == pytest.approx(900.0)
    assert np.isnan(texp.values[0, 1])


def test_ufile_1d(discharge):
    ip = discharge.one_d["IP"]
    assert ip.rho is None
    np.testing.assert_allclose(ip.values, [8.0e5, 8.1e5])


def test_ufile_count_mismatch_is_an_error(tmp_path):
    block = _block_1d("IP", "Amps", [0.1, 0.2], [1.0, 2.0]).replace("          2    ", "          3    ")
    path = tmp_path / "bad_1d.dat"
    path.write_text(block)
    with pytest.raises(ValueError, match="expected"):
        read_ufiles(path)


def test_0d_csv_missing_markers(discharge):
    row = discharge.zero_d.iloc[0]
    assert row["TOK"] == "TEST"
    assert row["EVAP"] is None          # ????????
    assert np.isnan(row["BT"])          # -9.999E-09
    assert row["PGASA"] == pytest.approx(2.0)


def test_0d_fixed_width_wraps_seven_to_a_line(tmp_path):
    # The DIII-D layout: names wrapped seven per line, then values the same way,
    # including a blank cell that must stay in its column.
    names = [f"V{i:02d}" for i in range(9)]
    values = ["1.000E+00", "2.000E+00", "", "4.000E+00", "????????", "6.000E+00", "7.000E+00",
              "8.000E+00", "9.000E+00"]
    def wrap(cells):
        return "\n".join("".join(f"{c:>11}" for c in cells[i:i + 7]) for i in range(0, len(cells), 7))
    path = tmp_path / "wrapped_0d.dat"
    path.write_text(wrap(names) + "\n" + wrap(values) + "\n")
    row = read_pr08_0d(path).iloc[0]
    assert row["V01"] == pytest.approx(2.0)
    assert row["V02"] is None                 # blank cell stays in place
    assert row["V03"] == pytest.approx(4.0)
    assert row["V08"] == pytest.approx(9.0)   # second line lands in the right columns


def test_0d_fixed_width_with_name_like_values(tmp_path):
    # Values such as 'D3D' look like names; the first line with a number or
    # a missing marker starts the values.
    names = ["TOK", "SHOT", "TIME", "PHASE", "PGASA", "ECHMODE", "IGRADB"]
    values = ["D3D", "81507", "3.800E+00", "HSELM", "2.000E+00", "????????", "-9999999"]
    text = "".join(f"{n:>11}" for n in names) + "\n" + "".join(f"{v:>11}" for v in values) + "\n"
    path = tmp_path / "fixed_0d.dat"
    path.write_text(text)
    frame = read_pr08_0d(path)
    row = frame.iloc[0]
    assert list(frame.columns) == names
    assert row["TOK"] == "D3D" and row["PHASE"] == "HSELM"
    assert row["TIME"] == pytest.approx(3.8)
    assert row["ECHMODE"] is None and np.isnan(row["IGRADB"])


def test_file_hashes_are_recorded(discharge):
    assert set(discharge.files) == {"0d", "1d", "2d", "com"}
    assert all(len(v) == 64 for v in discharge.files.values())


# ---------------------------------------------------------------- ODS mapping


def test_core_profiles_on_the_reference_grid(discharge):
    ods = pr08_to_omas(discharge)
    slice0 = "core_profiles.profiles_1d.0"
    np.testing.assert_allclose(ods[f"{slice0}.grid.rho_tor_norm"], RHO_C)  # rho = sqrt(Phi/Phi_a)
    np.testing.assert_allclose(ods[f"{slice0}.electrons.temperature"], _profile(1000.0)[0])
    np.testing.assert_allclose(ods[f"{slice0}.electrons.temperature_error_upper"], _profile(50.0)[0])
    np.testing.assert_allclose(ods[f"{slice0}.ion.0.temperature"], _profile(800.0)[0])
    assert ods[f"{slice0}.ion.0.element.0.a"] == pytest.approx(2.0)
    assert ods[f"{slice0}.ion.0.label"] == "D"
    np.testing.assert_allclose(ods["core_profiles.time"], TIMES)
    assert len(ods["core_profiles.profiles_1d"]) == 2


def test_equilibrium_keeps_its_own_grid_and_skips_off_grid_signals(discharge):
    ods = pr08_to_omas(discharge)
    prof = "equilibrium.time_slice.0.profiles_1d"
    np.testing.assert_allclose(ods[f"{prof}.rho_tor_norm"], RHO_B)
    np.testing.assert_allclose(ods[f"{prof}.volume"], _profile(3.0, RHO_B)[0])
    assert f"{prof}.pressure" not in ods  # PRES is on the centre grid: not interpolated
    assert "equilibrium.time_slice.0.profiles_1d.psi" not in ods  # nothing invented


def test_q_is_signed_by_current_and_field_under_cocos_11(discharge):
    # IP > 0, BT < 0: q = sign(Ip) sign(B0) |q| < 0.
    ods = pr08_to_omas(discharge)
    np.testing.assert_allclose(ods["equilibrium.time_slice.0.profiles_1d.q"], -_profile(2.0, RHO_B)[0])
    # 0.20 s has BT but no IP sample: no sign, so no q and no ip there.
    assert "equilibrium.time_slice.1.profiles_1d.q" not in ods
    assert "equilibrium.time_slice.1.global_quantities.ip" not in ods
    assert ods["equilibrium.time_slice.0.global_quantities.ip"] == pytest.approx(8.0e5)


@pytest.mark.parametrize("bt", [-0.5, 0.5])
def test_a_q_file_that_is_already_signed_gets_the_documented_sign_not_a_double_one(tmp_path, bt):
    (tmp_path / "pr08_test_9999_2d.dat").write_text(_block_2d("Q", "0", RHO_B, TIMES, _profile(-2.0, RHO_B)))
    (tmp_path / "pr08_test_9999_1d.dat").write_text(
        _block_1d("IP", "Amps", [0.19, 0.25], [8.0e5, 8.1e5]) + _block_1d("BT", "Tesla", [0.19, 0.20], [bt, bt])
    )
    (tmp_path / "pr08_test_9999_0d.dat").write_text(
        "TOK,SHOT,TIME,PGASA,PGASZ,EVAP,BT,IP,RGEO\n"
        f"TEST,9999,1.900E-01,2.000E+00,1.000E+00,????????,{bt:.3E},8.000E+05,1.000E+00\n"
    )
    ods = pr08_to_omas(read_pr08(tmp_path, "test", 9999))
    # sign(Ip) sign(B0) |q|, as documented, whatever sign the file stored q with.
    np.testing.assert_allclose(ods["equilibrium.time_slice.0.profiles_1d.q"], np.sign(bt) * _profile(2.0, RHO_B)[0])


def test_contradicting_current_signs_leave_sign_bearing_quantities_out(discharge_dir):
    # As in MAST 8302: the 0D IP is negative, the 1D IP positive.
    path = discharge_dir / "pr08_test_9999_0d.dat"
    path.write_text(path.read_text().replace("8.000E+05,1.000E+00", "-8.000E+05,1.000E+00"))
    discharge = read_pr08(discharge_dir, "test", 9999)
    ods = pr08_to_omas(discharge)
    assert "equilibrium.time_slice.0.profiles_1d.q" not in ods
    assert "equilibrium.time_slice.0.global_quantities.ip" not in ods
    coverage = pr08_mapping_coverage(discharge).set_index("variable")
    assert coverage.loc["Q", "status"] == "unmapped"
    assert "IP sign differs" in coverage.loc["Q", "reason"]


def test_curtot_is_the_toroidal_current_density(discharge):
    ods = pr08_to_omas(discharge)
    np.testing.assert_allclose(ods["core_profiles.profiles_1d.0.j_tor"], _profile(-3.0e5)[0])
    assert "core_profiles.profiles_1d.0.j_total" not in ods


def test_source_signs(discharge):
    ods = pr08_to_omas(discharge)
    by_name = {
        ods[f"core_sources.source.{i}.identifier.name"]: i for i in range(len(ods["core_sources.source"]))
    }
    eq = f"core_sources.source.{by_name['collisional_equipartition']}.profiles_1d.0"
    rad = f"core_sources.source.{by_name['radiation']}.profiles_1d.0"
    np.testing.assert_allclose(ods[f"{eq}.electrons.energy"], -_profile(1.0e4)[0])
    np.testing.assert_allclose(ods[f"{eq}.total_ion_energy"], _profile(1.0e4)[0])
    np.testing.assert_allclose(ods[f"{rad}.electrons.energy"], -_profile(2.0e3)[0])
    assert ods[f"core_sources.source.{by_name['radiation']}.identifier.index"] == 200
    assert ods[f"core_sources.source.{by_name['ohmic']}.identifier.index"] == 7


def test_transport_diffusivity(discharge):
    ods = pr08_to_omas(discharge)
    root = "core_transport.model.0"
    assert ods[f"{root}.identifier.name"] == "transport_solver"
    np.testing.assert_allclose(ods[f"{root}.profiles_1d.0.grid_d.rho_tor_norm"], RHO_B)
    np.testing.assert_allclose(ods[f"{root}.profiles_1d.0.electrons.energy.d"], _profile(1.5, RHO_B)[0])


def test_provenance_keeps_source_names_and_units(discharge):
    ods = pr08_to_omas(discharge)
    sources = [
        ods[f"core_profiles.ids_properties.provenance.node.{i}.sources"][0]
        for i in range(len(ods["core_profiles.ids_properties.provenance.node"]))
    ]
    assert "PR08 test/9999 TE [eV]" in sources
    assert "synthetic discharge" in ods["dataset_description.ids_properties.comment"]
    assert ods["dataset_description.data_entry.pulse"] == 9999


def test_the_comment_file_is_read_as_utf_8_whatever_the_locale_says(discharge_dir, monkeypatch):
    # The PR08 text files are UTF-8 regardless of the reading machine's locale;
    # a Windows code page decoded a Korean comment into mojibake.
    (discharge_dir / "pr08_test_9999_com.dat").write_text("  주석: 합성 방전\n", encoding="utf-8")
    original = Path.read_text

    def locale_is_cp1252(self, encoding=None, errors=None, **kwargs):
        return original(self, encoding=encoding or "cp1252", errors=errors, **kwargs)

    monkeypatch.setattr(Path, "read_text", locale_is_cp1252)
    ods = pr08_to_omas(read_pr08(discharge_dir, "test", 9999))
    assert "주석: 합성 방전" in ods["dataset_description.ids_properties.comment"]


def test_coverage_lists_every_variable_with_a_reason(discharge):
    coverage = pr08_mapping_coverage(discharge).set_index("variable")
    assert coverage.loc["TE", "status"] == "mapped"
    assert coverage.loc["TEEB", "target"].endswith("_error_upper")
    assert coverage.loc["PRES", "status"] == "unmapped"
    assert "not interpolated" in coverage.loc["PRES", "reason"]
    assert "own grid" in coverage.loc["TEXP", "reason"]
    assert "species" in coverage.loc["VROT", "reason"]
    assert (coverage["status"].isin({"mapped", "partial", "unmapped"})).all()
    assert not (coverage.loc[coverage.status != "mapped", "reason"] == "").any()


def test_coverage_is_honest_about_error_bars_and_partial_slices(discharge):
    coverage = pr08_mapping_coverage(discharge).set_index("variable")
    # NE is mapped, but its error bar sits on another grid and was not written.
    assert coverage.loc["NE", "status"] == "mapped"
    assert coverage.loc["NEEB", "status"] == "unmapped"
    # IP exists at 0.19 s only: one of two slices.
    assert coverage.loc["IP", "status"] == "partial"
    assert coverage.loc["IP", "slices"] == "1/2"


def test_mapped_ods_renders_with_the_canonical_profile_plots(discharge):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import vaft

    ods = pr08_to_omas(discharge)
    fig, _ = vaft.omas.plot_electron_temperature_profile(ods, label="PR08 test")
    plt.close(fig)


# ---------------------------------------------------------------- fetch


def test_fetch_refuses_an_unpinned_discharge(tmp_path):
    with pytest.raises(KeyError, match="No hash"):
        fetch_pr08("jet", "12345", cache=tmp_path)


def test_fetch_refuses_a_partial_pin_set(tmp_path):
    with pytest.raises(KeyError, match="com"):
        fetch_pr08("jet", "12345", cache=tmp_path, sha256={"0d": "0" * 64, "1d": "0" * 64, "2d": "0" * 64})


def test_unpinned_fetch_marks_files_unverified(tmp_path, discharge_dir):
    target = tmp_path / "pr08" / "test" / "9999"

    def fake_download(url, destination, timeout, sha256):
        assert sha256 is None
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((discharge_dir / destination.name).read_bytes())

    with mock.patch("vaft.data.public._fetch._download", side_effect=fake_download):
        with pytest.warns(UserWarning, match="without a known hash"):
            result = fetch_pr08("test", 9999, cache=tmp_path, allow_unpinned=True)
    assert set(result.verification.values()) == {"unverified"}
    assert "unverified" in pr08_to_omas(result)["dataset_description.ids_properties.provenance.node.0.sources"][0]
    # A second call reuses the cache but still says it is unverified.
    with pytest.warns(UserWarning, match="cached file"):
        fetch_pr08("test", 9999, cache=tmp_path, allow_unpinned=True)
    assert (target / "pr08_test_9999_2d.dat").exists()


def test_fetch_uses_the_pinned_hashes(tmp_path, discharge_dir):
    pins = {
        kind: itpa_profile.sha256_of(discharge_dir / f"pr08_test_9999_{kind}.dat")
        for kind in ("0d", "1d", "2d", "com")
    }

    def fake_fetch(url, *, sha256, filename, cache, timeout):
        target = cache / filename
        cache.mkdir(parents=True, exist_ok=True)
        target.write_bytes((discharge_dir / filename).read_bytes())
        assert sha256 == pins[filename.split("_")[-1][:-4]]
        return target

    with mock.patch.object(itpa_profile, "fetch", side_effect=fake_fetch) as fetched:
        result = fetch_pr08("test", 9999, cache=tmp_path, sha256=pins)
    assert fetched.call_count == 4
    assert "TE" in result.two_d
    assert set(result.verification.values()) == {"pinned"}


def test_pinned_registry_is_complete():
    for key, pins in itpa_profile.PR08_PINNED.items():
        assert set(pins) == {"0d", "1d", "2d", "com"}, key
        assert all(len(v) == 64 for v in pins.values())


@pytest.mark.skipif(
    not os.environ.get("VAFT_NETWORK_TESTS"),
    reason="set VAFT_NETWORK_TESTS=1 to fetch a pinned PR08 discharge",
)
def test_real_mast_discharge():
    discharge = fetch_pr08("mast", 8302)
    ods = pr08_to_omas(discharge)
    te = np.asarray(ods["core_profiles.profiles_1d.0.electrons.temperature"])
    assert te[0] > te[-1] > 0.0  # peaked
    coverage = pr08_mapping_coverage(discharge).set_index("variable")
    assert int((coverage.status == "mapped").sum()) == 16
    # The release contradicts itself on the current direction (0D IP < 0,
    # 1D IP > 0), so q, ip and b0 stay out.
    assert coverage.loc["Q", "status"] == "unmapped"
    assert "IP sign differs" in coverage.loc["Q", "reason"]
    assert "equilibrium.time_slice.0.profiles_1d.q" not in ods
