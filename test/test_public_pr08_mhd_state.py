"""PR08 global MHD states as a canonical table (#1736).

Offline: the 0D files are synthetic, written in the release's CSV and
fixed-width formats; the directory index and downloads are mocked.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

from vaft.data.public import (
    MHD_STATE_COLUMNS,
    PROVENANCE_KINDS,
    mhd_state_coverage,
    pr08_mhd_state_table,
    pr08_release_inventory,
    pr08_to_omas,
    projection_coverage,
    read_pr08,
    read_pr08_zero_d,
)
from vaft.data.public import pr08_mhd_state
from vaft.data.public.itpa_profile import read_pr08_0d
from vaft.formula import boundaries as B
from vaft.formula.stability import beta_N_from_beta_a_B0_Ip

HEADER = "TOK,SHOT,TIME,PHASE,STATE,RGEO,RMAG,AMIN,KAPPA,DELTA,AREA,VOL,BT,IP,Q95,QAXIS,BEPMHD,BETMHD,BEPDIA,BETNMHD,LI"
#: a JET-like H-mode record; BT and IP negative as in JET's conventions
JETLIKE = "JET,52009,5.500E+00,HGELM,STEADY,2.950E+00,3.050E+00,9.500E-01,1.700E+00,2.000E-01,4.500E+00,8.000E+01," \
          "-2.400E+00,-2.500E+06,-3.300E+00,1.000E+00,4.000E-01,1.500E-02,4.200E-01,-1.700E+00,9.000E-01"


def _zero_d(tmp_path, *rows, machine="jet", shot="52009", header=HEADER):
    path = tmp_path / f"pr08_{machine}_{shot}_0d.dat"
    path.write_text("\n".join([header, *rows]) + "\n")
    return read_pr08_zero_d(path, machine, shot)


def test_a_fixed_width_file_that_repeats_its_names_has_no_row_of_names(tmp_path):
    """T-10, TEXTOR, JT-60U and Tore Supra files repeat the names block before each record."""
    names = "".join(f"{n:>11}" for n in ("TOK", "SHOT", "TIME", "IP"))
    first = "".join(f"{v:>11}" for v in ("T10", "33957", "5.456E-01", "1.980E+05"))
    second = "".join(f"{v:>11}" for v in ("T10", "33957", "7.411E-01", "1.990E+05"))
    path = tmp_path / "pr08_t10_33957_0d.dat"
    path.write_text("\n".join([names, first, names, second]) + "\n")
    table = read_pr08_0d(path)
    assert list(table["TOK"]) == ["T10", "T10"]
    np.testing.assert_allclose(table["TIME"], [0.5456, 0.7411])


def test_direct_quantities_keep_their_definition_unit_and_provenance(tmp_path):
    table = pr08_mhd_state_table([_zero_d(tmp_path, JETLIKE)])
    row = table.iloc[0]
    assert row["record_id"] == "JET:52009:5500" and row["machine_class"] == "conventional_tokamak"
    assert row["dataset_type"] == "experimental" and row["source_kind"] == "0D" and len(row["source_sha256"]) == 64
    assert row["plasma_current"] == pytest.approx(2.5)               # |IP| in MA
    assert row["toroidal_field"] == pytest.approx(2.4)               # |BT| at RGEO
    assert row["edge_safety_factor_95"] == pytest.approx(3.3)        # |Q95|, not q_psi
    assert row["toroidal_beta"] == pytest.approx(1.5)                # BETMHD fraction -> %
    assert row["normalized_beta"] == pytest.approx(1.7)              # |BETNMHD| as given
    assert row["internal_inductance_li3"] == pytest.approx(0.9)
    assert row["li3_reference_radius"] == pytest.approx(2.95)        # LI is normalised by RGEO
    assert row["poloidal_beta"] == pytest.approx(0.4) and row["poloidal_beta_diamagnetic"] == pytest.approx(0.42)
    for column in ("plasma_current", "edge_safety_factor_95", "normalized_beta", "internal_inductance_li3"):
        assert row[f"{column}_provenance"] == "source_direct"
    assert "edge_safety_factor" not in table.columns                 # the source has no boundary q_psi
    units = table.attrs["units"]
    assert units["plasma_current"] == "MA" and units["normalized_beta"] == "% m T/MA" and units["toroidal_beta"] == "%"
    sources = table.attrs["quantity_sources"]
    assert sources["internal_inductance_li3"]["source_variable"] == "LI"
    assert "R_geo" in sources["internal_inductance_li3"]["definition"]
    assert sources["plasma_current"]["transformation"] == "magnitude x1e-06"
    assert set(table.filter(like="_provenance").stack()) <= set(PROVENANCE_KINDS)


def test_table_units_are_the_registry_axis_units():
    """Every column that is a projection axis is declared in the axis unit, so boundaries are drawn."""
    from vaft.diagram._op_space import get_projection, list_projections

    units = {k: v.unit for k, v in MHD_STATE_COLUMNS.items()}
    checked = 0
    for key in list_projections():
        p = get_projection(key)
        for q in (p.x, p.y):
            if q.name in units:
                assert units[q.name] == q.unit, (key, q.name)
                checked += 1
    assert checked >= 18


def test_deterministic_coordinates_come_from_the_registered_functions(tmp_path):
    row = pr08_mhd_state_table([_zero_d(tmp_path, JETLIKE)]).iloc[0]
    a, r, b, kappa, delta, ip = 0.95, 2.95, 2.4, 1.7, 0.2, 2.5
    assert row["normalized_current"] == pytest.approx(ip / (a * b))
    assert row["kink_safety_factor_elliptic"] == pytest.approx(float(B.kink_coordinates(a, r, b, kappa, ip)))
    assert row["kink_safety_factor_cylindrical"] == pytest.approx(
        float(B.cylindrical_kink_coordinates(a, r, b, kappa, ip)))
    assert row["edge_safety_factor_95_estimate_iter"] == pytest.approx(
        float(B.iter_q95_coordinates(a, r, b, kappa, delta, ip)))
    # no CONFIG, so START's configuration constant is unknown and its estimate is not formed
    assert math.isnan(row["edge_safety_factor_95_estimate_start"])
    assert row["area_elongation"] == pytest.approx(4.5 / (math.pi * a * a))
    assert row["kink_safety_factor_elliptic_provenance"] == "deterministic_derived"
    # the source q95 and the VAFT estimate stay two columns
    assert row["edge_safety_factor_95"] != row["edge_safety_factor_95_estimate_iter"]


def test_normalized_beta_is_derived_only_where_the_source_lacks_it(tmp_path):
    without = JETLIKE.replace("-1.700E+00,9.000E-01", "-9.999E-09,9.000E-01")
    row = pr08_mhd_state_table([_zero_d(tmp_path, without)]).iloc[0]
    assert row["normalized_beta_provenance"] == "deterministic_derived"
    assert row["normalized_beta"] == pytest.approx(beta_N_from_beta_a_B0_Ip(1.5, 0.95, 2.4, 2.5))


def test_missing_stays_missing_and_impossible_values_are_rejected(tmp_path):
    row_text = JETLIKE.replace(",1.700E+00,2.000E-01,", ",0.000E+00,2.000E-01,")      # KAPPA = 0
    row_text = row_text.replace(",9.000E-01", ",-9.999E-09")                          # LI missing
    row = pr08_mhd_state_table([_zero_d(tmp_path, row_text)]).iloc[0]
    assert math.isnan(row["elongation"]) and row["elongation_provenance"] == "source_invalid"
    assert math.isnan(row["internal_inductance_li3"]) and row["internal_inductance_li3_provenance"] == "missing"
    assert math.isnan(row["li3_reference_radius"])
    # what needs kappa is missing too, never estimated
    assert math.isnan(row["kink_safety_factor_elliptic"]) and row["kink_safety_factor_elliptic_provenance"] == "missing"
    assert row["plasma_current_provenance"] == "source_direct"


def test_a_current_not_in_amperes_and_a_beta_in_percent_are_rejected_with_a_note(tmp_path):
    """JT-60U 16107/16168 store IP in MA; ITER scenario records store BETMHD in percent."""
    in_ma = JETLIKE.replace("-2.500E+06", "-2.500E+00")
    in_percent = JETLIKE.replace(",1.500E-02,", ",1.500E+00,").replace("5.500E+00,HGELM", "6.000E+00,HGELM")
    table = pr08_mhd_state_table([_zero_d(tmp_path, in_ma, in_percent)])
    first, second = table.iloc[0], table.iloc[1]
    assert math.isnan(first["plasma_current"]) and first["plasma_current_provenance"] == "source_invalid"
    assert "IP is not in A" in first["state_notes"] and math.isnan(first["normalized_current"])
    assert math.isnan(second["toroidal_beta"]) and "BETMHD" in second["state_notes"]
    assert second["plasma_current_provenance"] == "source_direct"


def test_lower_case_names_and_the_sign_flipped_missing_marker_are_read(tmp_path):
    """A few files write their names in lower case; +9.999E-09 is the missing marker too."""
    lower = HEADER.lower()
    flipped = JETLIKE.replace(",4.200E-01,", ",9.999E-09,")          # BEPDIA
    row = pr08_mhd_state_table([_zero_d(tmp_path, flipped, header=lower)]).iloc[0]
    assert row["time_s"] == pytest.approx(5.5) and row["plasma_current_provenance"] == "source_direct"
    assert math.isnan(row["poloidal_beta_diamagnetic"])
    assert row["poloidal_beta_diamagnetic_provenance"] == "missing"


def test_a_rejected_value_says_why_and_is_not_replaced(tmp_path):
    zero = JETLIKE.replace(",1.700E+00,2.000E-01,", ",0.000E+00,2.000E-01,").replace("-1.700E+00,9.000E-01",
                                                                                      "0.000E+00,9.000E-01")
    row = pr08_mhd_state_table([_zero_d(tmp_path, zero)]).iloc[0]
    assert "KAPPA = 0 cannot be elongation" in row["state_notes"]
    # a rejected source beta_N stays rejected; the derived one does not paper over it
    assert row["normalized_beta_provenance"] == "source_invalid" and math.isnan(row["normalized_beta"])


def test_t10_q95_is_the_iter_estimate_and_not_taken_as_an_equilibrium_q95(tmp_path):
    t10 = JETLIKE.replace("JET,52009", "T10,33957")
    row = pr08_mhd_state_table([_zero_d(tmp_path, t10, machine="t10", shot="33957")]).iloc[0]
    assert math.isnan(row["edge_safety_factor_95"]) and row["edge_safety_factor_95_provenance"] == "source_invalid"
    assert "not an equilibrium q95" in row["state_notes"]
    assert np.isfinite(row["edge_safety_factor_95_estimate_iter"])   # the VAFT estimate is its own column


@pytest.mark.parametrize("config, expected", [("LIM", "limiter"), ("IN", "limiter"), ("DN", "double_null"),
                                              ("LSN", None), ("", None)])
def test_the_start_estimate_takes_its_configuration_from_config(tmp_path, config, expected):
    header = HEADER + ",CONFIG"
    row = pr08_mhd_state_table([_zero_d(tmp_path, JETLIKE + f",{config}", header=header)]).iloc[0]
    a, r, b, kappa, delta, ip = 0.95, 2.95, 2.4, 1.7, 0.2, 2.5
    if expected is None:
        assert math.isnan(row["edge_safety_factor_95_estimate_start"])
        assert row["edge_safety_factor_95_estimate_start_provenance"] == "missing"
    else:
        assert row["edge_safety_factor_95_estimate_start"] == pytest.approx(
            float(B.start_q95_coordinates(a, r, b, kappa, delta, ip, configuration=expected)))


def test_a_repeated_time_keeps_a_unique_record_id(tmp_path):
    table = pr08_mhd_state_table([_zero_d(tmp_path, JETLIKE, JETLIKE)])
    assert list(table["record_id"]) == ["JET:52009:5500", "JET:52009:5500#2"]
    assert table["source_record_id"].str.endswith(("#0", "#1")).all()


def test_iter_records_are_design_and_mast_is_a_spherical_tokamak(tmp_path):
    iter_row = JETLIKE.replace("JET,52009", "ITER,10010100")
    mast_row = JETLIKE.replace("JET,52009", "MAST,8302")
    table = pr08_mhd_state_table([_zero_d(tmp_path, iter_row, machine="iter", shot="10010100"),
                                  _zero_d(tmp_path, mast_row, machine="mast", shot="8302")])
    assert list(table["dataset_type"]) == ["design", "experimental"]
    assert list(table["machine_class"]) == ["conventional_tokamak", "spherical_tokamak"]


def test_coverage_counts_states_quantities_and_projection_eligibility(tmp_path):
    no_li = JETLIKE.replace(",9.000E-01", ",-9.999E-09").replace("5.500E+00", "6.500E+00")
    table = pr08_mhd_state_table([_zero_d(tmp_path, JETLIKE, no_li)])
    coverage = mhd_state_coverage(table)
    assert coverage.loc["JET", "states"] == 2 and coverage.loc["JET", "internal_inductance_li3"] == 1
    projections = projection_coverage(table).set_index("projection")
    assert projections.loc["q95_li", "eligible"] == 1
    assert projections.loc["q95_li", "excluded_by"] == "internal_inductance_li3"
    assert projections.loc["troyon", "eligible"] == 2
    assert "not a column" in projections.loc["hugill", "excluded_by"]     # density is not in the 0D MHD table
    assert projections.loc["li_qa_wesson", "eligible"] == 0                # no boundary q_psi in PR08


def test_the_release_inventory_is_the_crawled_release():
    inventory = pr08_release_inventory()
    assert len(inventory) == 348 and not inventory.duplicated(["machine_dir", "shot"]).any()
    assert inventory["sha256_0d"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert set(inventory["machine_dir"]) == set(pr08_mhd_state.PR08_MACHINES)
    # every discharge is read from its own directory, never from the second copy in another shot's
    assert (inventory["shot"] == inventory["directory"]).all()
    jet = inventory[(inventory.machine_dir == "jet") & (inventory.shot == "38287")].iloc[0]
    assert jet["sha256_0d"].startswith("dcc4c561b596") and jet["kinds"] == ("0d", "1d", "2d", "com")


def test_the_live_inventory_is_read_from_the_directory_index(monkeypatch):
    pages = {
        "https://x/pr08/": '<a href="?C=N">x</a><a href="/PR08/">up</a><a href="jet/">jet/</a>',
        "https://x/pr08/jet/": '<a href="38285/">38285/</a>',
        "https://x/pr08/jet/38285/": "".join(f'<a href="{n}">' for n in (
            "pr08_jet_38285_0d.dat", "pr08_jet_38285_1d.dat", "pr08_jet_38285_c0d.dat",
            "pr08_jet_38287_0d.dat", "pr08_jet_38287_com.dat", "P52009.NBI")),
    }

    class _Response:
        def __init__(self, text):
            self.text = text

        def read(self):
            return self.text.encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(pr08_mhd_state.urllib.request, "urlopen", lambda url, timeout: _Response(pages[url]))
    live = pr08_mhd_state.pr08_inventory(base_url="https://x/pr08")
    assert list(zip(live["shot"], live["directory"])) == [("38285", "38285"), ("38287", "38285")]
    assert live.iloc[0]["kinds"] == ("0d", "1d")
    assert "pr08_jet_38285_c0d.dat" in live.iloc[0]["other_files"] and "P52009.NBI" in live.iloc[0]["other_files"]


def test_the_population_is_fetched_against_the_pinned_hashes(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(url, *, sha256, filename, cache, timeout):
        calls.append((url, sha256, filename))
        path = cache / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER + "\n" + JETLIKE + "\n")
        return path

    monkeypatch.setattr(pr08_mhd_state, "fetch", fake_fetch)
    discharges = pr08_mhd_state.fetch_pr08_population(machines=["mast"], cache=tmp_path)
    inventory = pr08_release_inventory()
    mast = inventory[inventory.machine_dir == "mast"]
    assert len(discharges) == len(mast) == len(calls)
    assert {c[1] for c in calls} == set(mast["sha256_0d"])
    assert all(re.search(r"/mast/[^/]+/pr08_mast_[^/]+_0d\.dat$", c[0]) for c in calls)


def test_the_table_agrees_with_the_ods_mapping_where_definitions_match(tmp_path):
    """|I_p| and R_geo at the 0D time: the table and pr08_to_omas read the same discharge alike."""
    pytest.importorskip("omas")
    from test_public_profile import RHO_B, TIMES, _block_1d, _block_2d, _profile

    (tmp_path / "pr08_test_9999_2d.dat").write_text(_block_2d("Q", "0", RHO_B, TIMES, _profile(2.0, RHO_B)))
    (tmp_path / "pr08_test_9999_1d.dat").write_text(
        _block_1d("IP", "Amps", [0.19, 0.20], [-8.0e5, -8.0e5]) + _block_1d("BT", "Tesla", [0.19, 0.20], [-0.5, -0.5]))
    (tmp_path / "pr08_test_9999_0d.dat").write_text(
        "TOK,SHOT,TIME,BT,IP,RGEO,AMIN\nTEST,9999,1.900E-01,-5.000E-01,-8.000E+05,1.000E+00,3.000E-01\n")
    ods = pr08_to_omas(read_pr08(tmp_path, "test", 9999))
    row = pr08_mhd_state_table([read_pr08_zero_d(tmp_path / "pr08_test_9999_0d.dat", "test", "9999")]).iloc[0]
    assert ods["equilibrium.time"][0] == pytest.approx(row["time_s"])
    assert abs(ods["equilibrium.time_slice.0.global_quantities.ip"]) * 1e-6 == pytest.approx(row["plasma_current"])
    assert ods["equilibrium.vacuum_toroidal_field.r0"] == pytest.approx(row["major_radius"])
    assert abs(ods["equilibrium.vacuum_toroidal_field.b0"][0]) == pytest.approx(row["toroidal_field"])


@pytest.mark.skipif(not __import__("os").environ.get("VAFT_NETWORK_TESTS"),
                    reason="set VAFT_NETWORK_TESTS=1 to fetch the pinned PR08 discharges")
@pytest.mark.parametrize("machine, shot", [("mast", "8302"), ("d3d", "81507"), ("jet", "19649")])
def test_real_discharges_agree_with_the_ods_mapping_at_the_0d_time(machine, shot):
    """On the discharges the ODS mapping is checked against: |I_p|, R_geo and |B_T| at a shared time."""
    pytest.importorskip("omas")
    from vaft.data.public import fetch_pr08

    discharge = fetch_pr08(machine, shot)
    ods = pr08_to_omas(discharge)
    from vaft.data.public._fetch import cache_dir

    zero_d = cache_dir() / "pr08" / machine / shot / f"pr08_{machine}_{shot}_0d.dat"   # where fetch_pr08 put it
    table = pr08_mhd_state_table([read_pr08_zero_d(zero_d, machine, shot)])
    times = np.asarray(ods["equilibrium.time"], dtype=float) if "equilibrium.time" in ods else np.array([])
    compared = 0
    for row in table.itertuples():
        # the release's 2D time base is offset from the 0D TIME by under 1 ms (JET 19649: 48.7 vs 48.70097);
        # 1 ms is far inside the 50 ms sampling, so it is the same time, not a neighbour
        index = np.flatnonzero(np.isclose(times, row.time_s, rtol=0.0, atol=1e-3))
        if index.size == 0:
            continue
        k = int(index[0])
        path = f"equilibrium.time_slice.{k}.global_quantities.ip"
        if path in ods and np.isfinite(row.plasma_current):
            # the 0D IP is the record's value, the ODS ip the sampled 1D trace: they agree to ~0.2 % (DIII-D 81507)
            assert abs(float(ods[path])) * 1e-6 == pytest.approx(row.plasma_current, rel=1e-2)
            compared += 1
        if "equilibrium.vacuum_toroidal_field.r0" in ods:
            assert float(ods["equilibrium.vacuum_toroidal_field.r0"]) == pytest.approx(row.major_radius)
            compared += 1
        # B_T is not compared: the 0D BT is defined at RGEO, while the 1D BT trace the ODS takes as b0 states no
        # radius, and the two differ by 2.4 % on DIII-D 81507 (1.957 T against 1.91 T at RGEO = 1.606 m)
    # MAST 8302 contradicts itself on the current direction, so its ODS has no ip/b0 to compare
    assert compared or machine == "mast"


def test_a_pinned_whole_file_unit_slip_is_corrected_only_for_the_pinned_file(tmp_path, monkeypatch):
    """JT-60U 16107/16168 store IP in MA: rescaled while the file is the pinned one, rejected otherwise."""
    from vaft.data.public import _pr08_release

    in_ma = JETLIKE.replace("JET,52009", "JT60U,16107").replace("-2.500E+06", "-2.500E+00")
    discharge = _zero_d(tmp_path, in_ma, machine="jt60u", shot="16107")
    monkeypatch.setitem(_pr08_release.PR08_RELEASE, ("jt60u", "16107"), ("16107", discharge.sha256, ("0d",)))
    row = pr08_mhd_state_table([discharge]).iloc[0]
    assert row["plasma_current"] == pytest.approx(2.5) and row["plasma_current_provenance"] == "source_corrected"
    assert "stored in MA" in row["state_notes"] and np.isfinite(row["normalized_current"])

    # another file content under the same name is not corrected: the decisive check rejects it instead
    monkeypatch.setitem(_pr08_release.PR08_RELEASE, ("jt60u", "16107"), ("16107", "0" * 64, ("0d",)))
    row = pr08_mhd_state_table([discharge]).iloc[0]
    assert math.isnan(row["plasma_current"]) and row["plasma_current_provenance"] == "source_invalid"


def test_the_pinned_corrections_name_files_of_the_release():
    inventory = pr08_release_inventory().set_index(["machine_dir", "shot"])
    for key, corrections in pr08_mhd_state.SOURCE_CORRECTIONS.items():
        assert key in inventory.index, key
        assert all(name in ("IP", "BETMHD") and factor in (1e6, 0.01) for name, factor, _ in corrections)


def test_both_tables_declare_which_field_and_radius_normalize_the_current(tmp_path):
    # PR08 I_N = I_p / (a |BT|) with BT at RGEO; the equilibrium-state table's I_N uses b0
    # at R_ref (R_ref/R_geo = 1.00-1.65 on the VEST sample). Neither table said so in its
    # attrs, so a merge drew two conventions as one population
    # (cold review 0.8.0 delta-absorb-18 stability-opspace F2; the numbers are unchanged).
    from vaft.omas.equilibrium_state import EQUILIBRIUM_STATE_CONVENTIONS, EQUILIBRIUM_STATE_UNITS

    table = pr08_mhd_state_table([_zero_d(tmp_path, JETLIKE)])
    conventions = table.attrs["conventions"]
    assert conventions == pr08_mhd_state.PR08_CONVENTIONS
    for column in ("normalized_current", "normalized_beta"):
        assert "geometric" in conventions[column]["radius_reference"]
        assert "BT" in conventions[column]["b_field_definition"]
        assert "reference" in EQUILIBRIUM_STATE_CONVENTIONS[column]["radius_reference"]
        assert "b0" in EQUILIBRIUM_STATE_CONVENTIONS[column]["b_field_definition"]
        assert conventions[column]["radius_reference"] != EQUILIBRIUM_STATE_CONVENTIONS[column]["radius_reference"]
        assert table.attrs["units"][column] == EQUILIBRIUM_STATE_UNITS[column]   # same unit, different field
    [row] = table.to_dict("records")
    assert row["normalized_current"] == pytest.approx(row["plasma_current"] / (row["minor_radius"] * row["toroidal_field"]))
