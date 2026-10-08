"""Canonical transition table: TCV L-H reader, event semantics, margins (#1205).

Offline: the TCV rows are synthetic, written in the real file's layout (one
``(1, N)`` dataset per variable, MATLAB-style byte-string ``Units`` and
``Description`` attributes).  The test against the real Zenodo release runs
only with ``VAFT_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from vaft.data.public import (
    SOURCES,
    TRANSITION_COLUMNS,
    empty_transition_table,
    normalize_tcv_lh,
    read_tcv_lh,
    transition_margin,
    validate_transition_table,
)

h5py = pytest.importorskip("h5py")

# Three records: a D L-H transition, an H record without transition, and a
# record of an unmapped species with missing loss power and missing n_min.
_VARIABLES = {
    "SHOT": ([66444.0, 68001.0, 70000.0], ""),
    "TIME": ([0.910, 2.03, 1.5], "s"),
    "THL": ([1.30, 2.50, 2.50], "s"),
    "ILH": ([1.0, 0.0, 1.0], ""),
    "PLMW": ([0.31, 1.03, np.nan], "MW"),
    "PRAD": ([1.2e5, 2.0e5, 1.0e5], "W"),
    "IP": ([0.173, -0.20, 0.15], "MA"),
    "BT": ([-1.41, 1.42, 1.40], "T"),
    "NEL": ([3.5e19, 5.5e19, 4.0e19], "m^-3"),
    "SPLASMA": ([10.25, 10.1, 10.0], "m^2"),
    "RGEO": ([0.877, 0.88, 0.88], "m"),
    "AMIN": ([0.225, 0.23, 0.23], "m"),
    "KAPPA": ([1.64, 1.5, 1.5], ""),
    "DELTA": ([0.37, -0.2, 0.3], ""),
    "Q95": ([4.8, 3.3, 4.0], ""),
    "ZEFF": ([1.33, 1.5, 1.4], ""),
    "A": ([2.0, 1.0, 7.0], ""),
    "Z": ([1.0, 1.0, 3.0], ""),
    "cH": ([0.06, 0.9, 0.0], ""),
    "cHe": ([0.025, 0.02, 0.0], ""),
    "BAFFLES": ([0.0, 33.0, 0.0], ""),
    "nRyter": ([0.34, 0.60, np.nan], "10^20 m^-3"),
    "PLH": ([0.273, 0.40, 0.30], "MW"),
    "PNBI": ([4.2e5, 7.0e5, 0.0], "W"),
    "PECH": ([0.0, 3.0e5, 0.0], "W"),
}


def _write(path, variables=_VARIABLES):
    with h5py.File(path, "w") as handle:
        for name, (values, unit) in variables.items():
            dataset = handle.create_dataset(name, data=np.asarray(values, float).reshape(1, -1))
            dataset.attrs["Units"] = np.bytes_(unit) if unit else h5py.Empty("S1")
            dataset.attrs["Description"] = np.bytes_(f"synthetic {name}")
            dataset.attrs["MATLAB_class"] = np.bytes_("double")
    return path


@pytest.fixture()
def tcv_path(tmp_path):
    return _write(tmp_path / "lhdatabase.h5")


@pytest.fixture()
def tcv(tcv_path):
    return normalize_tcv_lh(read_tcv_lh(tcv_path))


# ---------------------------------------------------------------- reader


def test_reader_flattens_matlab_arrays_and_keeps_units(tcv_path):
    raw = read_tcv_lh(tcv_path)
    assert len(raw) == 3
    assert raw.attrs["units"]["IP"] == "MA"
    assert raw.attrs["units"]["ILH"] == ""  # h5py.Empty attribute
    assert raw.attrs["descriptions"]["PLMW"] == "synthetic PLMW"


def test_reader_rejects_a_file_without_the_tcv_variables(tmp_path):
    variables = {k: v for k, v in _VARIABLES.items() if k != "PLH"}
    with pytest.raises(ValueError, match="missing variables"):
        read_tcv_lh(_write(tmp_path / "other.h5", variables))


def test_reader_rejects_variables_of_different_lengths(tmp_path):
    variables = dict(_VARIABLES, SHOT=([1.0, 2.0], ""))
    with pytest.raises(ValueError, match="different lengths"):
        read_tcv_lh(_write(tmp_path / "ragged.h5", variables))


# ---------------------------------------------------------------- normalise


def test_units_become_si_and_signs_become_magnitudes(tcv):
    first = tcv.set_index("record_id").loc["TCV:66444:910"]
    assert first["i_p_A"] == pytest.approx(1.73e5)
    assert first["b_t_T"] == pytest.approx(1.41)
    assert first["p_loss_W"] == pytest.approx(3.1e5)
    assert first["p_lh_scaling_W"] == pytest.approx(2.73e5)
    assert first["n_e_min_m3"] == pytest.approx(3.4e19)
    assert first["n_e_line_avg_m3"] == pytest.approx(3.5e19)
    assert first["surface_area_m2"] == pytest.approx(10.25)
    assert "radiation NOT subtracted" in first["p_loss_definition"]
    assert "PTOTMW_A - DWMHDMW_A" in first["p_loss_definition"]
    assert "source-computed" in first["p_lh_scaling_definition"]
    assert first["p_rad_W"] == pytest.approx(1.2e5)
    # Derived from which heating powers are non-zero.
    assert tcv["auxiliary_heating"].tolist() == ["NB", "NBEC", "NONE"]


def test_every_row_names_its_event_and_a_miss_is_not_a_different_event(tcv):
    assert set(tcv["transition"]) == {"L_to_H"}
    assert set(tcv["source_regime"]) == {"L_mode"}
    assert set(tcv["target_regime"]) == {"H_mode"}
    assert tcv["transition_observed"].tolist() == [True, False, True]


def test_event_time_is_time_not_thl(tcv):
    # Every quantity belongs to TIME; THL has no conditions of its own and
    # must not leak into the record time.
    assert tcv["time_s"].tolist() == pytest.approx([0.910, 2.03, 1.5])
    # The transition time exists only where the transition was observed.
    assert tcv["transition_time_s"].iloc[0] == pytest.approx(0.910)
    assert np.isnan(tcv["transition_time_s"].iloc[1])
    assert tcv["source_phase"].tolist() == ["ILH=1", "ILH=0", "ILH=1"]


def test_a_blank_ilh_is_unknown_not_a_miss(tmp_path):
    variables = dict(_VARIABLES, ILH=([1.0, np.nan, 0.0], ""))
    with pytest.raises(ValueError, match="ILH"):
        normalize_tcv_lh(read_tcv_lh(_write(tmp_path / "blank.h5", variables)))


def test_a_record_without_shot_or_time_is_rejected(tmp_path):
    variables = dict(_VARIABLES, TIME=([0.9, np.nan, 1.5], "s"))
    with pytest.raises(ValueError, match="SHOT and a TIME"):
        normalize_tcv_lh(read_tcv_lh(_write(tmp_path / "notime.h5", variables)))


def test_hydrogenic_mixture_uses_the_authors_cuts(tmp_path):
    # A=2 with cH=0.5 is fuelled with D but is not a D plasma.
    variables = dict(_VARIABLES, cH=([0.5, 0.9, 0.0], ""))
    table = normalize_tcv_lh(read_tcv_lh(_write(tmp_path / "mix.h5", variables)))
    assert table["main_ion"].tolist()[:2] == ["D", "H"]
    assert table["hydrogenic_mix"].tolist()[:2] == ["mixed H/D", "H-dominated"]
    assert pd.isna(table["hydrogenic_mix"].iloc[2])  # not hydrogenic


def test_default_hydrogenic_mix(tcv):
    assert tcv["hydrogenic_mix"].tolist()[:2] == ["D-dominated", "H-dominated"]


def test_ilh_outside_zero_one_is_rejected(tmp_path):
    variables = dict(_VARIABLES, ILH=([1.0, 2.0, 0.0], ""))
    with pytest.raises(ValueError, match="ILH"):
        normalize_tcv_lh(read_tcv_lh(_write(tmp_path / "bad.h5", variables)))


def test_main_ion_is_mapped_or_left_missing(tcv):
    assert tcv["main_ion"].tolist()[:2] == ["D", "H"]
    assert pd.isna(tcv["main_ion"].iloc[2])  # A=7, Z=3: not guessed
    assert tcv["main_ion_mass_amu"].iloc[2] == pytest.approx(7.0)


def test_density_branch_is_relative_to_the_source_minimum(tcv):
    # 3.5e19 >= 3.4e19 -> high; 5.5e19 < 6.0e19 -> low; no n_min -> missing
    assert tcv["density_branch"].tolist()[:2] == ["high", "low"]
    assert pd.isna(tcv["density_branch"].iloc[2])


def test_missing_source_values_stay_missing(tcv):
    third = tcv.iloc[2]
    assert np.isnan(third["p_loss_W"])
    assert np.isnan(third["n_e_min_m3"])
    assert tcv["divertor_configuration"].isna().all()  # the file does not say
    assert tcv["divertor_closure"].tolist()[:2] == ["TCV BAFFLES=0", "TCV BAFFLES=33"]


def test_triangularity_keeps_its_sign(tcv):
    assert tcv["delta"].iloc[1] == pytest.approx(-0.2)


# ---------------------------------------------------------------- schema


def test_empty_transition_table_has_every_column():
    assert list(empty_transition_table().columns) == list(TRANSITION_COLUMNS)


def test_validation_requires_an_event_name_and_unique_ids(tcv):
    unnamed = tcv.copy()
    unnamed.loc[0, "transition"] = None
    with pytest.raises(ValueError, match="name its transition"):
        validate_transition_table(unnamed)
    with pytest.raises(ValueError, match="record_id"):
        validate_transition_table(pd.concat([tcv, tcv.iloc[:1]]))
    with pytest.raises(ValueError, match="missing"):
        validate_transition_table(tcv.drop(columns=["density_branch"]))


# ---------------------------------------------------------------- analysis


def test_margin_is_loss_power_over_the_scaling_threshold(tcv):
    margin = transition_margin(tcv)
    assert margin.iloc[0] == pytest.approx(0.31 / 0.273)
    assert np.isnan(margin.iloc[2])  # no loss power


def test_margin_is_missing_for_a_non_positive_threshold(tcv):
    table = tcv.copy()
    table.loc[0, "p_lh_scaling_W"] = 0.0
    assert np.isnan(transition_margin(table).iloc[0])


def test_registry_entry_for_tcv_is_pinned_and_cited():
    source = SOURCES["tcv_lh_2025"]
    assert len(source.sha256) == 64
    assert source.licence == "CC BY 4.0"
    assert source.doi == "10.1088/1361-6587/adc8ce"


# ---------------------------------------------------------------- plots


def test_transition_renderers_draw_the_canonical_table(tcv):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vaft.plot import population

    for fig, _ in (
        population.lh_threshold_population(tcv),
        population.transition_predicted_vs_measured(tcv),
        population.transition_margin(tcv, transition_margin(tcv)),
    ):
        assert fig.axes
        plt.close(fig)


def test_margin_plot_refuses_a_margin_from_another_table(tcv):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from vaft.plot import population

    other = tcv.iloc[::-1].reset_index(drop=True)
    shifted = transition_margin(other)
    shifted.index = shifted.index + 1
    with pytest.raises(ValueError, match="not indexed like the table"):
        population.transition_margin(tcv, shifted)


# ---------------------------------------------------------------- network


@pytest.mark.skipif(
    not os.environ.get("VAFT_NETWORK_TESTS"),
    reason="set VAFT_NETWORK_TESTS=1 to fetch the real TCV L-H release from Zenodo",
)
def test_real_tcv_release_matches_its_own_martin_scaling():
    raw = read_tcv_lh()
    table = normalize_tcv_lh(raw)
    assert len(table) == 92
    # p_loss_definition claims PLMW = PTOTMW_A - DWMHDMW_A exactly.
    rebuilt = raw["PTOTMW_A"] - raw["DWMHDMW_A"]
    finite = raw["PLMW"].notna()
    assert np.allclose(raw.loc[finite, "PLMW"], rebuilt[finite], rtol=1e-9, atol=0.0)
    assert int(table["transition_observed"].sum()) == 84
    # The source's PLH is Martin 2008 in 1e20 m^-3, T and m^2: recomputing it
    # from the normalised SI columns pins the density, field and area units.
    martin = 0.0488e6 * (table.n_e_line_avg_m3 / 1e20) ** 0.717 * table.b_t_T ** 0.803 \
        * table.surface_area_m2 ** 0.941
    ratio = table.p_lh_scaling_W / martin
    assert float(np.abs(ratio - 1.0).max()) < 5e-3


# ---------------------------------------------------------------- TC-26

_TC26_HEADER = (
    "TOK,SHOT,TIME,PHASE,PGASA,RGEO,AMIN,KAPPA,SPLASMA,CONFIG,IGRADB,WALMAT,DIVMAT,"
    "LIMMAT,EVAP,BT,IP,Q95,NEL,ZEFF,DIVNAME,PL,PLTH,PFLOSS,PRADCORE,DIVCON,FRACNMIN,"
    "LHTIME,AUXHEAT,SELEC2024"
)
_TC26_ROWS = (
    # C-Mod: leading spaces as in the release, blank EVAP, FRACNMIN sentinel
    "CMOD,1050428004,0.63461, LH,2,0.672389,0.216421,1.69737,7.42158, SN,1, MO, MO, MO,,"
    "5.33923,931812,4.26029,1.97E+20,1.088, DIVNOSE,2289520,2289520,0,249783, VERTPLT,"
    "-1.00E-08,0.63561,IC,1",
    # JET D-T mixture, signed IP/BT, '-' ZEFF
    "JET,98969,50.135, LH,2.5,2.88557,0.951136,1.63504,138.813, SN(B),1, Be, W, Be, NONE,"
    "-2.98159,-2459620,3.66265,3.13E+19,-, MkII-HD,2880580,2822080,58506,287455, V5,0,"
    "50.17,NB,0",
    # AUG: back transition phase, LHTIME sentinel, '-' PRADCORE, ZEFF 0 as
    # every real AUG row has it
    "AUG,26359,3.75, LHL,1.0,1.65,0.5,1.7,43.0, SN(L),1, W, W, W, BOR,-2.5,1.0e6,4.0,"
    "4.0e19,0, DIV-IIc,1.5e6,1.3e6,0,-, STANDARD,0.9,-1.00E-08,EC,1",
    # AUG plain 'LH' whose LHTIME precedes TIME; PRADCORE 0; mass on the band edge
    "AUG,35241,2.27, LH,1.95,1.65,0.5,1.7,43.0, SN(L),1, W, W, W, BOR,-2.5,1.0e6,4.0,"
    "3.0e19,0, DIV-III,1.2e6,1.2e6,0,0, STANDARD,0,2.057,NB,0",
)


def _tc26_csv(path, rows=_TC26_ROWS):
    path.write_text(_TC26_HEADER + "\n" + "\n".join(rows) + "\n")
    return path


@pytest.fixture()
def tc26(tmp_path):
    from vaft.data.public import normalize_tc26, read_tc26

    with pytest.warns(UserWarning, match="SHA-256"):  # synthetic file != release
        raw = read_tc26(_tc26_csv(tmp_path / "tc26.csv"))
    return normalize_tc26(raw)


def test_tc26_missing_markers_become_missing(tc26):
    rows = tc26.set_index("record_id")
    assert np.isnan(rows.loc["JET:98969:50135", "z_eff"])          # '-'
    assert np.isnan(rows.loc["AUG:26359:3750", "p_rad_W"])         # '-'
    assert np.isnan(rows.loc["AUG:26359:3750", "transition_time_s"])  # -1e-08
    assert np.isnan(rows.loc["AUG:26359:3750", "z_eff"])           # 0: Z_eff >= 1
    assert np.isnan(rows.loc["AUG:35241:2270", "p_rad_W"])         # 0: not measured


def test_tc26_a_real_zero_stays_zero(tmp_path):
    # PFLOSS = 0 means no beam loss; it is data, not a marker.  Only quantities
    # that cannot be zero (ZEFF, PRADCORE, FRACNMIN) treat 0 as missing.
    from vaft.data.public import read_tc26

    with pytest.warns(UserWarning):
        raw = read_tc26(_tc26_csv(tmp_path / "zero.csv"))
    assert raw["PFLOSS"].tolist()[0] == 0.0
    assert pd.isna(raw["FRACNMIN"].iloc[3])  # 0 -> missing


def test_tc26_conditions_may_follow_the_recorded_transition(tc26):
    row = tc26.set_index("record_id").loc["AUG:35241:2270"]
    assert row["source_phase"] == "LH"
    assert row["transition_time_s"] < row["time_s"]  # kept as given


def test_tc26_times_keep_conditions_and_transition_apart(tc26):
    cmod = tc26.set_index("machine").loc["CMOD"]
    assert cmod["time_s"] == pytest.approx(0.63461)
    assert cmod["transition_time_s"] == pytest.approx(0.63561)
    assert cmod["record_id"] == "CMOD:1050428004:635"


def test_tc26_is_si_with_magnitudes_and_core_radiation(tc26):
    jet = tc26.set_index("machine").loc["JET"]
    assert jet["i_p_A"] == pytest.approx(2459620)
    assert jet["b_t_T"] == pytest.approx(2.98159)
    assert jet["p_loss_W"] == pytest.approx(2822080)  # PLTH, not PL
    assert "CORE radiation" in jet["p_rad_definition"]
    assert jet["first_wall"] == "main=Be, divertor=W"
    assert jet["divertor_closure"] == "MkII-HD V5"


def test_tc26_event_labels_and_selection(tc26):
    assert set(tc26["transition"]) == {"L_to_H"}
    assert tc26["transition_observed"].all()
    assert tc26["source_phase"].tolist() == ["LH", "LH", "LHL", "LH"]
    assert tc26["selected"].tolist() == [True, False, True, False]
    # Outside the selection is not thereby low-density.
    assert tc26["density_branch"].iloc[0] == "high"
    assert pd.isna(tc26["density_branch"].iloc[1])
    assert tc26["grad_b_drift"].unique().tolist() == ["toward_x_point"]


def test_tc26_mass_maps_to_species_or_mixed(tc26):
    # 1.95 sits on the band edge and must count as D.
    assert tc26["main_ion"].tolist() == ["D", "mixed", "H", "D"]
    assert tc26["main_ion_mass_amu"].iloc[1] == pytest.approx(2.5)
    assert tc26["hydrogenic_mix"].iloc[1] == "M_eff 2-3"
    assert pd.isna(tc26["hydrogenic_mix"].iloc[0])


def test_tc26_has_no_threshold_so_no_margin(tc26):
    assert tc26["p_lh_scaling_W"].isna().all()
    assert transition_margin(tc26).isna().all()


def test_tc26_exact_duplicates_drop_and_conflicts_raise(tmp_path):
    from vaft.data.public import normalize_tc26, read_tc26

    with pytest.warns(UserWarning):
        raw = read_tc26(_tc26_csv(tmp_path / "dup.csv", _TC26_ROWS + (_TC26_ROWS[1],)))
    assert len(normalize_tc26(raw)) == 4
    conflicting = _TC26_ROWS[1].replace("2880580", "2880581")
    with pytest.warns(UserWarning):
        raw = read_tc26(_tc26_csv(tmp_path / "clash.csv", _TC26_ROWS + (conflicting,)))
    with pytest.raises(ValueError, match="share a key"):
        normalize_tc26(raw)


def test_tc26_rejects_a_file_without_its_columns(tmp_path):
    from vaft.data.public import read_tc26

    path = tmp_path / "other.csv"
    path.write_text("TOK,SHOT\nJET,1\n")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="not the TC-26"):
        read_tc26(path)


def test_tcv_and_tc26_share_one_table(tcv, tc26):
    table = validate_transition_table(pd.concat([tcv, tc26], ignore_index=True))
    assert set(table["machine"]) == {"TCV", "CMOD", "JET", "AUG"}


@pytest.mark.skipif(
    not os.environ.get("VAFT_TC26_CSV"),
    reason="set VAFT_TC26_CSV to a local nfae39f2supp2.csv (not redistributed)",
)
def test_real_tc26_release():
    import warnings

    from vaft.data.public import normalize_tc26, read_tc26

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the pinned release must not warn
        table = normalize_tc26(read_tc26(os.environ["VAFT_TC26_CSV"]))
    assert len(table) == 688  # 689 rows, one exact duplicate
    assert table.groupby("machine")["selected"].sum().to_dict() == {"AUG": 162, "CMOD": 59, "JET": 260}
    # The traps of this release, pinned.
    assert table.loc[table.machine == "AUG", "z_eff"].isna().all()     # ZEFF 0
    assert not (table["p_rad_W"] == 0.0).any()                         # PRADCORE 0
    assert int((table["transition_time_s"] < table["time_s"]).sum()) == 14
    assert table["main_ion"].value_counts().to_dict() == {"D": 520, "mixed": 77, "H": 75, "T": 16}
