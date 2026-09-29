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
    assert "source-computed" in first["p_lh_scaling_definition"]


def test_every_row_names_its_event_and_a_miss_is_not_a_different_event(tcv):
    assert set(tcv["transition"]) == {"L_to_H"}
    assert set(tcv["source_regime"]) == {"L_mode"}
    assert set(tcv["target_regime"]) == {"H_mode"}
    assert tcv["transition_observed"].tolist() == [True, False, True]


def test_thl_does_not_become_an_h_to_l_event(tcv):
    # THL is filled on no-transition rows too, so it is not event evidence.
    assert len(tcv) == 3
    assert "H_to_L" not in set(tcv["transition"])


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
    table = normalize_tcv_lh(read_tcv_lh())
    assert len(table) == 92
    assert int(table["transition_observed"].sum()) == 84
    # The source's PLH is Martin 2008 in 1e20 m^-3, T and m^2: recomputing it
    # from the normalised SI columns pins the density, field and area units.
    martin = 0.0488e6 * (table.n_e_line_avg_m3 / 1e20) ** 0.717 * table.b_t_T ** 0.803 \
        * table.surface_area_m2 ** 0.941
    ratio = table.p_lh_scaling_W / martin
    assert float(np.abs(ratio - 1.0).max()) < 5e-3
