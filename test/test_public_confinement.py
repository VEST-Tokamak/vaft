"""Canonical confinement table: DB5.2.3 reader, VEST adapters, predictions (#1205).

Everything here is offline.  The DB5 rows are synthetic, written in the real
file's layout (byte-order mark, a line of column numbers before the header,
CRLF line ends, blank cells); the network is mocked.  The one test that fetches
the real release runs only with ``VAFT_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import hashlib
import io
import os
from unittest import mock
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from vaft.data.public import (
    CONFINEMENT_COLUMNS,
    ChecksumError,
    FetchError,
    SOURCES,
    confinement_coverage,
    empty_confinement_table,
    h_factor,
    normalize_db5,
    predict_confinement_time,
    read_db5,
    validate_confinement_table,
    vest_summary_to_confinement_table,
)
from vaft.data.public import _fetch
from vaft.formula import confinement_time_from_engineering_parameters

_HEADER = (
    "TOK", "SHOT", "TIME", "TIME_ID", "PHASE", "IP", "BT", "NEL", "PLTH", "WTH",
    "TAUTH", "RGEO", "AMIN", "KAPPA", "KAREA", "DELTA", "MEFF", "SELDB5",
)
_ROWS = (
    # JET-like H-mode, signed IP/BT as DB5 stores them; TIME_ID not round(1000 TIME)
    ("JET", 52000, 52.10, 52099, "HGELM", -2.5e6, -2.4, 6.0e19, 1.2e7, 5.0e6,
     0.4167, 2.9, 0.95, 1.7, 1.62, 0.25, 2.0, 1),
    # AUG-like, not in the standard set
    ("AUG", 30000, 3.0, 3000, "H", 1.0e6, -2.5, 8.0e19, 5.0e6, 5.0e5,
     0.1, 1.65, 0.5, 1.7, 1.6, 0.3, 2.0, 0),
    # blank loss power and confinement time, as in the real file
    ("NSTX", 120000, 0.5, 500, "H", 8.0e5, 0.45, 5.0e19, "", 1.0e5,
     "", 0.85, 0.65, 2.0, 1.9, 0.4, 2.0, 0),
)


def _db5_csv(rows=_ROWS) -> bytes:
    numbers = "," + ",".join(str(i + 1) for i in range(len(_HEADER)))
    lines = [numbers, "," + ",".join(_HEADER)]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index}," + ",".join(str(v) for v in row))
    return ("﻿" + "\r\n".join(lines) + "\r\n").encode("utf-8")


@pytest.fixture()
def db5_path(tmp_path):
    path = tmp_path / "DB5.2.3.csv"
    path.write_bytes(_db5_csv())
    return path


@pytest.fixture()
def db5(db5_path):
    return normalize_db5(read_db5(db5_path))


# ---------------------------------------------------------------- schema


def test_empty_table_has_every_canonical_column_and_units():
    table = empty_confinement_table()
    assert list(table.columns) == list(CONFINEMENT_COLUMNS)
    assert table.attrs["units"]["p_loss_W"] == "W"
    assert table.attrs["units"]["n_e_line_avg_m3"] == "m^-3"


def test_validation_rejects_duplicate_ids_and_negative_magnitudes(db5):
    with pytest.raises(ValueError, match="record_id"):
        validate_confinement_table(pd.concat([db5, db5.iloc[:1]]))
    bad = db5.copy()
    bad.loc[0, "i_p_A"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        validate_confinement_table(bad)
    with pytest.raises(ValueError, match="missing"):
        validate_confinement_table(db5.drop(columns=["kappa_area"]))


def test_triangularity_is_signed(db5):
    table = db5.copy()
    table.loc[0, "delta"] = -0.3
    assert validate_confinement_table(table).loc[0, "delta"] == pytest.approx(-0.3)


# ---------------------------------------------------------------- DB5


def test_read_db5_handles_bom_column_number_line_and_crlf(db5_path):
    raw = read_db5(db5_path)
    assert list(raw["TOK"]) == ["JET", "AUG", "NSTX"]
    assert raw.loc[0, "IP"] == pytest.approx(-2.5e6)
    assert not any(str(c).startswith("Unnamed") for c in raw.columns)


def test_read_db5_rejects_a_table_without_the_db5_columns(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("1,2\nA,B\n1,2\n")
    with pytest.raises(ValueError, match="not a DB5.2.3 table"):
        read_db5(path)


def test_normalize_db5_keeps_si_units_takes_magnitudes_and_derives_epsilon(db5):
    jet = db5.set_index("record_id").loc["JET:52000:52099"]
    assert jet["i_p_A"] == pytest.approx(2.5e6)
    assert jet["b_t_T"] == pytest.approx(2.4)
    assert jet["n_e_line_avg_m3"] == pytest.approx(6.0e19)
    assert jet["p_loss_W"] == pytest.approx(1.2e7)
    assert jet["tau_e_th_s"] == pytest.approx(0.4167)
    assert jet["epsilon"] == pytest.approx(0.95 / 2.9)
    assert jet["kappa_area"] == pytest.approx(1.62)
    assert jet["m_eff_amu"] == pytest.approx(2.0)
    assert bool(jet["selected"]) is True
    assert "radiation NOT subtracted" in jet["p_loss_definition"]
    assert jet["b_t_definition"].endswith("at RGEO")
    assert jet["source_release"] == "DB5.2.3"


def test_record_id_uses_the_source_time_key_not_a_rounded_time(db5):
    # 52.10 s rounds to 52100 ms; DB5's own TIME_ID is 52099.
    assert "JET:52000:52099" in set(db5["record_id"])


def test_blank_source_cells_stay_missing(db5):
    nstx = db5.set_index("machine").loc["NSTX"]
    assert np.isnan(nstx["p_loss_W"])
    assert np.isnan(nstx["tau_e_th_s"])
    assert nstx["w_th_J"] == pytest.approx(1.0e5)


# ---------------------------------------------------------------- analysis


def test_prediction_matches_a_direct_formula_call(db5):
    predicted = predict_confinement_time(db5, "H98y2")
    row = db5.iloc[0]
    expected = confinement_time_from_engineering_parameters(
        I_p=row.i_p_A, B_t=row.b_t_T, P_loss=row.p_loss_W, n_e=row.n_e_line_avg_m3,
        M=row.m_eff_amu, R=row.r_geo_m, epsilon=row.epsilon, kappa=row.kappa_area,
        scaling="H98y2",
    )
    assert predicted.iloc[0] == pytest.approx(expected, rel=1e-12)
    assert np.isnan(predicted.iloc[2])  # no loss power -> no prediction


def test_kappa_column_is_an_explicit_choice(db5):
    area = predict_confinement_time(db5, "H98y2")
    boundary = predict_confinement_time(db5, "H98y2", kappa_column="kappa")
    assert area.iloc[0] != pytest.approx(boundary.iloc[0])


def test_a_scaling_ignores_columns_it_does_not_use(db5):
    # NSTX2006H uses neither R, epsilon, kappa nor M: missing values there
    # must not blank the prediction.
    table = db5.copy()
    table["kappa_area"] = np.nan
    table["m_eff_amu"] = np.nan
    assert np.isfinite(predict_confinement_time(table, "NSTX2006H").iloc[0])
    assert np.isnan(predict_confinement_time(table, "H98y2").iloc[0])


def test_h_factor_is_measured_over_predicted(db5):
    h = h_factor(db5, "H98y2")
    predicted = predict_confinement_time(db5, "H98y2")
    assert h.iloc[0] == pytest.approx(db5.tau_e_th_s.iloc[0] / predicted.iloc[0])


def test_unknown_scaling_is_an_error(db5):
    with pytest.raises(ValueError, match="Unknown scaling"):
        predict_confinement_time(db5, "not-a-scaling")


def test_coverage_counts_finite_values_per_machine(db5):
    coverage = confinement_coverage(db5)
    assert coverage.loc["NSTX", "rows"] == 1
    assert coverage.loc["NSTX", "p_loss_W"] == 0
    assert coverage.loc["JET", "p_loss_W"] == 1


# ---------------------------------------------------------------- VEST


def _summary_rows():
    return pd.DataFrame({
        "shot": [48224, 48224],
        "cp_index": [0, 1],
        "eq_index": [0, 1],
        "time_s": [0.300, 0.310],
        "ip_kA": [143.0, 140.0],
        "b_t_T": [0.15, 0.15],
        "p_loss_MW": [0.05, -0.01],
        "tau_e_s": [1.8e-3, -9.0e-3],
        "ne_line_1e19_m3": [0.53, 0.50],
        "major_radius_m": [0.392, 0.392],
        "inverse_aspect_ratio": [0.73, 0.73],
        "elongation": [1.74, 1.74],
    })


def test_vest_summary_adapter_converts_units_and_records_definitions():
    table = vest_summary_to_confinement_table(_summary_rows(), effective_mass_amu=1.0)
    first = table.iloc[0]
    assert first["record_id"] == "VEST:48224:300"
    assert first["i_p_A"] == pytest.approx(1.43e5)
    assert first["n_e_line_avg_m3"] == pytest.approx(0.53e19)
    assert first["p_loss_W"] == pytest.approx(5.0e4)
    assert first["b_t_T"] == pytest.approx(0.15 * 0.4 / 0.392)
    assert first["a_m"] == pytest.approx(0.73 * 0.392)
    assert "radiation IS subtracted" in first["p_loss_definition"]
    assert first["m_eff_source"] == "user-specified"
    assert np.isnan(first["kappa_area"])


def test_vest_summary_adapter_blanks_non_positive_loss_power():
    table = vest_summary_to_confinement_table(_summary_rows(), effective_mass_amu=1.0)
    assert np.isnan(table.iloc[1]["p_loss_W"])
    assert np.isnan(table.iloc[1]["tau_e_th_s"])


def test_vest_summary_adapter_requires_an_explicit_mass():
    with pytest.raises(TypeError):
        vest_summary_to_confinement_table(_summary_rows())  # noqa: the point
    with pytest.raises(ValueError):
        vest_summary_to_confinement_table(_summary_rows(), effective_mass_amu=0.0)


def test_vest_and_db5_rows_share_one_table(db5):
    vest = vest_summary_to_confinement_table(_summary_rows(), effective_mass_amu=1.0)
    table = validate_confinement_table(pd.concat([db5, vest], ignore_index=True))
    assert set(table["machine"]) == {"JET", "AUG", "NSTX", "VEST"}
    h = h_factor(table, "H98y2", kappa_column="kappa")
    assert np.isfinite(h[table.machine == "VEST"].iloc[0])


def test_vest_ods_adapter_reads_without_mutating():
    omas = pytest.importorskip("omas")
    from vaft.data.public import vest_ods_to_confinement_rows

    ods = omas.ODS()
    ods["dataset_description.data_entry.pulse"] = 1
    before = sorted(ods.flat())
    table = vest_ods_to_confinement_rows(ods, effective_mass_amu=1.0)
    assert len(table) == 0
    assert sorted(ods.flat()) == before


# ---------------------------------------------------------------- fetch


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_writes_verifies_and_reuses_the_cache(tmp_path):
    payload = b"db5 bytes"
    digest = hashlib.sha256(payload).hexdigest()
    with mock.patch.object(_fetch, "urlopen", return_value=_Response(payload)) as opened:
        path = _fetch.fetch("https://example.invalid/x", sha256=digest, filename="x.csv", cache=tmp_path)
        again = _fetch.fetch("https://example.invalid/x", sha256=digest, filename="x.csv", cache=tmp_path)
    assert path == again == tmp_path / "x.csv"
    assert path.read_bytes() == payload
    assert opened.call_count == 1
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_fetch_removes_a_download_with_the_wrong_checksum(tmp_path):
    with mock.patch.object(_fetch, "urlopen", return_value=_Response(b"other")):
        with pytest.raises(ChecksumError):
            _fetch.fetch("https://example.invalid/x", sha256="0" * 64, filename="x.csv", cache=tmp_path)
    assert not (tmp_path / "x.csv").exists()
    assert not list(tmp_path.iterdir())


def test_fetch_reports_a_truncated_transfer_as_fetch_error(tmp_path):
    from http.client import IncompleteRead

    with mock.patch.object(_fetch, "urlopen", side_effect=IncompleteRead(b"part")):
        with pytest.raises(FetchError):
            _fetch.fetch("https://example.invalid/x", sha256="0" * 64, filename="x.csv", cache=tmp_path)


def test_fetch_reports_an_upstream_outage_as_fetch_error(tmp_path):
    with mock.patch.object(_fetch, "urlopen", side_effect=URLError("down")):
        with pytest.raises(FetchError):
            _fetch.fetch("https://example.invalid/x", sha256="0" * 64, filename="x.csv", cache=tmp_path)


def test_fetch_rejects_a_path_as_filename(tmp_path):
    with pytest.raises(ValueError):
        _fetch.fetch("https://example.invalid/x", sha256="0" * 64, filename="../x.csv", cache=tmp_path)


def test_registry_entry_for_db5_is_pinned_and_cited():
    source = SOURCES["itpa_db5.2.3"]
    assert len(source.sha256) == 64
    assert source.licence == "CC BY 4.0"
    assert source.doi == "10.1088/1741-4326/abdb91"


# ---------------------------------------------------------------- plots


def test_population_renderers_draw_the_canonical_table(db5):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vaft.plot import population

    vest = vest_summary_to_confinement_table(_summary_rows(), effective_mass_amu=1.0)
    table = pd.concat([db5, vest], ignore_index=True)
    predicted = predict_confinement_time(table, kappa_column="kappa")
    for fig, _ in (
        population.confinement_population(table),
        population.confinement_predicted_vs_measured(table, predicted),
        population.confinement_h_factor_distribution(table, h_factor(table, kappa_column="kappa")),
        population.confinement_coverage_strip(table),
    ):
        assert fig.axes
        plt.close(fig)


def test_population_renderers_refuse_a_series_from_another_table(db5):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from vaft.plot import population

    vest = vest_summary_to_confinement_table(_summary_rows(), effective_mass_amu=1.0)
    table = pd.concat([vest, db5], ignore_index=True)
    # Same integer labels, different rows: pairing by label would be silent.
    with pytest.raises(ValueError, match="not indexed like the table"):
        population.confinement_predicted_vs_measured(table, predict_confinement_time(db5))
    with pytest.raises(ValueError, match="not indexed like the table"):
        population.confinement_h_factor_distribution(table, h_factor(db5))


def test_vest_ods_adapter_on_the_48224_kinetic_equilibrium():
    from pathlib import Path

    import vaft
    from vaft.data.public import vest_ods_to_confinement_rows

    path = Path(vaft.data.data_path("kineticEfit/ods_48224_300ms.json"))
    if not path.exists():
        pytest.skip("repository-only kinetic EFIT sample")
    ods = vaft.omas.load(path)
    before = sorted(ods.flat())
    row = vest_ods_to_confinement_rows(ods, effective_mass_amu=1.0).iloc[0]
    assert sorted(ods.flat()) == before  # reading must not create paths

    assert row["record_id"] == "VEST:48224:300"
    assert row["kappa_area"] == pytest.approx(
        1.0223282582 / (2 * np.pi**2 * row["a_m"] ** 2 * row["r_geo_m"]), rel=1e-6
    )
    assert row["b_t_T"] == pytest.approx(0.150869643 * 0.4 / row["r_geo_m"], rel=1e-6)
    density = np.asarray(ods["core_profiles.profiles_1d.0.electrons.density"], float)
    assert density.min() < row["n_e_line_avg_m3"] < density.max()
    assert np.isnan(row["p_loss_W"]) and np.isnan(row["tau_e_th_s"])


# ---------------------------------------------------------------- network


@pytest.mark.skipif(
    not os.environ.get("VAFT_NETWORK_TESTS"),
    reason="set VAFT_NETWORK_TESTS=1 to fetch the real DB5.2.3 release from OSF",
)
def test_real_db5_release_reproduces_its_own_ipb98_h_factor():
    raw = read_db5()
    table = normalize_db5(raw)
    assert len(table) == 14153
    assert int(table["selected"].sum()) == 7568
    # DB5's HIPB98Y2 = TAUTH * TAUC92 / IPB98(y,2): the SI conversions, the
    # line-averaged density and kappa_area must reproduce it.  HIPB98Y2 is
    # stored to ~4 significant figures, and a couple of source rows disagree
    # with their own TAUTH (e.g. D3D 86209), so allow both.
    ratio = h_factor(table, "H98y2") / (raw["HIPB98Y2"] / raw["TAUC92"])
    ratio = ratio[np.isfinite(ratio)]
    assert len(ratio) > 11000
    assert int((np.abs(ratio - 1.0) >= 2e-3).sum()) <= 5


def test_the_subpackage_survives_the_sdist_prune_of_vaft_data():
    from pathlib import Path

    manifest = (Path(__file__).resolve().parents[1] / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include vaft/data/public/*.py" in manifest
    assert not any(Path(__file__).resolve().parents[1].joinpath("vaft/data/public").glob("*.csv"))


# --- lane D Tier A loader (#548) ---------------------------------------------


def test_tier_a_loader_selections_and_schema():
    from vaft.data import public
    from vaft.data.public.schema import CONFINEMENT_COLUMNS

    primary = public.load_vest_tier_a_confinement()
    assert list(primary.columns[: len(CONFINEMENT_COLUMNS)]) == list(CONFINEMENT_COLUMNS)
    assert (len(primary), primary["shot"].nunique()) == (59, 19)
    assert primary["selected"].all() and (primary["tau_e_th_s"] > 0).all()
    assert set(primary["tier_a_block"]) == {"399xx-403xx", "429xx-430xx"}
    assert primary["n_e_definition"].str.startswith("z = 0 chord").all()
    every = public.load_vest_tier_a_confinement(None)
    assert len(every) == 133 and int(every["selected"].sum()) == 59
    strict = public.load_vest_tier_a_confinement("sensitivity")
    assert (len(strict), strict["shot"].nunique()) == (14, 4)
    with pytest.raises(ValueError, match="selection"):
        public.load_vest_tier_a_confinement("nope")


def test_tier_a_loader_matches_the_process_layer_decision():
    """The loader's inline rules are the ones vaft.process.confinement applies."""
    import numpy as np
    import pandas as pd

    from vaft.data import data_path
    from vaft.data.public.vest_confinement import VEST_TIER_A_SELECTIONS, _tier_a_selected
    from vaft.process.confinement import ConfinementSliceEvidence, confinement_slice_decision

    table = pd.read_csv(data_path("confinement/vest_tier_a_confinement.csv"))
    for thresholds in VEST_TIER_A_SELECTIONS.values():
        evidence = ConfinementSliceEvidence(
            ip_abs=table["i_p_A"].abs().to_numpy(float), ip_rate=table["ip_rate_1_s"].to_numpy(float),
            ip_change_per_tau=table["ip_change_per_tau"].to_numpy(float),
            dwdt_fraction=table["dwdt_fraction"].to_numpy(float),
            finite=table["rule_finite"].astype("boolean").fillna(False).to_numpy(bool))
        decided = confinement_slice_decision(evidence, **thresholds)["accepted"]
        decided &= (table["quality_status"] == "evaluated").to_numpy() & (table["tau_e_th_s"] > 0).to_numpy()
        np.testing.assert_array_equal(_tier_a_selected(table, thresholds).to_numpy(), decided)


def test_tier_a_loader_reads_an_atlas_table_and_rejects_a_foreign_one(tmp_path):
    import pandas as pd

    from vaft.data import data_path, public

    table = pd.read_csv(data_path("confinement/vest_tier_a_confinement.csv"))
    copy = tmp_path / "table.csv"
    table.to_csv(copy, index=False)          # no manifest beside it: definitions stay absent
    loaded = public.load_vest_tier_a_confinement(path=copy)
    assert len(loaded) == 59 and loaded["n_e_definition"].isna().all()
    table.drop(columns=["dwdt_fraction"]).to_csv(copy, index=False)
    with pytest.raises(ValueError, match="slice-evidence"):
        public.load_vest_tier_a_confinement(path=copy)
