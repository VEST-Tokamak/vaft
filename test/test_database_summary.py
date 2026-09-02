from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from vaft import database
from vaft.database import summary as public_summary
from vaft.database import summary as summary_function
import vaft.database._summary as summary_module


def _preset(extractor):
    return summary_module.SummaryPreset(
        columns=("shot", "eq_index", "time_s", "value"),
        paths=("equilibrium.time_slice",),
        key_columns=("shot", "eq_index", "time_s"),
        replace_groups=("shot",),
        sort_columns=("shot", "time_s", "eq_index"),
        extractor=extractor,
    )


def test_summary_uses_inclusive_range_lazy_paths_and_skips_failures(
    monkeypatch, caplog
):
    opened = []

    def fake_open(shot, **kwargs):
        opened.append((shot, kwargs))
        if shot == 11:
            raise FileNotFoundError("missing")
        return nullcontext(object())

    def extract(_ods, shot):
        return [{"shot": shot, "eq_index": 0, "time_s": 0.2, "value": shot * 2}]

    monkeypatch.setitem(summary_module.PRESETS, "test", _preset(extract))
    monkeypatch.setattr(database, "open", fake_open)

    result = public_summary((10, 12), preset="test", source="public")

    assert result["shot"].tolist() == [10, 12]
    assert [call[0] for call in opened] == [10, 11, 12]
    assert all(call[1]["paths"] == ["equilibrium.time_slice"] for call in opened)
    assert "Shot 11" in caplog.text


@pytest.mark.parametrize("shot_range", [(1,), (1, 2, 3), (True, 2), (1.0, 2)])
def test_summary_rejects_invalid_ranges(shot_range):
    with pytest.raises(TypeError):
        summary_function(shot_range)


def test_summary_rejects_reversed_range():
    with pytest.raises(ValueError, match="start"):
        summary_function((3, 2))


def test_summary_empty_result_has_canonical_columns(monkeypatch):
    monkeypatch.setitem(
        summary_module.PRESETS, "empty", _preset(lambda _ods, _shot: [])
    )
    monkeypatch.setattr(database, "open", lambda *args, **kwargs: nullcontext(object()))

    result = public_summary((5, 5), preset="empty")

    assert result.empty
    assert tuple(result.columns) == summary_module.PRESETS["empty"].columns


def test_summary_without_range_discovers_all_available_shots(monkeypatch):
    opened = []

    monkeypatch.setitem(
        summary_module.PRESETS,
        "all",
        _preset(
            lambda _ods, shot: [
                {"shot": shot, "eq_index": 0, "time_s": 0.1, "value": shot}
            ]
        ),
    )
    monkeypatch.setattr(
        "vaft.database.utils.exist_shot",
        lambda **_kwargs: ["12", "metadata.h5", "10", "12"],
    )

    def fake_open(shot, **_kwargs):
        opened.append(shot)
        return nullcontext(object())

    monkeypatch.setattr(database, "open", fake_open)

    result = public_summary(preset="all")

    assert opened == [10, 12]
    assert result["shot"].tolist() == [10, 12]


def test_public_summary_remains_callable_after_internal_module_import():
    assert callable(database.summary)


def test_equilibrium_extractor_preserves_canonical_schema(monkeypatch):
    ods = {
        "equilibrium.time": [0.25],
        "equilibrium.time_slice": [
            {
                "time": 0.25,
                "global_quantities.ip": 100_000.0,
                "global_quantities.q_95": 3.2,
                "boundary.geometric_axis.r": 0.4,
                "boundary.minor_radius": 0.2,
            }
        ],
        "equilibrium.vacuum_toroidal_field.b0": 0.2,
        "equilibrium.vacuum_toroidal_field.r0": 0.2,
    }
    for name in (
        "update_equilibrium_profiles_1d_geometry",
        "update_equilibrium_global_quantities_beta_li",
        "update_equilibrium_boundary",
        "update_equilibrium_global_quantities_q_min",
        "update_equilibrium_global_quantities_volume",
        "update_equilibrium_global_quantities_area",
        "update_equilibrium_stored_energy",
        "update_equilibrium_constraints_diamagnetic_flux",
    ):
        monkeypatch.setattr(f"vaft.omas.{name}", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "vaft.omas.compute_virial_equilibrium_quantities_ods",
        lambda *_args, **_kwargs: {0: {"beta_p_vir_lao": 0.5, "li_vir_lao": 0.7}},
    )
    rows = summary_module.extract_equilibrium_global(ods, 42)

    assert set(rows[0]) == set(summary_module.EQUILIBRIUM_GLOBAL_COLUMNS)
    assert rows[0]["shot"] == 42
    assert rows[0]["ip_kA"] == 100.0
    assert rows[0]["aspect_ratio"] == 2.0
    assert rows[0]["vacuum_b0_T"] == pytest.approx(0.1)
    assert rows[0]["vacuum_r0_m"] == 0.4
    assert "psi_boundary_Wb" not in rows[0]
    assert "reconstructed_dia_flux_Wb" not in rows[0]
    assert "dia_flux_Wb" in rows[0]
    assert rows[0]["virial_beta"] == rows[0]["virial_beta_lao"] == 0.5


def test_core_profiles_extractor_uses_canonical_units(monkeypatch):
    ods = {
        "equilibrium.time_slice": [{"time": 0.1}],
        "core_profiles.profiles_1d": [
            {"time": 0.1, "electrons.temperature": [10.0, 20.0]}
        ],
    }
    monkeypatch.setattr(
        "vaft.omas.update_equilibrium_boundary", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "vaft.omas.general.find_matching_time_indices",
        lambda *_args, **_kwargs: (0, 0, 0.1),
    )
    monkeypatch.setattr(
        "vaft.omas.formula_wrapper.compute_tau_E_engineering_parameters",
        lambda *_args, **_kwargs: {
            "I_p": 80_000.0,
            "B_t": 0.1,
            "P_loss": 1_000_000.0,
            "n_e_line_avg": 2e19,
            "n_e_vol_avg": 3e19,
            "R": 0.4,
            "epsilon": 0.5,
            "kappa": 1.6,
        },
    )
    monkeypatch.setattr(
        "vaft.omas.formula_wrapper.compute_confiment_time_paramters",
        lambda *_args, **_kwargs: (1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 6.0),
    )

    rows = summary_module.extract_core_profiles(ods, 42)

    assert set(rows[0]) == set(summary_module.CORE_PROFILES_COLUMNS)
    assert rows[0]["ip_kA"] == 80.0
    assert rows[0]["p_loss_MW"] == 1.0
    assert rows[0]["ne_volume_1e19_m3"] == 3.0
    assert rows[0]["te_mean_eV"] == 15.0


def test_volume_averaged_extractor_matches_nearest_equilibrium_slice(monkeypatch):
    ods = {
        "core_profiles.profiles_1d": [{"time": 0.2}, {"time": 0.4}],
        "core_profiles": {"global_quantities": {}},
        "equilibrium.time_slice": [{"time": 0.19}, {"time": 0.41}],
    }

    def update(values):
        values["core_profiles"]["global_quantities"] = {
            "n_e_volume_average": [1e19, 2e19],
            "t_e_volume_average": [10.0, 20.0],
        }

    monkeypatch.setattr(
        "vaft.omas.update_core_profiles_global_quantities_volume_average", update
    )
    monkeypatch.setattr(
        "vaft.omas.compute_volume_averaged_pressure",
        lambda *_args, **_kwargs: [100.0, 200.0],
    )

    rows = summary_module.extract_volume_averaged(ods, 42)

    assert len(rows) == 2
    assert set(rows[0]) == set(summary_module.VOLUME_AVERAGED_COLUMNS)
    assert [row["eq_index"] for row in rows] == [0, 1]
    assert rows[0]["time_diff_s"] == pytest.approx(0.01)


def _reliability_ods():
    fitted = {
        "measured": 1.0,
        "reconstructed": 1.2,
        "measured_error_upper": 0.1,
        "weight": 2.0,
        "chi_squared": 4.0,
        "exact": 0,
        "identifier": "channel-1",
        "source": "current EFIT m-file",
    }
    disabled = dict(fitted, measured=2.0, reconstructed=2.1, weight=0.0)
    no_uncertainty = dict(fitted, measured=3.0, reconstructed=3.2)
    no_uncertainty.pop("measured_error_upper")
    return {
        "dataset_description.data_entry.machine": "VEST",
        "dataset_description.data_entry.run": 1,
        "equilibrium.ids_properties.comment": "current EFIT",
        "equilibrium.code.parameters.time_slice.0.meqdsk.mapping_source_revision": "efit-revision",
        "equilibrium.time": [0.1],
        "equilibrium.time_slice": [
            {
                "time": 0.1,
                "global_quantities.ip": 75_000.0,
                "constraints.chi_squared_reduced": 1.5,
                "convergence.iterations_n": 7,
                "convergence.grad_shafranov_deviation_value": 1e-5,
                "constraints.bpol_probe": [fitted, disabled],
                "constraints.flux_loop": {"2": no_uncertainty},
                "constraints.pf_current": [fitted],
                "constraints.ip": fitted,
                "constraints.diamagnetic_flux": fitted,
                "constraints.pressure": [fitted],
                "constraints.pressure_rotational": [fitted],
                "constraints.j_tor": [fitted],
                "constraints.mse_polarisation_angle": [fitted],
            }
        ],
    }


def test_efit_magnetic_reliability_extracts_all_mapped_families():
    rows = summary_module.extract_efit_magnetic_reliability(
        _reliability_ods(),
        42,
    )

    assert len(rows) == 6
    assert set(rows[0]) == set(summary_module.EFIT_RELIABILITY_COLUMNS)
    assert {row["measurement_type"] for row in rows} == {
        "bpol_probe",
        "flux_loop",
        "pf_current",
        "ip",
        "diamagnetic_flux",
    }
    assert sum(row["measurement_type"] == "bpol_probe" for row in rows) == 2
    assert sum(row["measurement_type"] == "flux_loop" for row in rows) == 1
    assert next(row for row in rows if row["disabled"])["weight"] == 0.0
    bpol = rows[0]
    assert bpol["residual"] == pytest.approx(0.2)
    assert bpol["absolute_residual"] == pytest.approx(0.2)
    assert bpol["normalized_residual"] == pytest.approx(2.0)
    assert bpol["identifier"] == "channel-1"
    assert bpol["constraint_source"] == "current EFIT m-file"
    assert bpol["chi_squared_reduced"] == 1.5
    assert bpol["convergence_iterations_n"] == 7
    assert bpol["convergence_deviation"] == 1e-5
    assert '"machine":"VEST"' in bpol["equilibrium_lineage"]
    assert '"mapping_source_revision":"efit-revision"' in bpol["equilibrium_lineage"]
    flux = next(row for row in rows if row["measurement_type"] == "flux_loop")
    assert flux["measurement_index"] == 2
    assert np.isnan(flux["uncertainty"])
    assert np.isnan(flux["normalized_residual"])


def test_efit_kinetic_reliability_extracts_all_mapped_families():
    rows = summary_module.extract_efit_kinetic_reliability(
        _reliability_ods(),
        42,
    )

    assert [row["measurement_type"] for row in rows] == [
        "pressure",
        "pressure_rotational",
        "j_tor",
        "mse_polarisation_angle",
    ]


def test_efit_reliability_skips_nonfinite_and_malformed_fits():
    ods = {
        "equilibrium.time": [0.1],
        "equilibrium.time_slice": [
            {
                "time": 0.1,
                "global_quantities.ip": 75_000.0,
                "constraints.bpol_probe": [
                    {"measured": 1.0},
                    {"measured": np.nan, "reconstructed": 1.1},
                    "malformed",
                ],
                "constraints.diamagnetic_flux": {"measured": 2.0},
            }
        ],
    }

    rows = summary_module.extract_efit_magnetic_reliability(ods, 42)

    assert rows == []


def test_split_reliability_summary_injects_database_source(monkeypatch):
    # An experiment namespace outside the catalog, opted into explicitly.
    monkeypatch.setenv("VAFT_HSDS_EXTRA_SOURCES", "private")
    monkeypatch.setattr(
        database,
        "open",
        lambda *_args, **_kwargs: nullcontext(_reliability_ods()),
    )

    magnetic = public_summary(
        (42, 42),
        preset="efit_magnetic_reliability",
        source="private",
    )
    kinetic = public_summary(
        (42, 42),
        preset="efit_kinetic_reliability",
        source="private",
    )

    assert set(magnetic["equilibrium_source"]) == {"private"}
    assert set(kinetic["equilibrium_source"]) == {"private"}
    with pytest.warns(DeprecationWarning):
        alias = public_summary((42, 42), preset="efit_reliability", source="private")
    pd.testing.assert_frame_equal(alias, magnetic)


def test_shot_overview_extractor_uses_diagnostic_signals(monkeypatch):
    ods = {
        "spectrometer_uv.time": [0.0, 0.1, 0.2, 0.3],
        "spectrometer_uv.channel.0.processed_line.0.intensity.data": [0, 1, 1, 0],
        "magnetics.ip.0.data": [0, 10_000, 20_000, 0],
        "tf.time": [0.0, 0.1, 0.2, 0.3],
        "tf.b_field_tor_vacuum_r.data": [0.0, 0.04, 0.08, 0.0],
        "tf.r0": 0.4,
    }
    monkeypatch.setattr(
        "vaft.process.signal_on_offset", lambda *_args, **_kwargs: (0.1, 0.2)
    )

    rows = summary_module.extract_shot_overview(ods, 42)

    assert set(rows[0]) == set(summary_module.SHOT_OVERVIEW_COLUMNS)
    assert rows[0]["pulse_duration_s"] == pytest.approx(0.1)
    assert rows[0]["max_ip_kA"] == 10.0
    assert rows[0]["mean_b_t_T"] == pytest.approx(0.15)


def test_export_replace_preserves_frame_and_column_order(tmp_path):
    frame = pd.DataFrame({"b": [2], "a": [1]})
    path = tmp_path / "summary.csv"

    written = database.export_summary(frame, path)

    pd.testing.assert_frame_equal(written, frame)
    assert pd.read_csv(path).columns.tolist() == ["b", "a"]


def test_export_upsert_replaces_whole_incoming_group_and_preserves_other_groups(
    tmp_path,
):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "shot": [1, 1, 2],
            "eq_index": [0, 1, 0],
            "time_s": [0.1, 0.2, 0.1],
            "value": [10, 11, 20],
        }
    ).to_csv(path, index=False)
    incoming = pd.DataFrame(
        {"shot": [1], "eq_index": [0], "time_s": [0.1], "value": [99]}
    )

    written = database.export_summary(
        incoming,
        path,
        mode="upsert",
        key_columns=("shot", "eq_index", "time_s"),
        replace_groups=("shot",),
    )

    assert written.to_dict("records") == [
        {"shot": 1, "eq_index": 0, "time_s": 0.1, "value": 99},
        {"shot": 2, "eq_index": 0, "time_s": 0.1, "value": 20},
    ]


def test_export_key_upsert_is_idempotent_and_reconciles_schema(tmp_path):
    path = tmp_path / "summary.xlsx"
    pd.DataFrame({"shot": [1], "eq_index": [0], "time_s": [0.1]}).to_excel(
        path, index=False
    )
    incoming = pd.DataFrame(
        {"shot": [1], "eq_index": [0], "time_s": [0.1], "value": [5.0]}
    )
    kwargs = {"mode": "upsert", "key_columns": ("shot", "eq_index", "time_s")}

    first = database.export_summary(incoming, path, **kwargs)
    second = database.export_summary(incoming, path, **kwargs)

    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    assert len(second) == 1
    assert second.columns.tolist() == incoming.columns.tolist()


def test_export_validates_upsert_contract(tmp_path):
    frame = pd.DataFrame({"shot": [1], "value": [2]})
    with pytest.raises(ValueError, match="key_columns"):
        database.export_summary(frame, tmp_path / "out.csv", mode="upsert")
    with pytest.raises(ValueError, match="missing merge columns"):
        database.export_summary(
            frame, tmp_path / "out.csv", mode="upsert", key_columns=("missing",)
        )
    with pytest.raises(ValueError, match=r"\.csv or \.xlsx"):
        database.export_summary(frame, tmp_path / "out.json")


def test_equilibrium_summary_fills_the_shape_and_volume_columns():
    """Regression for issue #290's wider half: eleven of the forty-four columns
    were empty on the packaged sample because an EFIT-sourced ODS stores no
    flux-surface geometry, not because nothing could compute it.

    Four of them -- the betas and li_3 -- outlived that fix because they needed a
    normative definition chosen first, and #238 chose it. Nothing is empty now.
    """
    import numpy as np

    from vaft.omas.sample import sample_ods

    try:
        ods = sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")

    row = summary_module.extract_equilibrium_global(ods, 39915)[0]
    empty = {
        name
        for name, value in row.items()
        if value is None or (isinstance(value, float) and not np.isfinite(value))
    }
    assert not empty, f"summary columns still empty: {sorted(empty)}"
    assert row["volume_m3"] > 0.0
    assert row["area_m2"] > 0.0
    assert row["energy_mhd_J"] > 0.0
    # The virial path read psi through a detector that assumed weber and got
    # beta_p = 30.5 on this Wb/rad sample (issue #278 follow-up).
    assert 0.0 < row["virial_beta"] < 10.0


def test_the_equilibrium_preset_loads_tf_so_the_r0_cross_check_can_run():
    """`beta_pol` and `li_3` divide by R_0, and the VEST database's
    `equilibrium.vacuum_toroidal_field.r0` is corrupt (#325). The resolver
    cross-checks it against `tf`, so a preset that does not load `tf` leaves the
    check blind and the corrupt value is used with no warning -- the summary was
    in exactly that state.
    """
    assert "tf" in summary_module.get_summary_preset("equilibrium_global").paths


def test_vacuum_b0_is_rescaled_with_the_cross_checked_radius():
    """`vacuum_b0_T` rescales `b0` by `r0/0.4`, so it inherits the bad `r0`
    directly: on shot 39915 it reported 0.0866 T where `tf` says 0.150 T."""
    import numpy as np
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.array([0.3])
    ods["equilibrium.time_slice.0.time"] = 0.3
    ods["equilibrium.time_slice.0.global_quantities.ip"] = 80_000.0
    # The corruption as the database holds it: b0 is the field at R = 0.4, but
    # r0 says 0.2313, so b0*r0 disagrees with tf's B*R by that ratio.
    ods["equilibrium.vacuum_toroidal_field.r0"] = 0.231317
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.array([0.149799])
    ods["tf.r0"] = 0.4
    ods["tf.b_field_tor_vacuum_r.data"] = np.full(4, 0.149799 * 0.4)

    row = summary_module.extract_equilibrium_global(ods, 39915)[0]
    # Rescaled with R0 = 0.4, so b0 passes through unchanged rather than being
    # shrunk by 0.2313/0.4.
    assert row["vacuum_b0_T"] == pytest.approx(0.149799, rel=1e-6)
    assert row["vacuum_r0_m"] == pytest.approx(0.4)
