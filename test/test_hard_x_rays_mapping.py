import json
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.hard_x_rays import (
    HXR_FLUX_ENERGIES_KEV,
    hard_x_rays,
    hard_x_rays_from_flux_csv,
    load_hxr_flux_csv,
    load_hxr_raw_csv,
    resolve_hxr_time_alignment,
)


def _write_flux(root: Path, shot: int, data: np.ndarray, *, per_shot: bool = True) -> Path:
    directory = root / str(shot) if per_shot else root
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"digitizer_hxr_Eflux_17592_{shot}.csv"
    np.savetxt(path, data, delimiter=",")
    return path


def _flux(rows: int = 40) -> np.ndarray:
    return np.arange(rows * 5, dtype=float).reshape(rows, 5) * 1e-5


def test_eflux_maps_to_one_channel_with_energy_major_radiance(tmp_path):
    _write_flux(tmp_path, 12345, _flux())
    ods = ODS()
    hard_x_rays(ods, 12345, data_root=tmp_path, time_offset=0.3)

    radiance = ods["hard_x_rays.channel.0.radiance.data"]
    assert radiance.shape == (5, 40)
    np.testing.assert_allclose(radiance, _flux().T)
    energies = [ods[f"hard_x_rays.channel.0.energy_band.{i}.energies"][0] for i in range(5)]
    np.testing.assert_allclose(energies, np.array(HXR_FLUX_ENERGIES_KEV) * 1e3)
    # Nominal labels only: no invented band edges.
    assert "lower_bound" not in ods["hard_x_rays.channel.0.energy_band.0"]
    assert ods["hard_x_rays.ids_properties.homogeneous_time"] == 1


def test_time_is_bin_centre_after_the_trigger(tmp_path):
    _write_flux(tmp_path, 12345, _flux())
    ods = ODS()
    hard_x_rays(ods, 12345, data_root=tmp_path, time_offset=0.3)
    assert ods["hard_x_rays.channel.0.radiance.validity"] == 0
    time = ods["hard_x_rays.time"]
    assert time[0] == pytest.approx(0.3005)
    assert time[-1] == pytest.approx(0.3395)
    np.testing.assert_allclose(ods["hard_x_rays.channel.0.radiance.time"], time)


def test_flat_directory_and_archive_time_reference(tmp_path):
    _write_flux(tmp_path, 12345, _flux(), per_shot=False)
    ods = ODS()
    hard_x_rays(ods, 12345, data_root=tmp_path, time_reference="archive")
    assert ods["hard_x_rays.time"][0] == pytest.approx(0.0005)


def test_negative_bins_are_suspect_not_removed(tmp_path):
    data = _flux()
    data[3, 2] = -1e-4
    _write_flux(tmp_path, 12345, data)
    ods = ODS()
    hard_x_rays(ods, 12345, data_root=tmp_path, time_offset=0.0)
    flags = ods["hard_x_rays.channel.0.radiance.validity_timed"]
    assert flags.tolist() == [0, 0, 0, -1] + [0] * 36
    assert ods["hard_x_rays.channel.0.radiance.validity"] == -1
    assert ods["hard_x_rays.channel.0.radiance.data"][2, 3] == pytest.approx(-1e-4)


def test_appended_records_are_rejected(tmp_path):
    path = _write_flux(tmp_path, 12345, _flux(120))
    with pytest.raises(ValueError, match="3 appended records"):
        load_hxr_flux_csv(path)
    with pytest.raises(ValueError, match="append mode"):
        hard_x_rays(ODS(), 12345, data_root=tmp_path, time_offset=0.0)


def test_missing_file_names_the_searched_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="digitizer_hxr_Eflux_17592_12345.csv"):
        hard_x_rays(ODS(), 12345, data_root=tmp_path)


def test_packaged_hxr_trigger_is_authoritative():
    alignment = resolve_hxr_time_alignment(40587)
    assert alignment.source == "hxr_trigger"
    assert alignment.offset_seconds == pytest.approx(0.300)


def test_sxr_trigger_is_the_cotrigger_fallback():
    # 40568 logs only SXR; HXR is read from the same digitizer-17592 record.
    alignment = resolve_hxr_time_alignment(40568)
    assert alignment.source == "sxr_cotrigger"
    assert alignment.offset_seconds == pytest.approx(0.300)


def test_hxr_entry_wins_over_sxr(tmp_path):
    settings = tmp_path / "trigger-settings.yaml"
    settings.write_text(
        "shots:\n  47370:\n    SXR:\n      start_time_ms: 287\n    HXR:\n      start_time_ms: 285\n",
        encoding="utf-8",
    )
    alignment = resolve_hxr_time_alignment(47370, trigger_settings_path=settings)
    assert (alignment.source, alignment.offset_seconds) == ("hxr_trigger", pytest.approx(0.285))


def test_unlogged_shot_stays_trigger_relative_with_a_warning():
    with pytest.warns(RuntimeWarning, match="trigger-relative"):
        alignment = resolve_hxr_time_alignment(12345)
    assert (alignment.source, alignment.offset_seconds) == ("trigger_relative", 0.0)


def test_provenance_is_recorded(tmp_path):
    path = _write_flux(tmp_path, 12345, _flux())
    ods = ODS()
    hard_x_rays(ods, 12345, data_root=tmp_path, time_offset=0.0)
    assert ods["hard_x_rays.ids_properties.source"] == str(path)
    comment = ods["hard_x_rays.ids_properties.comment"]
    assert "arbitrary units" in comment and "time_alignment=explicit" in comment
    params = json.loads(ods["hard_x_rays.code.parameters"])
    assert params["dropped_energy_keV"] == [210.0]
    assert params["unfolding_model"]["name"] == "deconvolution1.model"


def test_consistency_checked_ods_round_trips(tmp_path):
    _write_flux(tmp_path, 12345, _flux())
    ods = hard_x_rays_from_flux_csv(12345, data_root=tmp_path, time_offset=0.3, consistency_check=True)
    output = tmp_path / "hxr.json"
    ods.save(str(output))
    loaded = ODS().load(str(output))
    np.testing.assert_allclose(loaded["hard_x_rays.channel.0.radiance.data"], _flux().T)


def test_raw_csv_is_transposed_to_samples_by_channel(tmp_path):
    path = tmp_path / "digitizer_hxr_raw_17592_40140.csv"
    np.savetxt(path, np.ones((4, 10)), delimiter=",")
    raw = load_hxr_raw_csv(path, sample_rate=10.0)
    assert raw["data"].shape == (10, 4)
    assert raw["time"][-1] == pytest.approx(0.9)
    np.savetxt(path, np.ones((5, 10)), delimiter=",")
    with pytest.raises(ValueError, match="at most 4"):
        load_hxr_raw_csv(path)
