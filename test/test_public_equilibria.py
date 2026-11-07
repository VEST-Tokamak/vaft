"""Licence-verified public equilibria: registry, adapters, failure stages (#2285).

Offline: the DIII-D parquet record and the FAIR-MAST Zarr store are synthetic,
written in the audited layouts; the g-file adapter reads the packaged VEST
g-file.  Network access is mocked at ``_fetch.urlopen``, so the real checksum
path runs.  Loading the real artefacts runs only with ``VAFT_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import re
from unittest import mock
from urllib.error import URLError

import numpy as np
import pytest

from vaft.data.public import (
    EXCLUDED_EQUILIBRIUM_SOURCES,
    PUBLIC_EQUILIBRIUM_DEVICES,
    PUBLIC_EQUILIBRIUM_SOURCES,
    ChecksumError,
    EquilibriumCompatibilityError,
    EquilibriumLicenceError,
    EquilibriumParseError,
    FetchError,
    load_public_equilibrium,
    public_equilibrium_provenance,
)
from vaft.data.public import _fetch, equilibria
from vaft.ods_access import path_value

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(store: dict[str, bytes]):
    """A urlopen stand-in that serves ``store`` by full URL."""

    def opener(request, timeout=None):
        url = request.full_url
        if url not in store:
            raise URLError(f"not served: {url}")
        return _Response(store[url])

    return mock.patch.object(_fetch, "urlopen", side_effect=opener)


# ----------------------------------------------------------------- registry


def test_every_external_source_is_pinned_licensed_and_attributed():
    externals = [s for s in PUBLIC_EQUILIBRIUM_SOURCES.values() if s.reader != "vaft_sample"]
    assert {s.device for s in externals} == {"TCV", "DIII-D", "MAST", "STEP", "SPARC", "ITER"}
    for source in externals:
        assert _HEX64.match(source.sha256), source.device
        assert source.size_bytes and source.size_bytes > 0
        assert source.licence and source.licence_url.startswith("https://")
        assert source.licence_basis and source.attribution
        assert source.data_kind and source.contract in {"full", "flux-map+lcfs"}
        assert source.selection and source.convention
        if source.reader not in ("fair_mast_zarr", "imas_netcdf"):
            # A revision-pinned URL, never a moving branch.
            assert source.revision in source.url
            assert "/main/" not in source.url and "/master/" not in source.url
    assert PUBLIC_EQUILIBRIUM_SOURCES["MAST"].share_alike is True
    assert PUBLIC_EQUILIBRIUM_SOURCES["DIII-D"].contract == "flux-map+lcfs"
    assert "train" in PUBLIC_EQUILIBRIUM_SOURCES["DIII-D"].artefact
    assert all(_HEX64.match(sha) for sha in equilibria._MAST_CHUNKS.values())
    assert PUBLIC_EQUILIBRIUM_DEVICES[0] == "VEST"


def test_excluded_and_unknown_devices_are_licence_failures():
    assert set(EXCLUDED_EQUILIBRIUM_SOURCES) == {"JET", "MAST-U"}
    assert not set(EXCLUDED_EQUILIBRIUM_SOURCES) & set(PUBLIC_EQUILIBRIUM_SOURCES)
    for device in ("JET", "MAST-U", "NSTX"):
        with pytest.raises(EquilibriumLicenceError):
            load_public_equilibrium(device)


def test_no_active_source_points_at_fuse():
    for source in PUBLIC_EQUILIBRIUM_SOURCES.values():
        assert "FUSE" not in source.url and "ProjectTorreyPines" not in source.url


def test_flat_top_midpoint_rule():
    time = np.arange(10.0)
    ip = np.array([0, 1, 5, 9.5, 10, 9.8, 9.6, 9.2, 4, 0.0])
    # window |Ip| >= 9 is indices 3..7, middle 5
    assert equilibria._flat_top_midpoint(time, ip) == 5
    assert equilibria._flat_top_midpoint(time, -ip) == 5
    with pytest.raises(EquilibriumCompatibilityError):
        equilibria._flat_top_midpoint(time, np.zeros(10))


# ----------------------------------------------------------------- DIII-D


def _diiid_row(source_name="DIII-D", n_time=5, n_grid=9):
    r = np.linspace(1.0, 2.4, n_grid)
    z = np.linspace(-1.2, 1.2, n_grid + 2)
    rr, zz = np.meshgrid(r, z)  # (Z, R), as efit_psirz is stored
    times = np.linspace(100.0, 500.0, n_time)
    psirz = [((rr - 1.7) ** 2 + (zz / 1.5) ** 2) * (k + 1) - 0.3 for k in range(n_time)]
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    lcfs_r = np.full((n_time, 15), np.nan)
    lcfs_z = np.full((n_time, 15), np.nan)
    lcfs_r[:, :12] = 1.7 + 0.4 * np.cos(theta)
    lcfs_z[:, :12] = 0.6 * np.sin(theta)
    return {
        "source": source_name,
        "efit_times": times.tolist(),
        "efit_psirz": [p.tolist() for p in psirz],
        "efit_grid_R": r.tolist(),
        "efit_grid_Z": z.tolist(),
        "efit_lcfs_n": [12.0] * n_time,
        "efit_lcfs_r": lcfs_r.tolist(),
        "efit_lcfs_z": lcfs_z.tolist(),
        "efit_r_axis": [1.7] * n_time,
        "efit_z_axis": [0.0] * n_time,
        "efit_q95": [4.0 + k for k in range(n_time)],
        "efit_beta_n": [1.5] * n_time,
        "efit_li": [0.9] * n_time,
        "magnetics_plasma_current": [0.0, 1000.0, 1000.0, 1000.0, 0.0],
        "magnetics_plasma_current_times": [0.0, 150.0, 300.0, 450.0, 600.0],
    }


def _write_parquet(path, row):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist([row]), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diiid_source(sha):
    return dataclasses.replace(PUBLIC_EQUILIBRIUM_SOURCES["DIII-D"], sha256=sha)


def test_diiid_record_maps_flux_map_and_lcfs_without_inventing_profiles(tmp_path):
    row = _diiid_row()
    sha = _write_parquet(tmp_path / "rec.parquet", row)
    source = _diiid_source(sha)
    with _serve({source.url: (tmp_path / "rec.parquet").read_bytes()}):
        result = equilibria._read_neurips_parquet(source, cache=tmp_path / "cache")
    ods = result.ods
    # |Ip| >= 900 A on efit_times 200, 300, 400 ms -> middle is 300 ms (index 2)
    assert result.time_s == pytest.approx(0.3)
    psi = np.asarray(ods["equilibrium.time_slice.0.profiles_2d.0.psi"])
    expected = np.asarray(row["efit_psirz"][2]).T * 2 * np.pi
    assert psi.shape == (9, 11)
    np.testing.assert_allclose(psi, expected)
    assert len(ods["equilibrium.time_slice.0.boundary.outline.r"]) == 12
    assert ods["equilibrium.time_slice.0.global_quantities.q_95"] == pytest.approx(6.0)
    for leaf in ("profiles_1d.q", "profiles_1d.pressure", "profiles_1d.f", "global_quantities.li_3"):
        assert path_value(ods, f"equilibrium.time_slice.0.{leaf}") is None
    assert path_value(ods, "equilibrium.vacuum_toroidal_field.b0") is None
    assert {"q profile", "pressure profile", "F profile", "vacuum B0"} <= set(result.missing)
    assert result.cocos is None
    assert result.sha256 == sha
    assert ods["dataset_description.data_entry.machine"] == "DIII-D"
    assert "CC BY 4.0" in ods["equilibrium.ids_properties.comment"]


def test_missing_vacuum_field_is_not_read_as_zero(tmp_path):
    from vaft.process.equilibrium import as_equilibrium

    sha = _write_parquet(tmp_path / "rec.parquet", _diiid_row())
    source = _diiid_source(sha)
    with _serve({source.url: (tmp_path / "rec.parquet").read_bytes()}):
        result = equilibria._read_neurips_parquet(source, cache=tmp_path / "cache")
    eq = as_equilibrium(result.ods)
    assert eq.bt0 is None
    assert eq.q is None and eq.f is None and eq.pressure is None
    # reading must not leave phantom nodes behind
    assert path_value(result.ods, "equilibrium.vacuum_toroidal_field.b0") is None
    assert equilibria._missing(result.ods, 0) == result.missing


def test_diiid_record_from_another_machine_is_a_parse_error(tmp_path):
    sha = _write_parquet(tmp_path / "rec.parquet", _diiid_row(source_name="MAST"))
    source = _diiid_source(sha)
    with _serve({source.url: (tmp_path / "rec.parquet").read_bytes()}):
        with pytest.raises(EquilibriumParseError):
            equilibria._read_neurips_parquet(source, cache=tmp_path / "cache")


def test_diiid_checksum_and_network_failures_are_distinct(tmp_path):
    sha = _write_parquet(tmp_path / "rec.parquet", _diiid_row())
    source = _diiid_source("0" * 64)
    with _serve({source.url: (tmp_path / "rec.parquet").read_bytes()}):
        with pytest.raises(ChecksumError):
            equilibria._read_neurips_parquet(source, cache=tmp_path / "cache")
    with _serve({}):
        with pytest.raises(FetchError):
            equilibria._read_neurips_parquet(_diiid_source(sha), cache=tmp_path / "cache2")


# ----------------------------------------------------------------- FAIR-MAST


def _zarr_store(*, licence="Creative Commons 4.0 BY-SA", codecs=("bytes", "zstd")):
    pa = pytest.importorskip("pyarrow")
    nt, nr, nz, npsi, nb = 5, 6, 7, 4, 10
    r = np.linspace(0.2, 1.5, nr)
    z = np.linspace(-1.0, 1.0, nz).astype("float32")
    zz, rr = np.meshgrid(z, r, indexing="ij")
    psi = np.stack([0.05 - ((rr - 0.9) ** 2 + zz**2) * (k + 1) * 0.01 for k in range(nt)], axis=-1)
    theta = np.linspace(0, 2 * np.pi, nb - 2, endpoint=False)
    lcfs_r = np.full((nb, nt), np.nan)
    lcfs_z = np.full((nb, nt), np.nan)
    lcfs_r[: nb - 2, :] = (0.9 + 0.5 * np.cos(theta))[:, None]
    lcfs_z[: nb - 2, :] = (0.9 * np.sin(theta))[:, None]
    arrays = {
        "time": np.linspace(0.0, 0.4, nt),
        "ip": np.array([1e5, 7e5, 7.2e5, 7.1e5, 2e5]),
        "psi": psi,
        "major_radius": r,
        "z": z,
        "psi_axis": np.full(nt, 0.05),
        "psi_boundary": np.full(nt, -0.03),
        "magnetic_axis_r": np.full(nt, 0.9),
        "magnetic_axis_z": np.zeros(nt),
        "lcfs_r": lcfs_r,
        "lcfs_z": lcfs_z,
        "psi_norm": np.linspace(0, 1, npsi).astype("float32"),
        "f": np.full((npsi, nt), -0.4),
        "pressure": np.tile(np.linspace(8e3, 0, npsi)[:, None], (1, nt)),
        "q95": np.full(nt, 5.3),
        "x_point_r": np.full((2, nt), 0.55),
        "x_point_z": np.tile(np.array([[-1.1], [1.1]]), (1, nt)),
        "bvac_val": np.full(nt, -0.41),
        "bvac_r": np.full(nt, 1.0),
    }
    metadata = {
        "equilibrium": {
            "node_type": "group",
            "attributes": {
                "license_name": licence,
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            },
        }
    }
    chunks: dict[str, bytes] = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        # one chunk per array except psi, split in two along time
        chunk = list(value.shape)
        if name == "psi":
            chunk = [4, 4, 3]  # padded edges on every axis, as the real 33x33x49 grid
        metadata[f"equilibrium/{name}"] = {
            "node_type": "array",
            "shape": list(value.shape),
            "data_type": str(value.dtype),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "codecs": [{"name": codecs[0], "configuration": {"endian": "little"}}]
            + [{"name": c} for c in codecs[1:]],
        }
        counts = [-(-s // c) for s, c in zip(value.shape, chunk)]
        for flat in range(int(np.prod(counts))):
            index = np.unravel_index(flat, counts)
            block = np.zeros(chunk, dtype=value.dtype)
            region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunk, value.shape))
            block[tuple(slice(0, rg.stop - rg.start) for rg in region)] = value[region]
            key = f"equilibrium/{name}/c/" + "/".join(str(int(i)) for i in index)
            chunks[key] = pa.Codec("zstd").compress(block.astype(value.dtype.newbyteorder("<")).tobytes()).to_pybytes()
    root = json.dumps({"zarr_format": 3, "consolidated_metadata": {"metadata": metadata}}).encode()
    return root, chunks, arrays


def _mast(tmp_path, monkeypatch, **store_kwargs):
    root, chunks, arrays = _zarr_store(**store_kwargs)
    monkeypatch.setattr(
        equilibria, "_MAST_CHUNKS", {k: hashlib.sha256(v).hexdigest() for k, v in chunks.items()}
    )
    served = {equilibria._MAST_STORE + "zarr.json": root}
    served.update({equilibria._MAST_STORE + k: v for k, v in chunks.items()})
    source = dataclasses.replace(
        PUBLIC_EQUILIBRIUM_SOURCES["MAST"], sha256=hashlib.sha256(root).hexdigest()
    )
    return source, served, arrays, chunks


def test_fair_mast_store_maps_one_reconstructed_slice(tmp_path, monkeypatch):
    source, served, arrays, _ = _mast(tmp_path, monkeypatch)
    with _serve(served):
        result = equilibria._read_fair_mast(source, cache=tmp_path)
    ods = result.ods
    # |Ip| >= 0.9 * 7.2e5 at indices 1, 2, 3 -> index 2
    assert result.time_s == pytest.approx(0.2)
    psi = np.asarray(ods["equilibrium.time_slice.0.profiles_2d.0.psi"])
    np.testing.assert_allclose(psi, arrays["psi"][:, :, 2].T * 2 * np.pi)
    assert psi.shape == (6, 7)
    np.testing.assert_allclose(ods["equilibrium.time_slice.0.profiles_2d.0.grid.dim2"], arrays["z"])
    assert ods["equilibrium.time_slice.0.global_quantities.psi_axis"] == pytest.approx(0.05 * 2 * np.pi)
    assert len(ods["equilibrium.time_slice.0.boundary.outline.r"]) == 8
    assert ods["equilibrium.vacuum_toroidal_field.b0"][0] == pytest.approx(-0.41)
    assert ods["equilibrium.time_slice.0.global_quantities.q_95"] == pytest.approx(5.3)
    # q lives on a midplane coordinate upstream and is deliberately not mapped
    assert path_value(ods, "equilibrium.time_slice.0.profiles_1d.q") is None
    assert result.missing == ("q profile",)
    assert ods["dataset_description.data_entry.pulse"] == 30420


def test_fair_mast_licence_change_is_a_licence_error(tmp_path, monkeypatch):
    source, served, _, _ = _mast(tmp_path, monkeypatch, licence="Creative Commons 4.0 BY-NC")
    with _serve(served):
        with pytest.raises(EquilibriumLicenceError):
            equilibria._read_fair_mast(source, cache=tmp_path)


def test_fair_mast_unexpected_codec_is_a_parse_error(tmp_path, monkeypatch):
    source, served, _, _ = _mast(tmp_path, monkeypatch, codecs=("bytes", "blosc"))
    with _serve(served):
        with pytest.raises(EquilibriumParseError):
            equilibria._read_fair_mast(source, cache=tmp_path)


def test_fair_mast_changed_chunk_is_a_checksum_error(tmp_path, monkeypatch):
    source, served, _, chunks = _mast(tmp_path, monkeypatch)
    key = equilibria._MAST_STORE + "equilibrium/psi/c/0/0/1"
    served[key] = served[key] + b"\x00"
    with _serve(served):
        with pytest.raises(ChecksumError):
            equilibria._read_fair_mast(source, cache=tmp_path)


def test_fair_mast_unpinned_chunk_is_refused(tmp_path, monkeypatch):
    source, served, _, _ = _mast(tmp_path, monkeypatch)
    pins = dict(equilibria._MAST_CHUNKS)
    pins.pop("equilibrium/q95/c/0")
    monkeypatch.setattr(equilibria, "_MAST_CHUNKS", pins)
    with _serve(served):
        with pytest.raises(EquilibriumParseError):
            equilibria._read_fair_mast(source, cache=tmp_path)


# ----------------------------------------------------------------- GEQDSK


def _gfile_source(device, **changes):
    from vaft.data import data_path

    path = data_path("efit/g039915.00319")
    payload = open(path, "rb").read()
    source = dataclasses.replace(
        PUBLIC_EQUILIBRIUM_SOURCES[device],
        sha256=hashlib.sha256(payload).hexdigest(),
        **changes,
    )
    return source, payload


def test_geqdsk_adapter_reads_and_records_provenance(tmp_path):
    source, payload = _gfile_source("TCV")
    with _serve({source.url: payload}):
        result = equilibria._read_geqdsk(source, cache=tmp_path)
    assert result.missing == ()
    assert result.time_s is None or result.time_s == pytest.approx(0.319)
    assert len(result.ods["equilibrium.time_slice.0.boundary.outline.r"]) > 3
    assert result.source.device == "TCV"
    assert result.ods["dataset_description.data_entry.machine"] == "TCV"
    table = public_equilibrium_provenance([result])
    assert list(table["device"]) == ["TCV"]
    for column in ("licence", "licence_basis", "attribution", "share_alike", "revision", "sha256", "missing"):
        assert column in table.columns


def test_stated_cocos_is_labelled_only_when_the_data_agree(tmp_path, monkeypatch):
    from vaft.omas.general import ods_cocos

    # The g-file reader labels identifiable files itself; switch that off so
    # the stated-index path is what decides.
    monkeypatch.setattr("vaft.data.eqdsk._label_cocos", lambda *a, **k: None)

    source, payload = _gfile_source("STEP")
    with _serve({source.url: payload}):
        result = equilibria._read_geqdsk(source, cache=tmp_path)
    candidates = result.cocos_candidates
    assert candidates, "the packaged g-file is identifiable"
    assert ods_cocos(result.ods) is not None
    agree = candidates[0] - 10 if candidates[0] > 10 else candidates[0]
    disagree = next(i for i in range(1, 9) if i + 10 not in candidates and i not in candidates)
    for stated, labelled in ((agree, True), (disagree, False)):
        changed = dataclasses.replace(source, stated_cocos=stated)
        with _serve({source.url: payload}):
            out = equilibria._read_geqdsk(changed, cache=tmp_path / str(stated))
        assert (out.cocos is not None) is labelled
        assert any("producer" in note.lower() or "Producer" in note for note in out.notes)


# ----------------------------------------------------------------- IMAS netCDF (ITER)


def _imas_nc_source():
    """The packaged VEST IMAS netCDF (DD 3.41.0, several IDSs) served as ITER."""
    import vaft.data

    payload = open(vaft.data.sample(39915, representation="imas"), "rb").read()
    source = dataclasses.replace(
        PUBLIC_EQUILIBRIUM_SOURCES["ITER"], sha256=hashlib.sha256(payload).hexdigest()
    )
    return source, payload


def test_imas_netcdf_adapter_keeps_only_the_equilibrium(tmp_path, monkeypatch):
    monkeypatch.setattr(equilibria, "_ITER_DD_VERSION", "3.41.0")
    source, payload = _imas_nc_source()
    with _serve({source.url: payload}):
        result = equilibria._read_imas_netcdf(source, cache=tmp_path)
    assert set(result.ods.keys()) == {"equilibrium", "dataset_description"}
    assert result.ods.imas_version == "3.41.0"  # the artefact's DD, not omas's default
    assert any("not adapted" in note and "magnetics" in note for note in result.notes)
    times = np.asarray(result.ods["equilibrium.time"], dtype=float)
    ip = [result.ods[f"equilibrium.time_slice.{i}.global_quantities.ip"] for i in range(len(times))]
    assert result.time_index == equilibria._flat_top_midpoint(times, ip)
    assert result.time_s == pytest.approx(times[result.time_index])
    assert result.ods["dataset_description.data_entry.machine"] == "ITER"
    assert "zenodo" in result.ods["equilibrium.ids_properties.comment"].lower()


def test_imas_netcdf_with_another_dd_version_is_a_parse_error(tmp_path):
    source, payload = _imas_nc_source()  # DD 3.41.0, audited as 3.39.0
    with _serve({source.url: payload}), pytest.raises(EquilibriumParseError):
        equilibria._read_imas_netcdf(source, cache=tmp_path)


def test_iter_record_is_the_iter_organization_release():
    source = PUBLIC_EQUILIBRIUM_SOURCES["ITER"]
    assert source.producer == "ITER Organization"
    assert source.licence == "CC BY 4.0" and source.stated_cocos == 11
    assert "zenodo.org/records/17062700/" in source.url and source.url.endswith("DD3.nc?download=1")
    assert "modelled" in source.data_kind


# ----------------------------------------------------------------- network


@pytest.mark.skipif(os.environ.get("VAFT_NETWORK_TESTS") != "1", reason="needs network")
@pytest.mark.parametrize("device", [d for d in PUBLIC_EQUILIBRIUM_DEVICES if d != "VEST"])
def test_real_public_equilibrium_loads_with_its_pinned_hash(device, tmp_path):
    result = load_public_equilibrium(device, cache=tmp_path)
    assert result.sha256 == PUBLIC_EQUILIBRIUM_SOURCES[device].sha256
    assert len(result.ods["equilibrium.time_slice.0.boundary.outline.r"]) > 3
    psi = np.asarray(path_value(result.ods, "equilibrium.time_slice.0.profiles_2d.0.psi"))
    r = np.asarray(result.ods["equilibrium.time_slice.0.profiles_2d.0.grid.dim1"])
    z = np.asarray(result.ods["equilibrium.time_slice.0.profiles_2d.0.grid.dim2"])
    assert psi.shape == (r.size, z.size)
    # psi orientation: the flux extremum inside the LCFS sits at the magnetic axis
    axis = (
        result.ods["equilibrium.time_slice.0.global_quantities.magnetic_axis.r"],
        result.ods["equilibrium.time_slice.0.global_quantities.magnetic_axis.z"],
    )
    rr, zz = np.meshgrid(r, z, indexing="ij")
    near = (rr - axis[0]) ** 2 + (zz - axis[1]) ** 2 < 0.3**2 * (r[-1] - r[0]) ** 2
    psi_axis = result.ods["equilibrium.time_slice.0.global_quantities.psi_axis"]
    local = psi[near]
    extremum = local.max() if abs(local.max() - psi_axis) < abs(local.min() - psi_axis) else local.min()
    i, j = np.unravel_index(np.flatnonzero(near & (psi == extremum))[0], psi.shape)
    cell = max(r[1] - r[0], z[1] - z[0])
    assert np.hypot(r[i] - axis[0], z[j] - axis[1]) < 3 * cell


def test_geqdsk_without_a_parsable_time_reports_none(tmp_path):
    from vaft.data import data_path

    payload = open(data_path("efit/g039915.00319"), "rb").read()
    lines = payload.split(b"\n")
    lines[0] = b"VAFT TEST HEADER WITHOUT A TIME".ljust(48) + lines[0][48:]
    payload = b"\n".join(lines)
    source = dataclasses.replace(
        PUBLIC_EQUILIBRIUM_SOURCES["SPARC"], sha256=hashlib.sha256(payload).hexdigest()
    )
    with _serve({source.url: payload}):
        result = equilibria._read_geqdsk(source, cache=tmp_path)
    assert result.time_s is None
    assert any("placeholder" in note for note in result.notes)


def test_conversion_notes_reach_the_ods_comment(tmp_path):
    sha = _write_parquet(tmp_path / "rec.parquet", _diiid_row())
    source = _diiid_source(sha)
    with _serve({source.url: (tmp_path / "rec.parquet").read_bytes()}):
        result = equilibria._read_neurips_parquet(source, cache=tmp_path / "cache")
    comment = result.ods["equilibrium.ids_properties.comment"]
    assert "derived" in comment and "psi_boundary" in comment


def test_vest_sample_uses_the_flat_top_midpoint():
    result = load_public_equilibrium("VEST")
    ip = [
        result.ods[f"equilibrium.time_slice.{i}.global_quantities.ip"]
        for i in range(len(result.ods["equilibrium.time"]))
    ]
    assert result.time_index == equilibria._flat_top_midpoint(result.ods["equilibrium.time"], ip)
    assert _HEX64.match(result.sha256)
    assert result.missing == ()
