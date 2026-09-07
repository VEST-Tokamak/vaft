"""The EFUND adapter: config, input writer, output contract, manifest (issue #194).

Everything here runs without the binary; the one test that needs ``efund``
is marked ``integration`` and skips when ``$EFITHOME`` resolves no such role.
"""

from __future__ import annotations

import json
import os
import struct

import f90nml
import numpy as np
import pytest

from external_code_stubs import write_launchable_stub
from vaft.code.efit import efund as efund_module
from vaft.code.efit.efund import (
    MHDIN_NAME,
    TABLE_MANIFEST_NAME,
    EFUNDConfig,
    EFUNDInputs,
    collect_efund_outputs,
    efund_namelist,
    expected_table_files,
    prepare_efund_inputs,
    fortran_byte_order,
    read_fortran_arrays,
    read_fortran_records,
    read_table_manifest,
    run_efund,
    table_identity,
    write_mhdin,
    write_table_manifest,
)
from vaft.code.efit.toolchain import resolve_role
from vaft.machine_mapping.efund_geometry import efund_geometry_from_static

LEGACY_ERA = "vest-pre-43017-pf1906"
# Captured before the autouse fixture clears the environment, so the
# integration test can opt back in to the toolchain the session configured.
_EFITHOME_AT_IMPORT = os.environ.get("EFITHOME")
# EFIT's own public DIII-D 181292 support file: the counts the checker's
# smoke run uses, so the size formulas here and there cannot disagree.
SMOKE_COUNTS = {"nsilop": 44, "magpri": 76, "nfsum": 18, "nfcoil": 18, "nesum": 6, "nvsum": 28, "nvesel": 28}


@pytest.fixture(scope="module")
def static():
    from vaft.omas.vest_upstream import build_static_ods

    return build_static_ods(LEGACY_ERA)


@pytest.fixture(scope="module")
def geometry(static):
    ods, manifest = static
    return efund_geometry_from_static(ods, manifest=manifest)


@pytest.fixture(autouse=True)
def no_toolchain_environment(monkeypatch):
    for name in ("EFITHOME", "EFIT"):
        monkeypatch.delenv(name, raising=False)


# --- configuration ---------------------------------------------------------


def test_defaults_are_the_legacy_table_with_its_echo_corrected():
    config = EFUNDConfig()
    assert (config.nw, config.nh) == (129, 129)
    assert (config.rleft, config.rright, config.zbotto, config.ztop) == (0.05, 1.2, -1.5, 1.5)
    assert config.ivesel == 1 and config.iecoil == 0
    assert (config.mgaus1, config.mgaus2, config.nsmp2, config.isize) == (8, 10, 1, 0)
    assert config.table_suffix == "129129"
    assert config.argv == ["129"]
    assert EFUNDConfig(nw=65, nh=33).argv == ["65", "33"]


@pytest.mark.parametrize(
    "bad",
    [
        {"nw": 0},
        {"nw": 10000},
        {"rleft": 0.0},
        {"rleft": 1.2, "rright": 0.05},
        {"zbotto": 1.5, "ztop": -1.5},
        {"ivesel": 2},
        {"igrid": True},
        {"mgaus1": 0},
        {"device": " "},
        {"timeout": 0},
        {"stack_size_kb": -1},
    ],
)
def test_config_refuses_out_of_contract_values(bad):
    with pytest.raises(ValueError):
        EFUNDConfig(**bad)


def test_config_hash_is_stable_and_scientific_only():
    base = EFUNDConfig()
    assert base.sha256 == EFUNDConfig().sha256
    assert base.sha256 == EFUNDConfig(workdir="/elsewhere", executable="/x/efund", timeout=5, stack_size_kb=1).sha256
    assert base.sha256 != EFUNDConfig(ivesel=0).sha256
    assert base.sha256 != EFUNDConfig(nw=65, nh=65).sha256
    assert base.sha256 != EFUNDConfig(mgaus2=12).sha256


# --- input -----------------------------------------------------------------


def test_namelist_carries_the_counts_and_the_geometry_in_efund_order(geometry):
    namelist = efund_namelist(geometry, EFUNDConfig())
    assert list(namelist) == ["machinein", "in5", "in3"]
    machine = namelist["machinein"]
    assert machine["device"] == "VEST"
    assert (machine["nfcoil"], machine["nfsum"], machine["nsilop"], machine["magpri"]) == (302, 16, 11, 64)
    assert (machine["necoil"], machine["nesum"], machine["nvesel"], machine["nvsum"], machine["nacoil"]) == (
        0,
        0,
        950,
        950,
        0,
    )
    in3 = namelist["in3"]
    assert in3["rf"] == list(map(float, geometry.fcoil_r))
    assert in3["fcid"] == list(map(int, geometry.fcoil_group))
    assert in3["turnfc"] == [1.0] * 16
    assert in3["vsid"] == list(range(1, 951))
    assert in3["amp2"] == [90.0] * 64
    assert all(len(name) <= 10 for name in in3["mpnam2"] + in3["lpname"] + in3["vsname"])


def test_write_mhdin_round_trips_through_f90nml_and_documents_itself(geometry, tmp_path):
    config = EFUNDConfig(workdir=tmp_path)
    path = write_mhdin(geometry, config, tmp_path / MHDIN_NAME, header=["extra note"])
    text = path.read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("!")]
    assert any("canonical VAFT static geometry" in line for line in header)
    assert any(LEGACY_ERA in line for line in header)
    assert any(config.sha256 in line for line in header)
    assert any("extra note" in line for line in header)
    parsed = f90nml.read(str(path))
    assert list(parsed) == ["machinein", "in5", "in3"]
    np.testing.assert_array_equal(parsed["in3"]["rvs"], geometry.vessel_r)
    np.testing.assert_array_equal(parsed["in3"]["zvs"], geometry.vessel_z)
    np.testing.assert_array_equal(parsed["in3"]["xmp2"], geometry.probe_r)
    np.testing.assert_array_equal(parsed["in3"]["rsi"], geometry.loop_r)
    np.testing.assert_array_equal(parsed["in3"]["fcturn"], geometry.fcoil_turns)
    assert parsed["in5"]["ivesel"] == 1
    # islpfc belongs to &in3; under &in5 this EFUND rejects the group and crashes.
    assert "islpfc" not in parsed["in5"] and parsed["in3"]["islpfc"] == 0
    assert parsed["in5"]["rleft"] == 0.05


def test_prepare_efund_inputs_writes_and_hashes_the_input(static, tmp_path):
    ods, manifest = static
    config = EFUNDConfig(workdir=tmp_path / "run")
    inputs = prepare_efund_inputs(ods, config, manifest=manifest)
    assert inputs.mhdin == tmp_path / "run" / MHDIN_NAME
    assert inputs.mhdin.is_file()
    assert len(inputs.mhdin_sha256) == 64
    assert inputs.counts["nvesel"] == 950
    assert inputs.geometry.machine["era"] == LEGACY_ERA
    again = prepare_efund_inputs(ods, config, manifest=manifest)
    assert again.mhdin_sha256 == inputs.mhdin_sha256


# --- outputs ---------------------------------------------------------------


def test_expected_sizes_follow_the_record_layout_the_checker_uses():
    config = EFUNDConfig(nw=33, nh=33, ivesel=1, iecoil=1)
    expected = expected_table_files(config, SMOKE_COUNTS)
    nsilop, magpri, nfsum, nesum, nvsum = 44, 76, 18, 6, 28
    nwnh = 33 * 33
    assert expected["rfcoil.ddd"] == (nsilop * nfsum + magpri * nfsum) * 8 + 16
    assert expected["brzgfc.dat"] == 2 * nwnh * nfsum * 8 + 16
    assert expected["ep3333.ddd"] == (nsilop + magpri) * nwnh * 8 + 16
    assert expected["ec3333.ddd"] == 2 * 4 + 8 + (33 * 2) * 8 + 8 + nwnh * nfsum * 8 + 8 + nwnh * 33 * 8 + 8
    assert expected["re3333.ddd"] == (nsilop * nesum + magpri * nesum + nwnh * nesum) * 8 + 24
    assert expected["rv3333.ddd"] == (
        nsilop * nvsum + magpri * nvsum + nwnh * nvsum + nfsum * nvsum + nesum * nvsum + nvsum * nvsum
    ) * 8 + 48
    assert expected["mhdout.dat"] is None
    assert "fc3333.ddd" not in expected


def test_flags_decide_which_files_are_expected():
    counts = SMOKE_COUNTS
    assert "rv6565.ddd" not in expected_table_files(EFUNDConfig(nw=65, nh=65, ivesel=0), counts)
    assert "re6565.ddd" not in expected_table_files(EFUNDConfig(nw=65, nh=65, iecoil=0), counts)
    assert "rfcoil.ddd" not in expected_table_files(EFUNDConfig(nw=65, nh=65, ifcoil=0), counts)
    assert "fc6565.ddd" in expected_table_files(EFUNDConfig(nw=65, nh=65, islpfc=1), counts)


def _write_records(path, arrays, order="<"):
    with path.open("wb") as stream:
        for array in arrays:
            payload = np.asarray(array, dtype=order + "f8").tobytes()
            stream.write(struct.pack(order + "i", len(payload)) + payload + struct.pack(order + "i", len(payload)))


def _synthetic_table(directory, config, counts):
    for name, size in expected_table_files(config, counts).items():
        if size is None:
            (directory / name).write_text("echo\n")
        else:
            (directory / name).write_bytes(b"\0" * size)


def test_collect_reports_every_missing_or_wrong_sized_file(tmp_path):
    config = EFUNDConfig(nw=5, nh=5)
    counts = {"nsilop": 2, "magpri": 3, "nfsum": 2, "nfcoil": 4, "nesum": 0, "nvsum": 3, "nvesel": 3}
    _synthetic_table(tmp_path, config, counts)
    good = collect_efund_outputs(tmp_path, config, counts, returncode=0)
    assert good.ok and good.problems == ()
    assert set(good.files) == set(expected_table_files(config, counts))

    (tmp_path / "rfcoil.ddd").write_bytes(b"\0" * 7)
    (tmp_path / "ep55.ddd").unlink()
    bad = collect_efund_outputs(tmp_path, config, counts, returncode=0)
    assert not bad.ok
    assert any(item.startswith("rfcoil.ddd is 7 bytes, expected") for item in bad.problems)
    assert "missing ep55.ddd" in bad.problems
    assert bad.reason == "; ".join(bad.problems)

    crashed = collect_efund_outputs(tmp_path, config, counts, returncode=139)
    assert crashed.problems[0] == "efund exited 139"


@pytest.mark.parametrize("order", ["<", ">"])
def test_read_fortran_records_detects_byte_order_and_refuses_torn_files(tmp_path, order):
    path = tmp_path / "t.ddd"
    _write_records(path, [np.arange(3.0), np.arange(5.0) * 2], order=order)
    assert fortran_byte_order(path) == order
    records = read_fortran_records(path)
    assert len(records) == 2
    arrays = read_fortran_arrays(path)
    np.testing.assert_array_equal(arrays[1], np.arange(5.0) * 2)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="runs past the end"):
        read_fortran_records(path)


def test_packaged_tables_are_big_endian_and_frame_as_their_counts_say():
    from vaft.data.resources import data_path

    path = data_path("efit/rfcoil.ddd")
    assert fortran_byte_order(path) == ">"
    arrays = read_fortran_arrays(path)
    assert [array.size for array in arrays] == [11 * 16, 64 * 16]


# --- running ---------------------------------------------------------------


def test_run_without_a_toolchain_is_skipped_with_directions(geometry, tmp_path):
    config = EFUNDConfig(workdir=tmp_path)
    inputs = EFUNDInputs(
        workdir=tmp_path, mhdin=tmp_path / MHDIN_NAME, mhdin_sha256="0" * 64, geometry=geometry, counts=geometry.counts()
    )
    result = run_efund(inputs, config)
    assert result.status == "skipped"
    assert "EFITHOME" in result.reason
    assert result.returncode is None
    assert "rv129129.ddd" in result.expected


def test_run_refuses_a_workdir_without_the_input(geometry, tmp_path, monkeypatch):
    fake = write_launchable_stub(tmp_path / "bin" / "efund")
    config = EFUNDConfig(workdir=tmp_path / "run", executable=str(fake))
    inputs = EFUNDInputs(
        workdir=tmp_path / "run", mhdin=tmp_path / "run" / MHDIN_NAME, mhdin_sha256="0" * 64, geometry=geometry, counts=geometry.counts()
    )
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match="prepare_efund_inputs"):
        run_efund(inputs, config)


def test_command_raises_the_stack_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(efund_module.compat, "IS_WINDOWS", False)
    executable = tmp_path / "efund"
    with_stack = efund_module._efund_command(EFUNDConfig(nw=65, nh=33, stack_size_kb=4096), executable)
    assert with_stack[:2] == ["bash", "-lc"]
    assert "ulimit -s 4096" in with_stack[2]
    assert with_stack[-3:] == [str(executable), "65", "33"]
    assert efund_module._efund_command(EFUNDConfig(stack_size_kb=None), executable) == [str(executable), "129"]
    hard = efund_module._efund_command(EFUNDConfig(), executable)
    assert "ulimit -s $(ulimit -Hs)" in hard[2] and "ulimit -s 3" not in hard[2]
    with pytest.raises(ValueError):
        EFUNDConfig(stack_size_kb="soft")


def test_windows_reserves_the_stack_at_link_time_not_through_bash(tmp_path, monkeypatch):
    """`ulimit` cannot raise a native Windows image's stack.

    The reserve is written into the PE header by the linker, so
    install_efit_windows.ps1 passes `-Wl,--stack` and nothing at run time can
    change it. Wrapping anyway would be worse than useless: it would make every
    EFUND run depend on an MSYS2 bash that the external-code installers
    deliberately keep off PATH, so the command would fail before reaching a
    setting that could not have worked.
    """
    monkeypatch.setattr(efund_module.compat, "IS_WINDOWS", True)
    executable = tmp_path / "efund.exe"

    for config in (
        EFUNDConfig(nw=65, nh=33, stack_size_kb=4096),
        EFUNDConfig(),  # the "hard" default
    ):
        command = efund_module._efund_command(config, executable)
        assert command[0] == str(executable)
        assert "bash" not in command
        assert not any("ulimit" in part for part in command)


# --- manifest --------------------------------------------------------------


def test_manifest_round_trip_records_machine_config_and_file_hashes(geometry, tmp_path):
    config = EFUNDConfig(workdir=tmp_path, nw=5, nh=5)
    counts = geometry.counts()
    _synthetic_table(tmp_path, config, counts)
    inputs = EFUNDInputs(
        workdir=tmp_path, mhdin=tmp_path / MHDIN_NAME, mhdin_sha256="a" * 64, geometry=geometry, counts=counts
    )
    result = collect_efund_outputs(tmp_path, config, counts, returncode=0)
    path = write_table_manifest(result, inputs, config, label="unit")
    assert path == tmp_path / TABLE_MANIFEST_NAME
    assert result.manifest == path
    manifest = read_table_manifest(tmp_path)
    assert manifest["code"] == "efund" and manifest["label"] == "unit"
    assert manifest["machine"]["era"] == LEGACY_ERA
    assert manifest["machine"]["counts"]["nvesel"] == 950
    assert manifest["efund"]["config_sha256"] == config.sha256
    assert manifest["efund"]["input"]["sha256"] == "a" * 64
    assert manifest["efund"]["executable"] is None
    assert set(manifest["table"]["files"]) == set(result.files)
    assert manifest["table"]["files"]["rv55.ddd"]["expected_size"] == result.expected["rv55.ddd"]
    assert len(manifest["table"]["identity"]) == 64
    assert [row["name"] for row in manifest["geometry_groups"]][:2] == ["PF1-1", "PF1-2"]
    json.dumps(manifest)


def test_manifest_refuses_an_incomplete_table(geometry, tmp_path):
    config = EFUNDConfig(workdir=tmp_path, nw=5, nh=5)
    inputs = EFUNDInputs(
        workdir=tmp_path, mhdin=tmp_path / MHDIN_NAME, mhdin_sha256="a" * 64, geometry=geometry, counts=geometry.counts()
    )
    result = collect_efund_outputs(tmp_path, config, geometry.counts(), returncode=0)
    with pytest.raises(ValueError, match="failed"):
        write_table_manifest(result, inputs, config)


def test_table_identity_distinguishes_manifested_from_legacy_directories(geometry, tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / MHDIN_NAME).write_text("&machinein\n/\n")
    record = table_identity(legacy)
    assert record["provenance"] == "unrecorded"
    assert record["identity"] is None and record["manifest"] is None
    assert len(record["mhdin_sha256"]) == 64
    assert table_identity(tmp_path / "absent")["mhdin_sha256"] is None

    generated = tmp_path / "generated"
    generated.mkdir()
    config = EFUNDConfig(workdir=generated, nw=5, nh=5)
    _synthetic_table(generated, config, geometry.counts())
    (generated / MHDIN_NAME).write_text("&machinein\n/\n")
    inputs = EFUNDInputs(
        workdir=generated, mhdin=generated / MHDIN_NAME, mhdin_sha256="b" * 64, geometry=geometry, counts=geometry.counts()
    )
    write_table_manifest(collect_efund_outputs(generated, config, geometry.counts(), returncode=0), inputs, config, label="x")
    record = table_identity(generated)
    assert record["provenance"] == "manifest" and record["label"] == "x"
    assert record["identity"] == read_table_manifest(generated)["table"]["identity"]


def test_packaged_table_directory_has_no_recorded_provenance():
    from vaft.data.resources import data_path

    record = table_identity(data_path("efit"))
    assert record["provenance"] == "unrecorded"
    assert record["mhdin_sha256"] is not None


# --- with the binary -------------------------------------------------------


@pytest.mark.integration
def test_efund_generates_a_complete_small_table_for_the_legacy_era(static, tmp_path, monkeypatch):
    home = _EFITHOME_AT_IMPORT
    if not home:
        pytest.skip("EFITHOME is not set")
    monkeypatch.setenv("EFITHOME", home)
    try:
        executable = resolve_role("efund")
    except FileNotFoundError as error:
        pytest.skip(str(error))
    if executable is None:
        pytest.skip("no efund under EFITHOME")
    ods, manifest = static
    config = EFUNDConfig(workdir=tmp_path, nw=33, nh=33, timeout=600)
    inputs = prepare_efund_inputs(ods, config, manifest=manifest)
    result = run_efund(inputs, config)
    assert result.ok, result.reason
    path = write_table_manifest(result, inputs, config, label="integration-33")
    manifest_out = read_table_manifest(tmp_path)
    assert manifest_out["efund"]["executable"]["sha256"]
    assert manifest_out["table"]["files"]["rv3333.ddd"]["size"] == result.expected["rv3333.ddd"]
    arrays = read_fortran_arrays(result.files["rfcoil.ddd"])
    assert [array.size for array in arrays] == [11 * 16, 64 * 16]
    assert np.all(np.isfinite(arrays[0])) and np.any(arrays[0] != 0.0)
    assert path.is_file()
