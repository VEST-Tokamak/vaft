"""Per-era EFIT Green tables built by the routine pipeline (#805 follow-up).

The library half -- config reuse, the limiter file, the 100-character
``TABLE_DIR`` guard, and the build itself against a stand-in EFUND -- and the
pipeline half: which table a shot is given, and that a shot of the base
table's own era is given exactly what it was given before.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vaft.code.efit import efund as E
from vaft.data.resources import data_path

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / "workflow" / "automatic_pipeline_1_routine_data_processing"
LEGACY_ERA = "vest-pre-43017-pf1906"
PF2507_ERA = "vest-45968-plus-pf2507"


def _paths_module():
    spec = importlib.util.spec_from_file_location("_pipeline1_paths_for_era_tables", WORKFLOW / "paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def packaged():
    return Path(data_path("efit"))


@pytest.fixture()
def short_tmp():
    """A temporary directory short enough for EFIT's 100-character TABLE_DIR.

    pytest's own ``tmp_path`` is not, on macOS -- which the guard under test
    duly refuses.
    """
    import shutil
    import tempfile

    roots = [root for root in ("/tmp", tempfile.gettempdir()) if Path(root).is_dir()]
    directory = Path(tempfile.mkdtemp(prefix="et", dir=min(roots, key=len)))
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #
def test_the_base_tables_configuration_is_recovered_exactly(packaged):
    manifest = E.read_table_manifest(packaged)
    config = E.efund_config_from_manifest(manifest)
    assert config.sha256 == manifest["efund"]["config_sha256"]
    assert E.efund_config_from_manifest(manifest, nw=65, nh=65).table_suffix == "6565"


def test_a_manifest_without_a_configuration_is_refused():
    with pytest.raises(ValueError, match="no EFUND configuration"):
        E.efund_config_from_manifest({"efund": {}})


def test_the_limiter_file_is_the_packaged_one_byte_for_byte(packaged, tmp_path):
    """lim.dat must equal the static wall limiter (#965); writing it reproduces it."""
    from vaft.omas.vest_upstream import build_static_ods

    ods, _ = build_static_ods(LEGACY_ERA)
    written = E.write_limiter_file(ods, tmp_path / E.LIMITER_NAME)
    assert written.read_bytes() == (packaged / E.LIMITER_NAME).read_bytes()


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.105, "  0.105000E+00"),
        (-0.7279, " -0.727900E+00"),
        (1.185, "  0.118500E+01"),
        (0.0, "  0.000000E+00"),
        (0.0999999999, "  0.100000E+00"),
        (9.9999999, "  0.100000E+02"),
    ],
)
def test_fortran_e_format(value, text):
    assert E._fortran_e(value) == text


def test_table_dir_gets_its_separator_and_a_length_check():
    assert E.table_dir_text("/srv/tables/x") == "/srv/tables/x/"
    assert E.table_dir_text("/srv/tables/x/") == "/srv/tables/x/"
    with pytest.raises(ValueError, match="100 characters"):
        E.table_dir_text("/" + "a" * 100)


# The 5 x 5 grids below are too coarse for the machine-derived acceptance
# envelope, so they build without it; the 129 x 129 tests keep it.
def _stand_in_efund(monkeypatch, *, ok=True):
    """Write a correctly sized synthetic table instead of running EFUND."""
    calls = []

    def run(inputs, config):
        calls.append(Path(config.workdir))
        workdir = Path(config.workdir)
        if ok:
            for name, size in E.expected_table_files(config, inputs.counts).items():
                target = workdir / name
                if size is None:
                    target.write_text("echo\n")
                else:
                    target.write_bytes(b"\0" * size)
        return E.collect_efund_outputs(workdir, config, inputs.counts, returncode=0 if ok else 1)

    monkeypatch.setattr(E, "run_efund", run)
    return calls


def test_a_table_is_built_for_the_requested_era_and_renamed_into_place(monkeypatch, packaged, short_tmp):
    calls = _stand_in_efund(monkeypatch)
    output = short_tmp / "tables" / f"{PF2507_ERA}-55"
    config = E.EFUNDConfig(nw=5, nh=5)
    result = E.generate_era_table(PF2507_ERA, output, config=config, base_table_dir=packaged, acceptance_envelope=False)

    assert result.ok and result.workdir == output and result.manifest == output / E.TABLE_MANIFEST_NAME
    assert calls and calls[0] != output  # built beside it, not in place
    manifest = json.loads(result.manifest.read_text())
    assert manifest["machine"]["era"] == PF2507_ERA == E.table_machine_era(output)
    assert manifest["machine"]["pf_geometry"] == "2507"
    assert "seconds" in manifest["extra"]
    for name in (E.MHDIN_NAME, E.LIMITER_NAME, "dprobe.dat", *result.files):
        assert (output / name).is_file(), name
    assert all(Path(path).parent == output for path in result.files.values())
    assert [p.name for p in output.parent.iterdir()] == [output.name]  # no staging left

    with pytest.raises(FileExistsError, match="already holds a table"):
        E.generate_era_table(PF2507_ERA, output, config=config, base_table_dir=packaged, acceptance_envelope=False)


def test_a_failed_efund_run_leaves_no_table_behind(monkeypatch, packaged, short_tmp):
    _stand_in_efund(monkeypatch, ok=False)
    output = short_tmp / "t"
    result = E.generate_era_table(PF2507_ERA, output, config=E.EFUNDConfig(nw=5, nh=5), base_table_dir=packaged, acceptance_envelope=False)
    assert not result.ok
    assert not output.exists()


def test_an_incomplete_directory_is_replaced_only_by_its_owner(monkeypatch, packaged, short_tmp):
    _stand_in_efund(monkeypatch)
    output = short_tmp / "t"
    output.mkdir()
    (output / "ec55.ddd").write_bytes(b"torn")
    config = E.EFUNDConfig(nw=5, nh=5)
    with pytest.raises(FileExistsError, match="not empty"):
        E.generate_era_table(PF2507_ERA, output, config=config, base_table_dir=packaged, acceptance_envelope=False)
    result = E.generate_era_table(
        PF2507_ERA, output, config=config, base_table_dir=packaged, replace_incomplete=True,
        acceptance_envelope=False,
    )
    assert result.ok and (output / "ec55.ddd").stat().st_size != 4


def test_the_config_defaults_to_the_base_tables(monkeypatch, packaged, short_tmp):
    seen = []
    real = E.prepare_efund_inputs

    def spy(ods, config, **kwargs):
        seen.append(config)
        return real(ods, config, **kwargs)

    monkeypatch.setattr(E, "prepare_efund_inputs", spy)
    _stand_in_efund(monkeypatch, ok=False)  # the configuration is all this test needs
    E.generate_era_table(PF2507_ERA, short_tmp / "t", base_table_dir=packaged)
    (config,) = seen
    assert config.sha256 == E.read_table_manifest(packaged)["efund"]["config_sha256"]


def test_a_too_long_output_is_refused_before_efund_runs(monkeypatch, packaged, tmp_path):
    calls = _stand_in_efund(monkeypatch)
    with pytest.raises(ValueError, match="100 characters"):
        E.generate_era_table(PF2507_ERA, tmp_path / ("x" * 120), base_table_dir=packaged)
    assert calls == []


# --------------------------------------------------------------------------- #
# Pipeline: which table a shot gets
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def paths():
    module = _paths_module()
    return module, module.PipelinePaths("/srv/vest.filedb", "filedb", "magnetic", "chease")


def test_a_shot_of_the_base_era_keeps_its_configuration_verbatim(paths):
    """Anything else would change the rule's params or inputs and make
    Snakemake 7.32 rebuild every existing constraint product and all of EFIT."""
    module, p = paths
    base = "/opt/vaft/data/efit/"
    assert module.select_efit_table(
        LEGACY_ERA, base_table_dir=base, base_era=LEGACY_ERA, suffix="129129", paths=p
    ) == (base, [])


def test_a_shot_of_another_era_gets_its_generated_table_as_an_input(paths):
    module, p = paths
    table_dir, inputs = module.select_efit_table(
        PF2507_ERA, base_table_dir="/opt/vaft/data/efit/", base_era=LEGACY_ERA, suffix="129129", paths=p
    )
    assert table_dir == f"/srv/vest.filedb/pipeline/efit_tables/{PF2507_ERA}-129129/"
    assert inputs == [table_dir + E.TABLE_MANIFEST_NAME]
    assert E.table_dir_text(table_dir) == table_dir  # within EFIT's 100 characters


@pytest.mark.parametrize(
    "kwargs",
    [
        {"era_tables": False},
        {"base_era": None},
        {"suffix": None},
    ],
)
def test_without_era_tables_every_shot_gets_the_base_table(paths, kwargs):
    module, p = paths
    arguments = {"base_table_dir": "B/", "base_era": LEGACY_ERA, "suffix": "129129", "paths": p, **kwargs}
    assert module.select_efit_table(PF2507_ERA, **arguments) == ("B/", [])


def test_the_pipeline_script_builds_from_the_base_configuration(monkeypatch, packaged, short_tmp):
    import runpy
    import sys

    _stand_in_efund(monkeypatch)
    output = short_tmp / f"{PF2507_ERA}-129129"
    monkeypatch.setattr(sys, "argv", [
        "generate_efit_table.py", "--era", PF2507_ERA, "--base-table-dir", str(packaged),
        "--output-dir", str(output),
    ])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(WORKFLOW / "generate_efit_table.py"), run_name="__main__")
    assert exit_info.value.code == 0
    manifest = E.read_table_manifest(output)
    assert manifest["machine"]["era"] == PF2507_ERA
    assert manifest["efund"]["config_sha256"] == E.read_table_manifest(packaged)["efund"]["config_sha256"]
    assert manifest["extra"]["configuration_from"]["identity"] == E.table_identity(packaged)["identity"]
