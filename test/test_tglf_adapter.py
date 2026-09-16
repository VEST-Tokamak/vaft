"""The TGLF runner and its native output container (issue #553, increment 1).

The fixture under `test/data/gacode/tglf_reg05/` is GACODE's own `tglf05` regression
case -- the GA standard case with spectral-shift ExB shear -- run through the launcher
here, so these assert against output TGLF actually produced. Nothing needs GACODE
installed.

The claim under test is that a run's *outcome* is read from what it wrote, not from its
exit status. TGLF's launcher creates `out.tglf.run` before the solve starts, exactly as
NEO's does, so the presence of output proves nothing; `solved` is what answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.tglf import (
    GBFLUX_QUANTITIES,
    SCHEMA_VERSION,
    TGLFConfig,
    TGLFResult,
    TglfOutputs,
    collect_tglf_outputs,
    read_tglf_case,
)

FIXTURE = Path(__file__).parent / "data" / "gacode" / "tglf_reg05"


@pytest.fixture(scope="module")
def native():
    return collect_tglf_outputs(FIXTURE)


# --------------------------------------------------------------------------
# reading a finished run
# --------------------------------------------------------------------------


def test_the_fixture_is_a_real_solved_run(native):
    assert native is not None
    assert native.solved is True
    assert native.errors == ()
    assert native.version["revision"].startswith("6357db30")
    assert native.n_species == 2


def test_the_flat_gbflux_row_is_split_by_quantity_not_by_species(native):
    """`out.tglf.gbflux` is one row, quantity-major: every species' particle flux,
    then every species' energy flux, and so on.

    Splitting it species-major instead gives the same shapes and wrong physics, which
    is why it is checked against the per-species table TGLF prints in `out.tglf.run`.
    """
    assert set(native.gbflux) == set(GBFLUX_QUANTITIES)
    for values in native.gbflux.values():
        assert np.asarray(values).size == 2

    table = (FIXTURE / "out.tglf.run").read_text(encoding="utf-8").splitlines()
    electron_row = next(line for line in table if line.strip().startswith("elec"))
    printed = [float(value) for value in electron_row.split()[1:]]
    # The printed columns are Gam, Q, Q_low, Pi, S -- Q_low is not in gbflux.
    assert native.gbflux["particle"][0] == pytest.approx(printed[0], rel=1e-4)
    assert native.gbflux["energy"][0] == pytest.approx(printed[1], rel=1e-4)
    assert native.gbflux["momentum"][0] == pytest.approx(printed[3], rel=1e-4)
    assert native.gbflux["exchange"][0] == pytest.approx(printed[4], rel=1e-4)


def test_the_named_accessors_agree_with_the_mapping(native):
    np.testing.assert_array_equal(native.energy_flux, native.gbflux["energy"])
    np.testing.assert_array_equal(native.particle_flux, native.gbflux["particle"])


def test_a_directory_with_no_tglf_output_reads_as_nothing(tmp_path):
    assert collect_tglf_outputs(tmp_path) is None
    (tmp_path / "unrelated.txt").write_text("not TGLF", encoding="utf-8")
    assert collect_tglf_outputs(tmp_path) is None


def test_read_tglf_case_is_the_parse_only_path(native):
    assert read_tglf_case(FIXTURE).solved == native.solved


# --------------------------------------------------------------------------
# what `solved` refuses
# --------------------------------------------------------------------------


def test_a_logged_error_is_not_solved_however_the_files_look(tmp_path):
    """TGLF writes `out.tglf.run` before it solves, so files prove nothing."""
    for name in ("out.tglf.gbflux", "out.tglf.grid"):
        (tmp_path / name).write_text(
            (FIXTURE / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (tmp_path / "out.tglf.run").write_text(
        "[Parsing data in input.tglf]\nERROR: (TGLF) input is inconsistent\n",
        encoding="utf-8",
    )
    outputs = collect_tglf_outputs(tmp_path)
    assert outputs.errors
    assert outputs.solved is False


def test_a_run_with_no_fluxes_is_not_solved(tmp_path):
    (tmp_path / "out.tglf.run").write_text("[Parsing data]\n", encoding="utf-8")
    (tmp_path / "out.tglf.version").write_text("abc [2026]\nPLAT\ndate\n", encoding="utf-8")
    outputs = collect_tglf_outputs(tmp_path)
    assert outputs is not None
    assert outputs.gbflux is None
    assert outputs.solved is False
    assert "gbflux" in outputs.missing()


def test_non_finite_fluxes_are_not_solved(tmp_path):
    (tmp_path / "out.tglf.grid").write_text(" 2\n16\n", encoding="utf-8")
    (tmp_path / "out.tglf.gbflux").write_text(
        "1.0 2.0 NaN 4.0 5.0 6.0 7.0 8.0\n", encoding="utf-8"
    )
    outputs = collect_tglf_outputs(tmp_path)
    assert outputs.gbflux is not None
    assert outputs.solved is False


def test_a_row_that_does_not_divide_by_the_species_count_is_refused(tmp_path):
    """A truncated write would otherwise reshape into plausible nonsense."""
    (tmp_path / "out.tglf.grid").write_text(" 2\n16\n", encoding="utf-8")
    (tmp_path / "out.tglf.gbflux").write_text("1.0 2.0 3.0\n", encoding="utf-8")
    assert collect_tglf_outputs(tmp_path).gbflux is None


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def test_the_native_result_round_trips_through_json(native, tmp_path):
    path = native.write_json(tmp_path / "tglf.json")
    restored = TglfOutputs.read_json(path)
    assert restored.solved == native.solved
    assert restored.version == native.version
    for name in GBFLUX_QUANTITIES:
        np.testing.assert_allclose(restored.gbflux[name], native.gbflux[name])


def test_a_payload_from_a_newer_schema_is_refused_not_half_read(native):
    payload = native.to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema version"):
        TglfOutputs.from_dict(payload)


# --------------------------------------------------------------------------
# the result wrapper
# --------------------------------------------------------------------------


def test_ok_requires_a_solve_not_just_a_zero_exit(native, tmp_path):
    """The whole reason NEOResult overrides `ok`, and it applies here too."""
    solved = TGLFResult(returncode=0, workdir=tmp_path, outputs_native=native)
    assert solved.ok is True

    assert TGLFResult(returncode=0, workdir=tmp_path, outputs_native=None).ok is False
    assert TGLFResult(returncode=1, workdir=tmp_path, outputs_native=native).ok is False

    broken = TglfOutputs(directory=str(tmp_path), errors=("ERROR: (TGLF) bad input",))
    assert TGLFResult(returncode=0, workdir=tmp_path, outputs_native=broken).ok is False


# --------------------------------------------------------------------------
# the launcher contract
# --------------------------------------------------------------------------


def test_the_runner_passes_no_nomp_flag(monkeypatch, tmp_path):
    """`tglf/bin/tglf` parses -p -e -n -g -i -r -rc -reset -c -h and nothing else.

    NEO's runner passes `-nomp`, and copying it here would make TGLF exit 1 with
    "incorrect tglf syntax" before it ever read the input. Asserted on the argument
    list because the failure is otherwise indistinguishable from a bad case.
    """
    from vaft.code.gacode.tglf import runner as tglf_runner

    seen = {}

    def fake_run(executable, arguments, *, cwd, log_path, config=None, code="neo"):
        seen["arguments"] = list(arguments)
        seen["code"] = code
        seen["cwd"] = Path(cwd)
        Path(log_path).write_text("no mpi\n", encoding="utf-8")
        return 0, Path(log_path)

    monkeypatch.setattr(tglf_runner, "require_gacode_executable", lambda c, n: Path("/x/tglf"))
    monkeypatch.setattr(tglf_runner, "gacode_platform", lambda c: "TEST_PLATFORM")
    monkeypatch.setattr(tglf_runner, "run_gacode", fake_run)

    case = tmp_path / "case"
    case.mkdir()
    for name in ("out.tglf.gbflux", "out.tglf.grid", "out.tglf.version"):
        (case / name).write_text(
            (FIXTURE / name).read_text(encoding="utf-8"), encoding="utf-8"
        )

    from vaft.code.gacode.tglf.inputs import TGLFInputs

    staged = TGLFInputs(workdir=case, parameters={"NS": 2})
    result = tglf_runner.run_tglf(staged, TGLFConfig(n_mpi=1, n_omp=4), check=False)

    assert seen["arguments"] == ["-e", "case", "-n", "1"]
    assert "-nomp" not in seen["arguments"]
    assert seen["code"] == "tglf"
    assert seen["cwd"] == tmp_path
    assert result.provenance["platform"] == "TEST_PLATFORM"


def test_stale_output_is_deleted_before_a_rerun(monkeypatch, tmp_path):
    """Parsing is by filename, so a failed rerun would return the previous physics."""
    from vaft.code.gacode.tglf import runner as tglf_runner
    from vaft.code.gacode.tglf.inputs import TGLFInputs

    case = tmp_path / "case"
    case.mkdir()
    (case / "out.tglf.gbflux").write_text("9 9 9 9 9 9 9 9\n", encoding="utf-8")
    (case / "out.tglf.grid").write_text(" 2\n16\n", encoding="utf-8")

    def fake_run(executable, arguments, *, cwd, log_path, config=None, code="neo"):
        Path(log_path).write_text("ran\n", encoding="utf-8")
        return 0, Path(log_path)

    monkeypatch.setattr(tglf_runner, "require_gacode_executable", lambda c, n: Path("/x/tglf"))
    monkeypatch.setattr(tglf_runner, "gacode_platform", lambda c: "TEST")
    monkeypatch.setattr(tglf_runner, "run_gacode", fake_run)

    result = tglf_runner.run_tglf(TGLFInputs(workdir=case), TGLFConfig(), check=False)
    assert result.outputs_native is None, "the stale fluxes must not survive the rerun"


def test_a_failure_says_which_kind_it_was(monkeypatch, tmp_path):
    from vaft.code.gacode.tglf import runner as tglf_runner
    from vaft.code.gacode.tglf.inputs import TGLFInputs

    case = tmp_path / "case"
    case.mkdir()

    def fake_run(executable, arguments, *, cwd, log_path, config=None, code="neo"):
        Path(log_path).write_text("line one\nline two\n", encoding="utf-8")
        return 0, Path(log_path)

    monkeypatch.setattr(tglf_runner, "require_gacode_executable", lambda c, n: Path("/x/tglf"))
    monkeypatch.setattr(tglf_runner, "gacode_platform", lambda c: "TEST")
    monkeypatch.setattr(tglf_runner, "run_gacode", fake_run)

    with pytest.raises(tglf_runner.TGLFExecutionError, match="wrote no out.tglf"):
        tglf_runner.run_tglf(TGLFInputs(workdir=case), TGLFConfig(), check=True)


def test_the_package_imports_without_gacode_installed(monkeypatch):
    """`import vaft.code.gacode.tglf` must not need an installation."""
    for name in ("GACODEHOME", "GACODE_ROOT", "GACODE_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    import importlib

    module = importlib.import_module("vaft.code.gacode.tglf")
    assert module.TGLFConfig().home is None
