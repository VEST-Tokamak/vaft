import copy
import os
import shutil

import numpy as np
import pytest


def _assert_flat_ods_equal(a, b) -> None:
    flat_a, flat_b = a.flat(), b.flat()
    assert set(flat_a) == set(flat_b)
    for key, value in flat_a.items():
        np.testing.assert_array_equal(value, flat_b[key])


def test_chease_import_smoke_without_external_runner():
    import vaft.code.chease as chease

    assert chease.CHEASEConfig is not None
    assert chease.prepare_chease_inputs is not None


def test_chease_sign_forcing_roundtrip():
    from vaft.code import chease
    from vaft.data.resources import sample_geqdsk

    original = sample_geqdsk("efit/g039915.00319")
    original_info = chease._geqdsk_sign_info(original)

    flipped, _, flipped_info, _ = chease._force_geqdsk_signs(
        original,
        desired_dpsi_sign=-original_info.dpsi_sign,
        desired_bcentr_sign=-original_info.bcentr_sign,
        desired_current_sign=-original_info.current_sign,
        desired_fpol_sign=-original_info.fpol_sign,
        desired_q_sign=-original_info.q_sign,
    )
    assert flipped_info.dpsi_sign == -original_info.dpsi_sign
    assert flipped_info.bcentr_sign == -original_info.bcentr_sign
    assert flipped_info.current_sign == -original_info.current_sign
    assert flipped_info.fpol_sign == -original_info.fpol_sign
    assert flipped_info.q_sign == -original_info.q_sign

    restored, _, restored_info, _ = chease._force_geqdsk_signs(
        flipped,
        desired_dpsi_sign=original_info.dpsi_sign,
        desired_bcentr_sign=original_info.bcentr_sign,
        desired_current_sign=original_info.current_sign,
        desired_fpol_sign=original_info.fpol_sign,
        desired_q_sign=original_info.q_sign,
    )
    assert restored_info.as_dict() == original_info.as_dict()
    assert np.asarray(restored["PSIRZ"]).shape == np.asarray(original["PSIRZ"]).shape


def test_prepare_chease_inputs_from_path_geqdsk_and_ods(tmp_path):
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs
    from vaft.data.resources import data_path, sample_geqdsk

    source_path = data_path("efit/g039915.00319")
    for idx, source in enumerate(
        [
            source_path,
            sample_geqdsk("efit/g039915.00319"),
            sample_geqdsk("efit/g039915.00319").to_omas(),
        ]
    ):
        workdir = tmp_path / f"case_{idx}"
        inputs = prepare_chease_inputs(source, CHEASEConfig(workdir=workdir))
        assert inputs.input_geqdsk and inputs.input_geqdsk.exists()
        assert inputs.expeq and inputs.expeq.exists()
        assert inputs.namelist and inputs.namelist.exists()
        assert (workdir / "chease_cocos_transform.json").exists()
        assert "COCOS_IN = 2" in inputs.namelist.read_text()
        assert inputs.expeq.read_text().splitlines()[0]


def test_collect_chease_outputs_parses_refined_copy(tmp_path):
    from vaft.code.chease import CHEASEConfig, collect_chease_outputs, prepare_chease_inputs
    from vaft.data.resources import sample_geqdsk

    source = sample_geqdsk("efit/g039915.00319")
    inputs = prepare_chease_inputs(source, CHEASEConfig(workdir=tmp_path, create_plot=False))
    refined = tmp_path / "input_chease.geqdsk"
    shutil.copy2(inputs.input_geqdsk, refined)

    result = collect_chease_outputs(tmp_path, CHEASEConfig(workdir=tmp_path, create_plot=False))

    assert result.refined_geqdsk == refined
    assert result.refined_ods is not None
    assert result.comparison["boundary_points"] > 0


def test_collect_chease_outputs_matches_vest_time_suffixed_refined_names(tmp_path):
    """VEST gfiles have no .geqdsk/.gfile/.g suffix -- `g<shot>.<time>`.

    `_refined_output_path()` names the restored file `<stem>_chease<suffix>`,
    where `<suffix>` is whatever Path.suffix finds after the last dot in the
    input name -- for a VEST gfile that is the numeric time, giving e.g.
    `g039915_chease.00319`. A glob restricted to `*_chease.geqdsk` (etc.)
    never matches that name and silently falls back to the raw, pre-restore
    CHEASE output (`EQDSK_COCOS_02.OUT`), discarding run_chease()'s
    boundary/limiter restoration entirely.

    Also plants an AppleDouble-style dotfile sidecar (`._g039915_chease.00319`,
    the kind macOS/exFAT/network volumes create automatically) to confirm it
    is never selected instead of the real refined file.
    """
    from vaft.code.chease import CHEASEConfig, collect_chease_outputs, prepare_chease_inputs
    from vaft.data.resources import sample_geqdsk

    source = sample_geqdsk("efit/g039915.00319")
    inputs = prepare_chease_inputs(source, CHEASEConfig(workdir=tmp_path, create_plot=False))
    refined = tmp_path / "g039915_chease.00319"
    shutil.copy2(inputs.input_geqdsk, refined)
    (tmp_path / ".g039915_chease.00319.swp").write_bytes(b"not a geqdsk")
    (tmp_path / "._g039915_chease.00319").write_bytes(b"AppleDouble resource fork, not a geqdsk")

    result = collect_chease_outputs(tmp_path, CHEASEConfig(workdir=tmp_path, create_plot=False))

    assert result.refined_geqdsk == refined
    assert result.refined_ods is not None
    assert result.comparison["boundary_points"] > 0


def test_run_chease_skips_without_executable(tmp_path):
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs, run_chease
    from vaft.data.resources import sample_geqdsk

    inputs = prepare_chease_inputs(sample_geqdsk("efit/g039915.00319"), CHEASEConfig(workdir=tmp_path))
    with pytest.raises(FileNotFoundError):
        run_chease(inputs, CHEASEConfig(workdir=tmp_path, executable="/definitely/missing/chease"))


@pytest.mark.skipif(
    not (os.environ.get("CHEASEHOME") or os.environ.get("CHEASE_EXEC_DIR")),
    reason="CHEASE integration test requires CHEASEHOME or CHEASE_EXEC_DIR",
)
def test_run_chease_integration_when_available(tmp_path):
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs, run_chease
    from vaft.data.resources import sample_geqdsk

    config = CHEASEConfig(workdir=tmp_path, create_plot=False, timeout=60)
    inputs = prepare_chease_inputs(sample_geqdsk("efit/g039915.00319"), config)
    result = run_chease(inputs, config)

    assert result.ok
    assert result.refined_geqdsk is not None
    assert result.refined_geqdsk.exists()
    assert result.refined_ods is not None


@pytest.mark.skipif(
    not (os.environ.get("CHEASEHOME") or os.environ.get("CHEASE_EXEC_DIR") or os.environ.get("CHEASE")),
    reason="CHEASE integration test requires CHEASEHOME, CHEASE_EXEC_DIR, or CHEASE",
)
def test_run_chease_preserves_source_limiter_from_a_file_path(tmp_path):
    """End-to-end regression for the VEST-naming refined-file selection bug.

    `sample_geqdsk()` returns an in-memory GEQDSK, which
    `prepare_chease_inputs` names `input.geqdsk` -- a name the old glob
    already matched, so no in-memory-source test could have caught this.
    The real pipeline always passes a file path (a VEST gfile named
    `g<shot>.<time>`, no .geqdsk/.gfile/.g suffix), which is what actually
    triggers `_refined_output_path()`'s time-suffixed name. Pass a real path
    here, and confirm the restored file -- not the raw pre-restore CHEASE
    output -- is what collect_chease_outputs() actually returns, with the
    limiter round-tripped from the source unchanged.
    """
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs, run_chease
    from vaft.data.resources import data_path

    source_path = data_path("efit/g039915.00319")
    # Matches the settings run_chease_refinement.py's own defaults use and
    # this repository's real shot-39915 validation confirmed converges;
    # CHEASEConfig's bare defaults (nideal=11) are untuned for VEST and do
    # not converge here, which is a separate, pre-existing gap, not this fix.
    config = CHEASEConfig(
        workdir=tmp_path,
        create_plot=False,
        timeout=60,
        target_psin=0.993,
        relax=0.5,
        nideal=6,
        nw=513,
        preserve_boundary_limiter=True,
    )
    inputs = prepare_chease_inputs(source_path, config)
    assert inputs.input_geqdsk.name == "g039915.00319"

    result = run_chease(inputs, config)

    assert result.ok
    assert result.refined_geqdsk is not None
    assert result.refined_geqdsk.name == "g039915_chease.00319"
    assert result.refined_ods is not None

    source_rlim = np.asarray(inputs.geqdsk["RLIM"], dtype=float)
    source_zlim = np.asarray(inputs.geqdsk["ZLIM"], dtype=float)
    refined_rlim = np.asarray(result.refined_ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float)
    refined_zlim = np.asarray(result.refined_ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float)
    assert np.array_equal(source_rlim, refined_rlim)
    assert np.array_equal(source_zlim, refined_zlim)


@pytest.mark.skipif(
    not (os.environ.get("CHEASEHOME") or os.environ.get("CHEASE_EXEC_DIR") or os.environ.get("CHEASE")),
    reason="CHEASE integration test requires CHEASEHOME, CHEASE_EXEC_DIR, or CHEASE",
)
def test_run_chease_gfile_and_equivalent_ods_input_agree(tmp_path):
    """A g-file and the ODS built from that same g-file must both refine,
    reach the same equilibrium, and each preserve its own input's
    limiter/wall unchanged.

    This is an end-to-end regression for the from_omas() PSIRZ transpose
    bug (see test_eqdsk_omas_roundtrip.py): before that fix, this ODS input
    failed deep inside CHEASE's spline setup ("xin not in ascending order")
    on every run, an intrinsic-looking failure that was actually caused by
    VAFT feeding CHEASE a self-inconsistent equilibrium (RMAXIS/boundary
    computed from correctly-oriented PSIRZ, but the flux map itself
    transposed under it).
    """
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs, run_chease
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path

    source_path = data_path("efit/g039915.00319")
    source_ods = read_geqdsk(source_path).to_omas()

    def _run(source, workdir):
        config = CHEASEConfig(
            workdir=workdir,
            create_plot=False,
            timeout=60,
            target_psin=0.993,
            relax=0.5,
            nideal=6,
            nw=513,
            preserve_boundary_limiter=True,
        )
        inputs = prepare_chease_inputs(source, config)
        return inputs, run_chease(inputs, config)

    inputs_gfile, result_gfile = _run(source_path, tmp_path / "from_gfile")
    inputs_ods, result_ods = _run(source_ods, tmp_path / "from_ods")

    assert result_gfile.ok, result_gfile.stderr
    assert result_ods.ok, result_ods.stderr

    refined_gfile = read_geqdsk(result_gfile.refined_geqdsk)
    refined_ods_geqdsk = read_geqdsk(result_ods.refined_geqdsk)
    for key in ("RMAXIS", "ZMAXIS", "SIMAG", "SIBRY", "CURRENT", "BCENTR"):
        np.testing.assert_allclose(
            float(refined_gfile[key]), float(refined_ods_geqdsk[key]), err_msg=key
        )

    # Each path preserves its own input's limiter/wall, unchanged.
    np.testing.assert_array_equal(
        np.asarray(refined_gfile["RLIM"], dtype=float), np.asarray(inputs_gfile.geqdsk["RLIM"], dtype=float)
    )
    np.testing.assert_array_equal(
        np.asarray(result_ods.refined_ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float),
        np.asarray(source_ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float),
    )
    np.testing.assert_array_equal(
        np.asarray(result_ods.refined_ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float),
        np.asarray(source_ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float),
    )


def test_prepare_chease_inputs_keeps_the_original_ods_as_source(tmp_path):
    """`inputs.source` must stay the caller's actual ODS object (not a copy
    or a re-derived GEQDSK), since `_preserve_source_wall()` keys off it.
    """
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path

    source_ods = read_geqdsk(data_path("efit/g039915.00319")).to_omas()
    inputs = prepare_chease_inputs(source_ods, CHEASEConfig(workdir=tmp_path, create_plot=False))

    assert inputs.source is source_ods


def test_preserve_source_wall_copies_an_ods_sources_wall_onto_the_result():
    """CHEASE refines the equilibrium only -- an ODS input's `wall` IDS must
    come back out identical, not merely numerically close.

    Unit-tested directly against `_preserve_source_wall()` rather than a
    full `run_chease()` call: the ODS-derived EXPEQ this repository's own
    sample equilibrium produces does not currently converge in CHEASE's
    solver (a separate, pre-existing numerical gap in the ODS-to-GEQDSK
    round trip, unrelated to wall/limiter handling), so gating this
    assertion on a real CHEASE run would make the test depend on unrelated,
    unresolved convergence behavior instead of the invariant it checks.
    Routing the wall through the refined GEQDSK's own RLIM/ZLIM (as the
    g-file case does, and as this used to do before this fix) would also
    round-trip through EQDSK's E14.6 fixed-format text serialization, which
    only carries 6 significant digits -- this copies the *original* input
    ODS's `wall` IDS directly instead, so it is provably unchanged rather
    than a lossy reconstruction.
    """
    from omas import ODS
    from vaft.code.chease import CHEASEResult, _preserve_source_wall
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path

    source_ods = read_geqdsk(data_path("efit/g039915.00319")).to_omas()
    original_wall = copy.deepcopy(source_ods["wall"])

    refined_ods = ODS()
    refined_ods["wall.description_2d.0.limiter.unit.0.outline.r"] = [0.0, 1.0]
    refined_ods["wall.description_2d.0.limiter.unit.0.outline.z"] = [0.0, 1.0]
    result = CHEASEResult(returncode=0, workdir=None, refined_ods=refined_ods)

    _preserve_source_wall(result, source_ods)

    _assert_flat_ods_equal(result.refined_ods["wall"], original_wall)
    # The copy must be independent of the caller's object.
    result.refined_ods["wall.description_2d.0.limiter.unit.0.outline.r"] = [9.0]
    _assert_flat_ods_equal(source_ods["wall"], original_wall)


def test_preserve_source_wall_is_a_noop_for_a_geqdsk_path_source():
    """A g-file source has no ODS `wall` to copy -- must leave the
    to_omas()-reconstructed wall (already exact, per
    test_run_chease_preserves_source_limiter_from_a_file_path) untouched.
    """
    from vaft.code.chease import CHEASEResult, _preserve_source_wall
    from vaft.data.resources import sample_geqdsk

    reconstructed = object()
    result = CHEASEResult(returncode=0, workdir=None, refined_ods=reconstructed)

    _preserve_source_wall(result, sample_geqdsk("efit/g039915.00319"))

    assert result.refined_ods is reconstructed
