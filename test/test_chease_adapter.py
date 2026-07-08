import os
import shutil

import numpy as np
import pytest


def test_chease_import_smoke_without_external_runner():
    import vaft.code.chease as chease

    assert chease.CHEASEConfig is not None
    assert chease.prepare_chease_inputs is not None


def test_chease_sign_forcing_roundtrip():
    from vaft.code import chease
    from vaft.data.resources import sample_geqdsk

    original = sample_geqdsk("g039915.00319")
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

    source_path = data_path("g039915.00319")
    for idx, source in enumerate(
        [
            source_path,
            sample_geqdsk("g039915.00319"),
            sample_geqdsk("g039915.00319").to_omas(),
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

    source = sample_geqdsk("g039915.00319")
    inputs = prepare_chease_inputs(source, CHEASEConfig(workdir=tmp_path, create_plot=False))
    refined = tmp_path / "input_chease.geqdsk"
    shutil.copy2(inputs.input_geqdsk, refined)

    result = collect_chease_outputs(tmp_path, CHEASEConfig(workdir=tmp_path, create_plot=False))

    assert result.refined_geqdsk == refined
    assert result.refined_ods is not None
    assert result.comparison["boundary_points"] > 0


def test_run_chease_skips_without_executable(tmp_path):
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs, run_chease
    from vaft.data.resources import sample_geqdsk

    inputs = prepare_chease_inputs(sample_geqdsk("g039915.00319"), CHEASEConfig(workdir=tmp_path))
    with pytest.raises(FileNotFoundError):
        run_chease(inputs, CHEASEConfig(workdir=tmp_path, executable="/definitely/missing/chease"))


@pytest.mark.skipif(
    not os.environ.get("CHEASE_EXEC_DIR"),
    reason="CHEASE integration test requires CHEASE_EXEC_DIR",
)
def test_run_chease_integration_when_available(tmp_path):
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs, run_chease
    from vaft.data.resources import sample_geqdsk

    config = CHEASEConfig(workdir=tmp_path, create_plot=False, timeout=60)
    inputs = prepare_chease_inputs(sample_geqdsk("g039915.00319"), config)
    result = run_chease(inputs, config)

    assert result.ok
    assert result.refined_geqdsk is not None
    assert result.refined_geqdsk.exists()
    assert result.refined_ods is not None
