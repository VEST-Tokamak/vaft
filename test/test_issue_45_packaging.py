from pathlib import Path

import pytest

from vaft.machine_mapping import utils


def test_license_file_exists_for_packaging_metadata():
    assert (Path(__file__).resolve().parents[1] / "LICENSE").exists()


def test_vendor_omas_uv_source_removed():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'vendor/omas' not in pyproject


def test_machine_mapping_requires_explicit_info_file_without_vest_yaml_default():
    with pytest.raises(ValueError, match="info_file"):
        utils._resolve_info_file_path(None)


def test_efit_analysis_uses_supported_matplotlib_style_name():
    script = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "automatic_pipeline_3_data_summary"
        / "efit_analysis.py"
    ).read_text(encoding="utf-8")
    assert "plt.style.use('seaborn-v0_8')" in script
    assert "plt.style.use('seaborn')" not in script
