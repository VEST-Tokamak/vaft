"""Load the packaged sample ODS and check its basic structure."""

from omas import ODS

from vaft.data.resources import data_path


def test_load_sample_ods():
    sample = data_path("omas/39915.json")
    assert sample.is_file(), f"packaged sample missing: {sample}"

    ods = ODS()
    ods = ods.load(str(sample), consistency_check=False)

    assert len(ods.keys()) > 0
    assert "equilibrium" in ods
    assert len(ods["equilibrium"].keys()) > 0
