"""Smoke-test plot submodule functions against the packaged sample ODS."""

import matplotlib

matplotlib.use("Agg")

from omas import ODS

import vaft
from vaft.data.resources import data_path


def test_time_pf_active_current_plots():
    ods = ODS()
    ods = ods.load(str(data_path("omas/39915.json")), consistency_check=False)

    vaft.plot.time_pf_active_current(ods)
