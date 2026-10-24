"""The inferred-T_i stage against the #1839 archive (40326, 322 ms), where the notebook did it by hand.

The archive converter rebuilds the stored kinetic state with ``T_i`` inferred
point by point.  Here that ``T_i`` is taken away, the stage is run on what is
left (composed profiles, Thomson channels, the magnetic equilibrium), and the
two are compared.  The values must agree wherever both are valid; the stage may
void more points, because it carries a different ``sigma(p_EFIT)`` model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.core_profiles import classify_ti_record
from vaft.omas import vest_upstream as vu

WORKFLOW = Path(__file__).resolve().parents[1] / "workflow" / "kinetic_state"
ARCHIVE = WORKFLOW / "archive_data" / "vest_40326_322ms.json"

pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="the #1839 archive is repository-only")


@pytest.fixture(scope="module")
def archive_module():
    sys.path.insert(0, str(WORKFLOW))
    try:
        import archive_to_ods

        yield archive_to_ods
    finally:
        sys.path.remove(str(WORKFLOW))


def test_the_stage_reproduces_the_notebook_temperature_where_both_are_valid(tmp_path, archive_module):
    ods = archive_module.kinetic_state_ods(40326, "magnetics")
    base = "core_profiles.profiles_1d.0"
    reference = np.asarray(ods[f"{base}.ion.0.temperature"], dtype=float).copy()
    for k in range(len(ods[f"{base}.ion"])):
        for leaf in ("temperature", "temperature_fit"):
            if f"{base}.ion.{k}.{leaf}" in ods:
                del ods[f"{base}.ion.{k}.{leaf}"]
    for leaf in ("t_i_average", "t_i_average_fit", "pressure_ion_total"):
        if f"{base}.{leaf}" in ods:
            del ods[f"{base}.{leaf}"]
    composed = tmp_path / "composition.json"
    ods.save(str(composed))
    equilibrium = ODS(consistency_check=False)
    equilibrium["equilibrium"] = ods["equilibrium"]
    directory = tmp_path / "omas" / "efit" / "magnetic" / "40326" / "output"
    directory.mkdir(parents=True)
    efit = directory / "efit.json"
    equilibrium.save(str(efit))

    out, manifest = vu.build_ti_inferred_ods(shot=40326, core_profiles_product=composed, efit_product=efit)

    assert manifest["status"] == "success", manifest.get("error")
    inferred = np.asarray(out[f"{base}.ion.0.temperature"], dtype=float)
    both = np.isfinite(inferred) & np.isfinite(reference)
    assert both.sum() > 90
    assert np.allclose(inferred[both], reference[both], rtol=1e-6)
    assert np.isfinite(inferred).sum() <= np.isfinite(reference).sum(), "the stage is never less strict"
    assert classify_ti_record(out[f"{base}.ion.0.temperature_fit.parameters"]) == "inferred"
