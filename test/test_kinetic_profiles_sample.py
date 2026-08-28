"""Offline checks for the canonical kinetic-profile ODS sample (shot 48224 @ 300 ms).

``vaft/data/kineticEfit/ods_48224_300ms.json`` is the stored result of the kinetic
chain that ``test_kinetic_chain_data.py`` runs against the raw diagnostics. It exists
so notebooks, examples, and tests can consume representative ``core_profiles`` content
without ``omfit_classes`` or the paired ``.mat`` inputs.
"""
import numpy as np
import pytest

pytest.importorskip("omas")

from vaft.data.resources import data_path

SAMPLE = data_path("kineticEfit/ods_48224_300ms.json")

pytestmark = pytest.mark.skipif(
    not SAMPLE.exists(),
    reason="vaft/data/kineticEfit/ods_48224_300ms.json not present (repository-only sample)",
)

CP = "core_profiles.profiles_1d.0"


@pytest.fixture(scope="module")
def sample_ods():
    from omas import ODS

    ods = ODS()
    ods.load(str(SAMPLE), consistency_check=False)
    return ods


def test_sample_carries_equilibrium_and_diagnostic_inputs(sample_ods):
    assert len(sample_ods["equilibrium.time_slice"]) == 1
    assert len(sample_ods["thomson_scattering.channel"]) > 0
    assert len(sample_ods["charge_exchange.channel"]) > 0
    assert sample_ods["dataset_description.data_entry.pulse"] == 48224


def test_sample_core_profiles_are_physical(sample_ods):
    rho = np.asarray(sample_ods[f"{CP}.grid.rho_tor_norm"], dtype=float)
    ne = np.asarray(sample_ods[f"{CP}.electrons.density"], dtype=float)
    te = np.asarray(sample_ods[f"{CP}.electrons.temperature"], dtype=float)
    ti = np.asarray(sample_ods[f"{CP}.ion.0.temperature"], dtype=float)
    vtor = np.asarray(sample_ods[f"{CP}.ion.0.velocity.toroidal"], dtype=float)
    pth = np.asarray(sample_ods[f"{CP}.pressure_thermal"], dtype=float)

    assert rho.size > 10 and np.all(np.isfinite(rho))
    assert rho[0] == pytest.approx(0.0) and rho[-1] == pytest.approx(1.0)
    assert {ne.size, te.size, ti.size, vtor.size, pth.size} == {rho.size}
    assert 1e18 <= ne[0] <= 1e20, ne[0]        # VEST ohmic density
    assert 20.0 <= te[0] <= 300.0, te[0]        # eV
    assert 2.0 <= ti[0] <= 60.0, ti[0]          # eV (CX C3+, well below Te here)
    assert np.all(np.isfinite(vtor))
    assert np.all(pth >= 0.0) and pth[0] > 0.0


def test_sample_time_slice_matches_300_ms(sample_ods):
    times = np.asarray(sample_ods["core_profiles.time"], dtype=float)
    assert times.size == 1
    assert times[0] == pytest.approx(0.300, abs=3e-3)
    assert sample_ods[f"{CP}.ion.0.label"] == "H+"
