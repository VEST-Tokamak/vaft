"""DCON's two edge treatments, read back from real GPEC output (#792).

The fixtures under ``test/data/gpec/dcon_edge_792/`` are trimmed copies of two
real DCON runs. Equilibrium: shot 39915 at 319 ms, a "good" slice of the #1331
``statistical_891`` campaign, CHEASE-refined by the pipeline's
``run_chease_refinement.py``. Run: GPEC e68d7ac2 on vestserver, n=1, packaged
templates, with ``bal_flag=t``.

* ``full_edge/``: ``psiedge=1.0``, the packaged default (``dcon-peeling``).
* ``peak_dw_truncated/``: ``psiedge=0.95``. DCON scans ``dW_edge``, moves
  ``psilim`` to the peak and integrates again (``dcon-kink``).

Each directory holds only ``dcon.in`` and ``dcon_output_n1.nc``. The netCDF keeps
the eigenvalues, the edge scan and the 1-D profiles; the eigenvector and
``W_t`` matrices are dropped with ``nccopy -V``. That is enough for
:func:`read_dcon_output` and the edge classifier to run unchanged on what
GPEC actually wrote. The synthetic fixtures in ``gpec_nc_fixtures.py`` cannot
pin the one thing that matters here: where real DCON puts the truncated
boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gpec import DconOutput, read_dcon_output

DATA = Path(__file__).resolve().parent / "data" / "gpec" / "dcon_edge_792"
PSIHIGH = 0.994  # packaged equil.in


@pytest.fixture(scope="module")
def full() -> DconOutput:
    return read_dcon_output(DATA / "full_edge", mode=1)


@pytest.fixture(scope="module")
def truncated() -> DconOutput:
    return read_dcon_output(DATA / "peak_dw_truncated", mode=1)


def test_full_edge_is_classified_from_the_absent_edge_scan(full):
    assert full.evaluation.psiedge == 1.0
    assert full.edge_scan is None
    assert full.edge_treatment == DconOutput.FULL_EDGE
    assert full.psilim == pytest.approx(PSIHIGH)


def test_truncated_edge_sits_at_the_peak_of_the_real_edge_scan(truncated):
    assert truncated.evaluation.psiedge == 0.95
    assert truncated.edge_treatment == DconOutput.PEAK_DW_TRUNCATED
    scan = truncated.edge_scan
    peak = int(np.argmax(np.real(scan.dW)))
    # dcon.F:262-279 -- the file's psilim/qlim are the post-truncation values.
    assert truncated.psilim == pytest.approx(float(scan.psi_n[peak]), abs=1e-12)
    assert truncated.qlim == pytest.approx(float(scan.q[peak]), abs=1e-9)
    assert scan.psi_n[0] >= 0.95 and truncated.psilim < PSIHIGH


def test_the_two_treatments_give_different_least_stable_energies(full, truncated):
    # Reference values from GPEC e68d7ac2. Both runs are free-boundary stable
    # here; truncating at the dW peak raises W_t. Pinning both is what catches
    # a reader or classifier change that swaps or merges the two products.
    assert full.total1.real == pytest.approx(2.6923316420322827, rel=1e-9)
    assert truncated.total1.real == pytest.approx(6.497093004467069, rel=1e-9)
    assert full.stable_free_boundary and truncated.stable_free_boundary
    assert full.qlim == pytest.approx(11.687849635345607, rel=1e-9)
    assert truncated.qlim == pytest.approx(11.278671216358221, rel=1e-9)


def test_the_edge_scan_passes_through_negative_dw_inside_the_truncation_window(truncated):
    # The full and peak-truncated solutions are both stable, but some
    # truncations in [psiedge, psihigh] are not. That is why the atlas reports the
    # treatment as part of a result's identity rather than as one "stability"
    # verdict.
    dw = np.real(truncated.edge_scan.dW)
    assert dw.min() < 0 < dw.max()


def test_ballooning_was_evaluated_on_every_surface(full, truncated):
    for output in (full, truncated):
        assert output.evaluation.bal_flag is True
        assert bool(np.all(output.ca1_evaluated))
