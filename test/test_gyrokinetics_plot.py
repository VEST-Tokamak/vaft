"""vaft.plot.gyrokinetics draws on synthetic CGYRO-shaped data (#1354 stage 4)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.code.gacode.cgyro import collect_cgyro_outputs
from vaft.plot import gyrokinetics as gkplot

from test_cgyro_adapter import write_run


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_linear_spectrum_from_runs_and_overlay(tmp_path):
    runs = [
        collect_cgyro_outputs(write_run(tmp_path / f"ky{i}", gamma=0.1 * (i + 1),
                                        exit_message=("Linear converged" if i else
                                                      "Linear terminated at max time")))
        for i in range(3)
    ]
    spectrum = gkplot.cgyro_linear_spectrum(runs)
    assert spectrum["ky"].size == 3 and not spectrum["converged"][0]
    figure, axes = gkplot.plot_linear_spectrum(
        spectrum,
        references={"TGLF SAT2": {"ky": [0.2, 0.3, 0.4],
                                  "gamma": np.ones((3, 2)), "omega": -np.ones((3, 2))}},
    )
    assert len(axes) == 2
    labels = [line.get_label() for line in axes[0].get_lines()]
    assert "CGYRO" in labels and "TGLF SAT2" in labels


def test_eigenfunction_panels_per_field(tmp_path):
    run = collect_cgyro_outputs(write_run(tmp_path / "run"))
    figure, axes = gkplot.plot_eigenfunction(
        run.grid["thetab"], {"phi": run.ballooning["phi"], "a_parallel": 0.1j * run.ballooning["phi"]})
    assert len(axes) == 2


def test_convergence_and_flux_figures():
    gkplot.plot_convergence(
        {"n_theta": {"value": [24, 32, 48], "gamma": [0.24, 0.25, 0.25], "omega": [0.4, 0.4, 0.41]}},
        baseline={"gamma": 0.25, "omega": 0.4},
    )
    t = np.linspace(0, 100, 50)
    gkplot.plot_flux_trace(t, {"Q_i": np.sin(t) + 2}, window=(50, 100),
                           references={"TGLF SAT0": 1.66})
    gkplot.plot_flux_ky_spectrum([0.1, 0.2, 0.3], {"Q_i": [1, 2, 1]})


def test_a_supplied_axes_is_used():
    figure, ax = plt.subplots()
    out_figure, out_ax = gkplot.plot_flux_ky_spectrum([0.1], {"Q": [1]}, ax=ax)
    assert out_ax is ax and out_figure is figure


# --------------------------------------------------------------------------
# native TGLF views (#1591), on real VEST runs
# --------------------------------------------------------------------------

from pathlib import Path

from vaft.code.gacode.tglf.outputs import collect_tglf_outputs

_DATA = Path(__file__).parent / "data" / "gacode"
_SAT0 = _DATA / "tglf_vest_39915_r0.70_sat0-es"
_SAT2 = _DATA / "tglf_vest_39915_r0.70_sat2-em-bper"


def test_tglf_linear_spectrum_names_its_preset_family():
    spectrum = gkplot.tglf_linear_spectrum(collect_tglf_outputs(_SAT2))
    assert spectrum["gamma"].shape == (21, 2)
    assert "XNU_MODEL=3" in spectrum["preset"] and "SAT2" in spectrum["preset"]


def test_flux_contributors_show_only_the_fields_tglf_wrote():
    es = gkplot.tglf_flux_contributors(collect_tglf_outputs(_SAT0))
    em = gkplot.tglf_flux_contributors(collect_tglf_outputs(_SAT2))
    assert set(es["series"]["e"]) == {"phi", "total"}
    assert set(em["series"]["i"]) == {"phi", "a_par", "total"}
    # the bins sum to TGLF's own total energy flux
    run = collect_tglf_outputs(_SAT2)
    assert np.isclose(em["series"]["e"]["total"].sum(), run.energy_flux[0], rtol=1e-3)
    figure, axes = gkplot.plot_flux_contributors(em)
    labels = [line.get_label() for line in axes[1].get_lines()]
    assert "total" in labels and any("A_" in label for label in labels)


def test_mixing_length_proxy_states_its_definition():
    spectrum = gkplot.tglf_linear_spectrum(collect_tglf_outputs(_SAT0))
    figure, ax = gkplot.plot_mixing_length_proxy(spectrum)
    assert "k_x=0" in ax.get_ylabel()
    figure, ax = gkplot.plot_mixing_length_proxy(spectrum, definition="gamma_over_ky")
    assert "k_y^2" not in ax.get_ylabel()
    with pytest.raises(ValueError, match="definition"):
        gkplot.plot_mixing_length_proxy(spectrum, definition="gamma_over_k")


def test_fluctuation_model_and_saturation_panels_draw_from_the_native_run():
    run = collect_tglf_outputs(_SAT2)
    gkplot.plot_fluctuation_spectra(
        run.ky_spectrum, {"dn_e/n_e": run.density_spectrum[:, 0],
                          "dT_e/T_e": run.temperature_spectrum[:, 0]},
        cross_phase=run.nete_crossphase_spectrum)
    figure, panels = gkplot.plot_model_details(
        run.ky_spectrum, {"width": run.width_spectrum, "kx/ky shift": run.spectral_shift_spectrum,
                          "ave_p0": run.ave_p0_spectrum, "absent": None})
    assert len(panels) == 3
    figure, ax = gkplot.plot_saturation_parameters(run.saturation_parameters)
    assert "XNU_MODEL" in ax.texts[0].get_text()


def test_local_state_marks_provenance_kinds():
    from test_cgyro_adapter import tglf_local

    state = gkplot.tglf_local_state(tglf_local())
    assert [s["name"] for s in state["species"]] == ["e", "H+", "C6+"]
    assert state["scalars"]["ExB shear"] is None
    assert state["provenance"]["ExB shear"] == "unavailable"
    assert state["provenance"]["q"] is None or state["provenance"]["q"] != "unavailable"
    figure, axes = gkplot.plot_local_state(state)
    text = axes[1].texts[0].get_text()
    exb = [line for line in text.splitlines() if line.startswith("ExB shear")][0]
    assert "[unavailable]" in exb and "solver default" in exb


def test_flux_contributors_refuse_species_that_do_not_exist():
    run = collect_tglf_outputs(_SAT0)
    for bad in ((5,), (0,), (-1,)):
        with pytest.raises(ValueError, match="outside"):
            gkplot.tglf_flux_contributors(run, species=bad)


def test_flux_contributors_keep_the_total_unknown_when_no_field_carries_the_quantity():
    """A species whose selected quantity is NaN in every field slot (cold review 0.8.0 F6).

    ``sum({}.values())`` is the integer 0; the renderer then paired a (nky,) ky axis with
    a scalar and raised. The total is unknown, not zero: it keeps the ky shape as NaN
    and the panel says so.
    """
    from vaft.code.gacode.tglf.outputs import FLUX_SPECTRUM_QUANTITIES
    import types

    nky, nfield, nq = 5, 3, len(FLUX_SPECTRUM_QUANTITIES)
    spectrum = np.full((2, nfield, nky, nq), np.nan)      # (species, field, ky, quantity)
    spectrum[:, 0, :, :] = 1.0                            # phi written for every quantity...
    q = FLUX_SPECTRUM_QUANTITIES.index("toroidal_stress")
    spectrum[0, :, :, q] = np.nan                         # ...except the electrons' stress
    run = types.SimpleNamespace(sum_flux_spectrum=spectrum, ky_spectrum=np.linspace(0.1, 1.0, nky))
    contributors = gkplot.tglf_flux_contributors(run, quantity="toroidal_stress", species=("e", "i"))
    total = contributors["series"]["e"]["total"]
    assert isinstance(total, np.ndarray) and total.shape == (nky,) and np.all(np.isnan(total))
    assert np.allclose(contributors["series"]["i"]["total"], 1.0)
    figure, axes = gkplot.plot_flux_contributors(contributors)
    notes = [text.get_text() for text in axes[0].texts]
    assert any("toroidal_stress" in note and "e" in note for note in notes)
    assert not axes[1].texts
    plt.close(figure)
