"""Spectroscopy diagrams (#1046): the same vocabulary as emission=, and nothing fabricated."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _spectroscopy as sp
from vaft.diagram._scene import Label
from vaft.formula.atomic import hydrogenic_energy_level, hydrogenic_transition_wavelength
from vaft.spectroscopy import matches, parse_emission_term, parse_line_label


def test_hydrogenic_lines_are_the_known_vacuum_wavelengths():
    # Bohr-model vacuum values (NIST differs by the fine structure, ~0.02 nm): H-alpha 656.47, H-beta 486.27,
    # D-alpha 656.29 (isotope shift 0.179 nm); He II 4->3 at infinite nuclear mass 468.65
    assert hydrogenic_transition_wavelength(3, 2, 1, 1) * 1e9 == pytest.approx(656.470, abs=2e-3)
    assert hydrogenic_transition_wavelength(4, 2, 1, 1) * 1e9 == pytest.approx(486.274, abs=2e-3)
    assert hydrogenic_transition_wavelength(3, 2, 1, 2) * 1e9 == pytest.approx(656.291, abs=2e-3)
    assert hydrogenic_transition_wavelength(4, 3, 2, None) * 1e9 == pytest.approx(468.652, abs=2e-3)
    with pytest.raises(ValueError, match="hydrogen isotope"):
        hydrogenic_transition_wavelength(4, 3, 2, 2)  # a deuteron's mass is not He II's
    assert hydrogenic_energy_level(1, 1, None) == pytest.approx(-13.6057, abs=1e-3)
    assert hydrogenic_energy_level(1, 2, None) == pytest.approx(4 * hydrogenic_energy_level(1, 1, None))
    for bad in ((0, 1, None), (1, 0, None), (1, 1, 4)):
        with pytest.raises(ValueError):
            hydrogenic_energy_level(*bad)
    with pytest.raises(ValueError):
        hydrogenic_transition_wavelength(2, 3)


def test_the_declared_labels_are_the_machine_mappings():
    from vaft.machine_mapping.spectrometer_uv import SIGNALS

    declared = list(dict.fromkeys(row[3] for row in SIGNALS))
    assert list(sp.DECLARED_LABELS) == declared


SPELLINGS = ["H-alpha", "Halpha", "H\u03b1", "D-alpha", "C III", "C2+", "carbon", "OI", "O I", "O0", "O+", "O II",
             "O V", "hydrogen"]


def _plot_selection(term, labels):
    """The (channel, line) pairs vaft.plot's emission= resolver picks from these labels, or an empty list."""
    from omas import ODS

    from vaft.plot.backend.recipes import RECIPES, _resolve_emission

    ods = ODS()
    for k, lab in enumerate(labels):
        ods[f"spectrometer_uv.channel.0.processed_line.{k}.label"] = lab
    try:
        return [line for _ch, line in _resolve_emission(ods, RECIPES["spectrometer_uv_time_intensity"], [0], term, None)]
    except ValueError:
        return []


@pytest.mark.parametrize("term", SPELLINGS)
def test_diagram_and_plot_selectors_agree(term):
    # the diagram resolves a term to the identity emission= resolves, and so picks the same declared lines
    ident = sp.identity(term)
    labels = list(sp.DECLARED_LABELS)
    by_plot = _plot_selection(term, labels)
    by_diagram = [k for k, lab in enumerate(labels) if matches(ident, parse_line_label(lab))]
    assert by_plot == by_diagram


def test_ionization_stages_mark_the_named_stage():
    m = vaft.diagram.spectroscopy_ionization_stages("C2+").model
    assert m["element"] == "C" and m["selected"] == 3 and m["stages"] == list(range(1, 8))
    w = vaft.diagram.spectroscopy_ionization_stages("tungsten").model
    assert w["stages"][-1] == 75 and None in w["stages"]
    d = vaft.diagram.spectroscopy_ionization_stages("D-alpha").model
    assert d["element"] == "H" and d["stages"] == [1, 2] and d["selected"] == 1  # a series line is H I
    w20 = vaft.diagram.spectroscopy_ionization_stages("W XX").model
    assert 20 in w20["stages"]  # the named stage stays in view when the chain is elided


def test_transitions_never_invent_a_wavelength():
    h = vaft.diagram.spectroscopy_transitions("H-alpha").model
    assert h["hydrogenic"] and h["wavelength_m"] == pytest.approx(hydrogenic_transition_wavelength(3, 2, 1, 1))
    o = vaft.diagram.spectroscopy_transitions("OI_7770").model
    assert not o["hydrogenic"] and o["wavelength_m"] == pytest.approx(777.0e-9)
    c = vaft.diagram.spectroscopy_transitions("C III").model
    assert c["wavelength_m"] is None and c["source"] is None  # nothing declared, nothing drawn
    assert "assumed" not in h["source"] and o["source"] == "the IMAS processed_line label"
    # a stored label names hydrogen without its isotope: protium is assumed, and said so
    assert "protium assumed" in vaft.diagram.spectroscopy_transitions("H-alpha_6563").model["source"]
    for bad in ("H II", "C VII", "hydrogen"):
        with pytest.raises(ValueError):
            vaft.diagram.spectroscopy_transitions(bad)
    assert "not declared" in " ".join(i.text for i in vaft.diagram.spectroscopy_transitions("C III").scene.items
                                      if isinstance(i, Label))


def test_energy_levels_are_hydrogenic_only():
    m = vaft.diagram.spectroscopy_energy_levels("D-alpha").model
    assert m["selected"] == (3, 2)
    assert all(m["levels"][n] < m["levels"][n + 1] for n in range(1, 7))
    with pytest.raises(ValueError, match="ADF04"):
        vaft.diagram.spectroscopy_energy_levels("OI_7770")
    with pytest.raises(ValueError, match="bare nucleus"):
        vaft.diagram.spectroscopy_energy_levels("H II")


def test_the_spectrum_places_only_declared_or_hydrogenic_lines():
    m = vaft.diagram.spectroscopy_spectrum(["OI_7770", "H-beta", "C III", "CIII_1909"]).model
    placed = {lab: (nm, src) for lab, nm, src in m["placed"]}
    assert placed["OI_7770"] == (777.0, "declared") and placed["CIII_1909"] == (190.9, "declared")
    assert placed["H-beta"][1] == "computed"
    assert m["unplaced"] == ["C III"]
    assert np.all(np.diff([nm for _l, nm, _s in m["placed"]]) >= 0)


@pytest.mark.parametrize("name, kwargs", [("spectroscopy_ionization_stages", {}), ("spectroscopy_transitions", {}),
                                          ("spectroscopy_transitions", {"term": "OI_7770"}),
                                          ("spectroscopy_energy_levels", {}), ("spectroscopy_spectrum", {})])
def test_every_spectroscopy_diagram_is_deterministic_and_exported(name, kwargs):
    fn = getattr(vaft.diagram, name)
    assert fn(**kwargs).tikz == fn(**kwargs).tikz
    assert name in vaft.diagram.__all__
    assert fn(**kwargs).scene.role("note") and not fn(**kwargs, labels=False).scene.role("note")
    with pytest.raises(ValueError):
        fn(**kwargs, labels="yes")


def test_an_unknown_term_is_refused():
    with pytest.raises(ValueError):
        vaft.diagram.spectroscopy_transitions("unobtainium")
