"""Spectroscopy diagrams (#1046): the same vocabulary as emission=, and nothing fabricated."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _spectroscopy as sp
from vaft.diagram._scene import Label
from vaft.formula.atomic import hydrogenic_energy_level, hydrogenic_transition_wavelength
from vaft.spectroscopy import matches, parse_emission_term, parse_line_label


def test_hydrogenic_lines_are_the_known_vacuum_wavelengths():
    # NIST vacuum wavelengths: H-alpha 656.47 nm, H-beta 486.27 nm, D-alpha 656.29 nm, He II 4->3 468.7 nm
    assert hydrogenic_transition_wavelength(3, 2, 1, 1) * 1e9 == pytest.approx(656.47, abs=0.02)
    assert hydrogenic_transition_wavelength(4, 2, 1, 1) * 1e9 == pytest.approx(486.27, abs=0.02)
    assert hydrogenic_transition_wavelength(3, 2, 1, 2) * 1e9 == pytest.approx(656.29, abs=0.02)
    assert hydrogenic_transition_wavelength(4, 3, 2, None) * 1e9 == pytest.approx(468.7, abs=0.1)
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


@pytest.mark.parametrize("term", ["H-alpha", "D-alpha", "C III", "C2+", "carbon", "OI", "O V"])
def test_diagram_and_plot_selectors_agree(term):
    # the diagram resolves a term exactly as emission= does, and selects the same declared lines
    ident = sp.identity(term)
    assert ident == parse_emission_term(term)
    selected = [lab for lab in sp.DECLARED_LABELS if matches(parse_emission_term(term), parse_line_label(lab))]
    for lab in selected:
        assert matches(ident, parse_line_label(lab))


def test_ionization_stages_mark_the_named_stage():
    m = vaft.diagram.spectroscopy_ionization_stages("C2+").model
    assert m["element"] == "C" and m["selected"] == 3 and m["stages"] == list(range(1, 8))
    w = vaft.diagram.spectroscopy_ionization_stages("tungsten").model
    assert w["stages"][-1] == 75 and None in w["stages"]
    d = vaft.diagram.spectroscopy_ionization_stages("D-alpha").model
    assert d["element"] == "H" and d["stages"] == [1, 2]


def test_transitions_never_invent_a_wavelength():
    h = vaft.diagram.spectroscopy_transitions("H-alpha").model
    assert h["hydrogenic"] and h["wavelength_m"] == pytest.approx(hydrogenic_transition_wavelength(3, 2, 1, 1))
    o = vaft.diagram.spectroscopy_transitions("OI_7770").model
    assert not o["hydrogenic"] and o["wavelength_m"] == pytest.approx(777.0e-9)
    c = vaft.diagram.spectroscopy_transitions("C III").model
    assert c["wavelength_m"] is None  # nothing declared, nothing drawn
    assert "not declared" in " ".join(i.text for i in vaft.diagram.spectroscopy_transitions("C III").scene.items
                                      if isinstance(i, Label))


def test_energy_levels_are_hydrogenic_only():
    m = vaft.diagram.spectroscopy_energy_levels("D-alpha").model
    assert m["selected"] == (3, 2)
    assert all(m["levels"][n] < m["levels"][n + 1] for n in range(1, 7))
    with pytest.raises(ValueError, match="ADF04"):
        vaft.diagram.spectroscopy_energy_levels("OI_7770")


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
