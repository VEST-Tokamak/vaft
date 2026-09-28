"""Wall conditioning: baking, glow-discharge cleaning and boronization as wall-state transitions (#1051).

``wall_conditioning_baking``
    thermal desorption: external heat drives adsorbed water and gases off the
    wall into the pump;
``wall_conditioning_gdc``
    one glow-discharge template (gas inlet, glow, anode, wall as cathode,
    wall-directed ions, pump) with gas-specific content: H$_2$/D$_2$ for
    reactive cleaning, He for ion-induced release of retained H/D;
``wall_conditioning_boronization``
    a B-containing precursor in a deposition plasma leaves a B-rich layer on
    the wall;
``wall_conditioning_sequence``
    several of them in order, each a transition of the wall state.

Every method is a transition $S^{(0)}_\\mathrm{wall} \\to S^{(1)}_\\mathrm{wall}$
(``WALL_STATE_CHANGE``), not only "cleaning". The diagrams are deliberately
reduced and semantic: no temperature, precursor, pressure or thickness is
drawn unless the caller supplies it, and a number needs its source. The
downstream wall response is #1047's plasma-wall interaction.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the wall-state transition of each method: (what changes, the mechanism)
WALL_STATE_CHANGE = {
    "baking": ("adsorbate inventory decreases", "thermal desorption"),
    "H2": ("reactive contaminants (O, C) decrease", "reactive / chemical cleaning"),
    "D2": ("reactive contaminants (O, C) decrease", "reactive / chemical cleaning"),
    "He": ("retained H/D inventory decreases", "ion-induced (physical) desorption"),
    "boronization": ("a B-rich surface layer is created or renewed", "surface coating"),
}
GASES = ("H2", "D2", "He")
STEPS = ("baking", "H2", "D2", "He", "boronization")
#: vessel cross-section drawn by every diagram [cm]
_W, _H, _T = 7.0, 4.4, 0.25


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def _vessel(x0: float = 0.0, *, pump: bool = True) -> List:
    """The vessel wall as a closed band, and a pump port on its lower right."""
    outer = [(x0, 0.0), (x0 + _W, 0.0), (x0 + _W, _H), (x0, _H)]
    inner = [(x0 + _T, _T), (x0 + _W - _T, _T), (x0 + _W - _T, _H - _T), (x0 + _T, _H - _T)]
    items: List = [Polyline.of(outer, "section fill", role="wall", closed=True),
                   Polyline.of(inner, "concept band", role="vacuum", closed=True)]
    if pump:
        items += [Polyline.of([(x0 + _W - 1.6, 0.0), (x0 + _W - 1.6, -0.9), (x0 + _W - 0.9, -0.9),
                               (x0 + _W - 0.9, 0.0)], "machine", role="pump_port"),
                  Arrow((x0 + _W - 1.25, 0.4), (x0 + _W - 1.25, -1.25), "connector", role="pumped")]
    return items


def _require_source(value, source, name):
    if value is not None and not source:
        raise ValueError(f"{name} is drawn only with its source: pass {name}_source= as well")


def _header(x0: float, method: str, labels: bool, title: str) -> List:
    if not labels:
        return []
    change, mechanism = WALL_STATE_CHANGE[method]
    return [Label((x0 + _W / 2, _H + 0.35), title, "label", anchor="south", role="title"),
            Label((x0 + _W - 1.25, -1.3), "pump", "small label", anchor="north", role="pump"),
            Label((x0 + _W / 2, -1.8), f"{mechanism}: {change}", "small label", anchor="north", role="state_change")]


def _leaving_wall(x0: float, sites, style: str, role: str) -> List:
    """Species leaving the wall surface into the vessel: an arrow from the inner surface, a dot at its head."""
    items: List = []
    for side, s in sites:
        if side == "left":
            start, end = (x0 + _T, s), (x0 + _T + 0.75, s)
        elif side == "right":
            start, end = (x0 + _W - _T, s), (x0 + _W - _T - 0.75, s)
        elif side == "top":
            start, end = (x0 + s, _H - _T), (x0 + s, _H - _T - 0.75)
        else:
            start, end = (x0 + s, _T), (x0 + s, _T + 0.75)
        items += [Arrow(start, end, "connector", role=role), Marker(end, "o", style, role=role)]
    return items


def _baking_items(x0: float, labels: bool, temperature, temperature_source) -> List:
    items = _vessel(x0)
    for k in range(4):
        y = 0.8 + 0.95 * k
        items.append(Arrow((x0 - 1.1, y), (x0 - 0.05, y), "drift", role="heat"))
        items.append(Arrow((x0 + _W + 1.1, y), (x0 + _W + 0.05, y), "drift", role="heat"))
    # desorbed species leave the heated wall; qualitative examples
    sites = (("left", 3.0, "H$_2$O"), ("left", 1.3, "CO"), ("top", 4.6, "H$_2$O"), ("right", 2.2, "H$_2$"))
    items += _leaving_wall(x0, [(side, s) for side, s, _ in sites], "xpoint", "desorption")
    if labels:
        for side, s, name in sites:
            at = {"left": (x0 + _T + 0.9, s), "right": (x0 + _W - _T - 0.9, s),
                  "top": (x0 + s + 0.15, _H - _T - 0.75)}[side]
            anchor = {"left": "west", "right": "east", "top": "west"}[side]
            items.append(Label(at, name, "small label", anchor=anchor, role="species"))
        items.append(Label((x0 - 0.1, 0.55), "external heat", "small label", anchor="north east", role="heat"))
        if temperature is not None:
            items.append(Label((x0 + _W / 2, _H / 2), f"wall at {temperature:g} $^\\circ$C ({temperature_source})",
                               "small label", role="temperature"))
    items += _header(x0, "baking", labels, "baking: thermal desorption")
    return items


def _gdc_items(x0: float, gas: str, labels: bool) -> List:
    items = _vessel(x0)
    # glow plasma filling the vessel, anode in the middle, wall as cathode
    items.append(Polyline.of([(x0 + 0.6, 0.6), (x0 + _W - 0.6, 0.6), (x0 + _W - 0.6, _H - 0.6), (x0 + 0.6, _H - 0.6)],
                             "layer", role="glow", closed=True))
    ax, ay = x0 + _W / 2, _H / 2
    items.append(Polyline.of([(ax - 0.12, ay - 0.6), (ax + 0.12, ay - 0.6), (ax + 0.12, ay + 0.6),
                              (ax - 0.12, ay + 0.6)], "section fill", role="anode", closed=True))
    items.append(Arrow((x0 - 1.2, _H - 0.9), (x0 + 0.05, _H - 0.9), "vector", role="gas_inlet"))
    # ions fall through the cathode sheath onto the wall
    for (x1, y), x2 in (((ax - 0.9, ay + 0.5), x0 + _T + 0.05), ((ax + 0.9, ay - 0.5), x0 + _W - _T - 0.05)):
        items.append(Arrow((x1, y), (x2, y), "drift ion", role="ion_flux"))
    if gas == "He":
        items += _leaving_wall(x0, (("left", 1.2), ("right", 2.6)), "opoint", "released_hydrogen")
        product = "retained H/D released"
    else:
        items += _leaving_wall(x0, (("left", 1.2), ("right", 2.6)), "xpoint", "volatile_product")
        product = "volatile products, e.g. H$_2$O"
    ion = {"H2": "H$^+$", "D2": "D$^+$", "He": "He$^+$"}[gas]
    if labels:
        items += [Label((x0 + 0.75, _H - 0.75), f"{gas.replace('2', '$_2$')} feed", "small label",
                        anchor="north west", role="gas_inlet"),
                  Label((x0 + _W - 0.75, _H - 0.75), f"glow, {ion} to the wall", "small label", anchor="north east",
                        role="glow"),
                  Label((ax + 0.2, ay + 0.1), "anode (+)", "small label", anchor="south west", role="anode"),
                  Label((x0 + 0.3, 0.05), "wall = cathode", "small label", anchor="south west", role="cathode"),
                  Label((x0 + _W - 0.75, 0.75), product, "small label", anchor="south east", role="product")]
    title = {"H2": "H$_2$ glow discharge: reactive cleaning", "D2": "D$_2$ glow discharge: reactive cleaning",
             "He": "He glow discharge: releases retained H/D"}[gas]
    items += _header(x0, gas, labels, title)
    return items


def _boronization_items(x0: float, labels: bool, precursor, thickness_nm, thickness_source) -> List:
    items = _vessel(x0)
    layer = 0.12
    items.append(Polyline.of([(x0 + _T, _T), (x0 + _W - _T, _T), (x0 + _W - _T, _H - _T), (x0 + _T, _H - _T)],
                             "boundary", role="boron_layer", closed=True))
    items.append(Polyline.of([(x0 + _T + layer, _T + layer), (x0 + _W - _T - layer, _T + layer),
                              (x0 + _W - _T - layer, _H - _T - layer), (x0 + _T + layer, _H - _T - layer)],
                             "boundary", role="boron_layer", closed=True))
    items.append(Polyline.of([(x0 + 0.8, 0.8), (x0 + _W - 0.8, 0.8), (x0 + _W - 0.8, _H - 0.8), (x0 + 0.8, _H - 0.8)],
                             "layer", role="deposition_plasma", closed=True))
    items.append(Arrow((x0 - 1.2, _H - 0.9), (x0 + 0.05, _H - 0.9), "vector", role="precursor_inlet"))
    for (x, y), (dx, dy) in (((x0 + 1.9, _H / 2), (-1.4, 0.0)), ((x0 + _W - 1.9, _H / 2), (1.4, 0.0)),
                             ((x0 + 5.3, _H - 1.3), (0.0, 0.85)), ((x0 + 1.7, 1.3), (0.0, -0.85))):
        items.append(Arrow((x, y), (x + dx, y + dy), "vector", role="deposition"))
    if labels:
        items += [Label((x0 + 0.95, _H - 0.95), precursor if precursor else "B-containing precursor",
                        "small label", anchor="north west", role="precursor"),
                  Label((x0 + _W / 2, _H / 2), "deposition plasma", "small label", role="deposition_plasma"),
                  Label((x0 + 4.0, 0.95), "B-rich layer on the wall", "small label", anchor="south",
                        role="boron_layer")]
        if thickness_nm is not None:
            items.append(Label((x0 + _W / 2, _H / 2 - 0.3), f"layer {thickness_nm:g} nm ({thickness_source})",
                               "small label", anchor="north", role="thickness"))
    items += _header(x0, "boronization", labels, "boronization: a B-rich surface layer")
    return items


def wall_conditioning_baking(*, temperature: Optional[float] = None, temperature_source: Optional[str] = None,
                             labels: bool = True) -> Diagram:
    r"""Baking: external heat desorbs water and gases from the wall into the pump.

    Thermal desorption only -- no glow, anode, ion bombardment or coating.
    The wall temperature is drawn only if given, and then only with its
    source (``temperature_source``); no "typical" baking temperature is built
    in.
    """
    labels = _check_labels(labels)
    _require_source(temperature, temperature_source, "temperature")
    items = _baking_items(0.0, labels, temperature, temperature_source)
    if labels:
        items.append(_note("Reduced and semantic: species shown are examples; no temperature unless supplied with "
                           "its source", _W / 2, -2.4))
    return Diagram("wall_conditioning_baking", Scene(tuple(items)),
                   model={"method": "baking", "state_change": WALL_STATE_CHANGE["baking"], "temperature": temperature,
                          "temperature_source": temperature_source})


def wall_conditioning_gdc(gas: str = "H2", *, labels: bool = True) -> Diagram:
    r"""Glow-discharge cleaning: one template, gas-specific content.

    Gas inlet, glow plasma, anode, the wall as cathode, wall-directed ions and
    the pump are shared. H$_2$ or D$_2$: reactive cleaning -- the ions and
    radicals turn O and C contaminants into volatile products that are pumped
    out. He: the He$^+$ bombardment releases hydrogen isotopes retained in the
    wall (ion-induced desorption) -- no chemistry, a different purpose.
    """
    if gas not in GASES:
        raise ValueError(f"gas must be one of {GASES}, not {gas!r}")
    labels = _check_labels(labels)
    items = _gdc_items(0.0, gas, labels)
    if labels:
        items.append(_note("Same apparatus template for every gas; what the wall loses differs. Products are "
                           "qualitative", _W / 2, -2.4))
    return Diagram(f"wall_conditioning_gdc_{gas.lower()}", Scene(tuple(items)),
                   model={"method": "gdc", "gas": gas, "state_change": WALL_STATE_CHANGE[gas]})


def wall_conditioning_boronization(*, precursor: Optional[str] = None, thickness_nm: Optional[float] = None,
                                   thickness_source: Optional[str] = None, labels: bool = True) -> Diagram:
    r"""Boronization: a B-containing precursor in a deposition plasma coats the wall.

    The generic diagram names no precursor ("B-containing precursor");
    ``precursor="B2H6"`` (or another) is annotated when given. The layer
    thickness is drawn only when supplied with its source. The B-rich layer
    getters oxygen and changes the surface the plasma sees.
    """
    labels = _check_labels(labels)
    _require_source(thickness_nm, thickness_source, "thickness_nm")
    items = _boronization_items(0.0, labels, precursor, thickness_nm, thickness_source)
    if labels:
        items.append(_note("A surface-state change, not cleaning: the new layer (oxygen gettering, lower "
                           "high-Z content) is what the plasma then sees", _W / 2, -2.4))
    return Diagram("wall_conditioning_boronization", Scene(tuple(items)),
                   model={"method": "boronization", "state_change": WALL_STATE_CHANGE["boronization"],
                          "precursor": precursor, "thickness_nm": thickness_nm, "thickness_source": thickness_source})


def wall_conditioning_sequence(steps: Sequence[str] = ("baking", "He", "boronization"), *,
                               labels: bool = True) -> Diagram:
    r"""Several conditioning steps in order, each a transition of the wall state.

    ``steps`` from ``STEPS``: ``"baking"``, ``"H2"``, ``"D2"``, ``"He"``,
    ``"boronization"``. Each panel is the corresponding single-method diagram
    (its semantic content, no quantities); arrows between them are wall-state
    transitions $S^{(k)} \to S^{(k+1)}$. The order is the caller's, not a
    recommendation.
    """
    labels = _check_labels(labels)
    steps = tuple(steps)
    if not steps or any(s not in STEPS for s in steps):
        raise ValueError(f"steps must be a non-empty sequence of {STEPS}, not {steps!r}")
    items: List = []
    pitch = _W + 4.5
    for k, step in enumerate(steps):
        x0 = k * pitch
        if step == "baking":
            items += _baking_items(x0, labels, None, None)
        elif step == "boronization":
            items += _boronization_items(x0, labels, None, None, None)
        else:
            items += _gdc_items(x0, step, labels)
        if k:
            items.append(Arrow((x0 - 3.2, _H / 2), (x0 - 1.4, _H / 2), "connector", role="transition"))
            if labels:
                items.append(Label((x0 - 2.3, _H / 2 + 0.15), f"$S^{{({k - 1})}} \\to S^{{({k})}}$", "small label",
                                   anchor="south", role="transition"))
    if labels:
        items.append(_note("Each step changes the wall state; the order is the caller's, not a recommended "
                           "procedure", (len(steps) * pitch - 4.5) / 2, -2.4))
    return Diagram("wall_conditioning_sequence", Scene(tuple(items)),
                   model={"steps": steps, "state_changes": tuple(WALL_STATE_CHANGE[s] for s in steps)})
