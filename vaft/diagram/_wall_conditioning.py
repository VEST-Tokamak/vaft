"""Wall conditioning: baking, glow-discharge cleaning and boronization as wall-state transitions (#1051).

``wall_conditioning_baking``
    thermal desorption: external heat drives adsorbed water and gases off the
    wall into the pump;
``wall_conditioning_gdc``
    one glow-discharge template (gas inlet, glow, anode, wall as cathode,
    ions accelerated across the cathode sheath onto the wall, pump) with
    gas-specific content: H$_2$/D$_2$ for reactive cleaning into volatile O/C
    products, He for ion-induced release of retained H/D;
``wall_conditioning_boronization``
    a B-containing precursor in a deposition plasma leaves a B-rich layer on
    the wall;
``wall_conditioning_sequence``
    several of them in order, each a transition of the wall state, ending in
    plasma operation.

Every method is a transition $S^{(0)}_\\mathrm{wall} \\to S^{(1)}_\\mathrm{wall}$
(``WALL_STATE_CHANGE``), not only "cleaning". The diagrams are deliberately
reduced and semantic: species are examples, and no temperature, precursor,
pressure or thickness is drawn unless the caller supplies it -- a number only
with its source. What the conditioned wall then does under the plasma is the
plasma-wall interaction (``plasma_wall_interaction_*``, #1047).
"""

from __future__ import annotations

from numbers import Real
from typing import List, Optional, Sequence

from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the wall-state change of each stage: (mechanism, what changes)
WALL_STATE_CHANGE = {
    "baking": ("thermal desorption", "adsorbed H$_2$O and gases decrease"),
    "H2_gdc": ("reactive cleaning", "O and C contaminants decrease"),
    "D2_gdc": ("reactive cleaning", "O and C contaminants decrease"),
    "He_gdc": ("ion-induced desorption", "retained H/D decreases"),
    "boronization": ("surface coating", "a B-rich layer is created or renewed"),
}
GASES = ("H2", "D2", "He")
STAGES = tuple(WALL_STATE_CHANGE)
TEMPERATURE_UNITS = {"K": "K", "degC": "$^\\circ$C"}
#: vessel cross-section drawn by every diagram [cm]; the pump port sits under the lower right
_W, _H, _T = 7.0, 4.4, 0.25
_PORT = (_W - 1.6, _W - 0.9)
_INLET_Y = _H - 0.9
#: example volatile products of reactive cleaning, isotope-consistent with the feed gas
_PRODUCTS = {"H2": "H$_2$O, CH$_4$", "D2": "D$_2$O, CD$_4$"}
_IONS = {"H2": "H$^+$", "D2": "D$^+$", "He": "He$^+$"}


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _quantity(value, source, name):
    """A drawn number: finite, positive, and only together with its source."""
    if value is None:
        if source is not None:
            raise ValueError(f"{name}_source given without {name}")
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not value > 0 or value == float("inf"):
        raise ValueError(f"{name} must be a positive finite number, not {value!r}")
    if not source or not isinstance(source, str):
        raise ValueError(f"{name} is drawn only with its source: pass {name}_source= as well")
    return float(value)


def _vessel(x0: float, wall_role: str = "wall") -> List:
    """Vessel wall as a closed band, a gas-inlet port on the upper left and a pump port on the lower right."""
    outer = [(x0, 0.0), (x0 + _W, 0.0), (x0 + _W, _H), (x0, _H)]
    inner = [(x0 + _T, _T), (x0 + _W - _T, _T), (x0 + _W - _T, _H - _T), (x0 + _T, _H - _T)]
    return [Polyline.of(outer, "section fill", role=wall_role, closed=True),
            Polyline.of(inner, "concept band", role="vacuum", closed=True),
            Polyline.of([(x0 - 0.7, _INLET_Y - 0.18), (x0, _INLET_Y - 0.18)], "machine", role="inlet_port"),
            Polyline.of([(x0 - 0.7, _INLET_Y + 0.18), (x0, _INLET_Y + 0.18)], "machine", role="inlet_port"),
            Polyline.of([(x0 + _PORT[0], 0.0), (x0 + _PORT[0], -0.9), (x0 + _PORT[1], -0.9), (x0 + _PORT[1], 0.0)],
                        "machine", role="pump_port"),
            Arrow((x0 + sum(_PORT) / 2, 0.6), (x0 + sum(_PORT) / 2, -1.25), "connector", role="pumped")]


def _inlet(x0: float, role: str, text: Optional[str]) -> List:
    items: List = [Arrow((x0 - 1.1, _INLET_Y), (x0 + _T + 0.1, _INLET_Y), "vector", role=role)]
    if text:
        items.append(Label((x0 - 0.1, _INLET_Y + 0.25), text, "small label", anchor="south east", role=role))
    return items


def _header(x0: float, stage: str, labels: bool, title: str) -> List:
    if not labels:
        return []
    return [Label((x0 + _W / 2, _H + 0.35), title, "label", anchor="south", role="title"),
            Label((x0 + sum(_PORT) / 2, -1.3), "pump", "small label", anchor="north", role="pump"),
            Label((x0 + _W / 2, -1.8), "wall: " + WALL_STATE_CHANGE[stage][1], "small label", anchor="north",
                  role="state_change")]


def _leaving_wall(x0: float, sites, style: str, role: str) -> List:
    """Species leaving the wall surface into the vessel: an arrow from the inner surface, a dot at its head."""
    items: List = []
    for side, s in sites:
        if side == "left":
            start, end = (x0 + _T, s), (x0 + _T + 0.75, s)
        elif side == "right":
            start, end = (x0 + _W - _T, s), (x0 + _W - _T - 0.75, s)
        else:  # top
            start, end = (x0 + s, _H - _T), (x0 + s, _H - _T - 0.75)
        items += [Arrow(start, end, style, role=role), Marker(end, "o", "xpoint", role=role)]
    return items


def _baking_items(x0: float, labels: bool, temperature=None, unit="K", source=None) -> List:
    items = _vessel(x0)
    for k in range(4):
        y = 0.8 + 0.95 * k
        items.append(Arrow((x0 - 1.1, y), (x0 - 0.05, y), "drift", role="heat"))
        items.append(Arrow((x0 + _W + 1.1, y), (x0 + _W + 0.05, y), "drift", role="heat"))
    sites = (("left", 2.4, "H$_2$O"), ("left", 1.2, "CO"), ("top", 4.4, "H$_2$O"), ("right", 2.2, "H$_2$"))
    items += _leaving_wall(x0, [(side, s) for side, s, _ in sites], "connector", "desorption")
    if labels:
        for side, s, name in sites:
            at, anchor = {"left": ((x0 + _T + 0.9, s), "west"), "right": ((x0 + _W - _T - 0.9, s), "east"),
                          "top": ((x0 + s + 0.15, _H - _T - 0.75), "west")}[side]
            items.append(Label(at, name, "small label", anchor=anchor, role="species"))
        items.append(Label((x0 - 0.1, 0.55), "external heat", "small label", anchor="north east", role="heat"))
        if temperature is not None:
            items.append(Label((x0 + _W / 2, 0.45), f"wall at {temperature:g} {TEMPERATURE_UNITS[unit]} ({source})",
                               "small label", anchor="south", role="temperature"))
    items += _header(x0, "baking", labels, "Baking: thermal desorption")
    return items


def _gdc_items(x0: float, gas: str, labels: bool) -> List:
    items = _vessel(x0, wall_role="cathode")
    g = 0.95  # glow edge; the gap to the wall surface is the cathode sheath
    items.append(Polyline.of([(x0 + g, g), (x0 + _W - g, g), (x0 + _W - g, _H - g), (x0 + g, _H - g)],
                             "layer", role="glow", closed=True))
    ax, ay = x0 + _W / 2, _H / 2
    items.append(Polyline.of([(ax - 0.12, ay - 0.6), (ax + 0.12, ay - 0.6), (ax + 0.12, ay + 0.6),
                              (ax - 0.12, ay + 0.6)], "section fill", role="anode", closed=True))
    items += _inlet(x0, "gas_inlet", f"{gas.replace('2', '$_2$')} feed" if labels else None)
    # ions leave the glow edge and are accelerated across the sheath onto the whole cathode
    for x in (1.7, 3.0, 4.3):
        items.append(Arrow((x0 + x, _H - g), (x0 + x, _H - _T - 0.02), "drift ion", role="ion_flux"))
    for x in (1.4, 5.0):
        items.append(Arrow((x0 + x, g), (x0 + x, _T + 0.02), "drift ion", role="ion_flux"))
    items.append(Arrow((x0 + g, 1.3), (x0 + _T + 0.02, 1.3), "drift ion", role="ion_flux"))
    for y in (1.4, 3.0):
        items.append(Arrow((x0 + _W - g, y), (x0 + _W - _T - 0.02, y), "drift ion", role="ion_flux"))
    if gas == "He":
        role, style, name = "released_hydrogen", "orbit tip", "H/D"
    else:
        role, style, name = "volatile_product", "connector", _PRODUCTS[gas]
    items += _leaving_wall(x0, (("left", 2.6),), style, role)
    if labels:
        items += [Label((x0 + _T + 0.9, 2.6), name, "small label", anchor="west", role=role),
                  Label((x0 + _W - 1.1, _H - 1.1), f"glow: {_IONS[gas]}", "small label", anchor="north east",
                        role="glow"),
                  Label((ax + 0.2, ay + 0.1), "anode (+)", "small label", anchor="south west", role="anode"),
                  Label((ax + 0.3, 1.05), "wall = cathode ($-$)", "small label", anchor="south",
                        role="cathode")]
    title = {"H2": "H$_2$ glow discharge: reactive cleaning", "D2": "D$_2$ glow discharge: reactive cleaning",
             "He": "He glow discharge: releases retained H/D"}[gas]
    items += _header(x0, f"{gas}_gdc", labels, title)
    return items


def _layer_path(x0: float, inset: float) -> List:
    """The B-rich layer along the wall surface, open over the pump port."""
    lo, hi = _T + inset, _H - _T - inset
    return [(x0 + _PORT[0], lo), (x0 + _T + inset, lo), (x0 + _T + inset, hi), (x0 + _W - _T - inset, hi),
            (x0 + _W - _T - inset, lo), (x0 + _PORT[1], lo)]


def _boronization_items(x0: float, labels: bool, precursor=None, thickness_nm=None, source=None) -> List:
    items = _vessel(x0)
    items += [Polyline.of(_layer_path(x0, 0.0), "boundary", role="boron_layer"),
              Polyline.of(_layer_path(x0, 0.12), "boundary", role="boron_layer")]
    d = 0.8
    items.append(Polyline.of([(x0 + d, d), (x0 + _W - d, d), (x0 + _W - d, _H - d), (x0 + d, _H - d)],
                             "layer", role="deposition_plasma", closed=True))
    items += _inlet(x0, "precursor_inlet",
                    (precursor or "\\begin{tabular}{r}B-containing\\\\precursor\\end{tabular}") if labels else None)
    for (x, y), (dx, dy) in (((x0 + 1.9, _H / 2), (-1.4, 0.0)), ((x0 + _W - 1.9, _H / 2), (1.4, 0.0)),
                             ((x0 + 5.3, _H - 1.3), (0.0, 0.85)), ((x0 + 1.7, 1.3), (0.0, -0.85))):
        items.append(Arrow((x, y), (x + dx, y + dy), "vector", role="deposition"))
    if labels:
        items += [Label((x0 + _W / 2, _H / 2), "deposition plasma", "small label", role="deposition_plasma"),
                  Polyline.of([(x0 + 3.2, _T + 0.12), (x0 + 3.2, -0.45)], "leader line", role="boron_layer"),
                  Label((x0 + 3.1, -0.5), "B-rich layer", "small label", anchor="north east", role="boron_layer")]
        if thickness_nm is not None:
            items.append(Label((x0 + _W / 2, _H / 2 - 0.3), f"layer {thickness_nm:g} nm ({source})",
                               "small label", anchor="north", role="thickness"))
    items += _header(x0, "boronization", labels, "Boronization: B-rich surface layer")
    return items


def wall_conditioning_baking(*, temperature: Optional[float] = None, temperature_unit: str = "K",
                             temperature_source: Optional[str] = None, labels: bool = True) -> Diagram:
    r"""Baking: external heat desorbs water and gases from the wall into the pump.

    Thermal desorption only -- no glow, anode, ion bombardment or coating.
    The desorbed species are examples. The wall temperature is drawn only if
    given, in ``temperature_unit`` (``"K"`` or ``"degC"``), and then only with
    its ``temperature_source``; no "typical" baking temperature is built in.
    """
    labels = _check_labels(labels)
    if temperature_unit not in TEMPERATURE_UNITS:
        raise ValueError(f"temperature_unit must be one of {tuple(TEMPERATURE_UNITS)}, not {temperature_unit!r}")
    temperature = _quantity(temperature, temperature_source, "temperature")
    items = _baking_items(0.0, labels, temperature, temperature_unit, temperature_source)
    return Diagram("wall_conditioning_baking", Scene(tuple(items)),
                   model={"stage": "baking", "state_change": WALL_STATE_CHANGE["baking"], "temperature": temperature,
                          "temperature_unit": temperature_unit if temperature is not None else None,
                          "temperature_source": temperature_source})


def wall_conditioning_gdc(gas: str = "D2", *, labels: bool = True) -> Diagram:
    r"""Glow-discharge cleaning: one apparatus template, gas-specific content.

    Gas feed, glow, anode, the wall as cathode, ions accelerated across the
    cathode sheath onto the wall, and the pump are shared. H$_2$ or D$_2$:
    reactive cleaning -- O and C contaminants leave as volatile products
    (examples isotope-consistent with the feed: H$_2$O/CH$_4$ or D$_2$O/CD$_4$).
    He: He$^+$ bombardment releases hydrogen isotopes retained in the wall
    (ion-induced desorption), drawn with its own arrow style -- no chemistry,
    a different purpose.
    """
    if gas not in GASES:
        raise ValueError(f"gas must be one of {GASES}, not {gas!r}")
    labels = _check_labels(labels)
    stage = f"{gas}_gdc"
    return Diagram("wall_conditioning_gdc", Scene(tuple(_gdc_items(0.0, gas, labels))),
                   model={"stage": stage, "gas": gas, "state_change": WALL_STATE_CHANGE[stage]})


def wall_conditioning_boronization(*, precursor: Optional[str] = None, thickness_nm: Optional[float] = None,
                                   thickness_source: Optional[str] = None, labels: bool = True) -> Diagram:
    r"""Boronization: a B-containing precursor in a deposition plasma coats the wall.

    The generic diagram names no precursor ("B-containing precursor");
    ``precursor="B$_2$H$_6$"`` (or another) is annotated when given. The layer
    thickness is drawn only with its source. A surface-state change rather
    than cleaning: the B-rich layer getters oxygen and covers high-Z
    surfaces, and it is what the plasma then sees.
    """
    labels = _check_labels(labels)
    if precursor is not None and (not isinstance(precursor, str) or not precursor):
        raise ValueError(f"precursor must be a non-empty string, not {precursor!r}")
    thickness_nm = _quantity(thickness_nm, thickness_source, "thickness_nm")
    items = _boronization_items(0.0, labels, precursor, thickness_nm, thickness_source)
    return Diagram("wall_conditioning_boronization", Scene(tuple(items)),
                   model={"stage": "boronization", "state_change": WALL_STATE_CHANGE["boronization"],
                          "precursor": precursor, "thickness_nm": thickness_nm, "thickness_source": thickness_source})


def _stage_items(stage: str, x0: float, labels: bool) -> List:
    if stage == "baking":
        return _baking_items(x0, labels)
    if stage == "boronization":
        return _boronization_items(x0, labels)
    return _gdc_items(x0, stage.split("_")[0], labels)


#: horizontal pitch of the sequence panels [cm]
SEQUENCE_PITCH = _W + 5.0


def wall_conditioning_sequence(stages: Sequence[str] = ("baking", "D2_gdc", "He_gdc", "boronization"), *,
                               plasma_operation: bool = True, labels: bool = True) -> Diagram:
    r"""Conditioning stages in order, each a transition of the wall state.

    ``stages`` from ``STAGES``: ``"baking"``, ``"H2_gdc"``, ``"D2_gdc"``,
    ``"He_gdc"``, ``"boronization"``. Each panel is the corresponding
    single-stage diagram without quantities; arrows between them are
    wall-state transitions $S^{(k)} \to S^{(k+1)}$. With ``plasma_operation``
    a last arrow hands the conditioned wall to plasma operation, where its
    response is plasma-wall interaction. The order is the caller's, not a
    recommended procedure.
    """
    labels = _check_labels(labels)
    stages = tuple(stages)
    if not stages or any(s not in STAGES for s in stages):
        raise ValueError(f"stages must be a non-empty sequence of {STAGES}, not {stages!r}")
    items: List = []
    for k, stage in enumerate(stages):
        x0 = k * SEQUENCE_PITCH
        items += _stage_items(stage, x0, labels)
        if k:
            items.append(Arrow((x0 - 3.7, _H / 2), (x0 - 1.6, _H / 2), "connector", role="transition"))
            if labels:
                items.append(Label((x0 - 2.65, _H / 2 + 0.15), f"$S^{{({k - 1})}} \\to S^{{({k})}}$",
                                   "small label", anchor="south", role="transition"))
    if plasma_operation:
        x_end = (len(stages) - 1) * SEQUENCE_PITCH + _W
        items.append(Arrow((x_end + 1.3, _H / 2), (x_end + 3.0, _H / 2), "connector", role="plasma_operation"))
        if labels:
            items.append(Label((x_end + 3.1, _H / 2), "\\begin{tabular}{l}plasma\\\\operation\\end{tabular}",
                               "small label", anchor="west", role="plasma_operation"))
    return Diagram("wall_conditioning_sequence", Scene(tuple(items)),
                   model={"stages": stages, "state_changes": tuple(WALL_STATE_CHANGE[s] for s in stages),
                          "plasma_operation": plasma_operation})
