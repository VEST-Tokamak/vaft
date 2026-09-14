"""Issue #709: a recipe says what a colour means; the theme says what it is.

What has to hold first is parity: with no theme, every model the packaged
shot supports resolves to exactly the literal it carried before the tokens
existed, entry for entry (a snapshot taken from the unmodified tree).  Then
the vocabulary is closed, a theme reaches every token, and Plotly draws the
default look.
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model, entry_supports
from vaft.plot.intent import (
    DEFAULT_COLOURS,
    DEFAULT_PALETTE,
    TOKEN_NAMESPACES,
    active_theme,
    is_colour_token,
    palette,
    resolve_color,
    resolve_style,
    themed,
)
from vaft.plot.presentation import THEMES, Presentation
from vaft.plot.registry import canonical_names

SNAPSHOT = Path(__file__).parent / "data" / "plot_colour_snapshot.json"


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _walk(obj, path, out, theme):
    for attr in ("series", "reference_lines", "layers", "overlays", "fits", "models"):
        items = getattr(obj, attr, None)
        if items is None:
            continue
        for item in items:
            style = getattr(item, "style", None)
            if style is not None and hasattr(style, "get"):
                for key in ("color", "markerfacecolor", "markeredgecolor"):
                    # Continuous maps hand out RGBA tuples whose last digits
                    # depend on the platform's solve; they are outside the
                    # vocabulary and outside the snapshot.
                    if isinstance(style.get(key), str):
                        out[f"{path}.{attr}:{key}={resolve_color(style[key], theme)}"] += 1
            _walk(item, f"{path}.{attr}", out, theme)


# ---------------------------------------------------------------------------
# parity: nothing changes without a theme
# ---------------------------------------------------------------------------

def test_every_model_resolves_to_the_colours_it_had(sample):
    snapshot = json.loads(SNAPSHOT.read_text())["colours"]
    entries = normalize_entries(sample)
    seen = {}
    for name in canonical_names():
        if not entry_supports(sample, name):
            continue
        out = Counter()
        _walk(build_model(name, entries), "", out, theme=None)
        if out:
            seen[name] = dict(sorted(out.items()))
    assert set(seen) == set(snapshot), "the set of plots carrying colours changed"
    for name in snapshot:
        assert seen[name] == snapshot[name], name


def test_no_literal_colour_is_left_in_a_recipe_outside_the_allowed_blocks():
    source = Path(vaft.__file__).parent.joinpath("plot", "backend", "recipes.py").read_text()
    literal = re.compile(r"""["']color["']:\s*["'](?!(?:palette|role|feature|state|emphasis):)[^"']+["']""")
    # Only the camera overlay builders may name a colour: they contrast with a photograph.
    camera = [m.start() for m in re.finditer(r"\ndef _(?:efit_overlay|field_line)_layers\(", source)]
    def inside_camera(pos):
        return any(start < pos < source.find("\ndef ", start + 1) for start in camera)
    stray = [m.group(0) for m in literal.finditer(source) if not inside_camera(m.start())]
    assert stray == [], stray
    assert camera, "the camera builders moved; point the allowlist at them"


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------

def test_the_vocabulary_is_closed_and_every_default_is_a_colour():
    for token, colour in DEFAULT_COLOURS.items():
        assert token.partition(":")[0] in TOKEN_NAMESPACES
        assert matplotlib.colors.is_color_like(colour), token
    for colour in DEFAULT_PALETTE:
        assert matplotlib.colors.is_color_like(colour)
    for theme in THEMES.values():
        for token, override in theme.intents.items():
            assert token in DEFAULT_COLOURS, f"{theme.name} overrides an unknown token {token}"
            colour = override["color"] if isinstance(override, dict) or hasattr(override, "keys") else override
            assert matplotlib.colors.is_color_like(colour), (theme.name, token)
            if hasattr(override, "keys"):
                assert "markerfacecolor" not in override, "a patch must not fill a hollow marker"
    assert palette(3) == "palette:3" and is_colour_token("palette:3")
    for value in ("k", "0.4", "#e41a1c", (1.0, 0.0, 0.0, 0.5), "none", "C3", "", None):
        assert not is_colour_token(value)
        assert resolve_color(value, None) == value


def test_an_unknown_key_in_a_known_namespace_is_refused():
    with pytest.raises(ValueError, match="unknown colour intent"):
        resolve_color("feature:vessel", None)
    with pytest.raises(ValueError, match="palette token"):
        resolve_color("palette:x", None)


def test_resolve_style_touches_only_colour_keys_and_applies_a_patch():
    plain = resolve_style({"color": "state:missing", "marker": "s", "markerfacecolor": "none"}, None)
    assert plain == {"color": "tab:red", "marker": "s", "markerfacecolor": "none"}
    mono = resolve_style({"color": "role:reconstructed", "linestyle": "none"}, THEMES["monochrome"])
    assert mono == {"color": "0.45", "linestyle": "none", "marker": "^"}
    hollow = resolve_style({"color": "state:missing", "markerfacecolor": "none"}, THEMES["monochrome"])
    assert hollow["markerfacecolor"] == "none"


# ---------------------------------------------------------------------------
# a theme reaches every token
# ---------------------------------------------------------------------------

def test_monochrome_resolves_every_token_to_a_grey():
    theme = THEMES["monochrome"]
    for token in (*DEFAULT_COLOURS, *(palette(i) for i in range(10))):
        r, g, b, _ = matplotlib.colors.to_rgba(resolve_color(token, theme))
        assert r == g == b, token


def test_technical_palette_is_okabe_ito_and_defaults_stay_the_literals():
    assert resolve_color("palette:0", THEMES["technical"]) == "#000000"
    assert resolve_color("palette:0", None) == "#377eb8"
    assert resolve_color("feature:boundary", None) == "#e41a1c"
    assert resolve_color("feature:boundary", THEMES["technical"]) == "#D55E00"


def test_the_drawn_colours_follow_the_theme(sample):
    figure, axes = vaft.omas.plot_magnetics_geometry_poloidal(sample)
    plain = {matplotlib.colors.to_hex(line.get_color()) for line in axes.lines}
    assert "#377eb8" in plain or "#ff7f00" in plain
    figure, axes = vaft.omas.plot_magnetics_geometry_poloidal(sample, theme="technical")
    themed_colours = {matplotlib.colors.to_hex(line.get_color()) for line in axes.lines}
    assert themed_colours <= {c.lower() for c in (*THEMES["technical"].colors, "#000000")} | {"#666666"}
    assert not themed_colours & {"#377eb8", "#ff7f00"}
    figure, axes = vaft.omas.plot_equilibrium_geometry_boundary(sample, theme="monochrome")
    r, g, b, _ = matplotlib.colors.to_rgba(axes.lines[0].get_color())
    assert r == g == b and axes.lines[0].get_linestyle() == "--"


def test_the_theme_is_read_through_the_context_and_reset_after():
    assert active_theme() is None
    presentation = Presentation(None, THEMES["minimal"])
    with presentation.context():
        assert active_theme() is THEMES["minimal"]
        assert resolve_color("feature:boundary") == "#EE6677"
    assert active_theme() is None and resolve_color("feature:boundary") == "#e41a1c"
    with pytest.raises(RuntimeError):
        with themed(THEMES["technical"]):
            raise RuntimeError("inside")
    assert active_theme() is None


# ---------------------------------------------------------------------------
# Plotly draws the default look
# ---------------------------------------------------------------------------

def test_plotly_resolves_tokens_to_the_default_look():
    pytest.importorskip("plotly")
    from vaft.plot.plotly._style import color, translate_style

    assert color("palette:0") == "#377eb8"
    assert color("feature:wall") == "rgb(102,102,102)"
    assert color("state:disabled") == "#ff7f0e"
    style = translate_style({"color": "state:missing", "marker": "s", "markerfacecolor": "none"})
    text = json.dumps(style)
    assert "#d62728" in text


def test_a_theme_that_thins_markers_along_lines_keeps_every_point_of_a_scatter():
    """The monochrome cycle carries markevery=0.1 for dense traces; a point series is not a trace."""
    from vaft.plot.models import LineSeries, Series

    x = np.arange(20.0)
    model = LineSeries(
        series=(Series(x=x, y=x, label="points", style={"marker": "o", "linestyle": "none", "color": "state:enabled"}),),
        y_label="y",
    )
    from vaft.plot import renderers

    _, axes = renderers.render_line_series(model, theme="monochrome")
    assert axes.lines[0].get_markevery() in (None, 1)
    r, g, b, _ = matplotlib.colors.to_rgba(axes.lines[0].get_color())
    assert r == g == b == 0.0


def test_geometry_points_keep_every_marker_under_a_theme(sample):
    figure, axes = vaft.omas.plot_magnetics_geometry_poloidal(sample, theme="monochrome")
    points = [line for line in axes.lines if line.get_linestyle() == "None" and line.get_marker() not in ("", "None")]
    assert points and all(line.get_markevery() in (None, 1) for line in points)


def test_a_patch_evicts_the_alias_it_replaces():
    resolved = resolve_style({"color": "feature:boundary", "ls": "-"}, THEMES["monochrome"])
    assert resolved["linestyle"] == "--" and "ls" not in resolved
    from vaft.plot.models import GeometryLayer, GeometryLayers
    from vaft.plot import renderers

    layer = GeometryLayer(r=np.array([0.1, 0.8]), z=np.array([0.0, 0.5]), label="b", style={"color": "feature:boundary"})
    renderers.render_geometry_layers(GeometryLayers(layers=(layer,)), theme="monochrome", ls="-")


def test_themes_stay_hashable():
    assert len({theme for theme in THEMES.values()}) == 3
