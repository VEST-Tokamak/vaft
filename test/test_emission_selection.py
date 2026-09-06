"""Selecting a spectral line by what it is rather than where it is stored.

``emission=`` names the physics -- an element, an ion, a transition -- and
resolves against the labels the data itself records.  ``line_index=`` is the
storage-level position beneath it, kept so an array index stays reachable.

One selector may legitimately match several lines: carbon is measured twice at
different ionization stages, and H-alpha twice on different digitizers.  That
is not an ambiguity to refuse but a set of traces to draw, so the result
follows the same multi-trace convention ``selection=`` already uses.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import omas
import pytest

import vaft.data
import vaft.omas


#: What the VEST mapping writes, as ``(channel, line, label, wavelength_m)``.
SIGNAL_LAYOUT = (
    (0, 0, "H-alpha_6563", 656.3e-9),
    (1, 0, "OI_7770", 777.0e-9),
    (2, 0, "H-alpha_6563", 656.3e-9),
    (2, 1, "H-beta_4861", 486.1e-9),
    (2, 2, "H-gamma_4340", 434.0e-9),
    (2, 3, "CII_4267", 426.7e-9),
    (2, 4, "CIII_1909", 190.9e-9),
    (2, 5, "OII_3726", 372.6e-9),
    (2, 6, "OV_629", 62.9e-9),
)

CHANNEL_NAMES = {0: "H alpha Filterscope", 1: "O-I Filterscope", 2: "Versatile Filterscope"}


@pytest.fixture
def filterscope_ods():
    """A three-channel filterscope, one of which records seven lines.

    The packaged contract fixture carries a single line and so cannot exercise
    a selector that resolves to several.
    """
    ods = omas.ODS()
    ods["spectrometer_uv.ids_properties.homogeneous_time"] = 1
    ods["spectrometer_uv.time"] = np.linspace(0.24, 0.36, 8)
    for channel, name in CHANNEL_NAMES.items():
        ods[f"spectrometer_uv.channel.{channel}.name"] = name
    for channel, line, label, wavelength in SIGNAL_LAYOUT:
        base = f"spectrometer_uv.channel.{channel}.processed_line.{line}"
        ods[f"{base}.label"] = label
        ods[f"{base}.wavelength_central"] = wavelength
        ods[f"{base}.intensity.data"] = np.linspace(0.0, 1.0, 8) * (line + 1)
    return ods


def _drawn(ods, **options):
    figure, axes = vaft.omas.plot_spectrometer_uv_time_intensity(ods, show=False, **options)
    labels = [line.get_label() for line in axes.get_lines()]
    matplotlib.pyplot.close(figure)
    return labels


# ---------------------------------------------------------------------------
# The default must not move
# ---------------------------------------------------------------------------

def test_omitting_the_selector_draws_exactly_what_it_always_did(filterscope_ods):
    """One trace per channel, of that channel's first line, unnamed."""
    assert _drawn(filterscope_ods) == list(CHANNEL_NAMES.values())


# ---------------------------------------------------------------------------
# Resolving at each level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "emission, expected",
    [
        ("CIII", ["Versatile Filterscope CIII_1909"]),
        ("C III", ["Versatile Filterscope CIII_1909"]),
        ("C2+", ["Versatile Filterscope CIII_1909"]),
        ("CIII_1909", ["Versatile Filterscope CIII_1909"]),
        ("OII", ["Versatile Filterscope OII_3726"]),
        ("O II", ["Versatile Filterscope OII_3726"]),
        ("O+", ["Versatile Filterscope OII_3726"]),
        ("OV", ["Versatile Filterscope OV_629"]),
    ],
)
def test_an_ion_resolves_to_its_own_stage(filterscope_ods, emission, expected):
    assert _drawn(filterscope_ods, emission=emission) == expected


@pytest.mark.parametrize("emission", ["C", "Carbon", "carbon"])
def test_an_element_resolves_to_every_stage_of_it(filterscope_ods, emission):
    assert _drawn(filterscope_ods, emission=emission) == [
        "Versatile Filterscope CII_4267",
        "Versatile Filterscope CIII_1909",
    ]


@pytest.mark.parametrize("emission", ["O", "Oxygen", "oxygen"])
def test_an_element_spans_channels(filterscope_ods, emission):
    assert _drawn(filterscope_ods, emission=emission) == [
        "O-I Filterscope OI_7770",
        "Versatile Filterscope OII_3726",
        "Versatile Filterscope OV_629",
    ]


@pytest.mark.parametrize("emission", ["H_alpha", "H-alpha", "Halpha", "Hα"])
def test_a_transition_is_reachable_however_it_is_spelled(filterscope_ods, emission):
    """H-alpha is recorded twice, on two digitizers.  Both are drawn.

    They are different measurements of the same line, not a duplicate to
    refuse, and the channel name keeps them apart.
    """
    assert _drawn(filterscope_ods, emission=emission) == [
        "H alpha Filterscope H-alpha_6563",
        "Versatile Filterscope H-alpha_6563",
    ]


@pytest.mark.parametrize("emission", ["H", "Hydrogen", "hydrogen"])
def test_hydrogen_resolves_to_its_whole_series(filterscope_ods, emission):
    assert _drawn(filterscope_ods, emission=emission) == [
        "H alpha Filterscope H-alpha_6563",
        "Versatile Filterscope H-alpha_6563",
        "Versatile Filterscope H-beta_4861",
        "Versatile Filterscope H-gamma_4340",
    ]


def test_several_terms_keep_the_callers_order_and_collapse_repeats(filterscope_ods):
    assert _drawn(filterscope_ods, emission=["OV", "CIII", "OV"]) == [
        "Versatile Filterscope OV_629",
        "Versatile Filterscope CIII_1909",
    ]


# ---------------------------------------------------------------------------
# The storage-level index stays reachable
# ---------------------------------------------------------------------------

def test_an_index_still_selects_a_line(filterscope_ods):
    """``line_index=`` means that position wherever a channel has one."""
    assert _drawn(filterscope_ods, line_index=4) == ["Versatile Filterscope CIII_1909"]
    assert _drawn(filterscope_ods, line_index=0) == [
        "H alpha Filterscope H-alpha_6563",
        "O-I Filterscope OI_7770",
        "Versatile Filterscope H-alpha_6563",
    ]


def test_an_index_no_channel_holds_is_refused(filterscope_ods):
    """It used to draw an empty figure and report nothing (issue #290)."""
    with pytest.raises(ValueError, match="index 99"):
        _drawn(filterscope_ods, line_index=99)


def test_a_negative_or_non_integer_index_is_refused(filterscope_ods):
    with pytest.raises(ValueError, match="non-negative"):
        _drawn(filterscope_ods, line_index=-1)
    with pytest.raises(TypeError, match="integer position"):
        _drawn(filterscope_ods, line_index=True)


def test_the_two_selectors_are_alternatives(filterscope_ods):
    with pytest.raises(TypeError, match="not both"):
        _drawn(filterscope_ods, emission="CIII", line_index=4)


# ---------------------------------------------------------------------------
# Refusing, and saying what exists
# ---------------------------------------------------------------------------

def test_an_unknown_term_names_the_choices_that_exist(filterscope_ods):
    with pytest.raises(ValueError, match="unknown emission"):
        _drawn(filterscope_ods, emission="Xq")
    # The message reports species, not only indices or raw labels.
    with pytest.raises(ValueError, match="C III"):
        _drawn(filterscope_ods, emission="Xq")


@pytest.mark.parametrize("emission", ["D", "Deuterium", "D_alpha", "HeII", "CIV"])
def test_a_species_this_input_does_not_record_is_refused(filterscope_ods, emission):
    """Structurally supported, absent from the data: an honest empty answer.

    VEST runs hydrogen, so no deuterium line is mapped; helium would resolve
    the moment a shot recorded one.
    """
    with pytest.raises(ValueError, match="is recorded in this input"):
        _drawn(filterscope_ods, emission=emission)


def test_a_malformed_value_is_refused(filterscope_ods):
    with pytest.raises(TypeError, match="species or line name"):
        _drawn(filterscope_ods, emission=3.7)
    with pytest.raises(TypeError, match="species or line name"):
        _drawn(filterscope_ods, emission=["CIII", 4])


# ---------------------------------------------------------------------------
# Plots that read no spectral line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("option", ["emission", "line_index"])
def test_a_plot_that_reads_no_line_refuses_the_selector(option):
    ods = omas.ODS()
    ods["magnetics.ids_properties.homogeneous_time"] = 1
    ods["magnetics.time"] = np.linspace(0.0, 1.0, 4)
    ods["magnetics.ip.0.data"] = np.linspace(0.0, 1.0, 4)
    ods["magnetics.ip.0.time"] = np.linspace(0.0, 1.0, 4)
    value = "CIII" if option == "emission" else 4
    with pytest.raises(ValueError, match="does not read one"):
        vaft.omas.plot_plasma_current_time(ods, show=False, **{option: value})


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_a_composite_passes_the_selector_only_to_the_panel_that_reads_a_line():
    """The selector used to reach the plasma-current panel and raise there."""
    ods = vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))
    figure, _ = vaft.omas.plot_spectrometer_uv_time_impurity(
        ods, emission="CIII", show=False
    )
    drawn = [axis for axis in figure.axes if axis.get_lines()]
    assert len(drawn) == 2
    assert [line.get_label() for line in drawn[1].get_lines()] == [
        "Versatile Filterscope CIII_1909"
    ]
    matplotlib.pyplot.close(figure)


def test_each_matched_line_can_have_its_own_panel(filterscope_ods):
    """Layout decides presentation; the selector decides what was selected."""
    figure, _ = vaft.omas.plot_spectrometer_uv_time_intensity(
        filterscope_ods, emission="Carbon", layout="subplots", show=False
    )
    titles = [axis.get_title() for axis in figure.axes if axis.get_lines()]
    assert titles == [
        "Versatile Filterscope CII_4267",
        "Versatile Filterscope CIII_1909",
    ]
    matplotlib.pyplot.close(figure)


# ---------------------------------------------------------------------------
# The metadata the selector reads
# ---------------------------------------------------------------------------

def test_no_two_lines_of_one_channel_claim_the_same_wavelength():
    """C II was mapped at 3726 A, which is the [O II] line beside it.

    Two species at one wavelength on one channel is a mapping error, and a
    species-aware selector would resolve it to the wrong ion.
    """
    from vaft.machine_mapping.spectrometer_uv import SIGNALS

    seen: dict[int, set[float]] = {}
    for _, channel, _, label, wavelength in SIGNALS:
        assert wavelength not in seen.setdefault(channel, set()), (channel, label)
        seen[channel].add(wavelength)


def test_the_label_agrees_with_the_wavelength_it_is_stored_beside():
    """The Angstrom suffix and ``wavelength_central`` state the same thing."""
    from vaft.machine_mapping.spectrometer_uv import SIGNALS
    from vaft.spectroscopy import parse_line_label

    for _, _, _, label, wavelength in SIGNALS:
        identity = parse_line_label(label)
        assert identity is not None, label
        assert identity.wavelength_angstrom == pytest.approx(wavelength * 1e10, rel=1e-3)
