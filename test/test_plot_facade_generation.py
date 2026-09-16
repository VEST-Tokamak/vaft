"""All three verbs of a canonical plot come from one registry pass (#815).

`dd_*` and `extract_*` were generated; `plot_*` was written out by hand. So
registering a plot created two of the three and silently not the third, and the
drift surfaced as a failure of `test_the_three_verbs_cover_the_same_plots` on a
push to develop rather than on the PR -- that test is outside the `core`
selection. #810 did exactly that and #814 patched the instance.

`install_facades` now generates `plot_*` too, but only where the namespace has
not written one out: an adapter that does more than call `render`, or whose
prose says something the registry description does not, keeps its own body.
These tests pin both halves of that rule, because either half alone is a bug --
generating nothing brings the drift back, generating over everything silently
deletes documentation.
"""

import pytest

from vaft.plot.backend.facade import install_facades
from vaft.plot.registry import canonical_names, specs


pytestmark = pytest.mark.core


def _install(module_globals):
    return install_facades(
        module_globals, normalize=lambda *a, **k: (), namespace="vaft.omas", subject="ods"
    )


def test_a_plot_the_namespace_has_not_written_gets_generated():
    """The gap that #810 fell into: registered, two verbs, no third."""
    stem = next(iter(canonical_names()))
    namespace = {"render": lambda *a, **k: ("figure", "axes")}

    bound = _install(namespace)

    assert f"plot_{stem}" in namespace
    assert f"plot_{stem}" in bound, "a generated name must reach __all__"
    assert namespace[f"plot_{stem}"].__name__ == f"plot_{stem}"
    assert namespace[f"plot_{stem}"].__module__ == "vaft.omas.plotting"


def test_a_hand_written_adapter_is_left_alone():
    """Generating over it would silently replace a body that does more.

    Four adapters in `vaft.omas.plotting` have real logic, and about fifty carry
    prose the registry description does not. Overwriting either is a
    documentation or behaviour regression that nothing else would catch.
    """
    stem = next(iter(canonical_names()))

    def mine(source, **options):
        return "the hand-written one"

    namespace = {"render": lambda *a, **k: None, f"plot_{stem}": mine}

    bound = _install(namespace)

    assert namespace[f"plot_{stem}"] is mine
    assert f"plot_{stem}" not in bound, (
        "a name the module already exports must not be added to __all__ again -- "
        "that is the duplicate the three-verbs test refuses"
    )


def test_a_generated_adapter_passes_its_arguments_through_to_render():
    stem = next(iter(canonical_names()))
    seen = {}

    def render(name, source, **options):
        seen.update(name=name, source=source, **options)
        return ("figure", "axes")

    namespace = {"render": render}
    _install(namespace)

    result = namespace[f"plot_{stem}"]("the-ods", ax="an-axes", show=True, label="pulse")

    assert result == ("figure", "axes")
    assert seen == {
        "name": stem,
        "source": "the-ods",
        "ax": "an-axes",
        "show": True,
        "label": "pulse",
    }


def test_a_generated_docstring_carries_the_registry_description_and_its_twins():
    stem = next(iter(canonical_names()))
    description = next(spec.description for spec in specs() if spec.name == stem)
    namespace = {"render": lambda *a, **k: None}

    _install(namespace)

    doc = namespace[f"plot_{stem}"].__doc__
    assert doc.startswith(description)
    assert f"vaft.omas.extract_{stem}" in doc
    assert f"vaft.omas.dd_{stem}" in doc


def test_installing_without_a_render_says_so_rather_than_binding_something_broken():
    """The generated adapters are wrappers around the module's own `render`.

    Without it they would bind and then fail at call time, one plot at a time,
    far from the cause.
    """
    with pytest.raises(TypeError, match="must define `render`"):
        _install({})

    with pytest.raises(TypeError, match="must define `render`"):
        _install({"render": "not callable"})


def test_a_generated_adapter_reads_render_at_call_time():
    """Hand-written adapters read `render` as a module global.

    A generated one that closed over it instead would make the two halves of
    one `__all__` disagree the moment anything rebinds the name -- a test
    monkeypatching `render` would see the hand-written adapters diverted and
    the generated ones still calling the real renderer, which draws.
    """
    calls = []
    namespace = {"render": lambda *a, **k: ("first", a, k)}
    _install(namespace)

    stem = next(iter(canonical_names()))
    plot = namespace[f"plot_{stem}"]
    assert plot("ods")[0] == "first"

    namespace["render"] = lambda *a, **k: calls.append((a, k)) or ("second", a, k)
    assert plot("ods")[0] == "second"
    assert calls and calls[0][0][0] == stem
