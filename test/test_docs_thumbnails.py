"""The committed plot thumbnails of the documentation site and their freshness check.

Nothing here renders the whole set: that takes minutes and is what
``python -m vaft.plot.docs_thumbnails`` is for.  The committed manifest is
checked structurally, every failure mode is provoked on a copy of the assets,
and one cheap plot is rendered from the smallest packaged sample.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import vaft
from vaft.plot import docs_thumbnails, registry

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "plots"
#: The packaged sample that loads fastest, and a plot it can draw.
SMALL_SHOT = 48224
CHEAP_PLOT = "equilibrium_time_li"

pytestmark = pytest.mark.skipif(not ASSETS.is_dir(), reason="this branch commits no plot thumbnails")


@pytest.fixture
def assets(tmp_path):
    copy = tmp_path / "plots"
    shutil.copytree(ASSETS, copy)
    return copy


def _manifest(directory: Path) -> dict:
    return json.loads((directory / docs_thumbnails.MANIFEST).read_text(encoding="utf-8"))


def _rewrite(directory: Path, mutate) -> None:
    document = _manifest(directory)
    mutate(document["plots"])
    (directory / docs_thumbnails.MANIFEST).write_text(json.dumps(document), encoding="utf-8")


def _first_rendered(directory: Path) -> str:
    return next(name for name, entry in sorted(_manifest(directory)["plots"].items())
                if entry["status"] == "rendered")


# --------------------------------------------------------------------------
# the committed set
# --------------------------------------------------------------------------


def test_every_registered_plot_has_a_manifest_entry():
    recorded = _manifest(ASSETS)["plots"]
    assert set(recorded) == {spec.name for spec in registry.specs(status=None)}
    for name, entry in recorded.items():
        assert entry["status"] in docs_thumbnails.STATUSES, name
        assert (entry["status"] == "no_figure") == (
            registry.get_spec(name).view in registry.NON_GRAPHICAL_VIEWS
        ), name
        if entry["status"] == "rendered":
            assert (ASSETS / f"{name}.png").is_file(), name
            assert entry["shot"] in vaft.data.available_samples()
        else:
            assert entry["reason"], name


def test_the_committed_thumbnails_pass_the_structural_check():
    problems, _warnings = docs_thumbnails.check(ASSETS, full=False)
    assert problems == []


# --------------------------------------------------------------------------
# what fails, and what only warns
# --------------------------------------------------------------------------


def test_a_missing_png_is_a_problem(assets):
    name = _first_rendered(assets)
    (assets / f"{name}.png").unlink()
    problems, _ = docs_thumbnails.check(assets)
    assert f"{name}: {docs_thumbnails.MANIFEST} records a thumbnail but {name}.png is missing" in problems


def test_an_orphaned_png_is_a_problem(assets):
    (assets / "no_such_plot.png").write_bytes(b"\x89PNG")
    problems, _ = docs_thumbnails.check(assets)
    assert "no_such_plot.png: orphaned thumbnail, not a registered plot" in problems


def test_a_manifest_entry_for_an_unknown_plot_is_a_problem(assets):
    _rewrite(assets, lambda plots: plots.update(no_such_plot={"status": "no_sample", "reason": "x"}))
    problems, _ = docs_thumbnails.check(assets)
    assert f"no_such_plot: recorded in {docs_thumbnails.MANIFEST} but no longer a registered plot" in problems


def test_a_newly_registered_plot_without_an_entry_only_warns(assets):
    """A plot registered after the last render must not break the docs build (#1270)."""
    name = _first_rendered(assets)
    _rewrite(assets, lambda plots: plots.pop(name))
    (assets / f"{name}.png").unlink()
    problems, notes = docs_thumbnails.check(assets)
    assert problems == []
    assert any(note.startswith(f"{name}: registered plot has no entry") for note in notes)


def test_a_png_without_its_manifest_entry_is_still_a_problem(assets):
    name = _first_rendered(assets)
    _rewrite(assets, lambda plots: plots.pop(name))
    problems, _ = docs_thumbnails.check(assets)
    assert f"{name}.png: committed but {docs_thumbnails.MANIFEST} records it as not rendered" in problems


def test_a_hand_edited_png_is_a_problem(assets):
    name = _first_rendered(assets)
    png = assets / f"{name}.png"
    png.write_bytes(png.read_bytes() + b"\0")
    problems, _ = docs_thumbnails.check(assets)
    assert f"{name}.png: does not match {docs_thumbnails.MANIFEST} (edited by hand?)" in problems


def test_a_changed_renderer_is_only_a_warning(assets):
    name = _first_rendered(assets)
    _rewrite(assets, lambda plots: plots[name].update(renderer_sha256="0" * 64))
    problems, notes = docs_thumbnails.check(assets)
    assert problems == []
    assert any(note.startswith(f"{name}: stale (renderer or style changed)") for note in notes)


def test_a_changed_sample_is_only_a_warning(assets):
    name = _first_rendered(assets)
    _rewrite(assets, lambda plots: plots[name].update(sample_sha256="0" * 64))
    problems, notes = docs_thumbnails.check(assets)
    assert problems == []
    assert any(f"{name}: stale (sample" in note for note in notes)


# --------------------------------------------------------------------------
# rendering and the recipe hash
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_sample():
    return vaft.omas.load(vaft.data.sample(SMALL_SHOT))


def test_rendering_one_plot_writes_a_png_and_a_complete_entry(tmp_path, small_sample):
    spec = registry.get_spec(CHEAP_PLOT)
    first = docs_thumbnails.render_one(spec, SMALL_SHOT, small_sample, tmp_path)
    png = (tmp_path / f"{CHEAP_PLOT}.png").read_bytes()
    assert png.startswith(b"\x89PNG")
    assert first["status"] == "rendered" and first["shot"] == SMALL_SHOT
    assert first["renderer_sha256"] == docs_thumbnails.renderer_sha256(spec)
    second = docs_thumbnails.render_one(spec, SMALL_SHOT, small_sample, tmp_path)
    assert second == first, "rendering is deterministic on one machine"


def test_the_view_model_hash_is_stable_and_sensitive(small_sample):
    one = docs_thumbnails.model_sha256(vaft.plot.extract(CHEAP_PLOT, small_sample))
    two = docs_thumbnails.model_sha256(vaft.plot.extract(CHEAP_PLOT, small_sample))
    assert one == two
    other = docs_thumbnails.model_sha256(vaft.plot.extract("equilibrium_time_beta_p", small_sample))
    assert other != one


def test_the_view_model_hash_is_blind_to_the_last_bits_of_a_build():
    """26 of 109 committed hashes disagreed between numpy 2.5.3 and 2.4.3 (cold review 0.8.0 docs F5).

    ``_feed`` hashed exact float bytes, so an FFT or a Green's function that
    differs in its last bit on another numpy/scipy build read as "the data it
    draws changed".  The hash now quantises floats: one ulp is nothing, one
    element moved by a percent, or the whole array by a factor, is a change.
    """
    import numpy as np

    x = np.linspace(-3.0, 7.0, 1001) ** 3 * 1e-9
    model = {"spectrum": x, "limits": [1.2, "label", None, True, 3], "complex": np.array([1 + 2j, 3 - 4j]),
             "edge": np.array([np.nan, 0.0, -0.0, np.inf])}
    reference = docs_thumbnails.model_sha256(model)
    assert docs_thumbnails.model_sha256({
        **model, "spectrum": np.nextafter(x, np.inf), "edge": np.array([np.nan, -0.0, 0.0, np.inf]),
        "limits": [1.2 + 1e-12, "label", None, True, 3],
    }) == reference
    assert docs_thumbnails.model_sha256({**model, "spectrum": 2.0 * x}) != reference
    assert docs_thumbnails.model_sha256({**model, "spectrum": x * (1.0 + 1e-4)}) != reference
    assert docs_thumbnails.model_sha256(
        {**model, "spectrum": np.where(np.arange(x.size) == 500, 1.01 * x, x)}) != reference
    assert docs_thumbnails.model_sha256({**model, "limits": [1.3, "label", None, True, 3]}) != reference


def test_the_fallback_hash_carries_no_memory_address():
    class Opaque:
        pass

    assert docs_thumbnails.model_sha256([Opaque()]) == docs_thumbnails.model_sha256([Opaque()])


def test_the_committed_model_hashes_follow_the_current_rule(small_sample):
    """A hashing-rule change without ``--rehash`` would report every thumbnail stale on ``--check``."""
    entry = _manifest(ASSETS)["plots"][CHEAP_PLOT]
    assert entry["status"] == "rendered" and entry["shot"] == SMALL_SHOT
    assert entry["model_sha256"] == docs_thumbnails.model_sha256(vaft.plot.extract(CHEAP_PLOT, small_sample))


def test_rehash_rewrites_only_the_model_hashes(assets, small_sample):
    _rewrite(assets, lambda plots: plots[CHEAP_PLOT].update(model_sha256="0" * 64))
    before = _manifest(assets)
    changed = docs_thumbnails.rehash(assets, only=[CHEAP_PLOT])
    after = _manifest(assets)
    assert changed == [CHEAP_PLOT]
    assert after["plots"][CHEAP_PLOT]["model_sha256"] == docs_thumbnails.model_sha256(
        vaft.plot.extract(CHEAP_PLOT, small_sample))
    assert after["toolchain"] == before["toolchain"], "nothing was rendered"
    for key, value in before["plots"][CHEAP_PLOT].items():
        if key != "model_sha256":
            assert after["plots"][CHEAP_PLOT][key] == value, key
    assert {name: entry for name, entry in after["plots"].items() if name != CHEAP_PLOT} == {
        name: entry for name, entry in before["plots"].items() if name != CHEAP_PLOT}
    assert (assets / f"{CHEAP_PLOT}.png").read_bytes() == (ASSETS / f"{CHEAP_PLOT}.png").read_bytes()


def test_a_crlf_checkout_hashes_like_an_lf_one(tmp_path):
    """Windows runners check out with autocrlf; the renderer hash must not notice."""
    spec = registry.get_spec(CHEAP_PLOT)
    for path in docs_thumbnails._drawing_sources(spec):
        crlf = tmp_path / path.name
        crlf.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert docs_thumbnails._text_bytes(crlf) == docs_thumbnails._text_bytes(path)


def test_a_change_to_a_shared_render_body_marks_composites_stale(monkeypatch):
    """Composite renderers draw through other modules' bodies (render_line_series, ...)."""
    composite = next(spec for spec in registry.specs(status=None)
                     if spec.renderer.__module__.endswith("renderers.panels"))
    before = docs_thumbnails.renderer_sha256(composite)
    lines = docs_thumbnails._PACKAGE / "renderers" / "lines.py"
    original = docs_thumbnails._text_bytes

    def edited(path):
        data = original(path)
        return data + b"\n# a change to render_line_series\n" if path == lines else data

    monkeypatch.setattr(docs_thumbnails, "_text_bytes", edited)
    assert docs_thumbnails.renderer_sha256(composite) != before


def test_a_rendered_entry_missing_its_hashes_is_a_problem(assets):
    name = _first_rendered(assets)
    _rewrite(assets, lambda plots: plots[name].pop("model_sha256"))
    problems, _ = docs_thumbnails.check(assets)
    assert f"{name}: rendered entry lacks model_sha256" in problems
