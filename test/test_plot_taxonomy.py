"""Subject taxonomy contract (issue #251, phase B1).

Canonical plot identity is ``subject / view / [quantity]``: every registered
spec declares a subject from :mod:`vaft.plot.taxonomy`, aliases resolve
deterministically to exactly one canonical subject, and quantity families are
groups, never synonyms.  Policy: ``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

import vaft.plot
from vaft.plot import registry, taxonomy


def test_every_spec_declares_a_registered_subject():
    for spec in registry.specs():
        assert spec.subject, spec.name
        assert spec.subject in taxonomy.SUBJECTS, (spec.name, spec.subject)


def test_registry_rejects_missing_and_unknown_subjects():
    spec = registry.specs()[0]
    for bad_subject, match in (("", "declares no subject"), ("warp_core", "unregistered subject")):
        bad = registry.PlotSpec(
            name="no_such_plot_for_subject_test", model=spec.model,
            renderer=lambda model, **kw: None, domain=spec.domain,
            subject=bad_subject, view=spec.view, description="bad subject",
        )
        with pytest.raises(ValueError, match=match):
            registry.register(bad)
    assert "no_such_plot_for_subject_test" not in registry.canonical_names()


def test_aliases_resolve_to_exactly_one_canonical_subject():
    seen: dict[str, str] = {}
    for subject in taxonomy.SUBJECTS.values():
        for alias in subject.aliases:
            assert alias not in taxonomy.SUBJECTS, alias
            assert seen.setdefault(alias, subject.name) == subject.name, alias
    assert taxonomy.resolve_subject("ip").name == "plasma_current"
    assert taxonomy.resolve_subject("bpol_probe").name == "b_field_probe"
    assert taxonomy.resolve_subject("mirnov_coil").name == "mirnov"
    assert taxonomy.resolve_subject("pf_active").name == "pf_coil"
    # A canonical name resolves to itself.
    assert taxonomy.resolve_subject("flux_loop").name == "flux_loop"


def test_unknown_subject_terms_raise_with_the_vocabulary():
    # Related-but-distinct concepts must not silently resolve (issue #251):
    # a Rogowski coil measures plasma current but is not an alias of it.
    for term in ("rogowski_coil", "line_radiation", ""):
        with pytest.raises(KeyError, match="unknown subject"):
            taxonomy.resolve_subject(term)


def test_families_are_groups_not_aliases():
    alias_terms = {
        alias for subject in taxonomy.SUBJECTS.values() for alias in subject.aliases
    }
    family_terms = {
        term
        for family in taxonomy.FAMILIES.values()
        for term in (family.name, *family.aliases)
    }
    # Family names and family aliases must not collide with subject names,
    # subject aliases, or quantity aliases: a term resolves in one map only.
    assert not family_terms & set(taxonomy.SUBJECTS), family_terms
    assert not family_terms & alias_terms, family_terms
    assert not family_terms & set(taxonomy.QUANTITY_ALIASES), family_terms
    for family in taxonomy.FAMILIES.values():
        assert len(family.members) > 1, family.name
        for member in family.members:
            assert member != family.name
    beta = taxonomy.resolve_family("beta")
    assert set(beta.members) == {"beta_n", "beta_p", "beta_t"}
    assert taxonomy.resolve_family("w").name == "energy"
    with pytest.raises(KeyError, match="unknown quantity family"):
        taxonomy.resolve_family("beta_n")


def test_quantity_aliases_resolve_deterministically():
    assert taxonomy.resolve_quantity("safety_factor") == "q"
    assert taxonomy.resolve_quantity("beta_pol") == "beta_p"
    # A canonical quantity resolves to itself.
    assert taxonomy.resolve_quantity("beta_p") == "beta_p"
    with pytest.raises(KeyError, match="unknown quantity"):
        taxonomy.resolve_quantity("pressure_gradient")
    # No alias may itself be a canonical quantity.
    assert not set(taxonomy.QUANTITY_ALIASES) & set(taxonomy.QUANTITY_ALIASES.values())


def test_evolution_is_a_view_and_capabilities_are_not():
    assert "evolution" in registry.VIEWS
    for capability in ("interactive", "3d", "comparison", "validation"):
        assert capability not in registry.VIEWS, capability


def test_specs_and_available_plots_filter_by_subject_and_alias():
    canonical = registry.specs(subject="plasma_current")
    assert canonical and all(s.subject == "plasma_current" for s in canonical)
    assert registry.specs(subject="ip") == canonical
    rows = vaft.plot.available_plots(subject="ip")
    assert {row["name"] for row in rows} == {s.name for s in canonical}
    assert all(row["subject"] == "plasma_current" for row in rows)


def test_subject_separates_physical_concept_from_ids_domain():
    # ``domain`` records where the data lives; ``subject`` what it represents.
    flux_loop = registry.get_spec("flux_loop_time_flux")
    assert flux_loop.domain == "magnetics"
    assert flux_loop.subject == "flux_loop"
    density = registry.get_spec("electron_density_profile")
    assert density.domain == "core_profiles"
    assert density.subject == "electron_density"
