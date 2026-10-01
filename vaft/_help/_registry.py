"""The help topic registry: names, one-line summaries and where to look.

Every topic's content comes from the subsystem's own describe/catalog API,
reached through the ``provider`` import string only when the topic is asked
for.  What lives here is routing metadata -- the part the Python help, the
``vaft help`` command and later the docs site share -- and nothing that an
existing catalog already states.

Standard library only; importing this module imports no VAFT subsystem.
"""

from __future__ import annotations

from ._model import Topic

__all__ = ["TOPICS", "topic"]

_P = "vaft._help._providers"

TOPICS: dict[str, Topic] = {
    t.name: t
    for t in (
        Topic(
            "overview",
            "VAFT: VEST analysis toolkit. Pick a topic below with vaft.help('<topic>') or `vaft help <topic>`.",
            provider=f"{_P}:overview",
            cli=("vaft help <topic>", "vaft setup", "vaft --help"),
            setup=("vaft.setup()  (auto: live notebook figures when ipympl is installed)",),
        ),
        Topic(
            "formula",
            "Scientific formulas with units, definitions and references.",
            provider=f"{_P}:formula",
            item=f"{_P}:formula_item",
            entry_points=("vaft.formula.<name>", "vaft.formula.catalog.search(text)", "vaft.formula.catalog.show(name)"),
            see_also=("vaft.help('formula', '<name>')", "vaft.help('process')"),
        ),
        Topic(
            "process",
            "Signal and data processing functions on arrays and ODS.",
            provider=f"{_P}:process",
            item=f"{_P}:process_item",
            entry_points=("vaft.process.<name>", "vaft.process.catalog.search(text)", "vaft.process.catalog.describe(name)"),
            see_also=("vaft.help('process', '<name>')", "vaft.help('formula')"),
        ),
        Topic(
            "plot",
            "Canonical scientific plots of ODS, database and array data.",
            provider=f"{_P}:plot",
            item=f"{_P}:plot_item",
            entry_points=(
                "vaft.plot.<canonical-name>",
                "vaft.omas.plot_<canonical-name>",
                "vaft.database.plotting.plot_<canonical-name>",
                "vaft.plot.available_plots(query=...)",
            ),
            cli=("vaft plot --help",),
            setup=("vaft.setup('notebook')  (live figures via ipympl)", "vaft.setup('batch')  (headless Agg)"),
            see_also=("vaft.help('plot', '<query>')", "vaft.help('omas')"),
        ),
        Topic(
            "database",
            "HSDS-backed VEST shot database: named sources, load/open/save.",
            provider=f"{_P}:database",
            item=f"{_P}:database_item",
            entry_points=(
                "vaft.database.load(shot, source=...)",
                "vaft.database.open(shot, source=...)",
                "vaft.database.sources.known_sources()",
            ),
            cli=("vaft hsds configure", "vaft summary --help", "vaft export --help"),
            setup=("vaft.setup('database')  (diagnosis only)",),
            see_also=("vaft.help('database', '<source>')", "vaft.help('data')"),
        ),
        Topic(
            "code",
            "Adapters for external physics codes (EFIT, CHEASE, GPEC, GACODE, ...).",
            provider=f"{_P}:code",
            entry_points=("vaft.code.<adapter>",),
            see_also=("vaft.help('code', probe=True)", "install/README.md"),
        ),
        Topic(
            "data",
            "Packaged sample shots for offline tutorials and tests.",
            provider=f"{_P}:data",
            item=f"{_P}:data_item",
            entry_points=("vaft.data.resources.available_samples()", "vaft.data.resources.sample(shot)", "vaft.omas.sample_ods()"),
            see_also=("vaft.help('data', '<shot>')", "vaft.help('omas')"),
        ),
        Topic(
            "omas",
            "OMAS (ODS) adapters: loading, sample data, plotting entry points.",
            provider=f"{_P}:omas",
            item=f"{_P}:omas_item",
            entry_points=("vaft.omas.plot_<canonical-name>(ods)", "vaft.omas.discovery.describe(ods)", "vaft.omas.sample_ods()"),
            see_also=("vaft.help('omas', '<query>')", "vaft.help('plot')"),
        ),
        Topic(
            "imas",
            "IMAS (IDS) adapters and the IMAS view of the plot catalog.",
            provider=f"{_P}:imas",
            item=f"{_P}:imas_item",
            entry_points=("vaft.imas.discovery.describe(source)",),
            see_also=("vaft.help('imas', '<query>')", "vaft.help('omas')"),
        ),
        Topic(
            "validation",
            "Verification and validation checks with declared tolerances.",
            provider=f"{_P}:validation",
            item=f"{_P}:validation_item",
            entry_points=("vaft.validation.registry.CHECKS", "vaft.validation.registry.describe(key)"),
            see_also=("vaft.help('validation', '<check>')",),
        ),
        Topic(
            "cli",
            "Command-line workflows installed as `vaft <command>`.",
            provider=f"{_P}:cli",
            item=f"{_P}:cli_item",
            entry_points=("vaft <command> --help",),
            see_also=("vaft help cli <command>",),
        ),
    )
}


def topic(name: str) -> Topic:
    try:
        return TOPICS[name]
    except KeyError:
        raise KeyError(f"unknown help topic {name!r}; choose from: {', '.join(TOPICS)}") from None
