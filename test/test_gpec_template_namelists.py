"""The packaged GPEC namelists must be readable by the GPEC revisions VAFT runs.

A Fortran namelist READ stops the program on any name it does not declare:
``Fortran runtime error: Cannot match namelist object name ...``. So a key in
a template is a contract with every GPEC build that will read it.

``out_ahg2msc`` is the case that broke: GPEC develop (``e68d7ac2``, upstream
commit ``9e107194``) moved it out of ``DCON_OUTPUT``/``RDCON_OUTPUT``/
``GPEC_OUTPUT`` into ``EQUIL_OUTPUT``, and every DCON run on vestserver died
reading ``dcon.in`` -- stability products came back empty for every shot.

The templates therefore name it nowhere: naming it in either place makes one
of the two revisions stop at the READ.  The defaults differ, though.  Develop
defaults it to ``.FALSE.`` (``equil/global.f``), which is what the templates
used to ask for.  The pre-move layout (``f06e6abd``) defaults it to
``.TRUE.`` (``dcon/dcon_mod.f``, ``gpec/gpec.f``), so an old build now takes
the deprecated file-based vacuum path -- it writes ``ahg2msc_*.out`` and ideal
GPEC reads them back, which ``vaft.code.gpec._solvers.stage_dcon_products``
already stages between cells.  It still runs, but slower; GPEC develop is the
supported revision.
"""

from __future__ import annotations

import re

import pytest

from vaft.data.resources import data_path

#: Keys a GPEC revision in use has dropped or moved, with the namelists no
#: template may name them in.
REMOVED_OR_MOVED = {
    "out_ahg2msc": ("dcon_output", "rdcon_output", "gpec_output", "stride_output", "equil_output"),
}


def _namelist_keys(text: str) -> dict[str, set[str]]:
    keys: dict[str, set[str]] = {}
    block = None
    for line in text.splitlines():
        body = line.split("!", 1)[0].strip()
        if body.startswith("&"):
            block = body[1:].split()[0].lower()
            keys.setdefault(block, set())
        elif body.startswith("/"):
            block = None
        elif block is not None:
            for match in re.finditer(r"([A-Za-z_][\w%]*)\s*(?:\([^)]*\))?\s*=", body):
                keys[block].add(match.group(1).lower())
    return keys


@pytest.mark.parametrize("template", ["dcon.in", "rdcon.in", "gpec.in", "stride.in", "equil.in"])
def test_no_template_names_a_key_a_gpec_revision_in_use_rejects(template):
    keys = _namelist_keys((data_path("gpec") / template).read_text(encoding="utf-8"))
    for key, namelists in REMOVED_OR_MOVED.items():
        for namelist in namelists:
            assert key not in keys.get(namelist, set()), f"{template}: &{namelist.upper()} names {key}"


def test_the_reader_sees_the_blocks_it_checks():
    """Guard the guard: a parser that found nothing would pass the test above."""
    keys = _namelist_keys((data_path("gpec") / "dcon.in").read_text(encoding="utf-8"))
    assert {"dcon_control", "dcon_output"} <= set(keys)
    assert "out_fund" in keys["dcon_output"]
