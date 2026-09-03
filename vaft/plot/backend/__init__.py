"""Backend-neutral extraction: IMAS-DD path recipes turned into view models.

The recipes here are tables of IMAS Data Dictionary paths -- the vocabulary
every VAFT data model shares -- and the builders that turn what those paths
hold into the typed view models of :mod:`vaft.plot`.  They belong to no data
model: each namespace (``vaft.omas``, ``vaft.imas``, ``vaft.database``)
normalises its own inputs into ``(label, object)`` entries and supplies the
path accessor for its objects through :mod:`vaft.plot.backend.access`, and
the same recipes read them all.  That is what lets equivalent inputs from
different models be checked for equal view models before anything is drawn
(issue #63).

:mod:`vaft.plot` itself never imports this package, and nothing here imports
``omas`` or ``imas`` at module level.
"""
