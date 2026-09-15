# The legacy VEST `mhdin.dat`

The namelist that shipped beside the packaged Green tables until issue #708,
describing the PF set as **16 lumped conductors** where the canonical static
geometry resolves **530 filaments**, and declaring `nfsum = 16` where the
table now carries 26 current groups.

It is kept for one reason: it describes the same machine, written by other
hands, so agreeing with it on coil centroids, turns, probe positions and
angles is evidence that the canonical projection is right rather than merely
self-consistent. `test/test_efund_geometry.py` checks exactly that.

It is **not** loadable as a table directory any more: its `nfsum` no longer
matches the tables, which is the whole point of the change that moved it here.
Nothing in the pipeline reads it.
