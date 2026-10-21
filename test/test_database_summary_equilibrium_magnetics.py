"""equilibrium_global through the lazy database view reads the measured diamagnetic flux.

The lazy view serves only the root IDS named in the preset's ``paths``, and a
requested IDS the shot does not store fails on first access with
``OSError(404)`` (what ``h5pyd.File`` raises), not at open. The diamagnetic
update interpolates ``magnetics.diamagnetic_flux`` and ``virial_mui_measured``
is built on it, so without ``magnetics`` in the paths that column is NaN in
every database table while the same data offline fills it.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy

import numpy as np
import pandas as pd
import pytest

import vaft
from vaft import database
from vaft.database._summary import extract_equilibrium_global


def _head(location) -> str:
    first = location[0] if isinstance(location, (list, tuple)) and location else location
    return str(first).split(".", 1)[0].split("[", 1)[0]


def _lazy_view_class(missing: tuple[str, ...]):
    """An ODS whose top level answers a missing requested IDS with a 404."""
    from omas import ODS

    class LazyView(ODS):
        def __contains__(self, location):
            if not self.location and _head(location) in missing:
                raise OSError(404, "Not Found")
            return super().__contains__(location)

        def __getitem__(self, location, *args, **kwargs):
            if not self.location and _head(location) in missing:
                raise OSError(404, "Not Found")
            return super().__getitem__(location, *args, **kwargs)

    return LazyView


def _serve(stored, opened):
    """``open`` as the lazy database view behaves: only the requested root IDS."""

    @contextmanager
    def fake_open(shot, *, source=None, paths=None, **kwargs):
        opened.append(tuple(paths))
        ods = stored[shot]
        roots = tuple(dict.fromkeys(path.split(".", 1)[0] for path in paths))
        view = _lazy_view_class(tuple(name for name in roots if name not in ods))(
            imas_version=ods.imas_version, consistency_check=False
        )
        for name in roots:
            if name in ods:
                view[name] = copy.deepcopy(ods[name])
        yield view

    return fake_open


def _table(monkeypatch, shot):
    stored = {shot: vaft.omas.sample_ods(shot)}
    opened = []
    monkeypatch.setattr(database, "open", _serve(stored, opened))
    table = database.summary((shot, shot), preset="equilibrium_global", source="public")
    assert opened and len(table) > 0
    return stored[shot], opened, table


@pytest.mark.parametrize("shot", [39915, 41672])
def test_the_measured_mu_i_is_filled_through_the_lazy_view(monkeypatch, shot):
    ods, opened, table = _table(monkeypatch, shot)
    assert "magnetics.diamagnetic_flux.0.data" in ods
    assert all("magnetics" in paths for paths in opened)
    assert np.isfinite(table["virial_mui_measured"].to_numpy(dtype=float)).any()


def test_a_shot_without_magnetics_loses_only_the_measured_flux(monkeypatch):
    """48224 stores an equilibrium but no magnetics: the 404 costs nothing else.

    The reconstructed diamagnetic flux and every equilibrium-derived virial
    column must equal what the full in-memory ODS gives.
    """
    ods, _, table = _table(monkeypatch, 48224)
    assert "magnetics" not in ods
    offline = pd.DataFrame(extract_equilibrium_global(copy.deepcopy(ods), 48224))
    assert np.isfinite(table["dia_flux_Wb"].to_numpy(dtype=float)).all()
    assert np.isfinite(table["virial_beta"].to_numpy(dtype=float)).all()
    assert np.isnan(table["virial_mui_measured"].to_numpy(dtype=float)).all()
    columns = [column for column in offline.columns if column in table.columns]
    pd.testing.assert_frame_equal(
        table[columns].reset_index(drop=True), offline[columns], check_dtype=False
    )


def test_a_magnetics_read_failure_other_than_404_is_not_hidden():
    """A 403 or a dropped connection is not an ODS without a diamagnetic loop."""
    ods = vaft.omas.sample_ods(39915)
    from omas import ODS

    class Forbidden(ODS):
        def __contains__(self, location):
            if not self.location and _head(location) == "magnetics":
                raise OSError(403, "Forbidden")
            return super().__contains__(location)

    view = Forbidden(imas_version=ods.imas_version, consistency_check=False)
    view["equilibrium"] = copy.deepcopy(ods["equilibrium"])
    with pytest.raises(OSError, match="Forbidden"):
        vaft.omas.update_equilibrium_constraints_diamagnetic_flux(view)
