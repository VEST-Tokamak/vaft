"""``DDCollection``: a keyed, lazy aggregation of DDs (#1127).

The successor of the *collection role* of OMAS ``ODC``: many Data Entries,
keyed by whatever identifies them -- a shot, a ``(machine, shot)`` pair, a
simulation run (#1167). The key is opaque here on purpose.

A collection carries no scientific intent: it is what was loaded or what a
query matched. Reproducible scientific organisation is ``Study`` (#1165), a
separate layer, and one DD's alternative instances are views of that DD, never
members of a collection.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Iterable, Iterator, Mapping

from ._dd import DD

Member = DD | Callable[[], DD]


class DDCollection(Mapping[Hashable, DD]):
    """Mapping of key -> DD, materialising factory members on first access."""

    def __init__(
        self,
        members: Mapping[Hashable, Member] | Iterable[tuple[Hashable, Member]] = (),
        *,
        owns: bool = True,
    ):
        """``owns=False`` makes a non-owning collection whose :meth:`close` is a
        no-op -- what :meth:`select` returns, since its members stay the
        parent's."""
        self._owns = owns
        items = members.items() if isinstance(members, Mapping) else members
        self._members: dict[Hashable, Member] = {}
        for key, member in items:
            if key in self._members:
                raise ValueError(f"key {key!r} given twice")
            if not (isinstance(member, DD) or callable(member)):
                raise TypeError(f"member {key!r} must be a DD or a zero-argument DD factory")
            self._members[key] = member
        self._loaded: dict[Hashable, DD] = {
            key: member for key, member in self._members.items() if isinstance(member, DD)
        }

    def __getitem__(self, key: Hashable) -> DD:
        loaded = self._loaded.get(key)
        if loaded is not None:
            return loaded
        try:
            member = self._members[key]
        except KeyError:
            raise KeyError(f"no member {key!r}; keys: {list(self._members)}") from None
        dd = member()
        if not isinstance(dd, DD):
            raise TypeError(f"the factory for {key!r} returned {type(dd).__name__}, not DD")
        self._loaded[key] = dd
        return dd

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def loaded_keys(self) -> tuple[Hashable, ...]:
        """Keys whose DD has been materialised."""
        return tuple(key for key in self._members if key in self._loaded)

    def available_ids(self) -> tuple[str, ...]:
        """Union of IDS names over every member (materialises all of them)."""
        names: set[str] = set()
        for key in self._members:
            names.update(self[key].available_ids())
        return tuple(sorted(names))

    def select(self, *, has: Iterable[str] = ()) -> "DDCollection":
        """Members that store every IDS in ``has`` (materialises all of them).

        The result shares this collection's DDs and does not own them:
        closing it closes nothing; close the parent.
        """
        required = tuple(has)
        chosen = []
        for key in self._members:
            dd = self[key]
            if all(dd.has(ids) for ids in required):
                chosen.append((key, dd))
        return DDCollection(chosen, owns=False)

    def map(self, function: Callable[[DD], Any]) -> dict[Hashable, Any]:
        """``{key: function(dd)}`` over every member, in key order."""
        return {key: function(self[key]) for key in self._members}

    def close(self) -> None:
        """Close every materialised member; unmaterialised factories are dropped.

        A non-owning collection (from :meth:`select`) closes nothing.
        """
        if not self._owns:
            return
        failure: BaseException | None = None
        for dd in self._loaded.values():
            try:
                dd.close()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                failure = failure or exc
        if failure is not None:
            raise failure

    def __enter__(self) -> "DDCollection":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<DDCollection {len(self)} entries, {len(self._loaded)} loaded>"


__all__ = ["DDCollection"]
