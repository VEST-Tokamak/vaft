"""Data shapes for :func:`vaft.help`: machine-readable first, rendered second.

A :class:`HelpPage` holds plain strings and tuples, so :meth:`HelpPage.as_dict`
can go straight to JSON and tests can read fields instead of scraping text.
The text and Markdown renderings are derived from those fields only.

Standard library only: this module is imported by ``vaft.help`` itself and
by ``vaft help --help``, neither of which may pull in a scientific subsystem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["DEFAULT_KINDS", "Default", "HelpPage", "Section", "Topic"]

#: The four kinds of default #1203 asks help to keep apart, in display order.
#: ``scientific`` entries are listed so a reader can see that the choice is
#: *not* a convenience default: VAFT takes it explicitly or from the data.
DEFAULT_KINDS = {
    "runtime": "Runtime",
    "data-access": "Data access",
    "presentation": "Presentation",
    "scientific": "Scientific choices",
}


@dataclass(frozen=True)
class Default:
    """One default value, what kind of default it is, and why it holds."""

    name: str
    value: str
    kind: str
    because: str = ""

    def __post_init__(self):
        if self.kind not in DEFAULT_KINDS:
            raise ValueError(f"unknown default kind {self.kind!r}; choose from {tuple(DEFAULT_KINDS)}")


@dataclass(frozen=True)
class Section:
    """A titled table of ``(label, value)`` rows, e.g. per-category counts."""

    title: str
    rows: tuple[tuple[str, str], ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Topic:
    """Static registry entry: routing metadata only, no subsystem content.

    ``provider`` is an import string ``"module:function"`` resolved only when
    the topic is asked for; ``item`` names the function that answers
    ``vaft.help(topic, item)`` the same way, or is empty when the topic has no
    per-item view.
    """

    name: str
    summary: str
    provider: str = ""
    item: str = ""
    entry_points: tuple[str, ...] = ()
    cli: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    see_also: tuple[str, ...] = ()


@dataclass(frozen=True)
class HelpPage:
    """What ``vaft.help(topic)`` returns.

    Print it (or let the REPL show it) for text, display it in a notebook for
    Markdown, or call :meth:`as_dict` for data.
    """

    topic: str
    summary: str
    entry_points: tuple[str, ...] = ()
    defaults: tuple[Default, ...] = ()
    sections: tuple[Section, ...] = ()
    optional: tuple[tuple[str, str], ...] = ()
    cli: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    see_also: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    # -- data ---------------------------------------------------------------
    def as_dict(self) -> dict:
        data = asdict(self)
        data["sections"] = [
            {"title": s.title, "rows": [list(row) for row in s.rows], "note": s.note}
            for s in self.sections
        ]
        data["optional"] = [list(row) for row in self.optional]
        for key in ("entry_points", "cli", "setup", "see_also", "warnings"):
            data[key] = list(data[key])
        return data

    def _default_groups(self):
        for kind, title in DEFAULT_KINDS.items():
            rows = [d for d in self.defaults if d.kind == kind]
            if rows:
                yield title, rows

    # -- text ---------------------------------------------------------------
    def render(self) -> str:
        heading = "VAFT" if self.topic == "overview" else f"VAFT {self.topic}"
        lines = [heading, "", self.summary]

        def block(title: str, rows: list[tuple[str, str]], note: str = "") -> None:
            lines.extend(["", title])
            width = max((len(label) for label, _ in rows), default=0)
            for label, value in rows:
                lines.append(f"  {label.ljust(width)}  {value}".rstrip())
            if note:
                lines.append(f"  {note}")

        if self.entry_points:
            block("Entry points", [(e, "") for e in self.entry_points])
        if self.defaults:
            lines.extend(["", "Defaults"])
            for title, rows in self._default_groups():
                lines.append(f"  {title}")
                width = max(len(d.name) for d in rows)
                for d in rows:
                    lines.append(f"    {d.name.ljust(width)}  {d.value}")
                    if d.because:
                        lines.append(f"    {' ' * width}  because: {d.because}")
        for section in self.sections:
            block(section.title, list(section.rows), section.note)
        if self.optional:
            block("Optional", list(self.optional))
        if self.cli:
            block("CLI", [(c, "") for c in self.cli])
        if self.setup:
            block("Setup", [(s, "") for s in self.setup])
        if self.see_also:
            block("Next", [(s, "") for s in self.see_also])
        if self.warnings:
            block("Warnings", [(w, "") for w in self.warnings])
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()

    # -- notebook -----------------------------------------------------------
    def _repr_markdown_(self) -> str:
        heading = "VAFT" if self.topic == "overview" else f"VAFT `{self.topic}`"
        out = [f"### {heading}", "", _md(self.summary)]

        def bullets(title: str, items) -> None:
            out.extend(["", f"**{title}**", ""])
            out.extend(f"- `{item}`" for item in items)

        def table(title: str, rows, note: str = "") -> None:
            out.extend(["", f"**{title}**", "", "| | |", "|---|---|"])
            out.extend(f"| {_md(a)} | {_md(b)} |" for a, b in rows)
            if note:
                out.extend(["", _md(note)])

        if self.entry_points:
            bullets("Entry points", self.entry_points)
        if self.defaults:
            out.extend(["", "**Defaults**", "", "| kind | name | value | because |", "|---|---|---|---|"])
            for title, rows in self._default_groups():
                out.extend(
                    f"| {title} | {_md(d.name)} | {_md(d.value)} | {_md(d.because)} |" for d in rows
                )
        for section in self.sections:
            table(section.title, section.rows, section.note)
        if self.optional:
            table("Optional", self.optional)
        if self.cli:
            bullets("CLI", self.cli)
        if self.setup:
            bullets("Setup", self.setup)
        if self.see_also:
            bullets("Next", self.see_also)
        if self.warnings:
            out.extend(["", "**Warnings**", ""])
            out.extend(f"- {_md(w)}" for w in self.warnings)
        return "\n".join(out)


def _md(text: str) -> str:
    """Table-safe Markdown: ``<placeholder>`` survives outside code spans."""
    parts = str(text).replace("\n", " ").split("`")
    for index in range(0, len(parts), 2):  # outside backticks only
        parts[index] = parts[index].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return "`".join(parts).replace("|", "\\|")
