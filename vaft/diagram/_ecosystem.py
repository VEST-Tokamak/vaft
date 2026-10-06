"""Software dependencies and external scientific-code integration (#1648).

``software_dependency_ecosystem``
    which capabilities are part of the core and which are activated by an
    optional extra or used only for development, grouped by capability role;
``external_code_integration``
    how an independently maintained scientific code becomes a reproducible
    VAFT capability: upstream project, installation boundary, provenance,
    environment resolution, the ``vaft.code`` adapter, execution, the
    solver-native result, mapping and the standardized state.

Both read :mod:`vaft._ecosystem`, the registry the external-code reference
and the dependency reference are generated from, so the figures and the
pages cannot disagree.  Neither shows a version: those belong to
``pyproject.toml``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .. import _ecosystem as ecosystem
from ._concept import box, connector, escape_latex
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

#: (scope, column title, install command as LaTeX)
_SCOPES = (
    ("runtime", "REQUIRED CORE", r"pip install vaft"),
    ("optional", "OPTIONAL CAPABILITIES", r"pip install vaft[\textit{extra}]"),
    ("development", "DEVELOPMENT AND DOCUMENTATION", r"pip install -e .[dev]"),
)

_INSTALLATION_TITLES = ecosystem.INSTALLATION_LABELS


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _members(scope: str) -> Dict[str, List[str]]:
    """Capability id -> what represents it: packages for the core, extras otherwise."""
    members: Dict[str, List[str]] = {c.id: [] for c in ecosystem.CAPABILITIES if c.scope == scope}
    if scope == "runtime":
        for package, (capability, _why) in ecosystem.RUNTIME_ROLES.items():
            members[capability].append(package)
    else:
        for extra, (capability, _why) in ecosystem.EXTRA_ROLES.items():
            if capability in members:
                members[capability].append(f"vaft[{extra}]")
    return members


def _lines(text: str, width: float, per_cm: float) -> int:
    return max(1, -(-len(text) // max(1, int(width * per_cm))))


def _height(title: str, listing: str, width: float) -> float:
    """A box height for a bold ``title`` line(s) over a small ``listing`` wrapped to ``width`` cm."""
    return 0.35 + 0.45 * _lines(title, width - 0.3, 4.6) + 0.4 * _lines(listing, width - 0.3, 5.6)


def software_dependency_ecosystem(*, labels: bool = True) -> Diagram:
    r"""Which software capabilities are core, and which an extra or development install activates.

    VAFT at the top; below it three columns -- the required core, the
    optional capabilities, development and documentation -- each listing its
    capability roles with the packages (core) or extras (otherwise) that
    provide them.  Capability is the emphasis, not the package: optional means
    only that the core works without it, and packaging status is not a
    scientific ranking.  Versions are deliberately absent.
    """
    labels = _check_labels(labels)
    col_w, gap = 6.2, 0.7
    xs = [i * (col_w + gap) for i in range(3)]
    top = 0.0
    root = box(xs[1], top + 1.6, 3.2, 0.9, r"\textbf{VAFT}", role="vaft", latex=True)
    items: List = list(root.items)
    model: Dict[str, Sequence[str]] = {}
    for x, (scope, title, command) in zip(xs, _SCOPES):
        head = box(x, top, col_w, 0.9, rf"\textbf{{{escape_latex(title)}}}", style="concept strong",
                   text_style="concept plain", role=f"scope:{scope}", latex=True)
        items += list(head.items) + [connector(root, head, role=f"scope:{scope}")]
        if labels:
            items.append(Label((x, top - 0.75), command, "concept annotation", role=f"scope:{scope}:install"))
        y = top - 1.35
        capabilities = [c for c in ecosystem.CAPABILITIES if c.scope == scope]
        members = _members(scope)
        for capability in capabilities:
            listing = ", ".join(members[capability.id])
            text = rf"\textbf{{{escape_latex(capability.title)}}}\\{{\small {escape_latex(listing)}}}"
            h = _height(capability.title, listing, col_w)
            b = box(x, y - 0.5 * h, col_w, h, text, style="concept leaf", text_style="concept plain",
                    role=f"capability:{capability.id}", latex=True)
            items += list(b.items)
            y -= h + 0.25
        model[scope] = tuple(c.id for c in capabilities)
    if labels:
        bottom = min(item.at[1] for item in items if isinstance(item, Label)) - 1.1
        items.append(Label((xs[1], bottom),
                           "Optional is not experimental, and required is not more important: a capability is "
                           "optional when the core works without it. Versions: pyproject.toml.",
                           "note,text width=18cm", anchor="north", role="note"))
    return Diagram("software_dependency_ecosystem", Scene(tuple(items)), model=dict(model))


def external_code_integration(*, labels: bool = True) -> Diagram:
    r"""How an independently maintained scientific code becomes a reproducible VAFT capability.

    The integration lifecycle top to bottom: the upstream project, obtaining
    its source or package, the installation boundary, installation
    provenance and checker, ``{CODE}HOME`` resolution, the ``vaft.code``
    adapter, local or scheduler-backed execution, the solver-native result,
    mapping, and the standardized IMAS/ODS/FileDB state.  The upstream
    project owns the solver; VAFT owns the integration boundary.  Beside the
    installation step, the current codes grouped by who installs them.
    """
    labels = _check_labels(labels)
    w, h, step = 8.0, 0.85, 1.35
    boxes = {}
    items: List = []
    y = 0.0
    for key, text in ecosystem.INTEGRATION_LIFECYCLE:
        if key == "execute":
            local = box(-2.1, y, 3.8, h, "local subprocess", text_style="concept plain", role="lifecycle:execute:local")
            hpc = box(2.1, y, 3.8, h, "Slurm or remote Slurm", text_style="concept plain",
                      role="lifecycle:execute:slurm")
            items += list(local.items) + list(hpc.items)
            boxes[key] = (local, hpc)
        else:
            style = "concept strong" if key in {"adapter", "standard"} else "concept box"
            b = box(0.0, y, w, h, text, style=style, text_style="concept plain", role=f"lifecycle:{key}")
            items += list(b.items)
            boxes[key] = (b,)
        y -= step
    keys = [key for key, _ in ecosystem.INTEGRATION_LIFECYCLE]
    for upper, lower in zip(keys, keys[1:]):
        for a in boxes[upper]:
            for b in boxes[lower]:
                items.append(connector(a, b, role=f"flow:{upper}->{lower}"))
    # ownership brackets on the left
    def bracket(first: str, last: str, text: str, role: str) -> None:
        y0 = boxes[first][0].y + 0.5 * h
        y1 = boxes[last][0].y - 0.5 * h
        x = -0.5 * w - 0.45
        items.append(Polyline.of([(x + 0.2, y0), (x, y0), (x, y1), (x + 0.2, y1)], "connector line", role=role))
        items.append(Label((x - 0.2, 0.5 * (y0 + y1)), text, "concept group title,text width=2.6cm,align=right,"
                           "execute at begin node={\\hyphenpenalty=10000}",
                           anchor="east", role=role))
    bracket("upstream", "obtain", "The upstream project owns the solver", "owner:upstream")
    bracket("install", "mapping", "VAFT owns the integration boundary", "owner:vaft")
    # the codes, by who installs them
    install = boxes["install"][0]
    side_x = 0.5 * w + 4.4
    groups: Dict[str, List[str]] = {}
    for code in ecosystem.EXTERNAL_CODES:
        groups.setdefault(code.installation, []).append(code.name)
    gy = install.y + 1.9
    for installation in ("vaft_managed_source_build", "python_package", "site_managed", "reader_only"):
        names = groups.get(installation, [])
        if not names:
            continue
        text = (rf"\textbf{{{escape_latex(_INSTALLATION_TITLES[installation])}}}\\"
                rf"{{\small {escape_latex(', '.join(names))}}}")
        bh = _height(_INSTALLATION_TITLES[installation], ", ".join(names), 6.6)
        g = box(side_x, gy - 0.5 * bh, 6.6, bh, text, style="concept leaf", text_style="concept plain",
                role=f"installation:{installation}", latex=True)
        items += list(g.items)
        items.append(Arrow((side_x - 3.35, gy - 0.5 * bh), (0.5 * w + 0.1, install.y), "connector line",
                           role=f"installation:{installation}"))
        gy -= bh + 0.3
    if labels:
        items.append(Label((0.0, y + 0.2), "No step is skipped: a native result is never IMAS without an explicit "
                           "mapping, and the native result stays available for validation.",
                           "note,text width=14cm", anchor="north", role="note"))
    model = {"lifecycle": tuple(keys),
             "installation": {k: tuple(v) for k, v in sorted(groups.items())}}
    return Diagram("external_code_integration", Scene(tuple(items)), model=model)
