"""The VEST data platform as a database-centred scientific workflow (#1550).

``vest_data_platform``
    the reference architecture: the VEST machine outside a server boundary
    holding experimental data processing (left), the per-shot database
    (centre), reconstruction and physics inference (above) and simulation
    (right); access and analysis with VAFT below the server, and the
    cross-platform, scalable execution strip under everything;
``vest_data_platform_overview``
    the compact companion for papers and slides: experiment, processing,
    database, reconstruction and simulation, access and analysis.

The scientific content is declared once, in the module-level constants
below; the builders own layout and styling and read the constants, so the
figure is edited by editing data. Backend and storage implementation names
(the database technology, the workflow engine) stay out of the figure.

The VEST machine is the CAD render packaged as
``vaft/diagram/images/vest_machine.jpg``, embedded in the SVG.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ._concept import Box, band, box as _concept_box, database as _concept_database, escape_latex
from ._render import Diagram
from ._scene import Arrow, Image, Label, Polyline, Scene

# ---------------------------------------------------------------------------
# content: edit these, not the drawing code
# ---------------------------------------------------------------------------

SERVER_TITLE = "VEST DATA PLATFORM SERVER"

#: left: what turns raw machine signals into qualified experimental data
EXPERIMENTAL_PROCESSING: Tuple[str, ...] = (
    "Machine Model & History", "Signal Processing", "Quality & Validation",
    "Fault & Anomaly Detection", "Shot Classification", "Event Detection",
)

#: centre: one shot's files -- experimental, reconstructed state, physics products -- each group open-ended
DATABASE_ROOT = "{shot}/"
DATABASE_MASTER = "master.h5"
DATABASE_LAYOUT: Tuple[Tuple[str, ...], ...] = (
    ("magnetics.h5", "thomson_scattering.h5", "camera_visible.h5", "..."),
    ("equilibrium.h5", "core_profiles.h5", "..."),
    ("mhd_linear.h5", "core_transport.h5", "..."),
)

#: above: inference of the experimental plasma state, then quantities derived from it
RECONSTRUCTION: Tuple[str, ...] = ("Eddy Current Model", "Magnetic EFIT", "Profile Fitting",
                                   "Plasma Parameter Inference", "Kinetic EFIT")
DERIVED_PHYSICS: Tuple[str, ...] = ("Vacuum Field Proxies", "MHD Parameters", "Synthetic Diagnostics",
                                    "Coordinate Conversion", "Power Balance")

#: right: forward simulation by domain, as (concept, code or model authors) -- the concept first
SIMULATION: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "Equilibrium": (("Fixed Boundary", "CHEASE"), ("Free Boundary", "TokaMaker"),
                    ("Analytic GS", "Solov'ev · Guazzotto & Freidberg")),
    "Stability": (("Ideal", "DCON"), ("Resistive", "RDCON")),
    "3D Response & Topology": (("Plasma Response", "GPEC"), ("Field-Line Following", "FLARE")),
    "Transport": (("Classical", "Braginskii"), ("Neoclassical", "NEO / Sauter & Redl"),
                  ("Turbulent", "TGLF / CGYRO")),
}

#: below the server: equivalent user-facing access points, and what they are used for
ACCESS: Tuple[str, ...] = ("Python API", "CLI", "GUI", "MCP", "Documentation")
ACCESS_CAPTION = ("Data Access · Search · Visualization · Comparison · Statistics · Export · Tutorials · "
                  "Research Archive")

#: the foundation: two separate dimensions, never merged into one list
EXECUTION_TITLE = "CROSS-PLATFORM & SCALABLE EXECUTION"
EXECUTION_PLATFORMS: Tuple[str, ...] = ("Windows", "macOS", "Linux")
EXECUTION_BACKENDS: Tuple[str, ...] = ("Local", "HPC / Cluster (Slurm)")

#: names the figure must not show: backend and storage implementation details
IMPLEMENTATION_TERMS: Tuple[str, ...] = ("OMAS", "REST", "Snakemake", "h5pyd", "FileDB")
#: the one place the figure names the database technology: a muted caption under its title
DATABASE_TECHNOLOGY: Tuple[str, ...] = ("IMAS", "HDF5", "HSDS")
#: the drum's rim half-height [cm]: a thin rim leaves the body for the shot directory
_DRUM_RIM = 0.55

#: the companion figure's five stages
OVERVIEW_STAGES: Tuple[str, ...] = ("Experiment", "Experimental Data Processing", "Database",
                                    "Reconstruction & Simulation", "Access & Analysis with VAFT")

#: the VEST machine: a CAD render packaged under ``vaft/diagram/images``, and its pixel aspect (height / width)
MACHINE_IMAGE = "vest_machine.jpg"
MACHINE_ASPECT = 554.0 / 361.0


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _leaf(x, y, w, h, text, role, *, latex=False, style="concept leaf") -> Box:
    return _concept_box(x, y, w, h, text, style=style, text_style="concept plain", role=role, latex=latex)


def _bold(text: str) -> str:
    return "\\textbf{" + escape_latex(text) + "}"


def _machine(cx: float, cy: float, height: float, items: List) -> Tuple[float, float, float, float]:
    """The VEST render centred at ``(cx, cy)``, ``height`` cm tall; its extent (x0, x1, y0, y1)."""
    width = height / MACHINE_ASPECT
    items.append(Image((cx, cy), MACHINE_IMAGE, width, height, role="machine"))
    return (cx - 0.5 * width, cx + 0.5 * width, cy - 0.5 * height, cy + 0.5 * height)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _panel(x0, x1, y0, y1, title, role, items: List, *, style="concept group") -> Box:
    items.append(Polyline.of([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], style, role=role, closed=True))
    items.append(Label((0.5 * (x0 + x1), y1 - 0.12), "\\textbf{" + escape_latex(title) + "}",
                       "concept group title", anchor="north", role=role))
    return Box(0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0, ())


def _stack(names: Sequence[str], x, y_top, w, h, gap, role, items: List) -> List[Box]:
    boxes = []
    for i, name in enumerate(names):
        b = _leaf(x, y_top - i * (h + gap) - 0.5 * h, w, h, name, f"{role}:{name}")
        items += list(b.items)
        boxes.append(b)
    return boxes


def _database_text() -> str:
    rows = ["\\textbf{" + escape_latex(DATABASE_ROOT) + "}", escape_latex(DATABASE_MASTER)]
    lines = []
    for group in DATABASE_LAYOUT:
        lines.append("\\hline ")
        lines += [escape_latex(name) + "\\\\" for name in group]
    body = "\\\\".join(rows) + "\\\\" + "".join(lines)
    return "{\\ttfamily\\small\\begin{tabular}{l}" + body + "\\end{tabular}}"


#: heights of a simulation domain title, an entry, and the gaps [cm]
_SIM_TITLE, _SIM_ENTRY, _SIM_GAP, _SIM_DOMAIN_GAP = 0.6, 1.0, 0.15, 0.35


def _simulation_height(column) -> float:
    return sum(_SIM_TITLE + len(entries) * (_SIM_ENTRY + _SIM_GAP) + _SIM_DOMAIN_GAP for _, entries in column)


def _simulation(x0, x1, y_top, items: List) -> Dict[str, List[Box]]:
    """Two columns of domains; each entry the concept in bold, its code or model authors muted beneath."""
    domains = list(SIMULATION.items())
    columns = (domains[:2], domains[2:])
    width = 0.5 * (x1 - x0) - 0.4
    placed: Dict[str, List[Box]] = {}
    for c, column in enumerate(columns):
        x = x0 + 0.25 + width / 2 + c * (width + 0.2)
        y = y_top
        for domain, entries in column:
            items.append(Label((x, y), "\\textbf{" + escape_latex(domain) + "}", "concept plain",
                               anchor="north", role=f"simulation:{domain}"))
            y -= _SIM_TITLE
            boxes = []
            for concept, implementation in entries:
                text = _bold(concept) + "\\\\{\\footnotesize\\color{black!60}" + escape_latex(implementation) + "}"
                b = _leaf(x, y - 0.5 * _SIM_ENTRY, width, _SIM_ENTRY, text,
                          f"simulation:{domain}:{implementation}", latex=True)
                items += list(b.items)
                boxes.append(b)
                y -= _SIM_ENTRY + _SIM_GAP
            placed[domain] = boxes
            y -= _SIM_DOMAIN_GAP
    return placed


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

#: the cross: a centre row (processing | database | simulation) about y = 0, inference above, access below [cm]
_PROC_X = (4.6, 10.6)
_DB_X = (14.2, 18.8)
_SIM_X = (22.4, 33.2)
_DB_HALF = 3.95              # half-height of the database drum
_RECO_H = 5.3                # height of the inference panel
_GAP = 1.6                   # between the row and the panel above it


def vest_data_platform(*, labels: bool = True) -> Diagram:
    r"""The VEST data platform: a per-shot database at the centre of experiment, inference and simulation.

    A cross about the database: experimental data processing
    (``EXPERIMENTAL_PROCESSING``) on the left, fed by the VEST machine
    outside the server; reconstruction and physics inference above;
    simulation (``SIMULATION``: each entry a concept with its code or model
    authors beneath) on the right; access and analysis with VAFT
    (``ACCESS``) below the server, and the cross-platform, scalable execution
    strip (``EXECUTION_*``) under everything. The database panel shows one
    shot's directory (``DATABASE_LAYOUT``: experimental, reconstructed state
    and physics products, each open-ended). No storage or workflow-engine
    names are drawn.
    """
    labels = _check_labels(labels)
    items: List = []
    edges: List[Tuple[str, str, str]] = []
    dbx = 0.5 * sum(_DB_X)
    # centre row, all centred on y = 0
    # the database is one large drum holding the shot directory
    drum = _concept_database(dbx, 0.0, _DB_X[1] - _DB_X[0], 2.0 * _DB_HALF, "", role="database", ry=_DRUM_RIM)
    items += [it for it in drum.items if not isinstance(it, Label)]
    top = _DB_HALF - 2.0 * _DRUM_RIM
    items += [Label((dbx, top - 0.15), "{\\Large\\textbf{Database}}", "concept group title", anchor="north",
                    role="database"),
              Label((dbx, top - 0.95), escape_latex(" · ".join(DATABASE_TECHNOLOGY)), "concept annotation",
                    anchor="north", role="database:technology"),
              Label((dbx, top - 1.55), _database_text(), "concept plain", anchor="north", role="database:layout")]
    db_panel = Box(dbx, 0.0, drum.width, drum.height, ())
    proc_h, proc_gap = 0.66, 0.16
    proc_half = 0.5 * (0.75 + len(EXPERIMENTAL_PROCESSING) * (proc_h + proc_gap) + 0.3)
    proc_panel = _panel(_PROC_X[0], _PROC_X[1], -proc_half, proc_half, "Experimental Data Processing",
                        "processing", items)
    proc = _stack(EXPERIMENTAL_PROCESSING, proc_panel.x, proc_half - 0.75, _PROC_X[1] - _PROC_X[0] - 0.5, proc_h,
                  proc_gap, "processing", items)
    sim_half = 0.5 * (max(_simulation_height(list(SIMULATION.items())[:2]),
                          _simulation_height(list(SIMULATION.items())[2:])) + 0.75)
    sim_panel = _panel(_SIM_X[0], _SIM_X[1], -sim_half, sim_half, "Simulation", "simulation", items)
    sim = _simulation(_SIM_X[0], _SIM_X[1], sim_half - 0.75, items)
    # inference, above the database and centred on it
    rw = 11.6
    ry0 = max(_DB_HALF, sim_half, proc_half) + _GAP
    reco_panel = _panel(dbx - 0.5 * rw, dbx + 0.5 * rw, ry0, ry0 + _RECO_H, "Reconstruction & Physics Inference",
                        "inference", items)
    half = 0.5 * rw
    for g, (title, names) in enumerate((("Reconstruction", RECONSTRUCTION), ("Derived Physics", DERIVED_PHYSICS))):
        gx = dbx - 0.5 * rw + half * (g + 0.5)
        items.append(Label((gx, ry0 + _RECO_H - 0.75), "\\textbf{" + escape_latex(title) + "}", "concept plain",
                           anchor="north", role=f"inference:{title}"))
        _stack(names, gx, ry0 + _RECO_H - 1.4, half - 0.7, 0.58, 0.14, f"inference:{title}", items)
    items.append(Polyline.of([(dbx, ry0 + 0.3), (dbx, ry0 + _RECO_H - 0.8)], "concept tick",
                             role="inference:divider"))
    # the server frame around the cross's upper four arms
    server = (_PROC_X[0] - 0.6, _SIM_X[1] + 0.6, -max(_DB_HALF, sim_half, proc_half) - 0.7, ry0 + _RECO_H + 0.6)
    sx0, sx1, sy0, sy1 = server
    items.insert(0, Polyline.of([(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1)], "concept frame", role="server",
                                closed=True))
    # the machine, outside the server, level with the row
    machine = _machine(0.5 * (sx0 - 0.4), 0.0, 6.2, items)
    # access, below the server and centred on the database
    ay = sy0 - 1.9
    acc_x0, acc_x1 = dbx - 9.6, dbx + 9.6
    access_panel = _panel(acc_x0, acc_x1, ay - 1.75, ay + 1.15, "Access & Analysis with VAFT", "access", items)
    tw = (acc_x1 - acc_x0 - 0.6) / len(ACCESS) - 0.2
    access = []
    for i, name in enumerate(ACCESS):
        b = _leaf(acc_x0 + 0.4 + tw / 2 + i * (tw + 0.2), ay - 0.2, tw, 0.7, name, f"access:{name}",
                  style="concept actor")
        items += list(b.items)
        access.append(b)
    items.append(Label((dbx, ay - 0.75), escape_latex(ACCESS_CAPTION), "concept annotation", anchor="north",
                       role="access:caption"))
    # execution strip
    ey0, ey1 = ay - 3.6, ay - 2.2
    x_left = min(machine[0], 0.0)
    items += band(x_left, sx1, ey0, ey1, role="execution")
    mid = 0.5 * (x_left + sx1)
    items += [Label((mid, ey1 - 0.1), "\\textbf{" + escape_latex(EXECUTION_TITLE) + "}", "concept group title",
                    anchor="north", role="execution:title"),
              Label((0.5 * (x_left + mid), ey0 + 0.5), escape_latex(" · ".join(EXECUTION_PLATFORMS)),
                    "concept plain", role="execution:platforms"),
              Polyline.of([(mid, ey0 + 0.2), (mid, ey0 + 0.75)], "concept tick", role="execution:divider"),
              Label((0.5 * (mid + sx1), ey0 + 0.5), escape_latex(" · ".join(EXECUTION_BACKENDS)), "concept plain",
                    role="execution:backends")]
    # data relationships: only these
    items += [Arrow((machine[1] + 0.1, 0.0), (_PROC_X[0] - 0.1, 0.0), "connector strong",
                    role="edge:machine->processing"),
              Arrow((_PROC_X[1] + 0.1, 0.0), (_DB_X[0] - 0.1, 0.0), "connector strong",
                    role="edge:processing->database"),
              Arrow((dbx, _DB_HALF + 0.1), (dbx, ry0 - 0.1), "connector strong", role="edge:database->inference",
                    both=True),
              Arrow((_DB_X[1] + 0.1, 0.0), (_SIM_X[0] - 0.1, 0.0), "connector strong",
                    role="edge:database->simulation", both=True),
              Arrow((dbx, -_DB_HALF - 0.1), (dbx, ay + 1.15 + 0.1), "connector strong",
                    role="edge:database->access")]
    edges += [("machine", "processing", "forward"), ("processing", "database", "forward"),
              ("database", "inference", "both"), ("database", "simulation", "both"),
              ("database", "access", "forward")]
    if labels:
        items += [Label((sx0 + 0.2, sy1 - 0.15), "\\textbf{" + escape_latex(SERVER_TITLE) + "}",
                        "concept band label", anchor="north west", role="server:title"),
                  Label((0.5 * (machine[0] + machine[1]), machine[2] - 0.1), "\\textbf{VEST}", "concept plain",
                        anchor="north", role="machine:title")]
    else:
        items = [it for it in items if not isinstance(it, Label)]
    model = {"edges": tuple(edges), "server": server, "machine_extent": machine,
             "boxes": {"processing": proc_panel, "database": db_panel, "inference": reco_panel,
                       "simulation": sim_panel, "access": access_panel},
             "simulation": {k: tuple(b.center for b in v) for k, v in sim.items()},
             "processing": tuple(b.center for b in proc), "access": tuple(b.center for b in access),
             "execution": (ey0, ey1)}
    return Diagram("vest_data_platform", Scene(tuple(items)), model=model)


def vest_data_platform_overview(*, labels: bool = True) -> Diagram:
    r"""The VEST data platform in five stages, for papers, slides and introductions.

    Experiment (the VEST machine) $\to$ experimental data processing $\to$
    the per-shot database $\leftrightarrow$ reconstruction and simulation
    $\to$ access and analysis with VAFT. No file contents or sub-items: the
    reference view is ``vest_data_platform``.
    """
    labels = _check_labels(labels)
    items: List = []
    machine = _machine(1.4, 0.0, 4.0, items)
    proc = _leaf(5.6, 0.0, 3.6, 1.3, _bold("Experimental Data Processing"), "stage:processing", latex=True,
                 style="concept box")
    db = _concept_database(10.6, 0.0, 3.0, 1.9, "\\textbf{Database}\\\\{\\small per-shot}", role="stage:database",
                           text_style="concept plain", latex=True)
    reco = _leaf(10.6, 3.0, 4.6, 1.2, _bold("Reconstruction & Physics Inference"), "stage:reconstruction",
                 latex=True, style="concept box")
    sim = _leaf(15.9, 0.0, 3.2, 1.3, _bold("Simulation"), "stage:simulation", latex=True, style="concept box")
    acc = _leaf(10.6, -3.0, 4.6, 1.2, _bold("Access & Analysis with VAFT"), "stage:access", latex=True,
                style="concept actor")
    for b in (proc, db, reco, sim, acc):
        items += list(b.items)
    items += [Arrow((machine[1] + 0.1, 0.0), (proc.x - 0.5 * proc.width - 0.1, 0.0), "connector strong",
                    role="edge:machine->processing"),
              Arrow((proc.x + 0.5 * proc.width + 0.1, 0.0), (db.x - 0.5 * db.width - 0.1, 0.0), "connector strong",
                    role="edge:processing->database"),
              Arrow((db.x, 0.5 * db.height + 0.1), (reco.x, reco.y - 0.5 * reco.height - 0.1), "connector strong",
                    role="edge:database->reconstruction", both=True),
              Arrow((db.x + 0.5 * db.width + 0.1, 0.0), (sim.x - 0.5 * sim.width - 0.1, 0.0), "connector strong",
                    role="edge:database->simulation", both=True),
              Arrow((db.x, -0.5 * db.height - 0.1), (acc.x, acc.y + 0.5 * acc.height + 0.1), "connector strong",
                    role="edge:database->access")]
    edges = (("machine", "processing", "forward"), ("processing", "database", "forward"),
             ("database", "reconstruction", "both"), ("database", "simulation", "both"),
             ("database", "access", "forward"))
    if labels:
        items.append(Label((1.4, machine[2] - 0.05), "\\textbf{VEST}", "concept plain", anchor="north",
                           role="machine:title"))
    else:
        items = [it for it in items if not isinstance(it, Label)]
    return Diagram("vest_data_platform_overview", Scene(tuple(items)),
                   model={"edges": edges, "stages": OVERVIEW_STAGES, "machine_extent": machine})
