"""Workflow specification against execution, and a verified IMAS/VAFT stage chain (#2150).

``workflow_specification_and_run``
    one workflow definition instantiated by an orchestrator into many runs,
    each leaving scientific outputs, logs and structured provenance as
    different artifact classes;
``imas_scientific_workflow_example``
    the routine pipeline's EFIT -> CHEASE -> MHD stability chain as the code
    implements it: which stage reads native files and which publishes IDS,
    the configuration each stage takes, the digests that link the stages and
    the per-stage sidecar records.

The visual grammar and the ``Graph`` recorder are those of
``vaft.diagram._computation_contract``.
"""

from __future__ import annotations

from typing import List

from ._computation_contract import RELATIONS, Graph
from ._render import Diagram
from ._scene import Arrow, Label, Polyline
from ._vaft_concepts import _check_labels

#: the pipeline that the example chain is read from
PIPELINE = "workflow/automatic_pipeline_1_routine_data_processing"


def workflow_specification_and_run(*, labels: bool = True) -> Diagram:
    r"""A workflow specification instantiated into separate runs, each with its own artifact classes (#2150).

    The specification carries an identity and definition version, its
    components and dependencies (A -> B -> C), input and output expectations
    and parameter definitions. An orchestrator -- Snakemake or plain Python,
    with scheduling and resources -- instantiates it; one definition gives
    many runs, each with its actual inputs and resolved configuration.
    Every run leaves three different artifact classes: scientific outputs,
    logs and structured provenance; a log is not provenance.

    Each run should reference the definition it instantiates. Today
    Snakemake records the job graph and logs and VAFT records stage
    manifests, artifact digests and replication records, but no record links
    a run to a definition version: that reference is drawn as planned.
    """
    labels = _check_labels(labels)
    g = Graph()
    x0, x1, y0, y1 = -7.6, 7.6, 3.4, 7.3
    # solid: a dashed outline means planned in these figures, and the specification exists
    g.items.append(Polyline.of([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                               "draw=islanddark!70,line width=1.0pt,rounded corners=6pt", role="specification",
                               closed=True))
    g.label((x0 + 0.25, y1 - 0.15), "\\textbf{Workflow specification}", anchor="north west",
            style="concept group title", role="specification")
    for key, x in zip("ABC", (-2.6, 0.0, 2.6)):
        g.node(f"component:{key}", x, 6.0, 1.5, 0.75, key)
    g.link("component:A", "component:B", "data")
    g.link("component:B", "component:C", "data")
    g.label((0.0, 5.15), "identity and definition version $\\cdot$ components and dependencies\\\\"
            "input and output expectations $\\cdot$ parameter definitions", anchor="north",
            style="concept plain", role="specification")
    g.status["specification"] = "implemented"
    g.evidence["specification"] = ((f"{PIPELINE}/Snakefile", "rule run_chease"),
                                   ("vaft/_pipeline_graph.py", "def pipeline_snapshot"))
    g.node("orchestration", 0.0, 1.55, 9.2, 1.2, "Workflow orchestration",
           ("Snakemake or Python · scheduling and resources",),
           evidence=(("vaft/database/worker/state.py", "runs"),))
    g.link("specification", "orchestration", "configuration", start=(0.0, y0 - 0.08), end=(0.0, 1.55 + 0.6 + 0.08))
    g.label((0.15, 0.5 * (y0 + 2.15)), "instantiated by", anchor="west", role="instantiated")
    for i, x in enumerate((-4.6, 4.6), start=1):
        side = "A" if i == 1 else "B"
        run = f"run:{i}"
        g.node(run, x, -0.75, 7.2, 1.55, f"Run {i}", (f"input {side} · configuration {side}", "execution"))
        g.link("orchestration", run, "data")
        for key, title, dx in (("outputs", "Scientific outputs", -2.45), ("logs", "Logs", 0.0),
                               ("provenance", "Structured provenance", 2.45)):
            g.node(f"{run}:{key}", x + dx, -3.25, 2.3, 1.15, title, status="implemented")
            g.link(run, f"{run}:{key}", "data")
        # the run should point back to its definition: drawn as planned, it is not recorded today
        sign = -1.0 if i == 1 else 1.0
        xr, xe = x + sign * (3.6 + 0.08), sign * 8.9
        g.link(run, "specification", "provenance", start=(xr, -0.75),
               via=[(xe, -0.75), (xe, 5.35)], end=(sign * (7.6 + 0.08), 5.35))
    g.status["reference"] = "planned"
    g.label((-9.05, 2.3), "references its definition (planned)", anchor="south",
            style="concept annotation,rotate=90", role="reference")
    g.label((9.05, 2.3), "references its definition (planned)", anchor="south",
            style="concept annotation,rotate=-90", role="reference")
    g.legend(-9.2, -4.55)
    if labels:
        g.label((0.0, -5.15), "Snakemake records the job graph and logs; VAFT records stage manifests, artifact "
                "digests and replication records. A log is not structured provenance", anchor="north",
                style="note", role="note")
    return g.diagram("workflow_specification_and_run", relations=RELATIONS, runs=2)


#: the chain's stages: (key, title, subtitle, configuration title, configuration lines, evidence)
STAGES = (
    ("efit", "EFIT", "magnetic reconstruction", "EFIT scientific configuration",
     ("profile · constraints · numerics", "Green table · scientific SHA-256"),
     ((f"{PIPELINE}/Snakefile", "rule run_efit_reconstruction"), ("vaft/code/efit/config.py", "def sha256"))),
    ("chease", "CHEASE", "equilibrium refinement", "CHEASE configuration",
     ("boundary surface · mesh · relaxation",),
     ((f"{PIPELINE}/Snakefile", "rule run_chease"), ("vaft/code/chease.py", "class CHEASEConfig"))),
    ("stability", "MHD stability", "DCON · RDCON · STRIDE · GPEC", "Solver configuration",
     ("configured modules and modes",),
     ((f"{PIPELINE}/Snakefile", "rule run_gpec_module"), ("vaft/code/gpec/_types.py", "class GPECSuiteResult"))),
)
#: what each stage leaves as native files, and the digest recorded for the link
NATIVE = (
    ("gfiles", "EFIT g-files", ("sha256 in the EFIT artifact manifest",)),
    ("refined_gfiles", "Refined g-files", ("input and output sha256", "in the CHEASE runs")),
    ("stability_native", "Native stability outputs", ("module manifest with", "input_equilibrium_sha256")),
)
#: what each stage publishes as IDS, and how
PUBLISHED = (
    ("efit_ids", "equilibrium IDS (EFIT)", "published product"),
    ("chease_ids", "equilibrium IDS (CHEASE)", "parsed from the refined g-files"),
    ("stability_ids", "mhd_linear · ntms IDS", "built from the native outputs"),
)


def imas_scientific_workflow_example(*, labels: bool = True) -> Diagram:
    r"""The routine EFIT -> CHEASE -> MHD-stability chain as implemented: native files, IDS, digests (#2150).

    Device data pass through machine-specific mapping into the scientific
    input IDSs EFIT reads: magnetics (with the plasma current), pf_active,
    pf_passive and the vacuum toroidal field, the diamagnetic flux when
    present; the machine geometry enters through the Green-function table.
    EFIT, with its scientific configuration, writes native g-files and
    publishes an equilibrium IDS. CHEASE reads the native EFIT g-files --
    not the published IDS -- and writes refined g-files, from which its own
    equilibrium IDS is parsed. The MHD stability codes (the configured set
    of DCON, RDCON, STRIDE and GPEC) read the refined g-files and write
    native outputs and a module manifest, from which the mhd_linear and ntms
    IDS are built.

    The digests that link the stages are recorded: the EFIT g-file SHA-256
    in the EFIT artifact manifest, the CHEASE input and output SHA-256 in
    its runs, and the stability input equilibrium SHA-256.
    ``vaft.database.provenance.verify_chain`` compares these recorded
    digests (it never re-hashes); it is a library check with tests and is
    not yet wired into the pipeline, so the check is drawn as partial. Each
    stage also leaves sidecar records that are not scientific IDS:
    configuration and version, log and status file, digests, stage manifest
    and replication record.
    """
    labels = _check_labels(labels)
    g = Graph()
    g.node("device", 0.0, 13.0, 5.2, 0.9, "Experiment and device")
    g.node("mapping", 0.0, 11.3, 5.2, 0.9, "Machine-specific mapping",
           evidence=((f"{PIPELINE}/Snakefile", "rule generate_diagnostics_ods"),))
    g.node("inputs", 0.0, 9.35, 9.4, 1.35, "Scientific input IDSs",
           ("magnetics (incl. plasma current) · pf_active · pf_passive · tf;", "diamagnetic flux when present"),
           evidence=(("vaft/code/efit/kfile.py", "REQUIRED_CONSTRAINT_INPUTS"),))
    g.link("device", "mapping", "data")
    g.link("mapping", "inputs", "data")
    stage_y, side_y = (7.1, 3.5, -0.1), (5.3, 1.7, -1.9)
    previous = "inputs"
    for (key, title, sub, ctitle, clines, evidence), y, (nkey, ntitle, digest), (pkey, ptitle, how), ny in zip(
            STAGES, stage_y, NATIVE, PUBLISHED, side_y):
        g.node(key, 0.0, y, 6.4, 1.15, title, (sub,), style="concept strong", evidence=evidence)
        g.node(f"configuration:{key}", 7.4, y, 6.2, 1.25, ctitle, clines, style="concept leaf")
        g.link(f"configuration:{key}", key, "configuration")
        g.link(previous, key, "data")
        g.node(nkey, -7.4, ny, 6.2, 1.4, ntitle, digest, evidence=evidence)
        g.node(pkey, 7.4, ny, 6.2, 1.2, ptitle, (how,), evidence=evidence)
        g.link(key, nkey, "data")
        g.link(key, pkey, "data")
        previous = nkey
    g.node("digest_check", -13.6, 1.7, 4.0, 3.4, "Cross-stage digest check",
           ("verify_chain compares", "recorded digests"), status="partial",
           note="library and tests; not yet in the pipeline",
           evidence=(("vaft/database/provenance.py", "def verify_chain"),))
    for nkey in ("gfiles", "refined_gfiles", "stability_native"):
        g.link(nkey, "digest_check", "provenance")
    g.node("sidecar", 0.0, -3.95, 21.0, 1.25, "For each stage: sidecar records, not scientific IDS",
           ("configuration and version · log and status file · input and output digests · stage manifest · "
            "replication record",),
           evidence=(("vaft/omas/vest_upstream.py", "def write_manifest"),
                     ("vaft/database/replication.py", "product_sha256")))
    g.label((-7.4, 6.2), "native files", anchor="south", style="concept group title", role="column")
    g.legend(-9.2, -5.25)
    if labels:
        g.label((0.0, -5.85), "CHEASE and the stability codes read the native files, not the published IDS; "
                "the IDS products are built from those files", anchor="north", style="note", role="note")
    return g.diagram("imas_scientific_workflow_example", relations=RELATIONS,
                     stages=tuple(k for k, *_ in STAGES), native=tuple(k for k, *_ in NATIVE),
                     published=tuple(k for k, *_ in PUBLISHED), pipeline=PIPELINE)
