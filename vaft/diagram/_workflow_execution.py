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

from ._computation_contract import RELATIONS, Graph
from ._render import Diagram
from ._scene import Polyline
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
    Snakemake schedules the job graph, each rule writes its log, and VAFT
    writes stage manifests, artifact digests and replication records, but no
    record links a run to the workflow definition's identity: that reference
    is drawn as planned and listed in ``model["planned_edges"]``. Only the
    scientific outputs are scientific data; logs and structured provenance
    are part of the run's record.
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
    g.record("specification", "implemented", ((f"{PIPELINE}/Snakefile", "rule run_chease:"),
                                              ("vaft/_pipeline_graph.py", "pipeline_snapshot")))
    for key, x in zip("ABC", (-2.6, 0.0, 2.6)):
        g.node(f"component:{key}", x, 6.0, 1.5, 0.75, key, generic=True)
    g.link("component:A", "component:B", "data")
    g.link("component:B", "component:C", "data")
    g.label((0.0, 5.15), "identity and definition version $\\cdot$ components and dependencies\\\\"
            "input and output expectations $\\cdot$ parameter definitions", anchor="north",
            style="concept plain", role="specification")
    g.node("orchestration", 0.0, 1.55, 9.2, 1.2, "Workflow orchestration",
           ("Snakemake or Python · scheduling and resources",),
           evidence=((f"{PIPELINE}/Snakefile", "rule run_efit_reconstruction:"),
                     ("vaft/database/worker/state.py", "CREATE TABLE IF NOT EXISTS runs (")))
    g.link("specification", "orchestration", "configuration", start=(0.0, y0 - 0.08), end=(0.0, 1.55 + 0.6 + 0.08))
    g.label((0.15, 0.5 * (y0 + 2.15)), "instantiated by", anchor="west", role="instantiated")
    planned = []
    for i, x in enumerate((-4.6, 4.6), start=1):
        side = "A" if i == 1 else "B"
        run = f"run:{i}"
        g.node(run, x, -0.75, 7.2, 1.55, f"Run {i}", (f"input {side} · configuration {side}", "execution"),
               generic=True)
        g.link("orchestration", run, "execution")
        # three artifact classes: only the outputs are scientific data; logs and provenance record the run
        for key, title, dx, relation, evidence in (
                ("outputs", "Scientific outputs", -2.45, "data", ()),
                ("logs", "Logs", 0.0, "part", ()),
                ("provenance", "Structured provenance", 2.45, "part",
                 (("vaft/omas/vest_upstream.py", "write_manifest"),
                  ("vaft/database/replication.py", "ReplicationRecord")))):
            g.node(f"{run}:{key}", x + dx, -3.25, 2.3, 1.15, title, evidence=evidence, generic=not evidence)
            g.link(run, f"{run}:{key}", relation)
        # the run should point back to its definition: drawn as planned, it is not recorded today
        sign = -1.0 if i == 1 else 1.0
        xr, xe = x + sign * (3.6 + 0.08), sign * 8.9
        g.link(run, "specification", "provenance", start=(xr, -0.75),
               via=[(xe, -0.75), (xe, 5.35)], end=(sign * (7.6 + 0.08), 5.35))
        planned.append((run, "specification"))
    g.label((-9.05, 2.3), "references its definition (planned)", anchor="south",
            style="concept annotation,rotate=90", role="reference")
    g.label((9.05, 2.3), "references its definition (planned)", anchor="south",
            style="concept annotation,rotate=-90", role="reference")
    g.legend(-9.2, -4.55, relations=("data", "configuration", "provenance", "execution", "part"))
    if labels:
        g.label((0.0, -5.75), "Snakemake schedules the job graph; each rule writes its log, and VAFT writes stage "
                "manifests, artifact digests and replication records. A log is not structured provenance",
                anchor="north", style="note", role="note")
    return g.diagram("workflow_specification_and_run", relations=RELATIONS, runs=2, planned_edges=tuple(planned))


#: the chain's stages: (key, title, subtitle, evidence, configuration title, lines, configuration evidence)
STAGES = (
    ("efit", "EFIT", "magnetic reconstruction", ((f"{PIPELINE}/Snakefile", "rule run_efit_reconstruction:"),),
     "EFIT scientific configuration",
     ("profile · initialization · numerics ·", "constraints: the scientific SHA-256;", "Green table recorded apart"),
     (("vaft/code/efit/config.py", "EFITScientificConfig.sha256"),)),
    ("chease", "CHEASE", "equilibrium refinement", ((f"{PIPELINE}/Snakefile", "rule run_chease:"),),
     "CHEASE configuration", ("boundary surface · mesh · relaxation",),
     (("vaft/code/chease.py", "CHEASEConfig"),)),
    ("stability", "MHD stability", "DCON · RDCON · STRIDE · GPEC", ((f"{PIPELINE}/Snakefile", "rule run_gpec_module:"),),
     "Solver configuration", ("configured modules and modes",),
     (("vaft/code/gpec/_types.py", "GPECSuiteConfig"),)),
)
#: what each stage leaves as native files, the digest recorded for the link, and where it is written
NATIVE = (
    ("gfiles", "EFIT g-files", ("sha256 in the EFIT artifact manifest",),
     ((f"{PIPELINE}/run_efit_reconstruction.py", "_artifact_payload"),)),
    ("refined_gfiles", "Refined g-files", ("input and output sha256", "in the CHEASE runs"),
     ((f"{PIPELINE}/run_chease_refinement.py", "_write_runs_summary"),)),
    ("stability_native", "Native stability outputs", ("module manifest with", "input_equilibrium_sha256"),
     (("vaft/code/gpec/_types.py", "GPECSuiteResult.input_equilibrium_sha256"),)),
)
#: what each stage publishes as IDS, how, and the rule that builds it
PUBLISHED = (
    ("efit_ids", "equilibrium IDS (EFIT)", "built from the EFIT outputs", "rule generate_efit_ods:"),
    ("chease_ids", "equilibrium IDS (CHEASE)", "parsed from the refined g-files", "rule generate_chease_ods:"),
    ("stability_ids", "mhd_linear · ntms IDS", "built from the native outputs", "rule build_mhd_linear:"),
)


def imas_scientific_workflow_example(*, labels: bool = True) -> Diagram:
    r"""The routine EFIT -> CHEASE -> MHD-stability chain as implemented: native files, IDS, digests (#2150).

    Device data pass through machine-specific mapping and processing
    (diagnostics, eddy currents, constraints) into the scientific input IDSs:
    magnetics (with the plasma current), pf_active and the vacuum toroidal
    field, pf_passive from the eddy stage and the diamagnetic flux when
    present. EFIT reads native k-files written from them -- the machine
    geometry enters through the Green-function table, recorded apart from
    the scientific SHA-256 of its configuration -- writes native g-files, and
    an equilibrium IDS is built from its outputs. CHEASE reads the native EFIT g-files --
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
    snakefile = f"{PIPELINE}/Snakefile"
    g.node("device", 0.0, 15.0, 5.2, 0.9, "Experiment and device", generic=True)
    g.node("mapping", 0.0, 13.05, 8.0, 1.25, "Machine-specific mapping and processing",
           ("diagnostics · eddy currents · constraints",),
           evidence=((snakefile, "rule generate_diagnostics_ods:"), (snakefile, "rule generate_eddy_ods:"),
                     (snakefile, "rule generate_constraints_ods:")))
    g.node("inputs", 0.0, 10.85, 9.8, 1.4, "Scientific input IDSs",
           ("magnetics (incl. plasma current) · pf_active · tf;", "pf_passive from the eddy stage; diamagnetic flux "
            "when present"),
           evidence=(("vaft/code/efit/kfile.py", "REQUIRED_CONSTRAINT_INPUTS"),
                     ("vaft/code/efit/kfile.py", "generate_constraints_ods")))
    g.node("kfiles", -7.4, 8.95, 6.2, 1.2, "EFIT k-files", ("written from the input IDSs",),
           evidence=((snakefile, "rule generate_kfile:"),))
    g.link("device", "mapping", "data")
    g.link("mapping", "inputs", "data")
    g.link("inputs", "kfiles", "data")
    stage_y, side_y = (7.1, 3.5, -0.1), (5.3, 1.7, -1.9)
    previous = "kfiles"
    for (key, title, sub, evidence, ctitle, clines, cevidence), y, (nkey, ntitle, digest, nevidence), \
            (pkey, ptitle, how, rule), ny in zip(STAGES, stage_y, NATIVE, PUBLISHED, side_y):
        g.node(key, 0.0, y, 6.4, 1.15, title, (sub,), style="concept strong", evidence=evidence)
        g.node(f"configuration:{key}", 7.4, y, 6.2, 1.25 + 0.3 * (len(clines) - 1), ctitle, clines,
               style="concept leaf", evidence=cevidence)
        g.link(f"configuration:{key}", key, "configuration")
        g.link(previous, key, "data")
        g.node(nkey, -7.4, ny, 6.2, 1.4, ntitle, digest, evidence=nevidence)
        g.node(pkey, 7.4, ny, 6.2, 1.2, ptitle, (how,), evidence=((snakefile, rule),))
        g.link(key, nkey, "data")
        g.link(key, pkey, "data")
        previous = nkey
    g.node("digest_check", -13.6, 1.7, 4.0, 3.4, "Cross-stage digest check",
           ("verify_chain compares", "recorded digests"), status="partial",
           note="library and tests; not yet in the pipeline",
           evidence=(("vaft/database/provenance.py", "verify_chain"),))
    for nkey in ("gfiles", "refined_gfiles", "stability_native"):
        g.link(nkey, "digest_check", "provenance")
    g.node("sidecar", 0.0, -3.95, 21.0, 1.25, "For each stage: sidecar records, not scientific IDS",
           ("configuration and version · log and status file · input and output digests · stage manifest · "
            "replication record",),
           evidence=(("vaft/omas/vest_upstream.py", "write_manifest"),
                     ("vaft/database/replication.py", "ReplicationRecord.product_sha256")))
    g.label((-7.4, 9.8), "native files", anchor="south", style="concept group title", role="column")
    g.legend(-9.2, -5.25)
    if labels:
        g.label((0.0, -5.85), "EFIT reads k-files written from the input IDSs; CHEASE and the stability codes read "
                "native files, not the published IDS, and each IDS product is built from those files",
                anchor="north", style="note", role="note")
    return g.diagram("imas_scientific_workflow_example", relations=RELATIONS,
                     stages=tuple(k for k, *_ in STAGES), native=tuple(k for k, *_ in NATIVE),
                     published=tuple(k for k, *_ in PUBLISHED), pipeline=PIPELINE)
