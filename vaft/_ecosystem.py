"""What VAFT depends on and integrates with, and why (#1648).

Two different relations, kept apart on purpose:

* **Software dependencies** -- Python packages VAFT imports to implement part
  of VAFT itself.  Which packages, and which versions, has one owner:
  ``pyproject.toml``.  This module adds only what the packaging metadata
  cannot say: the *capability* each dependency or extra provides.  It never
  restates a version constraint.
* **External scientific codes** -- independently maintained solvers VAFT
  delegates a calculation to through an explicit integration boundary in
  :mod:`vaft.code`.  They are not Python dependencies, and the upstream
  project, not VAFT, owns the scientific implementation.  Each entry records
  the integration facts: the adapter, how it is executed, who installs it,
  where VAFT finds it, how to check it, and its upstream and literature.

Values that already have an owner are *referenced*, not copied: a code's
``{CODE}HOME`` variable is the adapter's own constant (``home``, read by the
catalog generator), the standardized result is a ``STAGE_REPLICATION`` stage
or a named mapping function, and installers and checkers are repository paths
the tests require to exist.  ``test/test_ecosystem_catalog.py`` holds every
entry against the repository, so this registry cannot silently drift.

``python -m vaft._ecosystem_catalog`` turns it, with ``pyproject.toml``, into
``docs/_data/ecosystem.yml`` for ``/reference/software-dependencies/`` and
``/reference/external-codes/``; :mod:`vaft.diagram` draws the two overview
figures from it.  Only integrations the repository implements are listed;
planned ones belong in issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# software dependencies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    """A capability role a group of dependencies provides."""

    id: str
    title: str
    scope: str  # "runtime", "optional" or "development"
    summary: str


#: Capability roles, in figure order within each scope.  Packaging status is
#: not a scientific ranking: a mandatory storage library is less specialized
#: than an optional physics backend, and "optional" means only that the core
#: works without it.
CAPABILITIES: Tuple[Capability, ...] = (
    Capability("representation", "Scientific data representation", "runtime",
               "The OMAS/IMAS data model every VAFT layer reads and writes"),
    Capability("numerics", "Numerical computing", "runtime",
               "Arrays, solvers, statistics and uncertainty propagation"),
    Capability("tabular", "Tabular and labelled data", "runtime",
               "Tables, labelled arrays and spreadsheet exchange"),
    Capability("images", "Image processing", "runtime",
               "Camera frames and contour extraction"),
    Capability("storage", "Storage and remote access", "runtime",
               "HDF5 files, the HSDS service and the encrypted raw-data access"),
    Capability("database", "Database access", "runtime", "The VEST shot database"),
    Capability("formats", "Configuration and code file formats", "runtime",
               "YAML configuration and the Fortran namelists of EFIT, EFUND and g-/k-files"),
    Capability("visualization", "Visualization", "runtime", "Static and interactive figures"),
    Capability("workflow", "Workflow execution", "runtime", "The production Snakemake pipelines"),
    Capability("interactive", "Interactive notebooks and GUI", "runtime",
               "The Jupyter kernel and widgets the notebooks use, and the browser GUI"),
    Capability("learning", "Machine learning and surrogates", "optional",
               "Learned backends, ONNX export and surrogate inference"),
    Capability("solvers", "In-process solver backends", "optional",
               "External solvers VAFT calls as a Python library"),
    Capability("visualization3d", "3-D and video output", "optional",
               "ParaView scenes, Jupyter 3-D and encoded animations"),
    Capability("interfaces", "Agent interfaces", "optional",
               "The MCP server for agent clients (vaft[gui] is kept as an empty alias)"),
    Capability("acceleration", "Acceleration", "optional", "JIT compilation, reserved until a measurement justifies it"),
    Capability("development", "Testing, quality and notebooks", "development",
               "The test suite, linters, formatters and notebook execution"),
    Capability("architecture", "Architecture and documentation tooling", "development",
               "The generated import graph of the dependency explorer"),
)

#: Runtime dependency (distribution name as ``pyproject.toml`` spells it) ->
#: (capability, what VAFT uses it for).  No versions here.
RUNTIME_ROLES = {
    "omas": ("representation", "OMAS data structures (ODS) and their IMAS mapping"),
    "imas_python": ("representation", "native IMAS IDS objects"),
    "imas_core": ("representation", "the IMAS access layer imas_python runs on"),
    "xmltodict": ("representation", "lets omas serialize equilibrium.code.parameters (EFIT namelists)"),
    "numpy": ("numerics", "arrays everywhere"),
    "scipy": ("numerics", "integration, interpolation, optimization and linear algebra"),
    "statsmodels": ("numerics", "regression for confinement scalings and statistical analysis"),
    "uncertainties": ("numerics", "uncertainty propagation for Thomson scattering and plotted errors"),
    "pandas": ("tabular", "tables of shots, signals and summaries"),
    "xarray": ("tabular", "labelled arrays for plotting and export"),
    "openpyxl": ("tabular", "Excel shot logs and summary sheets"),
    "scikit-image": ("images", "contours of flux surfaces and images"),
    "opencv-python": ("images", "visible-camera frames and geometry"),
    "h5py": ("storage", "HDF5 files"),
    "h5pyd": ("storage", "the HSDS service, pinned because newer releases empty attributes on it"),
    "cryptography": ("storage", "decrypting the stored raw-database credentials"),
    "mysql-connector-python": ("database", "the VEST raw shot database"),
    "f90nml": ("formats", "Fortran namelists of EFIT/EFUND inputs and g-/k-files"),
    "PyYAML": ("formats", "machine descriptions, configurations and the generated documentation catalogs"),
    "matplotlib": ("visualization", "every static figure"),
    "plotly": ("visualization", "interactive panels and the GUI"),
    "seaborn": ("visualization", "confinement-scaling regression and residual plots"),
    "snakemake": ("workflow", "the production pipelines' scheduler"),
    "tqdm": ("workflow", "progress bar in verbose OMAS/IMAS loading (load_omas_imas), reached through omas's star import"),
    "wexpect": ("workflow", "Windows-only; declared but imported by no VAFT module today"),
    "ipykernel": ("interactive", "the notebooks' kernel"),
    "ipython": ("interactive", "inline display in notebooks"),
    "ipywidgets": ("interactive", "notebook widgets"),
    "panel": ("interactive", "the browser GUI (vaft gui), served locally or over SSH port forwarding"),
}

#: Optional extra (``pip install "vaft[<extra>]"``) -> (capability, what it enables).
EXTRA_ROLES = {
    "sklearn": ("learning", "Gaussian-process profile fitting with scikit-learn"),
    "surrogate": ("learning", "TGLF neural-network surrogate inference"),
    "ml": ("learning", "the torch and scikit-learn backends of vaft.process.ml and ONNX export"),
    "tokamaker": ("solvers", "TokaMaker free-boundary equilibria, in process (OpenFUSIONToolkit)"),
    "vtk": ("visualization3d", "3-D scenes as PyVista multiblocks and ParaView files"),
    "jupyter3d": ("visualization3d", "interactive 3-D scenes in Jupyter"),
    "video": ("visualization3d", "encoding animations to .mp4/.webm"),
    "mcp": ("interfaces", "the local, read-only MCP server for agent clients"),
    "gui": ("interfaces", "nothing: Panel is core now; the empty extra keeps vaft[gui] working"),
    "accel": ("acceleration", "nothing yet: no VAFT module imports numba"),
    "architecture": ("architecture", "the import graph behind the dependency explorer"),
    "dev": ("development", "running the test suite and contributing"),
}


# ---------------------------------------------------------------------------
# external scientific codes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """One link of an external code, by what it is."""

    role: str  # repository, homepage, documentation, primary_reference, method_reference
    title: str
    url: str = ""
    doi: str = ""


@dataclass(frozen=True)
class Standardized:
    """A standardized result VAFT maps a code's native output into.

    ``via`` is ``stage:<name>`` (a ``STAGE_REPLICATION`` stage, whose owned IDS
    are the standardized subtree) or ``mapper:<module>.<function>``.
    """

    ids: Tuple[str, ...]
    via: str


@dataclass(frozen=True)
class ExternalCode:
    """An external scientific code and VAFT's integration boundary around it."""

    id: str
    name: str
    roles: Tuple[str, ...]
    adapter: str
    #: "subprocess_executable", "in_process_python" or "native_reader"
    mode: str
    #: ``module:CONSTANT`` naming the adapter's ``{CODE}HOME`` variable, or a
    #: literal variable name where the adapter has no constant
    home: str = ""
    #: "vaft_managed_source_build", "python_package", "site_managed" or "reader_only"
    installation: str = "site_managed"
    #: "public", "registration" (users agreement), "not_open_source", or
    #: "not_stated" where the repository records no distribution terms
    access: str = "public"
    installers: Tuple[str, ...] = ()
    #: the installers that write vaft-external-install.json (the build record the checker reads)
    provenance: Tuple[str, ...] = ()
    checker: str = ""
    extra: str = ""
    maturity: str = "supported"  # "supported", "experimental" or "read_only"
    native: str = ""
    standardized: Tuple[Standardized, ...] = ()
    #: rules of the production pipelines that run it (pipeline-graph node ids)
    workflow: Tuple[str, ...] = ()
    install_section: str = ""
    links: Tuple[Reference, ...] = ()
    note: str = ""

    @property
    def scheduler_backed(self) -> bool:
        """Subprocess codes launch through vaft.code.execution: local, Slurm or remote Slurm."""
        return self.mode == "subprocess_executable"


def _doi(title: str, doi: str, role: str = "primary_reference") -> Reference:
    return Reference(role, title, f"https://doi.org/{doi}", doi)


#: Every external code the repository integrates today, verified 2026-10-06
#: (DOIs against Crossref; repositories and homepages as the installers and
#: install/README.md name them).
EXTERNAL_CODES: Tuple[ExternalCode, ...] = (
    ExternalCode(
        "efit", "EFIT (with EFUND)", ("equilibrium reconstruction",), "vaft.code.efit", "subprocess_executable",
        home="vaft.code.efit.magnetic:EFIT_HOME_ENV", installation="vaft_managed_source_build", access="registration",
        installers=("install/install_efit.sh", "install/install_efit_windows.ps1"),
        provenance=("install/install_efit.sh", "install/install_efit_windows.ps1"), checker="install/check_efit.py",
        native="k-files in, g-/a-/m-files out (EFUND response tables beside them)",
        standardized=(Standardized(("equilibrium",), "stage:efit"),),
        workflow=("routine:generate_kfile", "routine:run_efit_reconstruction", "routine:generate_efit_ods"),
        install_section="efit-and-efund-licensed-software-obtain-it-yourself",
        links=(_doi("L. L. Lao et al., Nucl. Fusion 25, 1611 (1985)", "10.1088/0029-5515/25/11/007"),),
        note="Distributed by the EFIT-AI collaboration under the EFIT users agreement; "
             "VAFT never fetches, mirrors or bundles it.",
    ),
    ExternalCode(
        "chease", "CHEASE", ("equilibrium refinement",), "vaft.code.chease", "subprocess_executable",
        home="vaft.code.chease:CHEASE_HOME_ENV", installation="vaft_managed_source_build",
        installers=("install/install_chease.sh", "install/install_chease_windows.ps1"),
        provenance=("install/install_chease.sh", "install/install_chease_windows.ps1"), checker="install/check_chease.py",
        native="EXPEQ in, refined EQDSK and the CHEASE output files out",
        standardized=(Standardized(("equilibrium",), "stage:chease"),),
        workflow=("routine:run_chease", "routine:generate_chease_ods"),
        install_section="chease-and-its-nideal-selection",
        links=(Reference("repository", "SPC EPFL GitLab", "https://gitlab.epfl.ch/spc/chease"),
               _doi("H. Lütjens, A. Bondeson and O. Sauter, Comput. Phys. Commun. 97, 219 (1996)",
                    "10.1016/0010-4655(96)00046-X")),
    ),
    ExternalCode(
        "gpec", "DCON, RDCON, STRIDE and GPEC", ("MHD stability", "3-D plasma response"), "vaft.code.gpec",
        "subprocess_executable", home="vaft.code.gpec._types:GPEC_HOME_ENV", installation="vaft_managed_source_build",
        installers=("install/install_gpec.sh", "install/install_gpec_windows.ps1"),
        provenance=("install/install_gpec.sh", "install/install_gpec_windows.ps1"), checker="install/check_gpec.py",
        native="namelists and an EQDSK in, NetCDF/binary stability and response files out",
        standardized=(Standardized(("mhd_linear", "ntms"), "stage:mhd_linear"),
                      Standardized(("mhd_linear", "coils_non_axisymmetric"), "stage:gpec_ideal")),
        workflow=("routine:run_gpec_module", "routine:build_mhd_linear", "routine:build_gpec_ideal"),
        install_section="external-fusion-codes-chease-dcongpec-nubeam-gacode",
        links=(Reference("repository", "Princeton University GitHub", "https://github.com/PrincetonUniversity/GPEC"),
               _doi("A. H. Glasser, Phys. Plasmas 23, 072505 (2016) (DCON)", "10.1063/1.4958328", "method_reference"),
               _doi("J.-K. Park, A. H. Boozer and A. H. Glasser, Phys. Plasmas 14, 052110 (2007) (GPEC)",
                    "10.1063/1.2732170")),
    ),
    ExternalCode(
        "gacode", "GACODE: NEO, TGLF and CGYRO", ("neoclassical transport", "turbulent transport", "gyrokinetics"),
        "vaft.code.gacode", "subprocess_executable", home="vaft.code.gacode._types:GACODE_HOME_ENV",
        installation="vaft_managed_source_build", installers=("install/gacode/linux.sh", "install/gacode/macos.sh"),
        checker="install/check_gacode.py", native="input.gacode and per-code inputs in, per-code output files out",
        standardized=(Standardized(("core_profiles", "core_transport"), "stage:neoclassical"),),
        install_section="gacode",
        links=(Reference("repository", "gafusion GitHub", "https://github.com/gafusion/gacode"),
               _doi("E. A. Belli and J. Candy, Plasma Phys. Control. Fusion 50, 095010 (2008) (NEO)",
                    "10.1088/0741-3335/50/9/095010"),
               _doi("G. M. Staebler, J. E. Kinsey and R. E. Waltz, Phys. Plasmas 14, 055909 (2007) (TGLF)",
                    "10.1063/1.2436852"),
               _doi("J. Candy, E. A. Belli and R. V. Bravenec, J. Comput. Phys. 324, 73 (2016) (CGYRO)",
                    "10.1016/j.jcp.2016.07.039")),
    ),
    ExternalCode(
        "mitim", "MITIM-fusion", ("integrated transport modelling", "flux-matching optimization"), "vaft.code.mitim",
        "subprocess_executable", home="vaft.code.mitim.config:MITIM_PYTHON_ENV",
        installation="vaft_managed_source_build", installers=("install/install_mitim.sh",),
        provenance=("install/install_mitim.sh",), maturity="experimental",
        native="MITIM driver inputs in its own interpreter, its run directories out",
        links=(Reference("repository", "MITIM-fusion GitHub", "https://github.com/pabloprf/MITIM-fusion"),
               _doi("P. Rodriguez-Fernandez et al., Nucl. Fusion 64, 076034 (2024)", "10.1088/1741-4326/ad4b3d")),
        note="Runs in an isolated interpreter (VAFT_MITIM_PYTHON) that VAFT never imports, against the GACODE "
             "build $GACODEHOME names; check it with install_mitim.sh --check-only. PORTALS flux matching is a later "
             "stage of #1588.",
    ),
    ExternalCode(
        "tglf_nn", "TGLF neural-network surrogates", ("turbulent transport",), "vaft.code.gacode.tglf.surrogate",
        "in_process_python", home="vaft.code.gacode.tglf.surrogate.resolver:MODELS_HOME_ENV", installation="site_managed",
        extra="surrogate", native="ONNX models of TurbulentTransport.jl, evaluated locally",
        links=(Reference("repository", "TurbulentTransport.jl (models)",
                         "https://github.com/ProjectTorreyPines/TurbulentTransport.jl"),),
        note="Models are local files; none is trained on a VEST-like domain.",
    ),
    ExternalCode(
        "nubeam", "NUBEAM", ("fast-particle and neutral-beam source modelling",), "vaft.code.nubeam",
        "subprocess_executable", home="vaft.code.nubeam.config:NUBEAM_HOME_ENV", installation="vaft_managed_source_build",
        installers=("install/nubeam/linux.sh", "install/nubeam/macos.sh", "install/nubeam/windows.sh",
                    "install/nubeam/windows.ps1"),
        provenance=("install/nubeam/linux.sh", "install/nubeam/windows.ps1"), checker="install/check_nubeam.py", native="a Plasma State file in, NUBEAM's Plasma State and NetCDF out",
        install_section="nubeam",
        links=(Reference("homepage", "NTCC NUBEAM", "https://w3.pppl.gov/NTCC/NUBEAM/"),
               _doi("A. Pankin et al., Comput. Phys. Commun. 159, 157 (2004)", "10.1016/j.cpc.2003.11.002")),
    ),
    ExternalCode(
        "genray", "GENRAY", ("RF wave propagation and current drive",), "vaft.code.genray", "subprocess_executable",
        home="vaft.code.genray.config:GENRAY_HOME_ENV", installation="vaft_managed_source_build",
        installers=("install/install_genray.sh",), provenance=("install/install_genray.sh",),
        checker="install/check_genray.py",
        native="genray.in and an EQDSK in, genray.nc out",
        standardized=(Standardized(("waves",), "mapper:vaft.code.genray.outputs.genray_to_waves"),),
        install_section="genray",
        links=(Reference("repository", "CompX GitHub", "https://github.com/compxco/genray"),),
    ),
    ExternalCode(
        "tes", "TES (RTES)", ("free-boundary equilibrium",), "vaft.code.tes", "subprocess_executable",
        home="vaft.code.tes.runner:TES_HOME_ENV", installation="site_managed", access="not_open_source",
        native="a namelist and cinput in, a g-file and result scalars out",
        standardized=(Standardized(("equilibrium",), "mapper:vaft.code.tes.outputs.collect_tes_outputs"),),
        install_section="tes",
        links=(_doi("Y. M. Jeon, J. Korean Phys. Soc. 67, 843 (2015)", "10.3938/jkps.67.843"),),
    ),
    ExternalCode(
        "tokamaker", "TokaMaker (OpenFUSIONToolkit)", ("free-boundary equilibrium",), "vaft.code.tokamaker",
        "in_process_python", installation="python_package", extra="tokamaker",
        native="OpenFUSIONToolkit objects in process; a g-file out",
        standardized=(Standardized(("equilibrium",), "mapper:vaft.code.tokamaker.outputs.collect_tokamaker_outputs"),),
        install_section="tokamaker-openfusiontoolkit",
        links=(Reference("repository", "OpenFUSIONToolkit GitHub", "https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit"),
               _doi("C. Hansen et al., Comput. Phys. Commun. 298, 109111 (2024)", "10.1016/j.cpc.2024.109111")),
        note="Located through OFT_* variables rather than a {CODE}HOME.",
    ),
    ExternalCode(
        "nice", "NICE", ("equilibrium reconstruction",), "vaft.code.nice", "subprocess_executable",
        home="vaft.code.nice.runner:NICE_HOME_ENV", installation="site_managed",
        maturity="experimental",
        native="NICE inputs in, its reconstruction files out",
        links=(Reference("repository", "Inria GitLab", "https://gitlab.inria.fr/blfauger/nice"),
               _doi("J. Blum, C. Boulbe and B. Faugeras, J. Comput. Phys. 231, 960 (2012)",
                    "10.1016/j.jcp.2011.04.005")),
        note="Open source under the LGPL-3.0 (the repository's LICENSE file). "
             "Experimental: none of the 34 VEST reference slices reconstructs yet.",
    ),
    ExternalCode(
        "flare", "FLARE", ("magnetic field-line tracing",), "vaft.code.flare", "subprocess_executable",
        home="FLAREHOME", installation="site_managed", access="not_stated", native="field-line and mesh files",
        links=(_doi("H. Frerichs, Nucl. Fusion 64, 106034 (2024)", "10.1088/1741-4326/ad7303"),),
    ),
    ExternalCode(
        "transp", "TRANSP (reader)", ("integrated modelling",), "vaft.code.transp", "native_reader",
        installation="reader_only", access="not_open_source", maturity="read_only",
        native="reads a finished run's .CDF output",
        install_section="transp",
        links=(Reference("homepage", "PPPL TRANSP", "https://transp.pppl.gov/"),
               _doi("A. Y. Pankin et al., Comput. Phys. Commun. 312, 109611 (2025)", "10.1016/j.cpc.2025.109611")),
        note="VAFT reads TRANSP results; it does not run TRANSP.",
    ),
    ExternalCode(
        "pentrc", "PENTRC", ("neoclassical toroidal viscosity",), "vaft.code.gpec", "subprocess_executable",
        # install_gpec.sh builds and installs pentrc into $GPECHOME/bin; the Windows installer does not
        # build it and check_gpec.py does not check it.
        home="vaft.code.gpec._types:GPEC_HOME_ENV", installation="vaft_managed_source_build",
        installers=("install/install_gpec.sh",), provenance=("install/install_gpec.sh",),
        native="pentrc.in and a .kin in a completed ideal-GPEC cell, pentrc_output_n*.nc out",
        install_section="external-fusion-codes-chease-dcongpec-nubeam-gacode",
        links=(Reference("repository", "Princeton University GitHub (in the GPEC suite)",
                         "https://github.com/PrincetonUniversity/GPEC"),
               _doi("N. C. Logan, J.-K. Park et al., Phys. Plasmas 20, 122507 (2013)", "10.1063/1.4849395")),
        note="Built with the GPEC suite and run in a completed ideal-GPEC cell by vaft.code.gpec.run_pentrc "
             "(not a GPECSuiteConfig module); vaft.code.gpec also reads its native output, and "
             "vaft.code.pentrc re-exports that reader.",
    ),
)

#: Who installs a code, in words (figure and pages).
INSTALLATION_LABELS = {
    "vaft_managed_source_build": "VAFT-managed source build (install/; build dependencies may be fetched)",
    "python_package": "Python package (an optional extra)",
    "site_managed": "Site- or user-managed (VAFT only resolves and runs it)",
    "reader_only": "Results reader (VAFT does not run the code)",
}

#: How VAFT calls a code, in words.
MODE_LABELS = {
    "subprocess_executable": "Executable, launched as a subprocess",
    "in_process_python": "Python library, called in process",
    "native_reader": "Reader of finished results",
}

#: The integration lifecycle every external code passes through, in figure order.
INTEGRATION_LIFECYCLE: Tuple[Tuple[str, str], ...] = (
    ("upstream", "Upstream scientific project: repository, release, literature"),
    ("obtain", "Obtain the source or package"),
    ("install", "Installation boundary: VAFT-managed build, Python package, or site-managed"),
    ("provenance", "Installation provenance (the build record) and checker"),
    ("resolve", "Environment resolution: {CODE}HOME"),
    ("adapter", "vaft.code adapter: prepare input, execute, collect"),
    ("execute", "Execution: local, Slurm, or remote Slurm"),
    ("native", "Solver-native result"),
    ("mapping", "Mapping and interpretation"),
    ("standard", "Standardized scientific state: IMAS / ODS / FileDB"),
)

__all__ = [
    "CAPABILITIES",
    "Capability",
    "EXTERNAL_CODES",
    "EXTRA_ROLES",
    "ExternalCode",
    "INSTALLATION_LABELS",
    "INTEGRATION_LIFECYCLE",
    "MODE_LABELS",
    "RUNTIME_ROLES",
    "Reference",
    "Standardized",
]

