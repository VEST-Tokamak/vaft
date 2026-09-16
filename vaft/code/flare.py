"""FLARE interoperability: the equilibrium handshake.

FLARE traces field lines through a 2-D equilibrium read from a GEQDSK plus a
3-D perturbation, and the two arrive in different conventions. This module is
the part that reconciles them.

**FLARE declares no COCOS.** The word appears nowhere in its source. What it
does instead is take two manual multipliers from its control file --
``scale_Ip`` applied to ``Simag``, ``Sibry`` and ``psirz``, and ``scale_Bt``
applied to ``Bcentr`` and ``fpol`` (``src/fortran/bfield/equi2d.f90:852``) --
and then derive the field directions from whatever comes out
(``:307-309``). Its index is therefore read off its behaviour rather than a
declaration, and :data:`~vaft.data.cocos` registers it as 3 with that
reasoning recorded.

**The multipliers are a convention conversion, and the legacy guessed them.**
``run_flare.py::_resolve_gpec_equilibrium_convention`` inferred ``scale_Ip``
from a g-file's metadata and fell back to ``1.0`` when the parse failed --
silently, so an unreadable header became "no conversion needed". The
difference is not cosmetic: measured on the DIII-D ideal GPEC example, the
same raw ``n = 1`` BRZPHI field traced through the converted and unconverted
equilibrium gives different island topology in the total, plasma and coil
fields alike. This module derives the pair from an identified COCOS and
refuses to proceed when the identification is ambiguous.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from vaft.data.cocos import convention_for
from vaft.process.cocos import (
    cocos_field_scales,
    identify_convention,
    identify_flux_exponent,
)

from ._executables import executable_from_home, missing_home_message

__all__ = [
    "BRZPHI_IMAGINARY_COLUMNS",
    "FLARE_TASKS",
    "FlareConfig",
    "FlareEquilibriumScales",
    "FlareResult",
    "flare_equilibrium_scales",
    "flare_executable",
    "run_flare",
    "write_helicity_flipped_field",
]

#: Zero-based positions of ``imag(b_r)``, ``imag(b_z)`` and ``imag(b_phi)`` in
#: a ``GPEC_BRZPHI`` data row, whose nine tokens are
#: ``l r z re(b_r) im(b_r) re(b_z) im(b_z) re(b_phi) im(b_phi)``.
BRZPHI_IMAGINARY_COLUMNS: tuple[int, ...] = (4, 6, 8)

#: How many tokens a ``GPEC_BRZPHI`` data row has. Header lines have other
#: counts, which is how the two are told apart -- the header length varies
#: between GPEC builds, so counting lines would not do.
_BRZPHI_TOKENS = 9

#: The subcommands ``flare`` dispatches, from its own usage text. ``run``
#: executes whatever the control file defines; the rest are shortcuts.
FLARE_TASKS: tuple[str, ...] = (
    "run",
    "equi2d_autoconf",
    "equi3d_autoconf",
    "find_resonances",
    "footprint_parameters",
    "geqdsk",
    "mgen",
    "poincare_plot",
)


@dataclass(frozen=True)
class FlareEquilibriumScales:
    """The two multipliers FLARE's control file needs, and where they came from."""

    scale_ip: int
    scale_bt: int
    source_cocos: int
    target_cocos: int
    psi_per_radian: bool


def flare_equilibrium_scales(
    equilibrium: Any, *, clockwise_phi: bool | None = None, source_cocos: int | None = None
) -> FlareEquilibriumScales:
    """Derive ``scale_Ip`` and ``scale_Bt`` for a GEQDSK FLARE is to trace.

    Parameters
    ----------
    equilibrium : EquilibriumData
        The equilibrium as read, in whatever convention its file used [n/a].
    clockwise_phi : bool, optional
        Whether the machine's toroidal angle runs clockwise seen from above. A
        fact about the machine, not about the file; without it a g-file
        identifies as a pair of indices and this refuses rather than picking
        one [n/a].
    source_cocos : int, optional
        The source index, when it is known from provenance rather than from
        the data. Supplying it skips the identification entirely [-].

    Returns
    -------
    FlareEquilibriumScales
        ``scale_ip`` multiplies the poloidal flux, ``scale_bt`` the toroidal
        field and ``F``; both are ``+1`` or ``-1`` [-].

    Raises
    ------
    ValueError
        The identification returns no candidate, or more than one and no
        ``clockwise_phi`` to separate them, or the flux normalization cannot
        be determined.

    Convention
    ----------
    **Fail closed.** A g-file carries no convention field, and the signs alone
    leave the odd and even index of a pair indistinguishable -- they differ by
    the handedness of the machine's toroidal angle, which is not in the file.
    The legacy resolved that by falling back to no conversion; this refuses,
    because a wrong sign here changes the traced topology rather than shifting
    it, and a silent default is indistinguishable from a correct answer.

    Applicability
    -------------
    Machine-independent. ``clockwise_phi`` is the one machine fact, and it is
    the caller's to supply.

    Processing steps
    ----------------
    1. Identify the source index, unless the caller states it.
    2. Refuse an empty or still-ambiguous candidate set.
    3. When the index came from the data, check the flux normalization is
       determined too -- the 1-8 and 11-18 families differ by ``2 pi`` on
       ``psi``, which these multipliers do not carry. A stated index already
       names its family.
    4. Take the sign multipliers to FLARE's registered index.

    Limitations
    -----------
    Carries signs only. A source in the 11-18 family also needs its flux
    divided by ``2 pi`` before FLARE, and this reports
    :attr:`~FlareEquilibriumScales.psi_per_radian` rather than applying it,
    because FLARE's two multipliers have nowhere to put a magnitude.

    Provenance
    ----------
    .. [flare] ``FLARE/src/fortran/bfield/equi2d.f90:852`` for where the two
       multipliers are applied, and ``:307-309`` for the direction derivation
       that follows.
    .. [measured] The DIII-D ideal GPEC example's ``g147131.02300_DIIID_KEFIT``
       identifies as COCOS 5 in weber per radian and needs ``(-1, +1)``; a
       COCOS 2 CHEASE equilibrium needs ``(+1, -1)``.
    """
    target = convention_for("flare").cocos
    if target is None:  # pragma: no cover - the registry fixes it at 3
        raise ValueError("the flare convention carries no COCOS index")

    if source_cocos is None:
        candidates = identify_convention(equilibrium, clockwise_phi=clockwise_phi)
        if not candidates:
            raise ValueError(
                "the equilibrium's COCOS cannot be identified: bt0, ip, q and "
                "psi_1d are what the signs are read from, and one of them is "
                "missing or inconsistent. FLARE's scale_Ip and scale_Bt are a "
                "convention conversion, so there is no safe default to fall "
                "back to"
            )
        if len(candidates) > 1:
            # Two different ambiguities reach here and they need different
            # facts to resolve, so the message says which one this is rather
            # than naming the commoner one and being wrong half the time.
            families = {index <= 10 for index in candidates}
            handed = {index % 2 for index in candidates}
            missing = []
            if len(handed) > 1:
                missing.append(
                    "the handedness of the machine's toroidal angle (pass "
                    "clockwise_phi)"
                )
            if len(families) > 1:
                missing.append(
                    "whether psi is in weber or weber per radian, which needs "
                    "an LCFS to measure the Ampere ratio against"
                )
            raise ValueError(
                f"the equilibrium identifies as COCOS {list(candidates)}, and "
                f"separating them needs {' and '.join(missing) or 'more than the signs'}"
                ". Pass source_cocos when provenance settles it; picking one "
                "here would change the traced topology on a guess"
            )
        source = int(candidates[0])
    else:
        source = int(source_cocos)

    if source_cocos is None:
        # Only when the index came from the data. A stated index already names
        # its family -- 1-8 is weber per radian, 11-18 weber -- so demanding
        # the residual as well would refuse a file whose provenance settles it.
        exponent, _residual = identify_flux_exponent(equilibrium)
        if exponent is None:
            raise ValueError(
                "the flux normalization is undetermined, so whether psi is in "
                "weber or weber per radian is unknown. The two families differ "
                "by 2*pi on psi, which scale_Ip and scale_Bt cannot carry; pass "
                "source_cocos when provenance settles it"
            )

    scale_ip, scale_bt = cocos_field_scales(source, int(target))
    return FlareEquilibriumScales(
        scale_ip=scale_ip,
        scale_bt=scale_bt,
        source_cocos=source,
        target_cocos=int(target),
        psi_per_radian=source <= 10,
    )


@dataclass(frozen=True)
class FlareConfig:
    """Where FLARE is and how to invoke it."""

    executable: Optional[str] = None
    workdir: Path | str = Path(".")
    #: ``flare -n <procs>``. FLARE's own flag; left unset it does not appear.
    processes: Optional[int] = None
    env: Mapping[str, str] = field(default_factory=dict)
    args: Sequence[str] = ()
    timeout: Optional[float] = None


@dataclass
class FlareResult:
    """One FLARE invocation: what it was asked, and what came back."""

    returncode: Optional[int]
    workdir: Path
    task: str
    command: tuple[str, ...] = ()
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def flare_executable(home: str | os.PathLike[str] | None = None) -> Path:
    """Locate the ``flare`` driver beneath its installation root.

    Parameters
    ----------
    home : str or path-like, optional
        FLARE's installation root. Falls back to ``$FLAREHOME`` [n/a].

    Returns
    -------
    Path
        The resolved driver.

    Raises
    ------
    RuntimeError
        Neither argument nor ``$FLAREHOME`` names a root.
    FileNotFoundError, PermissionError
        The root exists but the driver is missing or not executable.

    Notes
    -----
    ``bin/flare`` is a shell dispatcher, not the Fortran binary -- it selects a
    subcommand and execs the library's driver. **A build tree is not an
    installation**: CMake leaves ``build/flare`` without its executable bit,
    and only ``cmake --install`` produces the ``bin/flare`` this resolves.
    Pointing ``$FLAREHOME`` at a build tree therefore raises rather than
    silently running something else.
    """
    root = home if home is not None else os.environ.get("FLAREHOME")
    if root is None or not str(root).strip():
        raise RuntimeError(
            missing_home_message(
                home_variable="FLAREHOME",
                relative_path="bin/flare",
                code_name="FLARE",
            )
        )
    executable = executable_from_home(
        root,
        home_variable="FLAREHOME",
        relative_path="bin/flare",
        code_name="FLARE",
    )
    if executable is None:  # pragma: no cover - guarded by the check above
        raise RuntimeError(
            missing_home_message(
                home_variable="FLAREHOME",
                relative_path="bin/flare",
                code_name="FLARE",
            )
        )
    return executable


def run_flare(
    task: str,
    config: FlareConfig | None = None,
    *,
    home: str | os.PathLike[str] | None = None,
    extra_args: Sequence[str] = (),
) -> FlareResult:
    """Run one FLARE subcommand in a working directory.

    Parameters
    ----------
    task : str
        One of :data:`FLARE_TASKS` [n/a].
    config : FlareConfig, optional
        Working directory, process count, environment and timeout [n/a].
    home : str or path-like, optional
        FLARE's installation root, when ``config.executable`` is unset [n/a].
    extra_args : sequence of str, optional
        Arguments appended after the task, for the subcommands that take a
        file name [n/a].

    Returns
    -------
    FlareResult
        The return code, the command as run, and the captured streams.

    Raises
    ------
    ValueError
        ``task`` is not one of FLARE's subcommands, or ``processes`` is not
        positive.
    RuntimeError, FileNotFoundError, PermissionError
        As :func:`flare_executable`.

    Notes
    -----
    FLARE reads its control file from the working directory rather than from
    an argument, so ``workdir`` is what selects a case.

    The task name is checked against FLARE's own list so the refusal names the
    eight subcommands rather than surfacing ``error: invalid command``. Note
    what the dispatcher actually does, since the two cases differ: **no
    arguments at all prints usage and exits 0**, while an unknown task exits
    1. A caller that built its command line from a variable that came out
    empty would therefore see a successful run that did nothing.
    """
    if task not in FLARE_TASKS:
        raise ValueError(
            f"{task!r} is not a FLARE subcommand; expected one of "
            f"{FLARE_TASKS}"
        )
    config = config or FlareConfig()
    if config.processes is not None and int(config.processes) < 1:
        raise ValueError(f"processes must be at least 1, not {config.processes!r}")

    executable = (
        Path(config.executable) if config.executable else flare_executable(home)
    )
    command: list[str] = [str(executable)]
    if config.processes is not None:
        command += ["-n", str(int(config.processes))]
    command.append(task)
    command += [str(argument) for argument in (*config.args, *extra_args)]

    workdir = Path(config.workdir)
    environment = {**os.environ, **{k: str(v) for k, v in config.env.items()}}
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        env=environment,
        capture_output=True,
        text=True,
        timeout=config.timeout,
        check=False,
    )
    return FlareResult(
        returncode=completed.returncode,
        workdir=workdir,
        task=task,
        command=tuple(command),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _is_brzphi_row(tokens: Sequence[str]) -> bool:
    """Whether a line is a ``GPEC_BRZPHI`` data row rather than header text."""
    if len(tokens) != _BRZPHI_TOKENS:
        return False
    for token in tokens:
        try:
            float(token.replace("D", "E").replace("d", "e"))
        except ValueError:
            return False
    return True


def _negate(token: str) -> str:
    """Flip a numeric token's sign without touching the rest of its text."""
    token = token.strip()
    if token.startswith("-"):
        return token[1:]
    if token.startswith("+"):
        return "-" + token[1:]
    return "-" + token


def write_helicity_flipped_field(
    source: str | os.PathLike[str], destination: str | os.PathLike[str] | None = None
) -> Path:
    """Write the complex conjugate of a GPEC ``BRZPHI`` file.

    Parameters
    ----------
    source : str or path-like
        A ``gpec_*brzphi_n*.out`` as GPEC wrote it [n/a].
    destination : str or path-like, optional
        Where to write. Defaults to the source with ``_helicityflip`` before
        its extension, which is the name the legacy workflow used [n/a].

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    FileNotFoundError
        ``source`` does not exist.
    ValueError
        No data row was found, so nothing was conjugated and the copy would
        be a silent duplicate of the input.

    Convention
    ----------
    **FLARE reconstructs ``B = Re(C exp(-i n phi))``**, so conjugating the
    stored coefficients is the same as ``n -> -n``, which is the same as
    ``phi -> -phi``: it flips the helicity the trace assumes. Only the three
    imaginary columns change.

    Everything else is copied **as text, byte for byte** -- the ``l``, ``r``
    and ``z`` columns, the real parts, and every header line, whose ``n``,
    ``nr`` and ``nz`` sit at fixed character positions. Re-formatting the
    numbers would change the ASCII FLARE splines from, so the sign is flipped
    on the token rather than by parsing and printing a float.

    Data rows are recognised by content -- nine numeric tokens -- because the
    header's length varies between GPEC builds, so a fixed skip would silently
    treat a header line as data on one build and drop a data row on another.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``run_flare.py::_write_helicity_flipped_field`` and
       ``_flare_gpec_realspace``, whose reconstruction this matches.
    """
    origin = Path(source)
    if not origin.is_file():
        raise FileNotFoundError(f"GPEC perturbed-field input not found: {origin}")
    if destination is None:
        target = origin.with_name(
            f"{origin.stem}_helicityflip{origin.suffix}" if origin.suffix
            else f"{origin.name}_helicityflip"
        )
    else:
        target = Path(destination)

    lines: list[str] = []
    flipped = 0
    for line in origin.read_text().splitlines():
        tokens = line.split()
        if not _is_brzphi_row(tokens):
            lines.append(line)
            continue
        for column in BRZPHI_IMAGINARY_COLUMNS:
            tokens[column] = _negate(tokens[column])
        lines.append("  " + "  ".join(tokens))
        flipped += 1
    if flipped == 0:
        raise ValueError(
            f"{origin.name} holds no BRZPHI data row -- nine numeric tokens -- "
            "so nothing was conjugated. Writing the copy anyway would produce a "
            "file named helicityflip that is identical to its input"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
