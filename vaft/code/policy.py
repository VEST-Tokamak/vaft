"""Who may submit what to a shared cluster: limits, approvals and an audit trail.

A shared cluster has an owner. On tdst the administrator cancelled a run that
booked more than it used, and the run may resume only inside what the
administrator agreed to. That agreement has to bind every way VAFT submits a
job -- a script, a pipeline stage, an agent over MCP -- so it is checked here,
by the scheduler backends themselves, right before ``sbatch``.

A policy file (YAML, path in ``VAFT_EXECUTION_POLICY``) names clusters::

    audit_log: /path/to/audit.jsonl
    approvals_dir: /path/to/approvals      # one <id>.json per approval
    default: {requires_approval: false}
    clusters:
      tdst:
        hosts: [147.46.36.74, tdst.cluster.master]   # names a backend may know it by
        requires_approval: true
        max_cpus: 16
        max_mem_mb: 65536
        max_concurrent: 2
        partitions: [g1]

An approval (``<approvals_dir>/<id>.json``) records who agreed to what::

    {"id": "tdst-2147-20261104", "cluster": "tdst", "approver": "tdst admin",
     "scope": "benchmark + 2 x 16 CPU resume", "requester": "hs_yun",
     "expires": "2026-12-31T23:59:59", "max_cpus": 16, "max_mem_mb": 65536,
     "max_concurrent": 2}

The effective limits are the tighter of the cluster's and the approval's. No
policy file means no policy: submissions go ahead unchecked, as before. Every
decision taken under a policy -- refused or submitted -- is appended to the
audit log with the requester, the approval and the job id.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ._executables import ExecutableNotLaunchable

__all__ = [
    "APPROVAL_ENV",
    "POLICY_ENV",
    "Approval",
    "ClusterPolicy",
    "ExecutionPolicy",
    "PolicyDecision",
    "SubmissionGate",
    "SubmissionRefused",
    "load_policy",
    "open_gate",
]

POLICY_ENV = "VAFT_EXECUTION_POLICY"
APPROVAL_ENV = "VAFT_EXECUTION_APPROVAL"


class SubmissionRefused(ExecutableNotLaunchable):
    """The execution policy refused a submission before it reached the scheduler."""


@dataclass(frozen=True)
class ClusterPolicy:
    """Limits one cluster puts on a single submission; ``None`` is no limit."""

    name: str
    requires_approval: bool = False
    max_cpus: Optional[int] = None
    max_mem_mb: Optional[int] = None
    max_concurrent: Optional[int] = None
    partitions: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> "ClusterPolicy":
        unknown = set(data) - {"requires_approval", "max_cpus", "max_mem_mb", "max_concurrent", "partitions", "hosts"}
        if unknown:
            raise ValueError(f"cluster {name!r}: unknown policy keys {sorted(unknown)}")
        return cls(
            name=name,
            requires_approval=bool(data.get("requires_approval", False)),
            max_cpus=_optional_int(data.get("max_cpus")),
            max_mem_mb=_optional_int(data.get("max_mem_mb")),
            max_concurrent=_optional_int(data.get("max_concurrent")),
            partitions=tuple(str(p) for p in data.get("partitions") or ()),
            hosts=tuple(str(h) for h in data.get("hosts") or ()),
        )


@dataclass(frozen=True)
class Approval:
    """A recorded agreement to run within some limits on one cluster."""

    id: str
    cluster: str
    approver: str
    scope: str = ""
    requester: str = ""
    expires: Optional[datetime] = None
    max_cpus: Optional[int] = None
    max_mem_mb: Optional[int] = None
    max_concurrent: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Approval":
        expires = data.get("expires")
        return cls(
            id=str(data["id"]),
            cluster=str(data["cluster"]),
            approver=str(data["approver"]),
            scope=str(data.get("scope", "")),
            requester=str(data.get("requester", "")),
            expires=_parse_time(expires) if expires else None,
            max_cpus=_optional_int(data.get("max_cpus")),
            max_mem_mb=_optional_int(data.get("max_mem_mb")),
            max_concurrent=_optional_int(data.get("max_concurrent")),
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Whether a submission may go ahead, and why not when it may not."""

    allowed: bool
    cluster: str
    reasons: tuple[str, ...] = ()
    approval: Optional[Approval] = None
    limits: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPolicy:
    """The clusters a policy file names, its approvals and its audit log."""

    clusters: dict[str, ClusterPolicy]
    default: Optional[ClusterPolicy] = None
    approvals_dir: Optional[Path] = None
    audit_log: Optional[Path] = None
    source: Optional[Path] = None

    @classmethod
    def load(cls, path: Path) -> "ExecutionPolicy":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(data) - {"clusters", "default", "approvals_dir", "audit_log"}
        if unknown:
            raise ValueError(f"{path}: unknown policy keys {sorted(unknown)}")
        base = Path(path).parent
        clusters = {
            str(name): ClusterPolicy.from_mapping(str(name), entry or {})
            for name, entry in (data.get("clusters") or {}).items()
        }
        default = data.get("default")
        return cls(
            clusters=clusters,
            default=ClusterPolicy.from_mapping("default", default) if default is not None else None,
            approvals_dir=_relative(base, data.get("approvals_dir")),
            audit_log=_relative(base, data.get("audit_log")),
            source=Path(path),
        )

    def cluster(self, name: str) -> ClusterPolicy:
        """The policy for ``name`` -- an entry's name or one of its ``hosts`` --
        else the default, else no limits."""
        if name in self.clusters:
            return self.clusters[name]
        for entry in self.clusters.values():
            if name in entry.hosts:
                return entry
        if self.default is not None:
            return ClusterPolicy(**{**self.default.__dict__, "name": name})
        return ClusterPolicy(name=name)

    def approval(self, approval_id: str) -> Optional[Approval]:
        """The approval recorded as ``<approvals_dir>/<id>.json``, or ``None``.

        The record's own ``id`` must be the file's name, so a copied file
        cannot stand in for another approval.
        """
        if not approval_id or self.approvals_dir is None or "/" in approval_id or approval_id.startswith("."):
            return None
        path = self.approvals_dir / f"{approval_id}.json"
        try:
            approval = Approval.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            return None
        return approval if approval.id == approval_id else None

    def check(
        self,
        *,
        cluster: str,
        cpus: int,
        mem_mb: Optional[int],
        partition: Optional[str],
        approval_id: str = "",
        active_jobs: Optional[Callable[[Sequence[str]], Optional[int]]] = None,
        now: Optional[datetime] = None,
        options: Sequence[str] = (),
    ) -> PolicyDecision:
        """Decide one submission of ``cpus`` CPUs and ``mem_mb`` MiB on ``cluster``.

        ``active_jobs`` counts which of the job ids this log recorded for the
        cluster are still queued or running, or ``None`` when it cannot tell;
        ``max_concurrent`` then refuses rather than guess. ``options`` are the
        scheduler options the submission will carry: a later ``--partition``,
        ``--cpus-per-task`` or ``--mem`` (an ``extra_args`` entry) wins in
        sbatch, so it is what is checked.
        """
        cpus, mem_mb, partition = _effective(options, cpus, mem_mb, partition)
        rules = self.cluster(cluster)
        cluster = rules.name if rules.name != "default" else cluster
        reasons: list[str] = []
        approval = self.approval(approval_id) if approval_id else None
        current = now or datetime.now(timezone.utc)
        if rules.requires_approval:
            if approval is None:
                reasons.append(
                    f"cluster {cluster!r} requires a recorded approval ({APPROVAL_ENV} or approval=); "
                    + ("none given" if not approval_id else f"{approval_id!r} not found")
                )
            elif approval.cluster != cluster:
                reasons.append(f"approval {approval.id!r} is for cluster {approval.cluster!r}, not {cluster!r}")
            elif approval.expires is not None and current > approval.expires:
                reasons.append(f"approval {approval.id!r} expired at {approval.expires.isoformat()}")
            elif approval.requester and approval.requester != _requester():
                reasons.append(f"approval {approval.id!r} was given to {approval.requester!r}, not {_requester()!r}")
        limits = {
            "max_cpus": _tighter(rules.max_cpus, approval.max_cpus if approval else None),
            "max_mem_mb": _tighter(rules.max_mem_mb, approval.max_mem_mb if approval else None),
            "max_concurrent": _tighter(rules.max_concurrent, approval.max_concurrent if approval else None),
            "partitions": rules.partitions,
        }
        if limits["max_cpus"] is not None and cpus == WHOLE_NODE:
            reasons.append(f"--exclusive books a whole node, limit {limits['max_cpus']} CPUs")
        elif limits["max_cpus"] is not None and cpus > limits["max_cpus"]:
            reasons.append(f"{cpus} CPUs requested, limit {limits['max_cpus']}")
        if limits["max_mem_mb"] is not None:
            if mem_mb == WHOLE_NODE:
                reasons.append(
                    "the request books a node's whole memory (--mem=0, or --exclusive with"
                    f" --mem-per-cpu), limit {limits['max_mem_mb']} MiB"
                )
            elif mem_mb is None:
                reasons.append(f"no memory requested; cluster {cluster!r} requires --mem <= {limits['max_mem_mb']} MiB")
            elif mem_mb > limits["max_mem_mb"]:
                reasons.append(f"{mem_mb} MiB requested, limit {limits['max_mem_mb']} MiB")
        if rules.partitions and (partition or "") not in rules.partitions:
            reasons.append(f"partition {partition!r} not in {list(rules.partitions)}")
        if limits["max_concurrent"] is not None and active_jobs is not None:
            running = active_jobs(self.recorded_jobs(cluster))
            if running is None:
                reasons.append(f"cannot count active jobs on {cluster!r}; limit {limits['max_concurrent']}")
            elif running >= limits["max_concurrent"]:
                reasons.append(f"{running} jobs already active, limit {limits['max_concurrent']}")
        return PolicyDecision(
            allowed=not reasons, cluster=cluster, reasons=tuple(reasons), approval=approval, limits=limits
        )

    def recorded_jobs(self, cluster: str) -> list[str]:
        """Job ids the audit log recorded as submitted on ``cluster``."""
        jobs = []
        for entry in self._audit_entries():
            if entry.get("cluster") == cluster and entry.get("event") == "submitted" and entry.get("job_id"):
                jobs.append(str(entry["job_id"]))
        return jobs

    def record(self, event: str, decision: PolicyDecision, *, argv: Sequence[str], job_id: str = "") -> None:
        """Append one audit entry (``refused`` or ``submitted``)."""
        if self.audit_log is None:
            return
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "cluster": decision.cluster,
            "requester": _requester(),
            "approval": decision.approval.id if decision.approval else None,
            "approver": decision.approval.approver if decision.approval else None,
            "job_id": job_id or None,
            "reasons": list(decision.reasons),
            "request_digest": hashlib.sha256("\0".join(map(str, argv)).encode()).hexdigest()[:16],
            "argv": [str(part) for part in argv],
        }
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry, sort_keys=True) + "\n")

    def _audit_entries(self) -> list[dict[str, Any]]:
        if self.audit_log is None:
            return []
        try:
            lines = self.audit_log.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
        return entries


def load_policy(environ: Optional[Mapping[str, str]] = None) -> Optional[ExecutionPolicy]:
    """The policy ``VAFT_EXECUTION_POLICY`` names, or ``None`` when it is unset.

    A set but unreadable policy raises: a missing file must not silently lift
    the limits it was there to impose.
    """
    env = os.environ if environ is None else environ
    path = env.get(POLICY_ENV)
    if not path:
        return None
    return ExecutionPolicy.load(Path(path).expanduser())


class SubmissionGate:
    """A submission the policy let through, held under the policy's lock until it is recorded.

    The lock (``<audit_log>.lock``) serializes check-then-submit across threads
    and processes on this host, so concurrent submitters cannot all pass a
    ``max_concurrent`` that only one of them fits under.
    """

    def __init__(self, policy: Optional[ExecutionPolicy], decision: Optional[PolicyDecision], lock: Any = None):
        self.policy, self.decision, self._lock = policy, decision, lock

    def submitted(self, job_id: str, argv: Sequence[str]) -> None:
        """Record the job the scheduler accepted, then release the lock.

        Never raises: the job is already on the cluster, and a failure to write
        the audit log must not leave it running untracked by its caller.
        """
        try:
            if self.policy is not None and self.decision is not None:
                self.policy.record("submitted", self.decision, argv=argv, job_id=job_id)
        except OSError as error:
            import warnings

            warnings.warn(f"execution policy: could not audit job {job_id}: {error}", RuntimeWarning, stacklevel=2)
        finally:
            self.close()

    def close(self) -> None:
        if self._lock is not None:
            try:
                import fcntl

                fcntl.flock(self._lock, fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None

    def __enter__(self) -> "SubmissionGate":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def open_gate(
    *,
    cluster: str,
    cpus: int,
    mem_mb: Optional[int],
    partition: Optional[str],
    options: Sequence[str] = (),
    approval_id: Optional[str] = None,
    active_jobs: Optional[Callable[[Sequence[str]], Optional[int]]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> SubmissionGate:
    """Check one submission before any side effect; raise :class:`SubmissionRefused` if refused.

    For the scheduler backends. Without a configured policy the gate is a
    no-op. With one, the gate holds the policy lock until
    :meth:`SubmissionGate.submitted` records the job id (or it is closed).
    """
    env = os.environ if environ is None else environ
    policy = load_policy(env)
    if policy is None:
        return SubmissionGate(None, None)
    lock = _acquire(policy)
    try:
        approval = approval_id if approval_id is not None else env.get(APPROVAL_ENV, "")
        decision = policy.check(
            cluster=cluster, cpus=cpus, mem_mb=mem_mb, partition=partition,
            approval_id=approval, active_jobs=active_jobs, options=options,
        )
        if not decision.allowed:
            policy.record("refused", decision, argv=options)
            raise SubmissionRefused(
                f"execution policy ({policy.source}) refused the submission to {cluster!r}: "
                + "; ".join(decision.reasons)
            )
    except BaseException:
        SubmissionGate(None, None, lock).close()
        raise
    return SubmissionGate(policy, decision, lock)


def _acquire(policy: ExecutionPolicy) -> Any:
    if policy.audit_log is None:
        return None
    import fcntl

    policy.audit_log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(policy.audit_log.with_name(policy.audit_log.name + ".lock"), "a")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


#: A request for the whole node: ``--mem=0`` (all its memory) or ``--exclusive``.
WHOLE_NODE = -1


def _effective(
    options: Sequence[str], cpus: int, mem_mb: Optional[int], partition: Optional[str]
) -> tuple[int, Optional[int], Optional[str]]:
    """``(cpus, mem MiB, partition)`` an sbatch/srun option list really asks for (last wins).

    Short options may be joined (``-c32``, ``-pg1``). ``--nodes`` and
    ``--ntasks-per-node`` multiply the CPU count; ``--exclusive`` and
    ``--mem=0`` ask for a whole node and come back as :data:`WHOLE_NODE`.
    """
    values: dict[str, str] = {}
    short = {"-c": "--cpus-per-task", "-p": "--partition", "-n": "--ntasks", "-N": "--nodes"}
    known = ("--cpus-per-task", "--partition", "--ntasks", "--nodes", "--ntasks-per-node", "--mem", "--mem-per-cpu")
    items = [str(item) for item in options]
    exclusive = False
    for index, item in enumerate(items):
        if item == "--exclusive" or item.startswith("--exclusive="):
            exclusive = True
            continue
        name, _, value = item.partition("=")
        if name[:2] in short and len(name) > 2 and not name.startswith("--"):
            name, value = short[name[:2]], name[2:]   # -c32
        name = short.get(name, name)
        if name in known:
            if not value and index + 1 < len(items) and not items[index + 1].startswith("-"):
                value = items[index + 1]
            if value:
                values[name] = value
    nodes = _first_int(values.get("--nodes", "1"))
    if "--ntasks" in values:
        tasks = int(values["--ntasks"])
    elif "--ntasks-per-node" in values:
        tasks = int(values["--ntasks-per-node"]) * nodes
    else:
        tasks = nodes
    if "--cpus-per-task" in values:
        cpus = int(values["--cpus-per-task"]) * tasks
    elif tasks > 1:
        cpus = max(cpus, tasks)
    if "--mem" in values:
        mem_mb = _mib(values["--mem"]) * nodes
        if mem_mb == 0:
            mem_mb = WHOLE_NODE
    elif "--mem-per-cpu" in values:
        # Under --exclusive the job gets every CPU of the node, and memory per CPU with it.
        mem_mb = WHOLE_NODE if exclusive else _mib(values["--mem-per-cpu"]) * cpus
    if exclusive:
        cpus = WHOLE_NODE
    if "--partition" in values:
        partition = values["--partition"]
    return cpus, mem_mb, partition


def _first_int(text: str) -> int:
    # --nodes takes a range (2-4): the scheduler may give the upper bound.
    return int(str(text).split("-")[-1]) if str(text).replace("-", "").isdigit() else 1


def _mib(text: str) -> int:
    units = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    text = text.strip().upper()
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(float(text))


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _tighter(a: Optional[int], b: Optional[int]) -> Optional[int]:
    values = [value for value in (a, b) if value is not None]
    return min(values) if values else None


def _parse_time(text: Any) -> datetime:
    # A stamp without a zone is the local time of the host that checks it.
    parsed = datetime.fromisoformat(str(text))
    return parsed if parsed.tzinfo is not None else parsed.astimezone()


def _relative(base: Path, value: Any) -> Optional[Path]:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base / path


def _requester() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name in some containers
        return os.environ.get("USER", "")
