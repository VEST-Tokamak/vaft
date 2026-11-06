# Pipeline-1 regeneration on a Slurm cluster

Tooling for regenerating the whole VEST FileDB with the **unchanged** pipeline-1 Snakefile
(`workflow/automatic_pipeline_1_routine_data_processing/`) on a Slurm cluster, so the products
can be moved into the production FileDB afterwards (#2147). It never writes the production
FileDB and never contacts HSDS; moving and replicating are separate, separately approved steps.

## Design

- **Pinned code.** `setup_env.sh` pins one develop SHA (one variable, `PIN`): a private ref in an
  own clone, a detached worktree and a uv venv per SHA, the real EFIT Green tables (checked against
  their Git LFS oids, for hosts without git-lfs), and a record of the exact environment.
- **Production paths inside the products.** EFIT manifests, k-file `TABLE_DIR`/`INPUT_DIR`,
  `code.parameters` and the CHEASE/GPEC run records embed absolute FileDB paths. `srvns` runs a
  command in a private user + mount namespace where `/srv/vest.filedb` is bound to a lane's own
  root, as the caller's uid, so every product reads exactly as it would on the server.
- **Lanes, not one root.** The preflight and IMPA-selection checkpoints write run-scoped files
  (`pipeline/preflight/metadata/*.json`): two Snakemake runs on one root would overwrite each
  other's eligible list. A lane is one root, processed in sequential chunks (`CHUNK`, default
  500 shots) by `snakemake --cores N --scheduler greedy` inside one allocation. `merge_lanes.sh`
  joins the lanes by hard links (per-shot directories are disjoint) and rebuilds the run-wide
  preflight with the same scripts the checkpoints run.
- **Memory.** The Snakefile declares none. The lane caps the run at the allocation minus 8 GB
  (`--resources mem_mb`) and `regen-config.yaml`'s `resource_profile` books GPEC cells (24000:
  DCON/RDCON `match` peaks at 12-25 GB) and CHEASE (4000). The lane runner watches the solvers of
  its own process tree (`CAMPAIGN_SOLVER_PROCESSES`), and one above `SOLVER_RSS_KILL_GB` (64) is
  stopped: the stock RDCON grows to 60-210 GB on msing >= 25 cases
  before it aborts in `gal.f` anyway, and in the pilot such a case took a 64 GB allocation's
  Snakemake down with it.
- **Layers.** Chunks, resume, the Snakemake invocation, the chunk verdict and the CPU/memory
  measurement are `vaft.database.campaign` (scheduler-agnostic; it runs a lane on a workstation
  just the same). Submission and its policy gate are `vaft.code` (`SlurmBackend`,
  `vaft.code.policy`). What is this site's -- the pinned environment, `srvns`, raw staging,
  scratch pruning, hosts and accounts -- is the scripts here plus a site file kept outside the
  repository (`site.example.env` -> `~/.config/vaft/regen.env`).
- **CPU use.** `pestat`'s CPUload is a load average, and sacct `TotalCPU` is lost when a job is
  cancelled. The lane runner measures every chunk with `vaft.code.accounting.UsageMeter` (the
  job cgroup inside a Slurm job, `getrusage` elsewhere): `lane.jsonl` carries the chunk's
  `ResourceUsage` and `lane.log` its `cpu_s`, `cpu_util` (CPU-seconds / (wall x booked cores)),
  `mem_max_gb` (cgroup `memory.current`, page cache included) and `max_rss_gb` (largest process). The 2026-11 full run measured ~0.7 of the booked slots occupied by
  Snakemake jobs (about 10 % of each chunk is the serial preflight, the chunk tail and the gap to
  the next chunk) and a load of ~11 per 32-core lane: book what a lane uses, not more.
- **Validation plots** are 38-39 % of a chunk's job time. `validation_plots: false` in the run
  config leaves them out of `rule all` (and of the worker's verdict); a later run of the same
  config with the key removed is meant to render only the PNGs from the stage products already
  on disk (confirm with `snakemake -n` before relying on it).
- **Changing settings mid-run.** A lane refuses another pin, so a re-pinned remainder runs from a
  second root (`REGEN_ROOT=<another root>`, `PIN_BRANCH=ops/<name>` for a pin built from a
  frozen SHA plus cherry-picked fixes) over `SHOTS_ALL=<file>`: the shots the first run has not
  started. `MAX_CHUNKS=1` lets the first run's lanes finish only their interrupted chunk, so the
  two runs' shot sets stay disjoint for `merge_lanes.sh`.
- **Solver scratch.** DCON `euler.bin` and vacuum `vmat.bin` (~5 GB per MHD shot, 98 % of the
  GPEC tree) are deleted after each finished chunk (`PRUNE_SOLVER_SCRATCH=1`), with size, mtime
  and path kept in `<lane>/pruned.tsv`; no Snakemake output or stage product reads them. A later
  rerun of an ideal-GPEC cell alone then needs its DCON cell rerun first (euler.bin).
  `PRUNE_SOLVER_SCRATCH=2` drops the rest of the GPEC and CHEASE work trees too (ideal-GPEC
  profile netCDF, DCON native JSON, contour files, CHEASE run dirs: ~1.2 GB per MHD shot with
  ideal GPEC, ~0.5 GB without) once the chunk's stage products are built; logs, `run.json`,
  status files and the refined g-files stay.
- **Raw.** A shot without its dump and manifest is left out of its chunk (the raw rule would
  otherwise reach for VEST SQL) and keeps the chunk open (`ALLOW_MISSING_RAW=1` closes it).
  `launch_lanes.sh` freezes the shot list only with `RAW_VERIFIED=<copy_raw scratch>/missing_or_different.lst`
  present and empty.
- **Floating-point parity.** `env.example.sh` sets `OPENBLAS_CORETYPE=Haswell` and disables numpy's
  AVX-512 loops (`REGEN_CPU_PARITY=0` to opt out), which made magnetics products bit-identical to
  an AVX2-only server running the same SHA; see "Cross-environment differences" below.

| file | role |
|---|---|
| `setup_env.sh` | pin a develop SHA: clone/worktree/venv/tables/version record |
| `env.example.sh` | copy to `$REGEN_ROOT/env.sh`; sourced by every job (tools, isolation, CPU parity) |
| `srvns` | `srvns <real FileDB root> <cmd...>`: `/srv/vest.filedb` bound to that root |
| `regen-config.yaml` | run overrides deep-merged over the pinned workflow `config.yaml` |
| `site.example.env`, `site.sh` | the site file's template (copy outside the repo) and its loader |
| `run_lane.sh` | `run_lane.sh <lane dir>`: pinned env + `srvns` + hooks around `vaft.database.campaign run-lane` |
| `stage_raw.sh`, `prune_scratch.sh` | the lane's before/after-chunk hooks (raw hard links; solver scratch) |
| `launch_lanes.sh` | plan once (`campaign plan`), then `campaign launch --backend slurm` per pending lane |
| `harvest.py` | per-shot stage outcomes, failure classes, per-rule wall time, volume |
| `compare_stage.py` | stage-level ODS comparison (EFIT slices paired by time, never by index) |
| `kdiff.py` | numeric diff of two k-file directories |
| `merge_lanes.sh` | hard-link union of lane roots + run-wide preflight / IMPA selection |
| `copy_raw.sh`, `push_products.sh` | workstation relays (server raw -> cluster; products -> server staging), resumable, sha256-checked |

Hosts, ports, user names, accounts, partitions and cluster paths stay out of the repository:
every script reads them from `${VAFT_REGEN_ENV:-~/.config/vaft/regen.env}` (template
`site.example.env`; keep it `chmod 600`). The relays use `VAFT_REGEN_CLUSTER_SSH` /
`VAFT_REGEN_SERVER_SSH`; submission uses `VAFT_SLURM_ACCOUNT`/`_PARTITION`/`_QOS`/`_ARGS` and the
execution policy `VAFT_EXECUTION_POLICY` + `VAFT_EXECUTION_APPROVAL`.

## Use

```bash
cp workflow/cluster_regeneration/site.example.env ~/.config/vaft/regen.env   # fill in; chmod 600
. ~/.config/vaft/regen.env
mkdir -p $REGEN_ROOT/{raw,efit-ddd}            # efit-ddd: the four real *.ddd tables
cp workflow/cluster_regeneration/{*.sh,srvns,harvest.py,compare_stage.py,kdiff.py,regen-config.yaml} $REGEN_ROOT/
cp workflow/cluster_regeneration/env.example.sh $REGEN_ROOT/env.sh   # adapt EFITHOME/CHEASEHOME/GPECHOME
PIN=<develop sha> bash $REGEN_ROOT/setup_env.sh  # the pin must contain vaft.database.campaign
bash copy_raw.sh <scratch dir>                  # on the workstation
RAW_VERIFIED=<scratch>/missing_or_different.lst bash $REGEN_ROOT/launch_lanes.sh $REGEN_ROOT/full 2 16
#   monitor: tail $REGEN_ROOT/full/lane*/lane.log
#            python -m vaft.database.campaign benchmark $REGEN_ROOT/full/lane*
#   resume: run launch_lanes.sh again (same run dir); stop: touch <lane>/STOP or scancel <its job id>
sbatch ... --wrap "bash $REGEN_ROOT/merge_lanes.sh $REGEN_ROOT/full/merged $REGEN_ROOT/full/lane*"
bash push_products.sh $REGEN_ROOT/full/merged <staging dir on the server>   # after review only
```

The campaign runbook (measured costs, the move procedure, open decisions) is kept with the run
root and in #2147; it is not repeated here.

## Cross-environment differences (measured 2026-10-25, #2147)

Same SHA, same EFIT/EFUND binaries and Green tables, python 3.14 on both hosts, AVX-512 cluster
node versus an AVX2 server:

- With native kernels, probe signals differ by ~1e-13 T absolute (relative 5e-3 only on
  near-zero channels); with `REGEN_CPU_PARITY=1` magnetics are bit-identical and the remaining
  differences are 1-ULP leaves (Langmuir, barometry, EC) and ~1e-14 relative in `pf_passive`.
- The constraint stage's Gaussian recovery of rejected probes (`gaussian_fit_option: 1`)
  amplifies those ULP differences to 1e-4..5e-3 relative on the recovered probes, and EFIT's
  per-slice acceptance flips on marginal slices: shot yields move by +-1-2 slices (39915: 19 vs
  17), while the totals over many shots agree. Within one host and setting, reruns are
  byte-identical.

## Python API (`vaft.database.campaign`)

The library behind the scripts runs any Snakemake workflow whose `paths.py` states the
`CAMPAIGN_*` names (run-scoped rules and outputs, shot inputs, compared products, watched solvers;
see `vaft/database/campaign/spec.py`), on any host:

```bash
python -m vaft.database.campaign plan shots.txt RUN --lanes 2 --chunk 1000 --exclude OLD_RUN  # refuses shared shots
python -m vaft.database.campaign run-lane RUN/lane0 --cores 8 --workflow-dir WF --config run.yaml   # here, no scheduler
python -m vaft.database.campaign launch RUN --backend local --cores 8 --workflow-dir WF --config run.yaml
python -m vaft.database.campaign launch RUN --backend slurm --cores 16 --memory-mb 65536 --workflow-dir WF --config run.yaml
python -m vaft.database.campaign benchmark c1=bench/c1 c2=bench/c2 --workflow-dir WF   # cpu_util, CPU-h/shot, shots/h, per rule
python -m vaft.database.campaign compare NEW/filedb REF/lane0/filedb REF/lane1/filedb --shots-file shots.txt --workflow-dir WF
```

`--backend slurm` submits through `vaft.code.slurm.SlurmBackend`, so the execution policy applies;
account and partition come from `--account`/`--partition` or `VAFT_SLURM_*`, never from the
repository. `resource_profile` in the run config (rule -> `mem_mb`/`threads`) becomes
`--set-resources`/`--set-threads` for both the lane runner and the routine worker
(`paths.resource_profile_args`); bookings bind under the lane's `--mem-mb` (`--resources mem_mb`).
Lanes started by the earlier all-shell `run_lane.sh` (no `lane.jsonl`) are not read by `benchmark`.
