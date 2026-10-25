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
- **Memory.** The Snakefile declares none. `run_lane.sh` caps the run at the allocation
  (`--resources mem_mb`) and books GPEC cells (`GPEC_MEM_MB`, default 24000: DCON/RDCON `match`
  peaks at 12-25 GB; at 16000 a 64 GB pilot allocation lost its Snakemake process) and CHEASE (4000) with `--set-resources`; RSS of the solver binaries is
  sampled into `<lane>/logs/<chunk>.rss`.
- **Floating-point parity.** `env.example.sh` sets `OPENBLAS_CORETYPE=Haswell` and disables numpy's
  AVX-512 loops (`REGEN_CPU_PARITY=0` to opt out), which made magnetics products bit-identical to
  an AVX2-only server running the same SHA; see "Cross-environment differences" below.

| file | role |
|---|---|
| `setup_env.sh` | pin a develop SHA: clone/worktree/venv/tables/version record |
| `env.example.sh` | copy to `$REGEN_ROOT/env.sh`; sourced by every job (tools, isolation, CPU parity) |
| `srvns` | `srvns <real FileDB root> <cmd...>`: `/srv/vest.filedb` bound to that root |
| `regen-config.yaml` | run overrides deep-merged over the pinned workflow `config.yaml` |
| `run_lane.sh` | `run_lane.sh <lane dir> <shots file> [cores]`; resumable, `STOP` file honoured |
| `launch_lanes.sh` | deal all raw shots round-robin over lanes, submit/resume one allocation per lane |
| `harvest.py` | per-shot stage outcomes, failure classes, per-rule wall time, volume |
| `compare_stage.py` | stage-level ODS comparison (EFIT slices paired by time, never by index) |
| `kdiff.py` | numeric diff of two k-file directories |
| `merge_lanes.sh` | hard-link union of lane roots + run-wide preflight / IMPA selection |
| `copy_raw.sh`, `push_products.sh` | workstation relays (server raw -> cluster; products -> server staging), resumable, sha256-checked |

Hosts and accounts stay out of the repository: the relays read `VAFT_REGEN_TDST_SSH` /
`VAFT_REGEN_VEST_SSH` from the environment or `~/.config/vaft/regen.env`, and
`launch_lanes.sh` takes the Slurm account from `SLURM_ACCOUNT_NAME`.

## Use

```bash
export REGEN_ROOT=$HOME/regen2147
mkdir -p $REGEN_ROOT/{raw,efit-ddd}            # efit-ddd: the four real *.ddd tables
cp workflow/cluster_regeneration/{setup_env.sh,srvns,run_lane.sh,launch_lanes.sh,merge_lanes.sh,harvest.py,compare_stage.py,kdiff.py,regen-config.yaml} $REGEN_ROOT/
cp workflow/cluster_regeneration/env.example.sh $REGEN_ROOT/env.sh   # adapt EFITHOME/CHEASEHOME/GPECHOME
PIN=<develop sha> bash $REGEN_ROOT/setup_env.sh
bash copy_raw.sh <scratch dir>                  # on the workstation
SLURM_ACCOUNT_NAME=<account> bash $REGEN_ROOT/launch_lanes.sh $REGEN_ROOT/full 4 32
#   monitor: tail $REGEN_ROOT/full/lane*/lane.log ; python harvest.py $REGEN_ROOT/full/lane* --out s.json
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
