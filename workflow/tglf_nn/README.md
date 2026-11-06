# TGLF-NN training bounds (#2274, phase A)

`export_training_bounds.jl` writes the true per-input training range (`xbounds`) of every
TurbulentTransport.jl ONNX family next to a copy of that family. VAFT then tests
containment, not only the z-score distance from `xm`/`xsigma`.

```bash
julia --startup-file=no workflow/tglf_nn/export_training_bounds.jl \
    ~/.julia/packages/TurbulentTransport/<hash> ~/tglf-nn-models
export TURBULENTTRANSPORTHOME=~/tglf-nn-models   # or model_dir=~/tglf-nn-models/models
```

- The bounds exist only inside the Julia `.bson` twin of each family. The script reads them with BSON alone; neither Flux nor TurbulentTransport is loaded. BSON is resolved by UUID from the active Julia environment's manifest.
- A family is exported only when the `.bson` is provably its ONNX export's twin: identical `xnames`, in order, and `xm` matching `xm.txt`. Bounds that differ between ensemble members are combined by intersection; `xbounds.source` records which case applied and the `.bson` sha256.
- The source depot is never modified. VAFT vendors no weights; the copy is the user's.
- In VAFT, `load_metadata` reads `xbounds.txt` when present, and the file becomes part of the model's sha256. `DomainAudit` then reports `bounds_available=True`, `bound_violations` and `bound_positions` (`[0, 1]` inside the range). `run_surrogate` refuses an input that is outside the range (`DomainAudit.applicable`), even when it is within the z threshold.
