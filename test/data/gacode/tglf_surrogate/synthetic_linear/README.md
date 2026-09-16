# `synthetic_linear` — a two-member ONNX ensemble with a hand-checkable answer

Not a physics model. It exists so the surrogate backend's arithmetic — the log
transform, the input normalisation, the ensemble, the output denormalisation — can be
tested exactly, without VAFT vendoring pretrained weights (issue #553 sections 5 and
18). Both graphs together are ~500 bytes.

It mimics the layout upstream `TurbulentTransport.jl` ships: `*.onnx` members beside
`xnames/ynames/xm/xsigma/ym/ysigma` as whitespace-separated text.

Each member is a single `Linear(3, 2)` with integer weights:

    member 1:  y_raw = [ z0 + 0.5,  2*z1 - 0.5 ]
    member 2:  y_raw = [ z1 - 1.0,  4*z2 + 1.0 ]

where `z = (x - xm) / xsigma`, and the reported prediction is
`y = y_raw * ysigma + ym` averaged over the members.

So for `Q_LOC=3`, `RMIN_LOC=1`, `BETAE=10` (the third channel is `BETAE_log10`, so it
enters as `log10(10) = 1`):

    z    = [2, 2, 1]
    mean = [4.5, 1.125]
    std  = [1.5, 0.375]

Regenerate with (needs `torch` and `onnx`, neither of which VAFT depends on — the
committed bytes are what the tests use):

```python
import torch, torch.nn as nn
for index, (w, b) in {1: ([[1., 0., 0.], [0., 2., 0.]], [0.5, -0.5]),
                      2: ([[0., 1., 0.], [0., 0., 4.]], [-1.0, 1.0])}.items():
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(w)); layer.bias.copy_(torch.tensor(b))
    torch.onnx.export(
        layer, (torch.zeros(1, 3),), f"synthetic_linear_model_{index}.onnx",
        input_names=["data_0"], output_names=["dense_0"],
        dynamic_axes={"data_0": {0: "batch"}, "dense_0": {0: "batch"}},
        opset_version=13, dynamo=False,
    )
```
