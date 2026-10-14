"""Compare native fields to the actual pinned upstream pygacode reader and FFT.

No solver execution. Supply data.py and plot_fluct.py from GACODE b493397;
pygacode/gacodefuncs.py must also be importable beside their source tree.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from vaft.code.gacode.cgyro import read_field_history
from vaft.process.gyrokinetics import reconstruct_field


def verify(run, reader, plot, *, window=(100.0, 104.0)):
    reader, plot = Path(reader), Path(plot)
    expected = (
        "ba5c8003fb92cedcfdca4a7199a9512c10a3cf1c8b65d51c466ec586527b8286",
        "5d8b303dd3a518407f4405c6464a49fd94c1f0ae2c0bafc2af2764fa78f641f1",
    )
    if (
        tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in (reader, plot))
        != expected
    ):
        raise ValueError("oracle source differs from pinned b493397 upstream files")
    function_path = (
        reader.parent
        if (reader.parent / "gacodefuncs.py").exists()
        else reader.parent.parent
    )
    sys.path.insert(0, str(function_path.resolve()))
    try:
        spec = importlib.util.spec_from_file_location(
            "vaft_upstream_cgyro_oracle", reader
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        upstream = module.cgyrodata(
            str(Path(run).resolve()) + "/", silent=True, fast=True
        )
        upstream.getbigfield()
    finally:
        sys.path.pop(0)
    history = read_field_history(run, time_window=window)
    if not str(history.metadata["version"]["commit"]).startswith("b493397"):
        raise ValueError("this oracle is qualified for b493397 native output only")
    np.testing.assert_array_equal(history.raw, upstream.kxky_phi)
    tree = ast.parse(plot.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "maptoreal_fft"
    )
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(plot), "exec"), namespace)  # noqa: S102 - hash-pinned upstream oracle
    c = history.read(slice(0, 1))[..., 0]
    nr, nk = c.shape
    nx = nr + 1
    ny = 2 * nk - 1
    plane = reconstruct_field(
        c,
        history.kx,
        history.ky,
        length_x=history.metadata["length_rho_s"],
        nx=nx,
        ny=ny,
    )
    expected, _ = namespace["maptoreal_fft"](nr, nk, nx, ny, c)
    np.testing.assert_allclose(plane.values, expected, rtol=2e-5, atol=2e-6)
    return {
        "upstream_revision": "b49339750a4aa4cf2b089fa9ff3afe098005f0f8",
        "upstream_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (reader, plot)
        },
        "raw_field_array_equal": True,
        "reconstruction_max_abs_error": float(np.max(abs(plane.values - expected))),
        "frame_time": float(history.time[0]),
        "layout": list(history.raw.shape),
        "saved_theta": history.theta.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = verify(args.run, args.reader, args.plot)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
