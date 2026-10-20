#!/usr/bin/env python3
"""Generate a local, atomic two-renderer comparison; never publish its videos.

python docs/scripts/animation_preview.py --output /tmp/vaft-animation-preview
Use a Python environment with vaft[video] for video output.
Serve the directory with docs/scripts/serve_animation_preview.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import tempfile
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from vaft.diagram import magnetic_island
    from vaft.diagram._replay import (
        atomic_write,
        encode,
        scene_payload,
        make_payload,
        RECIPES,
        digest,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = {}
    for legacy in ("pyav", "manim", "html"):
        manifest.pop(legacy, None)

    def update(name, status, **fields):
        record = manifest.setdefault("results", {}).setdefault(name, {})
        if status != "failed":
            record.pop("error", None)
        if status != "pending":
            record.pop("reason", None)
        if status in ("validated", "generated"):
            record["scientific"] = dict(manifest.get("scientific", {}))
        record.update(status=status, **fields)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write(manifest_path, json.dumps(manifest, indent=2).encode())
        print(f"{name}: {status}", flush=True)

    parameters = dict(m=2, n=1, width=0.24, projection="poloidal", labels=False)
    phases = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    fps = 12
    sources = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    manifest["scientific"] = dict(
        parameters=parameters,
        states=len(phases),
        driver="phase",
        unit="rad",
        fps=fps,
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        worktree=str(ROOT),
        tracked_diff_sha256=hashlib.sha256(sources).hexdigest(),
    )
    scenes = [magnetic_island(phase=float(p), **parameters).scene for p in phases]
    payload = scene_payload(
        "magnetic_island",
        "comparison",
        {"name": "phase", "kind": "phase", "unit": "rad"},
        phases,
        scenes,
        parameters=parameters,
        fps=fps,
    )
    manifest["scientific"]["sources"] = payload["sources"]
    raw = encode(payload)
    atomic_write(output / "island-states.json", raw)
    exb = encode(make_payload(RECIPES[1]))
    atomic_write(output / "exb-states.json", exb)
    for name in ("player.js", "loader.js", "player.css"):
        atomic_write(
            output / name, (ROOT / "docs/assets/animations" / name).read_bytes()
        )
    for name in ("magnetic_island_phase_scan.svg", "exb_drift.svg"):
        shutil.copyfile(ROOT / "docs/assets/diagrams" / name, output / name)

    def player(builder, artifact, hash, poster):
        return f'''<div class="va-animation" data-animation="{builder}" data-artifact="{artifact}" data-sha256="{hash}" data-player="./player.js"><img class="va-poster" src="{poster}" alt="Canonical {builder} poster"><button data-activate>Interactive</button><p data-animation-status role="status">Static poster; no state download before activation.</p></div>'''

    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VAFT two-renderer comparison</title><link rel="stylesheet" href="player.css"><style>body{font:16px system-ui;margin:1.5rem;color:#172337}main{max-width:1300px;margin:auto}.comparisons{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:1rem}video{width:100%;background:#f1f5fa}pre{overflow:auto;max-height:25rem}img{max-width:100%}.stills{display:flex;flex-wrap:wrap}.stills img{width:32%}h2{font-size:1.2rem}</style><main><h1>VAFT animation comparison</h1><p>One magnetic-island model, 48 stored phase states, 12 presentation fps. No scientific interpolation. The same state coordinates feed both renderers.</p><div class="comparisons"><section><h2>Matplotlib → PyAV</h2><p data-result="pyav">Pending</p><video controls preload="none" src="island-pyav.mp4"></video><a href="island-pyav.mp4.json">Metadata</a></section><section><h2>Scene JSON → HTML/SVG</h2><p data-result="html">Pending</p>"""
        + player(
            "magnetic_island",
            "island-states.json",
            digest(raw),
            "magnetic_island_phase_scan.svg",
        )
        + """</section></div><h2>First / middle / final Python states</h2><div class="stills">"""
        + "".join(
            f'<img src="state-{i}.png" alt="Python phase state {i}">'
            for i in (0, 24, 47)
        )
        + """</div><h2>E×B trajectory and synchronized observables</h2><p>Normalized time and length; reduced mass ratio 4. Ion/electron coordinates are selected at exact common stored times.</p>"""
        + player("exb_drift", "exb-states.json", digest(exb), "exb_drift.svg")
        + """<h2>Status, scientific inputs and source fingerprint</h2><p><a href="generation.log">Generation log</a> · <a href="manifest.json">Manifest</a></p><pre id="status"></pre></main><script src="loader.js"></script><script>async function update(){const m=await(await fetch('manifest.json',{cache:'no-store'})).json();document.querySelector('#status').textContent=JSON.stringify(m,null,2);for(const n of ['pyav','html'])document.querySelector('[data-result='+n+']').textContent=m.results?.[n]?.status||'pending'}update();setInterval(update,5000)</script></html>"""
    )
    compare_controls = (
        "<p>Compare stored state: "
        + "".join(
            f'<button type="button" data-compare-state="{i}">{i}</button> '
            for i in (0, 24, 47)
        )
        + "</p>"
    )
    compare_script = """<script>
    document.addEventListener('click', async event => {
      const button=event.target.closest('[data-compare-state]');if(!button)return;
      const index=Number(button.dataset.compareState);
      for(const video of document.querySelectorAll('video')){
        video.pause();const seek=()=>video.currentTime=index/12;
        if(video.readyState>=3)seek();else{video.addEventListener('canplay',seek,{once:true});video.load();}
      }
      const root=document.querySelector('[data-animation="magnetic_island"]');
      if(!root.querySelector('.va-player'))root.querySelector('[data-activate]').click();
      for(let attempt=0;attempt<100;attempt++){
        const slider=root.querySelector('input[type=range]');
        if(slider){slider.value=index;slider.dispatchEvent(new Event('input',{bubbles:true}));break;}
        await new Promise(resolve=>setTimeout(resolve,100));
      }
    });
    </script>"""
    page = page.replace(
        '<div class="comparisons">', compare_controls + '<div class="comparisons">'
    )
    page = page.replace("</html>", compare_script + "</html>")
    atomic_write(output / "index.html", page.encode())
    update("html", "generated", artifact="island-states.json", sha256=digest(raw))
    animation = magnetic_island(
        phase=phases, animation=True, fps=fps, dpi=65, **parameters
    )
    from vaft.diagram._animation import draw_scene

    for index in (0, 24, 47):
        figure, _ = draw_scene(scenes[index])
        temporary = output / f".state-{index}.png"
        figure.savefig(temporary, dpi=100)
        temporary.replace(output / f"state-{index}.png")
    for name in ("pyav",):
        update(name, "generating", artifact=f"island-{name}.mp4")
        try:
            with tempfile.TemporaryDirectory(prefix=".render-", dir=output) as directory:
                path = animation.save(Path(directory) / f"island-{name}.mp4")
                import av

                with av.open(str(path)) as movie:
                    stream = movie.streams.video[0]
                    count = sum(1 for _ in movie.decode(video=0))
                    if count != len(phases):
                        raise ValueError(f"Expected {len(phases)} frames, decoded {count}")
                    rate = float(stream.average_rate)
                    if not np.isclose(rate, fps):
                        raise ValueError(f"Expected {fps} fps, found {rate}")
                destination = output / path.name
                os.replace(path, destination)
                os.replace(path.with_name(path.name + ".json"), destination.with_name(destination.name + ".json"))
                path = destination
            update(
                name,
                "validated",
                frames=count,
                fps=rate,
                bytes=path.stat().st_size,
                sha256=digest(path.read_bytes()),
            )
        except Exception as error:
            update(name, "failed", error=str(error))
            traceback.print_exc()
    print(f"Preview: {output}/index.html", flush=True)


if __name__ == "__main__":
    main()
