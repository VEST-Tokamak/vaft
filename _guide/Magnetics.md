---
title: Magnetics
author: VEST team
date: 2026-04-24 14:15
category: guide
layout: post
---

# Magnetics Data

The older magnetics page used legacy `vest.*` plotting calls that no longer match the current repository. This page keeps the focus on the data layout and a minimal working example.

## Load one shot

```python
import matplotlib.pyplot as plt
import vaft

ods = vaft.database.load_ods(39915, directory="public")
```

## Example: plasma current trace

```python
time = ods["magnetics.time"]
ip = ods["magnetics.ip.0.data"]

plt.figure(figsize=(8, 4))
plt.plot(time, ip)
plt.xlabel("Time [s]")
plt.ylabel("Plasma current [A]")
plt.title("VEST plasma current")
plt.tight_layout()
plt.show()
```

## Typical paths in the magnetics tree

| Quantity | Example ODS path |
| --- | --- |
| Common time base | `magnetics.time` |
| Plasma current | `magnetics.ip.0.data` |
| Poloidal field probe signal | `magnetics.b_field_pol_probe.<index>.field.data` |
| Flux loop signal | `magnetics.flux_loop.<index>.flux.data` |

## Example figures from the legacy guide

These figures are kept as quick visual references while the page content is being modernized.

![Inboard Bz]({{ site.baseurl }}/assets/images/magnetics/Inboard_B_z.png)
![Outboard Bz]({{ site.baseurl }}/assets/images/magnetics/Outboard_B_z.png)
![Side Bz]({{ site.baseurl }}/assets/images/magnetics/Side_B_z.png)
![Plasma current]({{ site.baseurl }}/assets/images/magnetics/plasma_current.png)

<div class="note-card">
  <strong>Tip:</strong> for richer plotting examples, use the repository notebook
  `notebooks/plotting_sample_using_vaft_plot_module.ipynb` together with a loaded ODS shot.
</div>
