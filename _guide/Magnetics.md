---
title: Magnetics
author: Sun jae Lee
date: 2024-08-08 20:11
category: guide
layout: post
---

__This section is about how to plot and analysis the magnetics data.__

Inboard $B_{z}$
=====
```python 
import matplotlib.pyplot as plt
import numpy as np
import vaft

ods = vaft.database.load_ods(39915, directory="public")

def plot_inboard_B_z(ods):  # 영역에 따라 분류
    index_inBz = np.where(ods['magnetics.b_field_pol_probe.:.position.r'] < 0.09)

    # Inboard B filed pol probe (pickup coil)
    n_rows = 9
    fig, axs = plt.subplots(n_rows, 3, figsize=(12, 3 * n_rows))

    for i, ax in zip(index_inBz[0], axs.flatten()):
        radial = ods[f'magnetics.b_field_pol_probe.{i}.position.r']
        vertical = ods[f'magnetics.b_field_pol_probe.{i}.position.z']
        position = '(' + str(radial) + ',' + str(vertical) + ')'
        ax.plot(ods['magnetics.time'], ods[f'magnetics.b_field_pol_probe.{i}.field.data'])
        ax.set_title(f'Inboard B Field Pol Probe at {position}')
        if i % 3 == 0:
            ax.set_ylabel("B Field [T]")
        if i >= 24:
            ax.set_xlabel("Time")

    plt.tight_layout()
    plt.show()

plot_inboard_B_z(ods)
```
![Inboard $B_z$ of shot #39915](https://vest-tokamak.github.io/vest/assets/images/magnetics/Inboard_B_z.png)

Outboard $B_{z}$
=====
```python 
# Use the same loaded ODS object and the corresponding
# `magnetics.b_field_pol_probe` paths for outboard probes.
```
![Outboard $B_z$ of shot #39915](https://vest-tokamak.github.io/vest/assets/images/magnetics/Outboard_B_z.png)


Side $B_{z}$
=====
```python 
# Use the same loaded ODS object and the corresponding
# `magnetics.b_field_pol_probe` paths for side probes.
```
![Side $B_z$ of shot #39915](https://vest-tokamak.github.io/vest/assets/images/magnetics/Side_B_z.png)



Inboard flux loop
=====
```python 
# Use the loaded ODS object and `magnetics.flux_loop`
# paths to inspect inboard flux loop signals.
```
![Side $B_z$ of shot #39915](https://vest-tokamak.github.io/vest/assets/images/magnetics/Inboard_flux_loop.png)


Outboard flux loop
=====
```python 
# Use the loaded ODS object and `magnetics.flux_loop`
# paths to inspect outboard flux loop signals.
```
![Side $B_z$ of shot #39915](https://vest-tokamak.github.io/vest/assets/images/magnetics/Outboard_flux_loop.png)



Plasma current
=====
```python 
plt.figure(figsize=(8, 4))
plt.plot(ods['magnetics.time'], ods['magnetics.ip.0.data'])
plt.xlabel("Time [s]")
plt.ylabel("Plasma current [A]")
plt.tight_layout()
plt.show()
```
![Plasma current of shot #39915](https://vest-tokamak.github.io/vest/assets/images/magnetics/plasma_current.png)


Diamagnetic Flux
=====
```python 
# Plot the corresponding diamagnetic flux path from `ods`
# with the same pattern used above.
```
![Diamagnetic Flux of shot #39915](https://vest-tokamak.github.io/vest/assets/images/magnetics/diamagnetic_flux.png)


Credit : Hongsik-yun (peppertonic18@snu.ac.kr)
