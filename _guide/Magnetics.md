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
>>> import vest
>>> shot_39915 = vest.load(39915)
>>> vest.plot_inboard_B_z(shot_39915)
```
![Inboard $B_z$ of shot #39915](https://satelite2517.github.io/vest/assets/images/magnetics/Inboard_B_z.png)

Outboard $B_{z}$
=====
```python 
>>> vest.plot_outboard_B_z(shot_39915)
```
![Outboard $B_z$ of shot #39915](https://satelite2517.github.io/vest/assets/images/magnetics/Outboard_B_z.png)


Side $B_{z}$
=====
```python 
>>> vest.plot_side_B_z(shot_39915)
```
![Side $B_z$ of shot #39915](https://satelite2517.github.io/vest/assets/images/magnetics/Side_B_z.png)



Inboard flux loop
=====
```python 
>>> vest.plot_inboard_flux_loop(shot_39915)
```
![Side $B_z$ of shot #39915](https://satelite2517.github.io/vest/assets/images/magnetics/Inboard_flux_loop.png)


Outboard flux loop
=====
```python 
>>> vest.plot_outboard_flux_loop(shot_39915)
```
![Side $B_z$ of shot #39915](https://satelite2517.github.io/vest/assets/images/magnetics/Outboard_flux_loop.png)



Plasma current
=====
```python 
>>> vest.plot_plasma_current(shot_39915)
```
![Plasma current of shot #39915](https://satelite2517.github.io/vest/assets/images/magnetics/plasma_current.png)


Diamagnetic Flux
=====
```python 
>>> vest.plot_diamagnetic_flux(shot_39915)
```
![Diamagnetic Flux of shot #39915](https://satelite2517.github.io/vest/assets/images/magnetics/diamagnetic_flux.png)


Credit : Hongsik-yun (peppertonic18@snu.ac.kr)
