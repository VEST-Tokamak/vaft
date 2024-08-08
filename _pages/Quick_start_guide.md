---
title: Quick start guide
author: Sun jae Lee
date: 2024-08-08 17:28
category: Jekyll
layout: post
---

This tool only support python.

Install
=====

If you are familiar with github then clone and install this [vest](https://github.com/satelite2517/vest). 

If you are not then write the below command in your cmd.

```bash
git clone https://github.com/satelite2517/vest.git
cd vest
pip install .
```
Currently it's not supported in pypi.


Load
=====
To load the data,

```python
>>> import vest
>>> ods_39020 = vest.load_shot(shot = 39020)
```

Plot
=====

Save 
=====
Saving the data in server is not supported yet. Currently you can only save in local.

```python
>>> import vest
>>> ods_39020 = vest.load_shot(shot = 39020)
>>> vest.save_local(ods_39020, './ods_39020.h5')
```