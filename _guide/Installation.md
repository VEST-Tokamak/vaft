---
title: Installation
author: Sun jae Lee
date: 2026-07-01 09:00
category: guide
layout: post
---

Install
=====

To use this tool you have to firstly install git. (You can skip this stage if you already used git before.) If you are familiar with github then clone and install this [vest](https://github.com/vest-tokamak/vaft).

If you are not then write the below command in your cmd.

Installing from source is the recommended route:

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
```

This pulls in every required dependency, including `aurorafusion` (the Open-ADAS interface used for radiative-power calculations).

Requirements: Python 3.10 -- 3.13, with NumPy 2.x (`numpy>=2,<3`) as the default numerical stack.

### HSDS database client

`h5pyd` is the one package deliberately left out of the dependency list. It is **not** exposed as an optional extra, because its own pins conflict with the rest of the stack. Install it separately with `--no-deps`:

```bash
python -m pip install --no-deps h5pyd==0.20.0
```

This is what provides the `hsconfigure` command used in the next section, so do not skip it if you intend to read from the VEST database.

### Development tooling

`dev` is the only optional-dependency group defined by the project:

```bash
python -m pip install -e ".[dev]"
```

### Install from PyPI (obsolete)

The published package still exists, but the source install above is the supported path:

```bash
pip install vaft
```

The PyPI route does not bring in the HSDS client either — you still need the `--no-deps h5pyd==0.20.0` step above.

Configuration
=====
Follow the below line in your command line. If you don't have any authentication just use :  
__username : reader__    
__password : test__  

```bash
>> hsconfigure
Enter new values or accept defaults in brackets with Enter.

Server endpoint []: http://147.46.36.244:5101
Username []: [your_username]
Password []: [your_password]
API Key [None]: 
Testing connection...
connection ok
Quit? (Y/N)Y
```

A `connection ok` message confirms you are connected.

If you want to store or share data then contact this email. (peppertonic18@snu.ac.kr)

Notebook example
=====
The installation and first-connection workflow is also summarized in the notebook examples page:

- [Examples]({{ site.baseurl }}/guide/examples/)
- [`database_initialization_and_load.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/database_initialization_and_load.ipynb)

Representative setup commands:

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
python -m pip install --no-deps h5pyd==0.20.0   # resolves version conflict (safe for usage)
hsconfigure
```
