---
title: Quick start guide
author: Sun jae Lee
date: 2026-07-01 09:10
category: guide
layout: redirect
redirect_to: /workflows/start-here/
---

__This tool only supports Python.__

Install
=====

To use this tool you have to firstly install git. To install git you can follow the
[Installation]({{ site.baseurl }}/guide/Installation/) guide. (You can skip this stage if you
already use git.) Supported Python versions are 3.10 -- 3.13.

Install from source (recommended):

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
```

`h5pyd`, the client for the remote HSDS database, is deliberately left out of the dependency
list: there is no packaging extra for it, and it must be installed with `--no-deps` so that its
own pins do not conflict with the ones VAFT already resolves. Without this step the
Configuration, Load and Save sections below cannot work.

```bash
# HSDS database client (required for remote data access)
python -m pip install --no-deps h5pyd==0.20.0
```

Development tooling (tests, formatters, notebook kernel) lives in the only declared extra:

```bash
python -m pip install -e ".[dev]"
```

Install from PyPI (obsolete — prefer the source install above):

```bash
python -m pip install vaft
python -m pip install --no-deps h5pyd==0.20.0
```

Update
=====
In the terminal, in the folder where you cloned the repository:

```bash
git pull
python -m pip install -e .
```

Configuration
=====
Follow the below line in your command line. If you don't have any authentication just use :
username : reader
password : test

```bash
>> hsconfigure
Enter new values or accept defaults in brackets with Enter.

Server endpoint []: http://147.46.36.244:5101
Username []: $your_username$
Password []: $your_password$
API Key [None]: 
Testing connection...
connection ok
Quit? (Y/N)Y
```

Load
=====
To load the data,

```python
>>> import vaft
>>> shot_39915 = vaft.database.load_ods(39915, directory="public")
```

Save
=====
`save_ods` defaults to `env="server"`, which uploads the shot to HSDS. That path is restricted to
authorized accounts — the public `reader` account is read-only. See
[Contacts]({{ site.baseurl }}/reference/contacts/) to request write access. Writing IMAS images to a
local directory with `env="local"` works with any account.

```python
>>> import vaft
>>> shot_39915 = vaft.database.load_ods(39915, directory="public")

>>> # Local IMAS images — available to every user
>>> vaft.database.save_ods(shot_39915, shot=39915, env="local")

>>> # Upload to HSDS — requires write access
>>> vaft.database.save_ods(shot_39915, shot=39915, directory="public")
```

The full `save_ods` signature is documented in
[Database and data access]({{ site.baseurl }}/guide/Database/).

Notebook example
=====
If you want the full walkthrough with database background, connection checks, and ODS / IDS examples, see the notebook-based [Examples]({{ site.baseurl }}/guide/examples/) guide and start from:

- [`database_initialization_and_load.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/database_initialization_and_load.ipynb)
- [`read_and_convert_data_structure.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/read_and_convert_data_structure.ipynb)

Representative notebook flow:

```python
import vaft

connected = vaft.database.is_connect()
shots = vaft.database.exist_shot("public")
ods = vaft.database.load_ods(39915, directory="public")
```
