---
title: Quick start guide
author: Sun jae Lee
date: 2024-08-08 17:28
category: guide
layout: post
---

__This tool only supports Python.__

Install
=====

To use this tool you have to firstly install git. To install the git you can follow this [link](./Installation.md).(You can skip this stage if you already used git before.) If you are familiar with github then clone and install this [vaft](https://github.com/vest-tokamak/vaft). 

If you are not then write the below command in your cmd.

```bash
git clone https://github.com/vest-tokamak/vaft.git
cd vaft
python -m pip install -e .
python -m pip install -e ".[hsds]"
```
or

```bash
python -m pip install vaft
python -m pip install "vaft[hsds]"
```

Update
=====
In the terminal, where you're folder for git is in. 
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
Saving the data in server is not supported yet. Currently you can only save in local.

```python
>>> import vaft
>>> shot_39915 = vaft.database.load_ods(39915, directory="public")
>>> vaft.database.save_ods(shot_39915, shot=39915, env="local")
```
