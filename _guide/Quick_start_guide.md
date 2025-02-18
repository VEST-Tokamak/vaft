---
title: Quick start guide
author: Sun jae Lee
date: 2024-08-08 17:28
category: guide
layout: post
---

__This tool only support python.__

Install
=====

To use this tool you have to firstly install git. To install the git you can follow this [link](./Installation.md).(You can skip this stage if you already used git before.) If you are familiar with github then clone and install this [vest](https://github.com/satelite2517/vaft). 

If you are not then write the below command in your cmd.

```bash
git init
git clone https://github.com/satelite2517/vaft.git
cd vest
pip install .
```
(Currently it's not supported in pypi.)

Update
=====
In the terminal, where you're folder for git is in. 
```bash
git pull 
pip install .
```

Configuration
=====
Follow the below line in your command line. If you don't have any authentication just use :
username : reader
passward : vest

```bash
>> hsconfigure
Enter new values or accept defaults in brackets with Enter.

Server endpoint []: http://vest.hsds.snu.ac.kr:5101
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
>>> import vest
>>> shot_39915 = vest.load_shot(shot = 39915)
```

Save 
=====
Saving the data in server is not supported yet. Currently you can only save in local.

```python
>>> import vest
>>> shot_39915 = vest.load_shot(shot = 39915)
>>> vest.save_local(shot_39915, './vest_39915.h5')
```
