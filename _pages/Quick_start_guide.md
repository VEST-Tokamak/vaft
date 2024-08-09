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

To use this tool you have to firstly install git. To install the git you can follow this [link](./Installation.md).(You can skip this stage if you already used git before.) If you are familiar with github then clone and install this [vest](https://github.com/satelite2517/vest). 

If you are not then write the below command in your cmd.

```bash
git init
git clone https://github.com/satelite2517/vest.git
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
>>> ods_39020 = vest.load_shot(shot = 39020)
```

Save 
=====
Saving the data in server is not supported yet. Currently you can only save in local.

```python
>>> import vest
>>> ods_39020 = vest.load_shot(shot = 39020)
>>> vest.save_local(ods_39020, './ods_39020.h5')
```