---
title: Installation
author: Sun jae Lee
date: 2024-08-08 17:29
category: guide
layout: post
---

Install
=====

To use this tool you have to firstly install git. (You can skip this stage if you already used git before.) If you are familiar with github then clone and install this [vest](https://github.com/vest-tokamak/vaft). 

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
If you want to store or share data then contact this email. (peppertonic18@snu.ac.kr)

Notebook example
=====
The installation and first-connection workflow is also summarized in the notebook examples page:

- [Examples]({{ site.baseurl }}/guide/examples/)
- [`database_initialization_and_load.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/database_initialization_and_load.ipynb)

Representative setup commands:

```bash
git clone https://github.com/vest-tokamak/vaft.git
cd vaft
python -m pip install -e .
python -m pip install -e ".[hsds]"
hsconfigure
```
