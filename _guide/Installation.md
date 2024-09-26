---
title: Installation
author: Sun jae Lee
date: 2024-08-08 17:29
category: guide
layout: post
---

Install
=====

To use this tool you have to firstly install git. (You can skip this stage if you already used git before.) If you are familiar with github then clone and install this [vest](https://github.com/vest-tokamak/vest). 

If you are not then write the below command in your cmd.

```bash
git init
git clone https://github.com/vest-tokamak/vest.git
cd vest
pip install .
```
(Currently it's not supported in pypi.)


Configuration
=====
First option is just download this
[file](https://github.com/vest-tokamak/vest/blob/main/.hscfg). and place this file to your home directory.

OR  

Follow the below line in your command line. If you don't have any authentication just use :  
__username : reader__    
__passward : vest__  

```bash
>> hsconfigure
Enter new values or accept defaults in brackets with Enter.

Server endpoint []: http://vest.hsds.snu.ac.kr:5101
Username []: [your_username]
Password []: [your_password]
API Key [None]: 
Testing connection...
connection ok
Quit? (Y/N)Y
```
If you want to store or share data then contact this email. (peppertonic18@snu.ac.kr)