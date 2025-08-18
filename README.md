# VEST(Versatile Experimental Spherical Torkamak)

This repository serves to provide users with data obtained from experiments on the VEST fusion ST tokamak.
All users are allowed to read these datasets, but if modification/deletion permissions are needed, please contact email. Databse system uses HSDS and if you want to get the h5 file format follow [h5pyd github](https://github.com/HDFGroup/h5pyd). This repository utilizes hdf5 and ODS data strucure ([omas github](https://github.com/gafusion/omas?tab=readme-ov-file)), if you need more information about this structure can be found on the following website: [omas](https://gafusion.github.io/omas/).

## Installation
```bash
git clone https://github.com/vest-tokamak/vaft.git
cd vest
pip install .
```
or

```bash
# Currently not supported planning to publish after beta test
pip install vaft

```

## Initial configuration

in your command line

```
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

## Usage

follow the [Learn Usage](https://github.com/vest-tokamak/vaft/blob/main/docs/load_save_example.md)


## Reporting bugs

Leave comment on [issue](https://github.com/vest-tokamak/vaft/issues) or mail me (satelite2517@snu.ac.kr)
If you need more infomation about VEST find [Nuplex](http://nuplex.snu.ac.kr)