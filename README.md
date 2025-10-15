# VAFT(Versatile Analytical Framework for Tokamak) 

This repository serves to provide users with data obtained from experiments on the VEST fusion ST tokamak.
All users are allowed to read these datasets, but if modification/deletion permissions are needed, please contact [email](peppertonic18@snu.ac.kr). Databse system uses HSDS and if you want to get the h5 file format follow [h5pyd github](https://github.com/HDFGroup/h5pyd). This repository utilizes hdf5 and ODS data strucure ([omas github](https://github.com/gafusion/omas?tab=readme-ov-file)), if you need more information about this structure can be found on the following website: [omas](https://gafusion.github.io/omas/).

## Quick start guide

The easiest was to use this framework is to install the framwork with `pip install vaft`. And if the download is finished, enter `hsconfigure` in command line. Then you will have to write `Server endpoint []`, `Username []`, `Password []`.
Write `http://147.46.36.244:5101` for endpoint, and if you have your own username write down, if not just write `reader`, `test` for each username and password. 

```
Testing connection...
connection ok
```

If you have this message, then you have connect to the server. This instruction can be viewed in [link](https://vest-tokamak.github.io/vaft/guide/Quick_start_guide/) too.


## Usage

We are providing you some examples too show what you can get access and to give some guides to anlaysis. 
Each samples are provided in [notebook](https://github.com/VEST-Tokamak/vaft/tree/main/notebooks) folders.
And also you can get access to the homepage for vaft and learn how to use this library. 
[Learn Usage](https://github.com/vest-tokamak/vaft)

Currently you can get access to these examples. 
 - [database_initialization_and_load](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/database_initialization_and_load.ipynb) : This will provide you the main and core functions to use this framework. 
 - [plotting_sample_using_vaft_plot_module](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/plotting_sample_using_vaft_plot_module.ipynb)
 - [profile_fitting_using_equilibrium_and_kinetic_diagnostics](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb)
 - [read_and_convert_data_structure](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/read_and_convert_data_structure.ipynb)
 - [vest_experimental_data_list.ipynb](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/vest_experimental_data_list.ipynb)

 And for the [publication_figures](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/publication_figures.ipynb) will show how to reproduce the results in [presentation]() 

## Reporting bugs

Leave comment on [issue](https://github.com/vest-tokamak/vaft/issues).
