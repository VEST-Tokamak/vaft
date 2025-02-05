# -----------------------------------
# Instructions for installing this package:
# 0. Run the required installations:
#    Run the following command in the terminal:
#    $ pip install -r requirements.txt
# 1. install VEST package:
#    $ pip install .
#   (if you install VEST package in editable mode for dev, Use the `-e` option: $ pip install -e .)
# Once installed, you can use the 'VEST' package and its associated dependencies in your project.
# -----------------------------------

from setuptools import setup, find_packages

setup(
    name="vest",
    version="0.1",
    packages=find_packages(),

    # These are commented out because the dependencies are already installed in the requirements.txt file.
    # install_requires=[
    #     "h5py",
    #     "numpy",
    #     "uncertainties",
    #     "omas",
    #     "matplotlib",
    # ],
    # dependency_links=[
    #     "git+https://github.com/hdfgroup/h5pyd.git#egg=h5pyd",
    # ],
    # extras_require={
    #     #    $ pip install .[dev] 
    #     "dev": [
    #         "os",
    #         "subprocess",
    #     ],
    # },
)