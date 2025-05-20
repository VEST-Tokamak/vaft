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
    name="vaft",
    version="0.1",
    packages=find_packages(),
    description="Versatile Analytical Framework for Tokamak",
    author="VEST team",
    author_email="satelite2517@snu.ac.kr",
    url="https://github.com/VEST-Tokamak/vaft",
    install_requires=[
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
)