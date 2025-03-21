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
        "h5py",
        "numpy",
        "uncertainties",
        "omas",
        "matplotlib",
        "h5pyd @ git+https://github.com/hdfgroup/h5pyd.git",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
)