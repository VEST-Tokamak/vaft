from setuptools import setup, find_packages

setup(
    name="VEST",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "h5py",
        "numpy",
        "uncertainties",
        "omas",
        "matplotlib",
    ],
    dependency_links=[
        "git+https://github.com/hdfgroup/h5pyd.git#egg=h5pyd",
    ],
    extras_require={
        "dev": [
            "os",
            "subprocess",
        ],
    },
)
