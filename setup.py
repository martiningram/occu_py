from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name="occu_py",
    version=getenv("VERSION", "LOCAL"),
    description="Occupancy detection modelling in python",
    packages=find_packages(),
)
