from distutils.core import setup
from setuptools import find_packages
import numpy

setup(
    include_dirs=[numpy.get_include(), '.'],
    name="darwinian_shift", packages=find_packages()
)