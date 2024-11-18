#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# this function uses information from package.xml to populate dict
requires = ["PyGeodesy", "shapely", "pyshp", "wxPython"]

d = generate_distutils_setup(
    packages=["wxpython_gui"], package_dir={"": "src"}, install_requires=[],
)

setup(**d)
