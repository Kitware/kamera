#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# this function uses information from package.xml to populate dict
d = generate_distutils_setup(packages=['sysinfo'],
                             package_dir={'': 'src'})

setup(**d)
