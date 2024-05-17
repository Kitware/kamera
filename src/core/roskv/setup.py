#!/usr/bin/env python
from setuptools import find_packages, setup
from catkin_pkg.python_setup import generate_distutils_setup

deps = [
    "python-benedict==0.23.2",
    "python-fsutil==0.4.0",
    "python-slugify==4.0.1",
    "typing",
    "six",
]

# this function uses information from package.xml to populate dict
packages = find_packages(where="src")
d = generate_distutils_setup(
    packages=packages,
    package_dir={"": "src"},
    install_requires=deps,
    # entry_points={"console_scripts": ["roskv=roskv.client:main"]},
    scripts=["scripts/roskv"],
    extras_require={"consul": ["python-consul"], "redis": ["redis"]},
)

setup(**d)
