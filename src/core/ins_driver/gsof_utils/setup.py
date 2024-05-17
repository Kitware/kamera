#!/usr/bin/env python
from setuptools import find_packages, setup

packages = find_packages()
deps = ["typing", "six", "enum34", "pytz", "python-dateutil", "boltons", "redis", "benedict"]

setup(
    name="gsof_utils",
    version="0.1.0",
    script_name="setup.py",
    python_requires=">2.7",
    zip_safe=False,
    packages=packages,
    install_requires=deps,
    include_package_data=True,
    entry_points={"console_scripts": ["avx_time=gsof_utils.avx_time:main",]},
)
