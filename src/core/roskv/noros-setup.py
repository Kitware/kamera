#!/usr/bin/env python
from setuptools import find_packages, setup

packages = find_packages()
deps = ["typing", "six", "enum34", "pytz", "python-dateutil", "boltons", "redis", "benedict"]

setup(
    name="roskv",
    version="0.1.0",
    script_name="noros-setup.py",
    python_requires=">2.7",
    zip_safe=False,
    packages=packages,
    install_requires=deps,
    include_package_data=True,
    entry_points={"console_scripts": ["roskv=roskv.client:main"]},
)
