#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# this function uses information from package.xml to populate dict
d = generate_distutils_setup(packages=['kamerahealth'],
                             package_dir={'': 'src'},
                             install_requires=['flask', 'redis', 'six'],
                             scripts=["scripts/y2j", "scripts/check_drop_rate.py"]
                             # entry_points={'console_scripts': [
                             #     'redis_arpd=kamerahealth.redis_arpd:main',
                             #     'y2j=kamerahealth.y2j:main',
                             # ]},
                             )

setup(**d)
