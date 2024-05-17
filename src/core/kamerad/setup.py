from setuptools import setup

pkgname = 'kamerad'

packages = [pkgname]
deps = ['pydantic',
        'flask',
        'loguru',
        'redis',
        'six'
]


setup(
    name=pkgname,
    version="0.1.0",
    script_name='setup.py',
    packages=packages,
    install_requires=deps,
)
