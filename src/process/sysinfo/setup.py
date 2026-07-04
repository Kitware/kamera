from glob import glob

from setuptools import setup

package_name = "sysinfo"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adam Romlein",
    maintainer_email="adam.romlein@kitware.com",
    description="Provides system info",
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "syscall_node = sysinfo.syscall_node:main",
        ],
    },
)
