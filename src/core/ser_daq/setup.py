from glob import glob

from setuptools import setup

package_name = "ser_daq"

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
    description="Drives I/O via serial port",
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "ser_daq_driver = ser_daq.ser_daq_driver:main",
        ],
    },
)
