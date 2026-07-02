from glob import glob

from setuptools import setup

package_name = "ins_driver"

setup(
    name=package_name,
    version="1.0.0",
    packages=["libnmea_navsat_driver"],
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adam Romlein",
    maintainer_email="adam.romlein@kitware.com",
    description="Trimble POS AVX GSOF socket driver",
    license="BSD",
    entry_points={
        "console_scripts": [
            "ins_socket_driver = libnmea_navsat_driver.ins_socket_driver:main",
            "spoof_events = libnmea_navsat_driver.spoof_events:main",
        ],
    },
)
