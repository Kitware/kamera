from glob import glob

from setuptools import setup

package_name = "kamcore"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
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
    description="Common library and monitor nodes for KAMERA",
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "fps_monitor = kamcore.fps_monitor_node:main",
            "cam_param_monitor = kamcore.cam_param_monitor_node:main",
            "shapefile_monitor = kamcore.shapefile_monitor_node:main",
            "seed_redis_config = kamcore.seed_redis_config:main",
        ],
    },
)
