from glob import glob

from setuptools import setup

package_name = "wxpython_gui"

setup(
    name=package_name,
    version="1.0.0",
    packages=[
        package_name,
        package_name + ".system_control_panel",
    ],
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.xml")),
        ("share/" + package_name + "/shapefiles", glob("shapefiles/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Adam Romlein",
    maintainer_email="adam.romlein@kitware.com",
    description="KAMERA system control panel GUI (wxPython)",
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "system_control_panel = wxpython_gui.system_control_panel_main:main",
        ],
    },
)
