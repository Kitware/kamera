from setuptools import setup

package_name = "nexus"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adam Romlein",
    maintainer_email="adam.romlein@kitware.com",
    description="KAMERA archiving and image-transport library",
    license="Apache 2.0",
)
