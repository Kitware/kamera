## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['rqt_sprokit_adapter'],
    package_dir={'': 'src'},
    requires=['rospy', 'std_msgs', 'std_srvs'],
    scripts=['scripts/rqt_sprokit_adapter']
)

setup(**setup_args)
