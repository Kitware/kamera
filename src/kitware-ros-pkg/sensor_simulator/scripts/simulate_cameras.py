#! /usr/bin/python
from __future__ import division, print_function
import numpy as np

# ROS imports
import rospy
import rospkg

# kitware-ros-pkg imports
import sensor_simulator.camera_simulator as camera_simulator

rospack = rospkg.RosPack()


def main():
    # Launch the node.
    node = 'simulate_cameras'
    rospy.init_node(node, anonymous=False)
    node = rospy.get_name()

    frame_rate = rospy.get_param('%s/frame_rate' % node, default=4)
    # ------------------------------------------------------------------------


    # Define camera simulator
    ir_cam = camera_simulator.CameraSimulator(res_x=540, res_y=512,
                                              encoding='mono8',
                                              image_topic='ir/image_raw')

    eo_cam = camera_simulator.CameraSimulator(res_x=6576, res_y=4384,
                                              encoding='bayer_grbg8',
                                              image_topic='rgb/image_raw')

    uv_cam = camera_simulator.CameraSimulator(res_x=6576, res_y=4384,
                                              encoding='mono8',
                                              image_topic='uv/image_raw')

    rate = rospy.Rate(frame_rate)
    rospy.loginfo('Publishing images at %0.1f Hz' % frame_rate)
    while not rospy.is_shutdown():
        ir_cam.publish_test_image()
        eo_cam.publish_test_image()
        uv_cam.publish_test_image()
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
