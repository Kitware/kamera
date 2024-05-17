#! /usr/bin/python
"""
ckwg +31
Copyright 2018 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

"""
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
