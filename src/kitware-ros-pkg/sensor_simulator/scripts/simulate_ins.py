#! /usr/bin/python
"""
ckwg +31
Copyright 2017 by Kitware, Inc.
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
import time

# ROS imports
import rospy
import genpy
from custom_msgs.msg import GSOF_INS
from kamera.sensor_models import quaternion_from_euler
from kamera.sensor_models.nav_conversions import ned_quat_to_enu_quat


def main():
    # Launch the node.
    node = 'simulate_ptz_camera'
    rospy.init_node(node, anonymous=False)
    node = rospy.get_name()

    lat = rospy.get_param('%s/lat' % node)
    lon = rospy.get_param('%s/lon' % node)
    height = rospy.get_param('%s/height' % node)

    yaw0 = rospy.get_param('%s/nominal_yaw' % node)
    pitch0 = rospy.get_param('%s/nominal_pitch'  % node)
    roll0 = rospy.get_param('%s/nominal_roll' % node)

    yaw_range = rospy.get_param('%s/yaw_range' % node)
    pitch_range = rospy.get_param('%s/pitch_range' % node)
    roll_range = rospy.get_param('%s/roll_range' % node)
    motion_rate = rospy.get_param('%s/motion_rate' % node)
    pub_rate = rospy.get_param('%s/pub_rate' % node)

    topic = rospy.get_param('%s/topic' % node)

    rospy.loginfo('lat (deg): %s' % str(lat))
    rospy.loginfo('lon (deg): %s' % str(lon))
    rospy.loginfo('height (m): %s' % str(height))
    rospy.loginfo('nominal_yaw (deg): %s' % str(yaw0))
    rospy.loginfo('nominal_pitch (deg): %s' % str(pitch0))
    rospy.loginfo('nominal_roll (deg): %s' % str(roll0))
    rospy.loginfo('yaw_range (deg): %s' % str(yaw_range))
    rospy.loginfo('pitch_range (deg): %s' % str(pitch_range))
    rospy.loginfo('roll_range (deg): %s' % str(roll_range))
    rospy.loginfo('Motion rate (deg/s): %s' % str(motion_rate))
    rospy.loginfo('Publish rate: %s' % str(pub_rate))
    rospy.loginfo('Odometry topic: %s' % str(topic))
    # ------------------------------------------------------------------------

    ins_state_pub = rospy.Publisher(topic, GSOF_INS, queue_size=1)

    rate = rospy.Rate(pub_rate)
    t0 = rospy.get_time()
    yaw = pitch = roll = 0
    while not rospy.is_shutdown():
        t = rospy.get_time() - t0
        yaw = yaw0 + yaw_range*np.sin(t*motion_rate/yaw_range*2*np.pi)
        pitch = pitch0 + pitch_range*np.sin(t*motion_rate/pitch_range*2*np.pi)
        roll = roll0 + roll_range*np.sin(t*motion_rate/roll_range*2*np.pi)

        print('yaw:', yaw, 'pitch:', pitch, 'roll:', roll)
        msg = GSOF_INS()
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = height
        msg.align_status = 4
        msg.gnss_status = 1
        msg.north_velocity = 50
        msg.east_velocity = 20
        msg.down_velocity = 1
        msg.total_speed = np.sqrt(msg.north_velocity**2 + msg.east_velocity**2 + msg.down_velocity**2)


        msg.heading = yaw
        msg.pitch = pitch
        msg.roll = roll
        msg.track_angle = 5
        msg.time = rospy.get_time()

        msg.header.stamp = genpy.Time.from_sec(rospy.get_time())
        ins_state_pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
