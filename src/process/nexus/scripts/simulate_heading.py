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

# ROS imports
import rospy
from std_msgs.msg import String


def main():
    # Launch the node.
    node = 'simulate_heading'
    rospy.init_node(node, anonymous=False)
    node = rospy.get_name()
    
    heading0 = rospy.get_param(''.join([node,'/nominal_heading']))
    heading_range = rospy.get_param(''.join([node,'/heading_range']))
    motion_rate = rospy.get_param(''.join([node,'/motion_rate']))
    pub_rate = rospy.get_param(''.join([node,'/pub_rate']))
    topic = rospy.get_param(''.join([node,'/topic']))
    
    print('Nominal heading (deg): ', heading0)
    print('Heading range: ', heading_range)
    print('Motion rate (deg/s): ', motion_rate)
    print('Publish rate: ', pub_rate)
    print('BaselineHeading topic: ', topic)
    # ------------------------------------------------------------------------
    
    heading_pub = rospy.Publisher(topic, String, queue_size=1)
    
    rate = rospy.Rate(pub_rate)
    t0 = rospy.get_time()
    heading = heading0
    while not rospy.is_shutdown():
        if heading_range > 0 and motion_rate > 0:
            t = rospy.get_time() - t0
            heading = heading0 + heading_range*np.sin(t*motion_rate/heading_range*2*np.pi)
        
        print('heading:', heading)
        msg = String("foo")
        # msg.heading = heading*1000
        # msg.n_sats = 10
        # t = rospy.get_time()
        # msg.header.stamp.secs = int(np.floor(t))
        # msg.header.stamp.nsecs = int((t - msg.header.stamp.secs)*1e9)
        heading_pub.publish(msg)
        rate.sleep()

    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
