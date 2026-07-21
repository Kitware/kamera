#! /usr/bin/python
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
