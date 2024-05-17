#!/usr/bin/env python
from __future__ import print_function
import time

import rospy
from sensor_msgs.msg import Image


def cb(msg):
    now = time.time()
    msg_time = msg.header.stamp.secs + (msg.header.stamp.nsecs / 1000000000.0)
    print("===")
    print("now        : %f" % now)
    print("msg time   : %f" % msg_time)
    print("msg seconds: %d" % time.gmtime(msg_time).tm_sec)
    print("now delta  : %f" % (now - msg_time))


rospy.init_node("latency_reader", anonymous=True)
rospy.Subscriber("/test/camera/cueing/0/image_raw", Image,
                 cb, queue_size=1)

rospy.spin()
