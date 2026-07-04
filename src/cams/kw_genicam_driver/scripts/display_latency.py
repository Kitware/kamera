#!/usr/bin/env python
from __future__ import print_function
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def cb(msg):
    now = time.time()
    msg_time = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1000000000.0)
    print("===")
    print("now        : %f" % now)
    print("msg time   : %f" % msg_time)
    print("msg seconds: %d" % time.gmtime(msg_time).tm_sec)
    print("now delta  : %f" % (now - msg_time))


def main(args=None):
    rclpy.init(args=args)
    node = Node("latency_reader")
    node.create_subscription(Image, "/test/camera/cueing/0/image_raw", cb, 1)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
