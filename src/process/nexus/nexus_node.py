#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy

from nodes.img_nexus import Nexus


def set_up_nexus():
    node = 'img_nexus'
    rospy.init_node(node)
    print('Parent', rospy.get_namespace())
    node_name = rospy.get_name()
    for param in rospy.get_param_names():
        print(param)
    verbosity = rospy.get_param('~verbosity')
    rgb_topic = rospy.get_param('~rgb_topic')
    ir_topic = rospy.get_param('~ir_topic')
    uv_topic = rospy.get_param('~uv_topic')
    out_topic = rospy.get_param('~out_topic')
    max_wait = rospy.get_param('/max_frame_period', 444) / 1000.0

    return Nexus(rgb_topic, ir_topic, uv_topic, out_topic, max_wait,
          verbosity=verbosity)


def main():
    nexus = set_up_nexus()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
