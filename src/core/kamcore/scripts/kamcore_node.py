#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy


def main():
    print("[i] started KAMCORE")
    rospy.init_node("kamcore")
    rospy.loginfo("started KAMCORE")
    rospy.spin()


if __name__ == "__main__":
    main()
