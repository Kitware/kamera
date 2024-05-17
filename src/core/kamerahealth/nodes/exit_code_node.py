#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import rospy
from std_msgs.msg import Int8


def cb_shutdown(msg):
    rospy.signal_shutdown("received shutdown request message: {}".format(msg.data))


def main():
    print("agrv: {}".format(sys.argv))
    rospy.init_node("exit_code_node")
    exit_code = rospy.get_param("exit_code_node/exit_code", 0)
    do_spin = rospy.get_param("exit_code_node/spin", False)
    rospy.Subscriber("/shutdown", Int8, cb_shutdown, queue_size=10)
    print("param spin: {} exit_code: {}".format(do_spin, exit_code))

    if do_spin:
        rospy.spin()

    if exit_code == 0:
        rospy.loginfo("Clean shutdown requested")
    else:
        rospy.logwarn("Code shutdown requested: {}".format(exit_code))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
    # test with the following:
    # roslaunch kamerahealth exit_code.launch exit_code:=1 spin:=0 || echo "exit=$?"
