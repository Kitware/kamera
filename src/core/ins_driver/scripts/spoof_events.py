#!/usr/bin/python
import rospy

import math
from custom_msgs.msg import GSOF_EVT
from std_msgs.msg import Header

print("Creating spoof publisher.")
spoof_pub = rospy.Publisher("/event", GSOF_EVT, queue_size=1)

def main():
    print("Initializing spoof node.")
    rospy.init_node("event_spoofer")
    sub = rospy.Subscriber("/trig", Header, pub)
    rospy.spin()


def pub(hmsg):
    t = hmsg.stamp
    s = t.to_sec()
    #ns = t.to_nsec()
    #s = math.floor(ns / 1e9)
    #t = rospy.Time.from_sec(s)
    msg = GSOF_EVT()
    msg.header.stamp = t
    msg.gps_time = t
    msg.sys_time = t
    msg.time = s
    spoof_pub.publish(msg)
    rospy.loginfo("Published event msg.")


if __name__ == "__main__":
    main()
