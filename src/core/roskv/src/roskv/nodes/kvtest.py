#! /usr/bin/python
# -*- coding: utf-8 -*-

import warnings
import rospy
from roskv.impl.rosparam_kv import RosParamKV


def ros_test():
    node = "kvtest"
    rospy.init_node(node)
    print("ROS Test. Parent: ", rospy.get_namespace())
    kv = RosParamKV()
    node_name = rospy.get_name()
    key, val = "foo", "bar_from_{}".format(node)
    x = kv.put(key, val)
    res = kv.get(key)
    print("{}: {}".format(key, res))

    key = "foostruct"
    val = {"name": "bar_from_{}".format(node), "nest": {"spam": "eggs", "num": 42}}
    x = kv.put(key, val)
    res = kv.get(key)
    print("{}: {}".format(key, res))
    res = kv.get("foostruct/nest/num")
    print("{}: {} ({})".format(key, res, type(res)))

    key = "test_list"
    kv.put(key, [1, 2, 3])
    res = kv.get(key)
    print("{}: {} ({})".format(key, res, type(res)))


def main():
    try:
        ros_test()
    except Exception as exc:
        warnings.warn("ros_test: {}: {}".format(exc.__class__.__name__, exc))


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
