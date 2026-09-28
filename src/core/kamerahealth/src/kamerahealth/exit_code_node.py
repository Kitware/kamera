#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8


class ExitCodeNode(Node):
    """Test node: exits with a configurable code, optionally spinning until a
    /shutdown message arrives."""

    def __init__(self):
        super().__init__("exit_code_node")
        self.exit_code = self.declare_parameter("exit_code", 0).value
        self.do_spin = self.declare_parameter("spin", False).value
        self.shutdown_requested = False
        self.create_subscription(Int8, "/shutdown", self.cb_shutdown, 10)
        print("param spin: {} exit_code: {}".format(self.do_spin, self.exit_code))

    def cb_shutdown(self, msg):
        self.get_logger().info(
            "received shutdown request message: {}".format(msg.data))
        self.shutdown_requested = True


def main(args=None):
    print("argv: {}".format(sys.argv))
    rclpy.init(args=args)
    node = ExitCodeNode()

    if node.do_spin:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)

    if node.exit_code == 0:
        node.get_logger().info("Clean shutdown requested")
    else:
        node.get_logger().warning(
            "Code shutdown requested: {}".format(node.exit_code))

    exit_code = node.exit_code
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
