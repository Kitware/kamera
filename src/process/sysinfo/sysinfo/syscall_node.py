#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import subprocess
import shlex

import rclpy
from rclpy.node import Node

from custom_msgs.srv import SysCall

USE_SHELL = False


class SysCallNode(Node):
    def __init__(self):
        super().__init__("syscall")
        self.srv = self.create_service(SysCall, "syscall", self.syscall_cb)

    def syscall_cb(self, msg, resp):
        cmdlist = shlex.split(msg.cmd)
        self.get_logger().info(str(cmdlist))
        try:
            proc = subprocess.Popen(
                cmdlist, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=USE_SHELL
            )
        except Exception:
            exc_type, value, traceback = sys.exc_info()
            self.get_logger().error("subprocess failed: {}: {}".format(exc_type, value))
            resp.stdout = ""
            resp.stderr = "{}: {}".format(exc_type, value)
            return resp
        try:
            outs, errs = proc.communicate()
        except Exception:
            exc_type, value, traceback = sys.exc_info()
            self.get_logger().error("subprocess failed: {}: {}".format(exc_type, value))
            proc.kill()
            outs, errs = proc.communicate()
        resp.stdout = outs.decode() if outs else ""
        resp.stderr = errs.decode() if errs else ""
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = SysCallNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
