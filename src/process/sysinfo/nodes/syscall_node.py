#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import subprocess
import shlex
import rospy
from custom_msgs.srv import SysCall

USE_SHELL = False


def syscall_cb(msg):
    cmdlist = shlex.split(msg.cmd)
    rospy.loginfo(cmdlist)
    try:
        proc = subprocess.Popen(
            cmdlist, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=USE_SHELL
        )
    except Exception as exc:
        exc_type, value, traceback = sys.exc_info()
        rospy.logerr("subprocess failed: {}: {}".format(exc_type, value))
        return ('', '{}: {}'.format(exc_type, value))
    try:
        outs, errs = proc.communicate()
    except Exception as exc:
        exc_type, value, traceback = sys.exc_info()
        rospy.logerr("subprocess failed: {}: {}".format(exc_type, value))
        proc.kill()
        outs, errs = proc.communicate()
    stdout = outs.decode() if outs else ""
    stderr = errs.decode() if errs else ""
    return (stdout, stderr)


def main():
    rospy.init_node("syscall")
    syscall_service = rospy.Service("syscall", SysCall, syscall_cb)
    rospy.spin()


if __name__ == "__main__":
    main()
