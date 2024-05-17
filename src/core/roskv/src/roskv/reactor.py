#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import sys
import json
from benedict import benedict
import threading
from six.moves.queue import Queue, Empty

import rospy
import rosservice

from roskv.impl.redis_envoy import RedisEnvoy, StateService
from kamcore.datatypes import TryIntoAttrMxn, ManditoryInitializer, ToDictMxn

# from roskv.util import redis_decode
from custom_msgs.srv import CamGetAttr, CamGetAttrRequest, CamSetAttr, CamSetAttrRequest
from custom_msgs.srv import SetTriggerRate, SetTriggerRateRequest

ROS_INSTANT = rospy.Duration(0, 1)

## Dataclass objects to parse messages
class DC_CamGetAttrReq(TryIntoAttrMxn, ManditoryInitializer, ToDictMxn):
    __slots__ = ["name"]
    __defaults__ = {"name": ""}


class DC_CamSetAttrReq(TryIntoAttrMxn, ManditoryInitializer, ToDictMxn):
    __slots__ = ["name", "value", "dtype"]
    __defaults__ = {"name": "", "value": "", "dtype": ""}


class DC_SetTriggerRateReq(TryIntoAttrMxn, ManditoryInitializer, ToDictMxn):
    __slots__ = ["rate"]
    __defaults__ = {"rate": 1.0}


topic_to_service = {
    "get_camera_attr": {"srv": CamGetAttr, "msg": CamGetAttrRequest, "dc": DC_CamGetAttrReq},
    "set_camera_attr": {"srv": CamSetAttr, "msg": CamSetAttrRequest, "dc": DC_CamSetAttrReq},
    "set_trigger_rate": {"srv": SetTriggerRate, "msg": SetTriggerRateRequest, "dc": DC_SetTriggerRateReq},
}


class Executor(threading.Thread):
    def __init__(self, cmd_queue):
        threading.Thread.__init__(self)
        self.cmd_queue = cmd_queue  # type: Queue

    def run(self):
        print("starting thread")
        while not rospy.core.is_shutdown():
            try:
                # print('waiting')
                cmdpair = self.cmd_queue.get(block=True, timeout=1.0)
                self.issue_cmd(cmdpair)
            except Empty:
                pass

    def issue_cmd(self, cmdpair):
        key, payload = cmdpair
        rospy.loginfo("Issuing: {}: {}".format(key, payload))
        topic = key.replace('/cmd', '')
        topic_kind = os.path.basename(topic)
        dispatch = topic_to_service.get(topic_kind, None)
        if dispatch is None:
            rospy.logwarn("Is not an allowed service kind: {}".format(key))
            return
        try:
            DC = dispatch['dc']
            # payload = json.loads(payload)
            dc = DC(**payload)
            req = dispatch['msg']()
            ## cast it into a message to make sure it works
            dc.into(req)
            print(req)
        except RuntimeError as e:
            rospy.logerr("{}: {}".format(e.__class__.__name__, e))
            return

        try:
            cls = rosservice.get_service_class_by_name(topic)
        except rosservice.ROSServiceException as e:
            rospy.logerr("{}: {}".format(e.__class__.__name__, e))
            return
        proxy = rospy.ServiceProxy(topic, cls, persistent=False)
        print(req)
        print(dc.to_dict())

        res = proxy(**dc.to_dict())
        print(res)




class KeyWatcher(object):
    def __init__(self, cmd_queue, host="nuvo0"):
        node_host = rospy.get_namespace().strip("/")
        self.envoy = RedisEnvoy(host, client_name=node_host + "-reactor")
        self.cmd_recv = benedict()
        self.cmd_queue = cmd_queue
        self.queue_timer = rospy.Timer(ROS_INSTANT, self.cb_sync, oneshot=True)

    def cb_sync(self, event=None):
        try:
            resp = self.envoy.get_dict("/cmd", flatten=True)
        except KeyError as e:
            resp = {}
        cmd_recv = benedict(resp)
        if cmd_recv:
            rospy.loginfo("Got command(s): {}".format(cmd_recv))
            self.cmd_recv.update(cmd_recv)
            try:
                self.envoy.delete_dict("/cmd")
            except KeyError:
                pass
            for cmdpair in cmd_recv.items():
                self.cmd_queue.put(cmdpair)
            self.queue_timer = rospy.Timer(ROS_INSTANT, self.cb_sync, oneshot=True)
        else:
            print('.', end='')

        sys.stdout.flush()


def main():
    # Launch the node.
    node = "redis_reactor"
    rospy.init_node(node, anonymous=False)
    node_name = rospy.get_name()
    cmd_queue = Queue()
    executor = Executor(cmd_queue)
    watcher = KeyWatcher(cmd_queue)
    sync_timer = rospy.Timer(rospy.Duration(0.5), watcher.cb_sync)

    executor.start()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("Interrupt")
        pass
