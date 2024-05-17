#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import json
from datetime import datetime

import rospy

from std_msgs.msg import Header
from custom_msgs.msg import SynchronizedImages, GSOF_EVT, GSOF_INS, SyncedPathImages, Stat
from nexus.archiver import make_path, ArchiveManager
from nexus.archiver_core import msg_as_dict


class EvtListener(object):
    """
    Listens to event messages, and if is_archiving, write a evt.json file alongside sync messages.
    This is mostly for debugging and profiling.

    """

    def __init__(self, verbosity=0):
        rospy.Subscriber('/event', GSOF_EVT, self.event_queue_callback)
        self.archiver = ArchiveManager(agent_name='evt_listener', verbosity=verbosity)
        self.last_gps_time = datetime.now()

    def event_queue_callback(self, msg):
        # type: (GSOF_EVT) -> None
        evttime  = datetime.utcfromtimestamp(msg.gps_time.to_sec())
        dt = evttime - self.last_gps_time
        self.last_gps_time = evttime
        rospy.loginfo("EVT[{: 4d}]: {} dt: {}".format(msg.header.seq, evttime, dt))

        if not self.archiver.is_archiving:
            return
        # - this is a debugging node so we can't actually interfere, lest that mess with something
        make_dir = False
        dd = msg_as_dict(msg)
        filename = self.archiver.dump_json(dd, evttime, mode='evt', make_dir=make_dir)
        rospy.loginfo(filename)


def main():
    rospy.init_node("evt_listener")
    EvtListener()
    rospy.spin()

if __name__ == "__main__":
    main()
