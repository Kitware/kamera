#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import time
import math
from typing import Optional, Dict
import datetime

# import threading
import cv2

from cv_bridge import CvBridge

# from profilehooks import timecall
from std_msgs.msg import UInt64 as MsgUInt64

from nexus.ros_numpy_lite import ImageEncodingMissingError
from nexus.archiver_core import ArchiverBase, pathsafe_timestamp, make_path, msg_as_dict

bridge = CvBridge()

chan_to_ext = {"rgb": "jpg", "uv": "jpg", "ir": "tif"}


def get_image_msg_meta(msg):
    fields = ["height", "width", "encoding", "is_bigendian", "step"]
    d = {"header": dict()}
    for field in fields:
        x = getattr(msg, field, None)
        d.update({field: str(x)})
    fields = ["seq", "stamp", "frame_id"]
    for field in fields:
        x = getattr(msg.header, field, None)
        d["header"].update({field: str(x)})
    return d


class ArchiveManager(ArchiverBase):
    abridge_fov = {
        "center_view": "C",
        "left_view": "L",
        "right_view": "R",
        "default_view": "D",
    }

    def __init__(
        self, agent_name="ArchiveManager", bytes_halt_archiving=1e9, verbosity=0
    ):
        """
        Class for managing the archiving of data coming from the system.
        By convention, paths are '/' terminated.

        :param agent_name:
        :param bytes_halt_archiving: When free bytes drops below this number, halt
        :param verbosity:
        """
        super(ArchiveManager, self).__init__(agent_name=agent_name, verbosity=verbosity)

    def dump_image_msg(self, msg, mode, fn_template, ext="tif"):
        # type: (genpy.msg, str, str, str) -> Optional[str]
        """
        Write image to disk. Return the path saved to if successful
        :param msg: sensor_msgs image message
        :param mode: rgb/ir etc
        :param fn_template: path to format
        :return: fully substituted path
        """
        if msg is None:
            return
        data = bridge.imgmsg_to_cv2(msg)
        filename = fn_template.format(mode=mode, ext=ext)

        # np.save(filename, data)
        start = time.time()
        # imageio.imwrite(filename, data)
        cv2.imwrite(filename, data, (cv2.IMWRITE_JPEG_QUALITY, 100))
        end = time.time()
        if self.verbosity >= 4:
            print("{} {} {:.3f} sec".format(msg.encoding, data.shape, end - start))
        if self.verbosity >= 2:
            print("Archiver saved: {}".format(filename))

        return filename

    def filename_from_msg(self, msg, mode, time=None):
        # type: (genpy.msg, str) -> str
        ext = self.image_formats[mode]
        if time is None:
            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs
            usecs = int(nsecs / 1e3)
            fracs = float(usecs / 1e6)
            t = secs + fracs
            now = datetime.datetime.utcfromtimestamp(t)
        else:
            now = time
        template = self.fmt_sync_path(now)
        return template.format(mode=mode, ext=ext)

    def dump_sync_image_messages(self, msg_dict, ext="jpg"):
        # type: (dict) -> Dict[str, str]
        """
        Writes images to disk. Returns the path template.
        :param msg_dict: Dictionary of { mode: msg }
        :return: Template path
        """
        start = time.time()
        # todo: dump the event
        metadata = dict()
        event = msg_dict.get("evt")
        ins = msg_dict.get("ins")
        metadata.update({"ins": msg_as_dict(ins), "evt": msg_as_dict(event)})

        # Always round down the last sigfig maintain parity between
        # Python / C++ saving.
        secs = event.header.stamp.secs
        nsecs = event.header.stamp.nsecs
        usecs = int(nsecs / 1e3)
        fracs = float(usecs / 1e6)
        t = secs + fracs
        now = datetime.datetime.utcfromtimestamp(t)
        # no longer flag if time was spoofed
        template = self.fmt_sync_path(now)
        pathdict = {}
        dirname = make_path(template, from_file=True)
        for mode, msg in msg_dict.items():
            if mode in ["ir", "rgb", "uv"]:
                filename = self.filename_from_msg(msg, mode)
                pathdict.update({mode: filename})
                metadata.update({mode: get_image_msg_meta(msg)})
        # Add NUC information, if available
        if "ir" in metadata:
            nucing = "unknown"
            if "nucing" in metadata["ir"]["header"]["frame_id"]:
                nucing = (
                    metadata["ir"]["header"]["frame_id"].split("?")[-1].split("=")[-1]
                )
            metadata["ir"]["is_nucing"] = nucing
            metadata["ir"]["header"]["frame_id"] = "ir"
        # add phase one exif data to meta json, if available
        if "rgb" in metadata:
            if "Phase" in metadata["rgb"]["header"]["frame_id"]:
                fid = metadata["rgb"]["header"]["frame_id"]
                for kv in fid.split(","):
                    k, v = kv.split(":")
                    metadata["rgb"][k] = v
                metadata["rgb"]["header"]["frame_id"] = "rgb"

        metadata["effort"] = self.effort
        metadata["collection_mode"] = self.collection_mode
        metadata["sys_cfg"] = self._sys_cfg
        metadata["save_every_x_image"] = self.save_every_x_image
        try:
            self.dump_json(metadata, time=now)
        except IOError:
            rospy.logger("DISK READ / WRITE ERROR, CHECK MOUNT.")
            self.disk_check(dirname)
        if self.verbosity >= 4:
            print(
                "<archiver> {} {:.3f} sec".format(msg_dict.keys(), time.time() - start)
            )

        return pathdict

    def save_sys_cfg_json(self, sys_dict):
        t = time.time()
        now = datetime.datetime.utcfromtimestamp(rgb_time)
        template = self.fmt_sync_path(now)
        dirname = make_path(template, from_file=True)
        sys_cfg_dir = os.path.dirname(dirname)
        fname = "%s/sys_config.json" % sys_cfg_dir
        print("Saving sys config json to %s" % fname)
        try:
            with open(fname, "w") as fn:
                json.dump(sys_dict[self._sys_cfg], fn)
        except IOError:
            rospy.logger("DISK READ / WRITE ERROR, CHECK MOUNT.")
            self.disk_check(dirname)


class SimpleFileDump(object):
    def __init__(self, archiver, suffix=".json", save_every=10, folder_time=True):
        self.save_every = save_every
        self.suffix = suffix
        self.data = []
        self.archiver = archiver
        self._enable = False

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, v):
        self._enable = bool(v)


class SimpleStatsLogger(SimpleFileDump):
    def __init__(self, archiver=None, suffix=".json", save_every=4, folder_time=True):
        SimpleFileDump.__init__(
            self,
            archiver=archiver,
            suffix=suffix,
            save_every=save_every,
            folder_time=folder_time,
        )

    def dump(self, purge=True, verbose=True):
        if not self.enable:
            return
        filename = self.archiver.dump_log_json(self.data)
        if verbose:
            print(
                "Wrote {} records to {}".format(
                    len(self.data), os.path.abspath(filename)
                )
            )
        if purge:
            self.data = []

    def append(self, x):
        self.data.append(x)
        if len(self.data) >= self.save_every:
            self.dump()


if __name__ == "__main__":
    raise NotImplementedError("Not for direct running at this time")
