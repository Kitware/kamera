#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
ckwg +31
Copyright 2018 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

"""
from __future__ import division, print_function, absolute_import
import os
import sys
import json
from typing import List, Tuple, Optional
from datetime import datetime
import threading
from functools import partial
import time

from roskv.impl.redis_envoy import RedisEnvoy
from six.moves import urllib_parse
from six.moves.queue import deque
import numpy as np
import cv2

# ROS imports
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Bool as MsgBool, Float64 as MsgFloat64, String as MsgString
from std_msgs.msg import Header
from sensor_msgs.msg import Image as MsgImage
from custom_msgs.msg import (
    SynchronizedImages,
    GSOF_EVT,
    GSOF_INS,
    SyncedPathImages,
    Stat,
)

# Project imports
from nexus.ros_numpy_lite import ImageEncodingMissingError
from nexus.archiver import make_path
from nexus.archiver import ArchiveManager, SimpleStatsLogger
from nexus.image_leveler import ir_trim_top, eliminate_inband
from nexus.pathimg_bridge import (
    PathImgBridge,
    ExtendedBridge,
    coerce_message,
    InMemBridge,
)
from libnmea_navsat_driver.gsof import GsofSpoofEventDispatch

G_DO_DEBAYER = True

MEM_TRANSPORT_DIR = os.environ.get("MEM_TRANSPORT_DIR", False)
MEM_TRANSPORT_NS = os.environ.get("MEM_TRANSPORT_NS", False)
if MEM_TRANSPORT_DIR:
    print("MEM_TRANSPORT_DIR", MEM_TRANSPORT_DIR)
    membridge = PathImgBridge(name="VS Nexus")
    membridge.dirname = MEM_TRANSPORT_DIR
    SyncedImageMsg = SyncedPathImages
elif MEM_TRANSPORT_NS:
    print("MEM_TRANSPORT_NS", MEM_TRANSPORT_NS)
    membridge = InMemBridge(name="VS Nexus")
    membridge.dirname = MEM_TRANSPORT_NS
    SyncedImageMsg = SyncedPathImages
else:
    membridge = ExtendedBridge(name="VS Nexus")
    SyncedImageMsg = SynchronizedImages

bridge = CvBridge()

bayer_patterns = {}
bayer_patterns["bayer_rggb8"] = cv2.COLOR_BayerBG2RGB
bayer_patterns["bayer_grbg8"] = cv2.COLOR_BayerGB2RGB
bayer_patterns["bayer_bggr8"] = cv2.COLOR_BayerRG2RGB
bayer_patterns["bayer_gbrg8"] = cv2.COLOR_BayerGR2RGB
bayer_patterns["bayer_rggb16"] = cv2.COLOR_BayerBG2RGB
bayer_patterns["bayer_grbg16"] = cv2.COLOR_BayerGB2RGB
bayer_patterns["bayer_bggr16"] = cv2.COLOR_BayerRG2RGB
bayer_patterns["bayer_gbrg16"] = cv2.COLOR_BayerGR2RGB

ros_immediate = rospy.Duration(nsecs=1)


def rostime_to_datetime(stamp):
    # type: (rospy.Time) -> datetime
    t = stamp.to_sec()
    return datetime.utcfromtimestamp(t)


def check_image_msg(msg, mode=""):
    # type: (MsgImage, Optional[str]) -> Optional[np.ndarray]
    """
    Checks validity (presence, size, encoding) of image before sending off
    :param msg: Image message object
    :param mode: [optional] Type of message - used for logging
    :return: Decoded image (this step is pretty fast) or None on failure
    """
    if not msg:
        rospy.logerr("Message {} is None".format(mode))
        return None
    if not msg.encoding:
        rospy.logerr("Message {} is missing encoding".format(mode))
        return None

    data = bridge.imgmsg_to_cv2(msg)  # type: np.ndarray
    rospy.logdebug(data.shape, data.size)
    if not data.shape:
        rospy.logerr("Message {} has no shape".format(mode))
        return None
    if not data.size:
        rospy.logerr("Message {} has no size".format(mode))
        return None

    if data.ndim not in (2, 3):
        rospy.logerr("Message {} has incorrect ndim: {}".format(mode, data.ndim))
        return None

    if not (np.prod(data.shape)):
        rospy.logerr("Message {} has null shape: {}".format(mode, data.shape))
        return None

    return data


def dump_image_array(filename, data, verbosity=0):
    # type: (str, MsgImage, int) -> None

    start = time.time()

    # Extra jpg params won't hurt writing other formats
    cv2.imwrite(filename, data, (cv2.IMWRITE_JPEG_QUALITY, 100))
    end = time.time()
    if verbosity >= 2:
        rospy.loginfo("Image Writer saved: {} in {:.3f}s".format(filename, end - start))


def dump_image_msg(filename, msg, mode="", verbosity=0):
    # type: (str, MsgImage, str, int) -> None
    if not msg.encoding:
        raise ImageEncodingMissingError

    start = time.time()
    data = bridge.imgmsg_to_cv2(msg)  # type: np.ndarray
    if mode == "ir":
        data = eliminate_inband(data)
    if verbosity >= 5:
        print(data.shape, data.size)

    # in the rare event of the filename being a dupe, just tag it as such
    if os.path.exists(filename):
        rospy.logerr("OOPS! Duplicate: {}".format(filename))
        fn, ext = os.path.splitext(filename)
        filename = fn + "_dupe" + ext

    # Extra jpg params won't hurt writing other formats
    cv2.imwrite(filename, data, (cv2.IMWRITE_JPEG_QUALITY, 100))
    end = time.time()
    if verbosity >= 4:
        print("{} {} {:.3f} sec".format(msg.encoding, data.shape, end - start))
    if verbosity >= 2:
        rospy.loginfo("Image Writer saved: {}".format(filename))


def debayer_image_msg(msg, do_debayer=G_DO_DEBAYER):
    # type: (MsgImage, bool) -> MsgImage
    """
    Optionally debayer an image message
    :param msg:
    :param do_debayer:
    :return: processed image message
    """
    if not do_debayer:
        return msg
    tic = time.time()
    if msg.encoding in bayer_patterns.keys():
        rospy.logdebug("DeBayering from encoding {}".format(msg.encoding))
        image = bridge.imgmsg_to_cv2(msg)

        # image = self.gamma_to_linear_lut[image]
        image = cv2.cvtColor(image, bayer_patterns[msg.encoding])
        # image = self.linear_to_gamma_lut[image]

        # White balance
        """
        RGB_rescale = [0.59987517, 1, 0.96323181]
        for i in range(3):
            lut = np.round(np.arange(256)*RGB_rescale[i]).astype(np.uint8)
            image[:,:,i] =  cv2.LUT(image[:,:,i], lut)
        """

        debayered_msg = bridge.cv2_to_imgmsg(image, encoding="rgb8")

        debayered_msg.header.stamp = msg.header.stamp
        debayered_msg.header.frame_id = msg.header.frame_id
        rospy.logdebug("Debayer Time elapsed: {:.3f} s".format(time.time() - tic))
    elif msg.encoding == "rgb8":
        # message is already decoded, just return
        return msg
    else:
        rospy.logwarn("Unrecognized Bayer encoding `{}`".format(msg.encoding))
        return msg
    return debayered_msg


class LowpassIIR(object):
    """
    Digital Infinite impulse response lowpass filter AKA exponential moving
    average. Smooths values.
    """

    def __init__(self, gamma=0.1, init_state=1.0):
        """
        :param gamma: Coefficient for lowpass, (0,1]
        gam=1 -> 100% pass
        """
        self.gamma = gamma
        self.state = init_state

    def update(self, x):
        """
        Push a value into the filter
        :param x: Value of input signal
        :return: Lowpassed signal output
        """
        self.state = (x * self.gamma) + (1.0 - self.gamma) * self.state
        return self.state


class Nexus(object):
    """
    Buffering camera stream. Will gather frames from an incoming topic, push
    them to a deque (automatically sheds to buffer_size) continuous. When rip()
    is called, the most recent frame is returned and the deque cleared.

    """

    symbol_dict = {"rgb": "█", "ir": "▒", "uv": "Ü", "evt": "E"}

    def __init__(
        self,
        rgb_topic,
        ir_topic,
        uv_topic,
        out_topic,
        compress_imagery,
        send_image_data,
        max_wait=0.66,
        rgb_queue=None,
        verbosity=0,
    ):
        """
        :param rgb_topic: Topic to receive RGB ROS Image messages on.
        :type rgb_topic: str

        :param ir_topic: Topic to receive IR ROS Image messages on.
        :type ir_topic: str

        :param uv_topic: Topic to receive UV ROS Image messages on.
        :type uv_topic: str

        :param out_topic: Topic to publish ROS SynchronizedImages messages on.
        :type out_topic: str

        :param send_image_data: Whether to send image bytes or just fname
                                in synced message.
        :type send_image_data: bool

        :param max_wait: Time to wait after receiving one image for the
            other-modality images to arrive (seconds).
        :type max_wait: float

        """
        redis_host = os.environ.get("REDIS_HOST", "nuvo0")
        node_host = rospy.get_namespace().strip("/")
        self.envoy = RedisEnvoy(redis_host, client_name=node_host + "_img_nexus")
        cam_fov = self.envoy.get(
            os.path.join("/sys", "arch", "hosts", node_host, "fov")
        )
        try:
            max_frame_rate = float(self.envoy.get("/sys/arch/max_frame_rate"))
        except Exception as e:
            print(e)
            max_frame_rate = 2.0

        if isinstance(SyncedImageMsg(), SyncedPathImages):
            out_topic += "_shm"

        if rgb_queue is None:
            raise ValueError("You must provide a rgb_queue parameter")
        self.rgb_queue = rgb_queue  # type: dequeue

        self.image_formats = {}
        for chan in ["rgb", "uv", "ir", "evt", "ins"]:
            self.image_formats[chan] = self.envoy.get("/sys/arch/ext_%s" % chan)

        max_wait = 1.0 / max_frame_rate
        rospy.loginfo(
            "node host: {} fov: {}   max_wait: {:.3f}".format(
                node_host, cam_fov, max_wait
            )
        )

        self.node_host = node_host
        self.cam_fov = cam_fov
        self.node_name = rospy.get_name()

        self.image_lock = threading.RLock()
        self.pub_timer = None
        self._current_epoch = rospy.Time.now()
        self.epoch_dict = dict()
        self._msg_dict = dict()
        self._recent_epochs = []
        self.max_wait = max_wait
        self.rolling_success = LowpassIIR()
        self.topics = {
            "rgb_topic": rospy.resolve_name(rgb_topic),
            "ir_topic": rospy.resolve_name(ir_topic),
            "uv_topic": rospy.resolve_name(uv_topic),
            "out_topic": rospy.resolve_name(out_topic),
        }
        topic_base = f"/sys/enabled/{cam_fov}"
        self.enabled = self.envoy.get(topic_base)
        # self.enabled = {
        #    'rgb': rospy.get_param(os.path.join('/cfg/enabled', cam_fov, 'rgb'), True),
        #    'ir': rospy.get_param(os.path.join('/cfg/enabled', cam_fov, 'ir'), True),
        #    'uv': rospy.get_param(os.path.join('/cfg/enabled', cam_fov, 'uv'), True)
        # }
        self.enabled_list = [k for k, v in self.enabled.items() if v]
        self.full_packet_list = self.enabled_list + ["evt"]
        self.skip_ir = not self.enabled["ir"]
        self.skip_uv = not self.enabled["uv"]
        self._is_archiving = False
        self.verbosity = verbosity
        self._pub_ir_leveled = True  # Outputs a stream of z-normalized IR
        self.archiver = ArchiveManager(agent_name="nexus", verbosity=verbosity)
        self.archiver.advertise_services()
        self.stats_logger = SimpleStatsLogger(archiver=self.archiver)
        self.pub_missed = {}  # publish when a frame is missed
        self.image_writers = {}

        if self.enabled["rgb"]:
            rospy.loginfo("Subscribing to Images topic '%s'" % rgb_topic)
            rospy.Subscriber(
                rgb_topic,
                MsgImage,
                self.any_queue_callback,
                callback_args="rgb",
                queue_size=1,
            )
            self.pub_missed["rgb"] = rospy.Publisher("rgb/missed", Header, queue_size=5)

        if self.enabled["ir"]:
            rospy.loginfo("Subscribing to Images topic '%s'" % ir_topic)
            rospy.Subscriber(
                ir_topic,
                MsgImage,
                self.any_queue_callback,
                callback_args="ir",
                queue_size=1,
            )
            self.pub_missed["ir"] = rospy.Publisher("ir/missed", Header, queue_size=5)

        if self.enabled["uv"]:
            rospy.loginfo("Subscribing to Images topic '%s'" % uv_topic)
            rospy.Subscriber(
                uv_topic,
                MsgImage,
                self.any_queue_callback,
                callback_args="uv",
                queue_size=1,
            )
            self.pub_missed["uv"] = rospy.Publisher("uv/missed", Header, queue_size=5)

        rospy.Subscriber(
            "/event", GSOF_EVT, self.any_queue_callback, callback_args="evt"
        )

        self.publisher = rospy.Publisher(out_topic, SyncedImageMsg, queue_size=1)

        self.pub_status = rospy.Publisher("status", MsgString, queue_size=3)

        self.stat_pub = rospy.Publisher("/stat", Stat, queue_size=3)
        self.pstat_pub = rospy.Publisher(self.node_name + "/stat", Stat, queue_size=3)
        self.stat_counter = 0
        self.compress_imagery = compress_imagery
        self.send_image_data = send_image_data

    @property
    def msg_dict(self):
        """Get the most recent message dict"""
        return self.epoch_dict.get(self._current_epoch, {})

    def get_spoof_event(self):
        rospy.logwarn("No event msg detected, generating spoof event")
        return GsofSpoofEventDispatch()

    def is_msg_dict_full(self):
        """Check if all requisite messages have been received (regardless of
        image message content)
        """
        check_evt = "evt" in self.msg_dict
        check_rgb = ("rgb" not in self.enabled_list) or ("rgb" in self.msg_dict)
        check_ir = ("ir" not in self.enabled_list) or ("ir" in self.msg_dict)
        check_uv = ("uv" not in self.enabled_list) or ("uv" in self.msg_dict)
        return all([check_evt, check_rgb, check_ir, check_uv])

    def reset_timer(self):
        if self.pub_timer is not None:
            self.pub_timer.shutdown()
            self.pub_timer = None

    def timer_writer_callback(self, timer_event=None, msg=None, mode=""):
        if not msg:
            raise RuntimeError("No message in timer callback, this should not happen")
        if not mode:
            raise RuntimeError("No mode in timer callback, this should not happen")
        raise NotImplementedError("timer_writer_callback is disabled!")

        self.image_writer_callback(msg=msg, args=mode)

    def image_writer_callback(self, msg, args):
        # type: (MsgImage, str) -> None
        rospy.logwarn("Archiving from nexus DEPRECATED")
        raise NotImplementedError("image_writer_callback is disabled!")
        return
        mode = args
        ext = self.image_formats[mode]

        data = check_image_msg(msg, mode)
        # We want to emit missed frame messages iff message is bad and we are archiving
        if data is None and self.archiver.is_archiving:
            self.pub_missed[mode].publish(msg.header or Header())

        if not self.archiver.is_archiving:
            return
        now = datetime.utcfromtimestamp(msg.header.stamp.to_sec())
        template = self.archiver.fmt_sync_path(now)
        filename = template.format(mode=mode, ext=ext)
        dirname = make_path(filename, from_file=True)
        try:
            dump_image_msg(filename, msg, mode, verbosity=self.verbosity)

        except ImageEncodingMissingError:
            pass  # we logged this with check_image
        except Exception:
            exc_type, value, traceback = sys.exc_info()
            rospy.logerr("dump_image_msg failed: {}: {}".format(exc_type, value))

    def end_of_turn(self, stale_time=1.5):
        """
        Finalize and publish completed packets
        :param stale_time:
        :param timeout_time:
        :return:
        """
        stale_time = rospy.Duration.from_sec(stale_time)
        now = rospy.Time.now()
        completed = []
        stale = []
        with self.image_lock:
            for ep in self.epoch_dict:
                msg_dict = self.epoch_dict.get(ep)
                age = now - ep
                if all(key in msg_dict for key in self.full_packet_list):
                    rospy.loginfo(
                        "[_] Comp {: >4}: {} {}".format(
                            msg_dict["evt"].event_num, ep, msg_dict.keys()
                        )
                    )
                    completed.append(ep)

                elif age > stale_time:
                    rospy.logerr(
                        "[_] Messages timed out, epoch {}: {}".format(
                            ep, msg_dict.keys()
                        )
                    )
                    stale.append(ep)
                else:
                    pass
                    # rospy.loginfo("[_] Partial  : {}".format(ep))

            for candidate in completed + stale:
                msg_dict = self.epoch_dict.pop(candidate)
                # rospy.loginfo("Publishing {}".format(candidate))
                self._publish(msg_dict=msg_dict)

        self._recent_epochs = self._recent_epochs[-20:]

    def any_queue_callback(self, msg, modality="evt"):
        modality = modality.lower()
        urlp = urllib_parse.urlparse(msg.header.frame_id)
        qs = urllib_parse.parse_qs(urlp.query)
        rospy.loginfo(
            "<^>{:>3} {:>6}: {:.6f} {}".format(
                modality, qs.get("eventNum", ["?"])[0], msg.header.stamp.to_sec(), qs
            )
        )

        if modality == "evt":
            self.event_queue_callback(msg, modality=modality)
        else:
            self.sync_queue_callback(msg, modality=modality)

    def event_queue_callback(self, event_msg, modality="evt"):
        modality = modality.lower()
        stat = Stat()
        stat.trace_header = event_msg.header
        stat.node = self.node_name
        stat.header.stamp = rospy.Time.now()
        stat.header.seq = self.stat_counter
        self.stat_counter += 1
        stat.trace_topic = self.node_name + "/queue/" + modality

        self.archiver.disk_check(self.archiver._base, every_nth=4)
        # rospy.loginfo('<^>{:>3} {:>6}: {:.6f}'.format(modality, event_msg.header.seq, event_msg.header.stamp.to_sec()))
        #        rospy.loginfo(modality + ': ' + str(image_msg.header))
        with self.image_lock:
            current_epoch = event_msg.header.stamp
            msg_dict = self.epoch_dict.get(current_epoch, {})
            if len(msg_dict):
                # If there are already entries in the dict, that means they arrived
                # before this event callback, which is concerning
                rospy.logwarn("Messages beat event: {}".format(msg_dict.keys()))
            msg_dict.update({"evt": event_msg})
            self.epoch_dict[current_epoch] = msg_dict
            rospy.loginfo(
                "Starting {: >4}: epoch {}, epochs: {}".format(
                    event_msg.header.seq, current_epoch, self.epoch_dict.keys()
                )
            )
            if current_epoch in self._recent_epochs:
                rospy.logerr("Duplicate event! {}".format(event_msg.header))
            else:
                self._recent_epochs.append(current_epoch)
            self._current_epoch = current_epoch

        self.stat_pub.publish(stat)
        self.end_of_turn()

    def insert_msg(self, image_msg, modality="rgb"):
        modality = modality.lower()
        stat = Stat()
        stat.trace_header = image_msg.header
        stat.node = self.node_name
        stat.header.stamp = rospy.Time.now()
        stat.header.seq = self.stat_counter
        self.stat_counter += 1
        stat.trace_topic = self.node_name + "/queue/" + modality

        # rospy.loginfo('<^>{:>3} {:>6}: {:.6f}'.format(modality, image_msg.header.seq, image_msg.header.stamp.to_sec()))
        #        rospy.loginfo(modality + ': ' + str(image_msg.header))
        with self.image_lock:
            epoch = image_msg.header.stamp
            msg_dict = self.epoch_dict.get(epoch, {})
            if "evt" not in msg_dict:
                rospy.logwarn(
                    "{} Message beat event: {}, epochs: {}".format(
                        modality, epoch, self.epoch_dict.keys()
                    )
                )
            if modality in msg_dict:
                rospy.logerr(
                    "Duplicate message {} in epoch: {}".format(modality, epoch)
                )
            if modality == "rgb":
                image_msg = debayer_image_msg(image_msg)
                self.rgb_queue.append(image_msg)

            msg_dict.update({modality: image_msg})
            self.epoch_dict[epoch] = msg_dict
            # send message off to be written in separate thread (hopefully)

        self.stat_pub.publish(stat)

    def sync_queue_callback(self, image_msg, modality):
        self.insert_msg(image_msg=image_msg, modality=modality)
        self.end_of_turn()

    def sync_queue_callback2(self, image_msg, modality):
        # type: (MsgImage, str) -> None
        """Method that receives messages published on self.image_topic

        :param image_msg: ROS image message.
        :type image_msg: Image

        :param modality: Which image stream from which to return an image view.
        :type modality: str {'EVT', 'RGB','IR','UV'}

        """
        modality = modality.lower()
        stat = Stat()
        stat.trace_header = image_msg.header
        stat.node = self.node_name
        stat.header.stamp = rospy.Time.now()
        stat.header.seq = self.stat_counter
        self.stat_counter += 1
        stat.trace_topic = self.node_name + "/queue/" + modality

        rospy.loginfo(
            "{:>3} {:>6}: {:.6f}".format(
                modality, image_msg.header.seq, image_msg.header.stamp.to_sec()
            )
        )
        #        rospy.loginfo(modality + ': ' + str(image_msg.header))
        with self.image_lock:
            header = image_msg.header
            t = header.stamp.secs + header.stamp.nsecs / 1e9
            t = datetime.utcfromtimestamp(t)
            if modality == "evt":
                raise NotImplementedError("Dead end! shouldn't happen")
            self.stat_pub.publish(stat)

            if header.stamp != self._current_epoch:
                if header.stamp in self._recent_epochs:
                    rospy.logerr("Stale epoch on {}: {}".format(modality, header.stamp))
                else:
                    rospy.logerr("Stale epoch on {}: {}".format(modality, header.stamp))

            #            rospy.loginfo('{:>3} {:>6} {}'.format(modality, image_msg.header.seq, t.isoformat()[11:24]))

            #            rospy.logdebug('{:>3} {:>6} {:.3f}'.format(modality, image_msg.header.seq, image_msg.header.stamp.to_sec()))

            if modality == "rgb":
                image_msg = debayer_image_msg(image_msg)

            if modality == "evt" and modality in self.msg_dict:
                # oops, we got double event before buffer filled
                # publish and roll over message
                raise NotImplementedError("Dead end! shouldn't happen")
                rospy.logwarn(
                    "OOPS double event! Missed packet?: {}".format(self.msg_dict.keys())
                )
                self.publish()
                self.msg_dict.update({modality: image_msg})
                return
            elif modality == "evt" and "ir" in self.msg_dict:
                evt_time = image_msg.header.stamp.to_sec()
                msg_time = self.msg_dict["ir"].header.stamp.to_sec()
                rospy.logwarn("IR beat event by {}".format(evt_time - msg_time))
                if (
                    abs(evt_time - msg_time) < 0.499
                ):  # empirically determined IR can lead by as much as 650 ms but system capped at 2 Hz
                    self.msg_dict.update({modality: image_msg})
                    rospy.logwarn("This is fine")
                else:
                    rospy.logerr(
                        "Publishing incomplete message: {}".format(self.msg_dict.keys())
                    )
                    self.publish()
                    self.msg_dict.update({modality: image_msg})
            elif modality == "evt" and (
                "rgb" in self.msg_dict or "uv" in self.msg_dict
            ):
                # ok we got event but there is stuff? reset the cycle
                # assume event always makes it first
                rospy.logwarn(
                    "got event but stuff in buffer: {}".format(self.msg_dict.keys())
                )
                self.publish()
                self.msg_dict.update({modality: image_msg})
            else:
                self.msg_dict.update({modality: image_msg})

            if self.verbosity > 10:
                # visual symbols for fast debugging
                smsg = "{} Rx {: >4} {}".format(
                    self.symbol_dict.get(modality),
                    modality,
                    rostime_to_datetime(image_msg.header.stamp).isoformat(),
                )
                rospy.loginfo("sync msg: {}".format(smsg))
            if self.is_msg_dict_full():
                self.publish()
            elif self.pub_timer is None:
                # Start a new timer to publish after 'max_wait'.
                self.pub_timer = rospy.Timer(
                    rospy.Duration(self.max_wait), self.publish, oneshot=True
                )
            self.end_of_turn()

    def check_success(self, msg_dict):
        # type: (dict) -> Tuple[list, list]
        """Returns list of names of all messages present and non-zero in message buffer
        dict, along with list of those that failed"""
        success_list = []
        fail_list = []
        for chan in self.enabled_list + ["evt", "ins"]:
            if chan not in msg_dict:
                rospy.logerr("Expecting {} Message, not in msg_dict ".format(chan))
                fail_list.append(chan)
                continue

            if chan in ["evt", "ins"]:
                result = True
            else:
                result = check_image_msg(msg_dict[chan], chan)

            if result is not None:
                success_list.append(chan)
            else:
                rospy.logerr(f"Registered {chan} as a miss.")

        return success_list, fail_list

    def publish(self, timer_event=None, record_stats=True):
        self._publish(timer_event, msg_dict=self.msg_dict, record_stats=record_stats)

    def _publish(self, timer_event=None, msg_dict=None, record_stats=True):
        self.reset_timer()
        print("topics: {}".format(self.topics))
        stat = Stat()
        with self.image_lock:
            stat.node = self.node_name
            stat.trace_topic = self.node_name + "/" + "sync"
            stat.header.stamp = rospy.Time.now()
            stat.header.seq = self.stat_counter
            self.stat_counter += 1
            if timer_event is not None:
                rospy.logerr("Publishing due to timer callback")
            msg_dict["ins"] = self.archiver.latch_ins
            rospy.logdebug("Pub'd: {}".format(msg_dict.keys()))

            if not any(msg_dict):
                # why does this happen?
                rospy.logerr("Tried to publish, but no data in buffer")
                return

            outmsg = SyncedImageMsg()

            success_list, fail_list = self.check_success(msg_dict)
            if self.archiver.is_archiving:
                for mode in fail_list:
                    if mode in self.pub_missed:
                        self.pub_missed[mode].publish(Header())

            success = float(not len(fail_list))
            success_rate = self.rolling_success.update(success)
            record = {
                "ts": datetime.now().isoformat(),
                "have_evt": "evt" in success_list,
                "have_rgb": "rgb" in success_list,
                "have_ir": "ir" in success_list,
                "have_uv": "uv" in success_list,
            }

            # keep only good data messages - this also should simplify dump_sync
            msg_dict = {k: msg_dict[k] for k in success_list}

            s = "img_nexus.py:publish() \n"
            for k, v in msg_dict.items():
                s += "||{:>3}: {:.3f}\n".format(k, v.header.stamp.to_sec())
            # rospy.loginfo(s)

            # Deal with missing event, we still need a header
            if "evt" not in msg_dict:
                rospy.loginfo("Exiting because dummy message.")
                return
                msg_dict["evt"] = self.get_spoof_event()
            event = msg_dict.get("evt")
            stat.meta_json = json.dumps(record)

            for mode in self.enabled_list:
                if mode in msg_dict:
                    if not self.send_image_data:
                        # Only send fname
                        msg = MsgImage()
                    else:
                        msg = msg_dict.get(mode)
                        msg = coerce_message(msg, membridge)
                    # hack to satisfy current file writing scheme
                    temp_header = msg.header
                    msg.header = event.header
                    filename = self.archiver.filename_from_msg(msg, mode)
                    msg.header = temp_header
                    setattr(outmsg, "image_" + mode, msg)
                    setattr(outmsg, "file_path_" + mode, filename)

            pathdict = {}
            if "ir" in msg_dict:
                msg_ir = msg_dict["ir"]
                msg_dict["ir"] = ir_trim_top(msg_ir)

            if self.archiver.is_archiving:
                pathdict = self.archiver.dump_sync_image_messages(msg_dict)
                self.stats_logger.append(record)
                if self.verbosity > 3:
                    rospy.loginfo("pathdict: {}".format(pathdict))
                else:
                    rospy.loginfo("archived")

            # Reset all the image buffers.
            msg_dict = dict()

        # For testing compression artifacts in detector, compress/decompress imagery
        # Takes about 0.7s
        if self.compress_imagery:
            # Process conversion outside of lock
            tic = time.time()
            for mode in self.enabled_list:
                if mode not in ["rgb", "uv"]:
                    continue
                encoding = "rgb8"
                msg = getattr(outmsg, "image_" + mode)
                if len(msg.data) > 0:
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                    img = cv2.imdecode(cv2.imencode(".jpg", cv_img, encode_param)[1], 1)
                    msg = bridge.cv2_to_imgmsg(img, encoding=encoding)
                    setattr(outmsg, "image_" + mode, msg)
            rospy.loginfo("All img conversions took %0.3fs." % (time.time() - tic))

        infostr = "SYN ({} {} {}{}) {: >3.0%}".format(
            "EVT" * record["have_evt"] or "   ",
            "RGB" * record["have_rgb"] or "   ",
            " IR" * record["have_ir"] or "  ",
            " UV" * record["have_uv"] or "",
            success_rate,
        )
        infomsg = MsgString()
        infomsg.data = infostr
        outmsg.header = event.header
        stat.trace_header = event.header
        seq = event.event_num
        stat.link = self.node_name + "/sync/event/{}".format(seq)
        stat.note = "success" if success else "sync_fail"
        self.stat_pub.publish(stat)
        self.pstat_pub.publish(stat)
        self.pub_status.publish(infomsg)
        self.publisher.publish(outmsg)
        rospy.loginfo(infostr)
