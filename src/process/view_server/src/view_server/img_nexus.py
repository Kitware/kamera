#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import sys
import json
import socket
from typing import List, Tuple, Optional
from datetime import datetime
import threading
import time
import urllib.parse as urllib_parse

from roskv.impl.redis_envoy import RedisEnvoy
import numpy as np
import cv2

# ROS imports
from cv_bridge import CvBridge
from std_msgs.msg import Bool as MsgBool, Float64 as MsgFloat64, String as MsgString
from std_msgs.msg import Header
from sensor_msgs.msg import Image as MsgImage
from custom_msgs.msg import (
    SynchronizedImages,
    GsofEvt,
    GsofIns,
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

log = None  # module logger, set by Nexus


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def stamp_key(stamp):
    """Hashable epoch key from a builtin_interfaces Time (ROS2 messages are unhashable)."""
    return (stamp.sec, stamp.nanosec)


def rostime_to_datetime(stamp):
    return datetime.utcfromtimestamp(stamp_to_sec(stamp))


def check_image_msg(msg, mode="", logger=None):
    # type: (MsgImage, Optional[str], object) -> Optional[np.ndarray]
    """
    Checks validity (presence, size, encoding) of image before sending off
    :param msg: Image message object
    :param mode: [optional] Type of message - used for logging
    :return: Decoded image (this step is pretty fast) or None on failure
    """
    logger = logger or log
    if not msg:
        logger.error("Message {} is None".format(mode))
        return None
    if not msg.encoding:
        logger.error("Message {} is missing encoding".format(mode))
        return None

    data = bridge.imgmsg_to_cv2(msg)  # type: np.ndarray
    if not data.shape:
        logger.error("Message {} has no shape".format(mode))
        return None
    if not data.size:
        logger.error("Message {} has no size".format(mode))
        return None

    if data.ndim not in (2, 3):
        logger.error("Message {} has incorrect ndim: {}".format(mode, data.ndim))
        return None

    if not (np.prod(data.shape)):
        logger.error("Message {} has null shape: {}".format(mode, data.shape))
        return None

    return data


def dump_image_array(filename, data, verbosity=0):
    # type: (str, MsgImage, int) -> None

    start = time.time()

    # Extra jpg params won't hurt writing other formats
    cv2.imwrite(filename, data, (cv2.IMWRITE_JPEG_QUALITY, 100))
    end = time.time()
    if verbosity >= 2:
        print("Image Writer saved: {} in {:.3f}s".format(filename, end - start))


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
        print("OOPS! Duplicate: {}".format(filename))
        fn, ext = os.path.splitext(filename)
        filename = fn + "_dupe" + ext

    # Extra jpg params won't hurt writing other formats
    cv2.imwrite(filename, data, (cv2.IMWRITE_JPEG_QUALITY, 100))
    end = time.time()
    if verbosity >= 4:
        print("{} {} {:.3f} sec".format(msg.encoding, data.shape, end - start))
    if verbosity >= 2:
        print("Image Writer saved: {}".format(filename))


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
    if msg.encoding in bayer_patterns.keys():
        image = bridge.imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, bayer_patterns[msg.encoding])
        debayered_msg = bridge.cv2_to_imgmsg(image, encoding="rgb8")
        debayered_msg.header.stamp = msg.header.stamp
        debayered_msg.header.frame_id = msg.header.frame_id
    elif msg.encoding == "rgb8":
        # message is already decoded, just return
        return msg
    else:
        if log is not None:
            log.warning("Unrecognized Bayer encoding `{}`".format(msg.encoding))
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
        node,
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
        :param node: rclpy node owning the ROS interfaces

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
        global log
        self.node = node
        self.log = node.get_logger()
        log = self.log
        redis_host = os.environ.get("REDIS_HOST", "nuvo0")
        node_host = os.environ.get("NODE_HOSTNAME") or socket.gethostname()
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
        self.rgb_queue = rgb_queue

        self.image_formats = {}
        for chan in ["rgb", "uv", "ir", "evt", "ins"]:
            self.image_formats[chan] = self.envoy.get("/sys/arch/ext_%s" % chan)

        max_wait = 1.0 / max_frame_rate
        self.log.info(
            "node host: {} fov: {}   max_wait: {:.3f}".format(
                node_host, cam_fov, max_wait
            )
        )

        self.node_host = node_host
        self.cam_fov = cam_fov
        self.node_name = node.get_name()

        self.image_lock = threading.RLock()
        self.pub_timer = None
        self._current_epoch = stamp_key(node.get_clock().now().to_msg())
        self.epoch_dict = dict()
        self._msg_dict = dict()
        self._recent_epochs = []
        self.max_wait = max_wait
        self.rolling_success = LowpassIIR()
        self.topics = {
            "rgb_topic": rgb_topic,
            "ir_topic": ir_topic,
            "uv_topic": uv_topic,
            "out_topic": out_topic,
        }
        topic_base = f"/sys/enabled/{cam_fov}"
        self.enabled = self.envoy.get(topic_base)
        self.enabled_list = [k for k, v in self.enabled.items() if v]
        self.full_packet_list = self.enabled_list + ["evt"]
        self.skip_ir = not self.enabled["ir"]
        self.skip_uv = not self.enabled["uv"]
        self._is_archiving = False
        self.verbosity = verbosity
        self._pub_ir_leveled = True  # Outputs a stream of z-normalized IR
        self.archiver = ArchiveManager(node, agent_name="nexus", verbosity=verbosity)
        self.archiver.advertise_services(namespace=node_host)
        self.stats_logger = SimpleStatsLogger(archiver=self.archiver)
        self.pub_missed = {}  # publish when a frame is missed
        self.image_writers = {}

        for mode, topic in (("rgb", rgb_topic), ("ir", ir_topic), ("uv", uv_topic)):
            if not self.enabled[mode]:
                continue
            self.log.info("Subscribing to Images topic '%s'" % topic)
            node.create_subscription(
                MsgImage,
                topic,
                lambda msg, m=mode: self.any_queue_callback(msg, m),
                1,
            )
            self.pub_missed[mode] = node.create_publisher(
                Header, "%s/missed" % mode, 5
            )

        node.create_subscription(
            GsofEvt, "/event", lambda msg: self.any_queue_callback(msg, "evt"), 10
        )

        self.publisher = node.create_publisher(SyncedImageMsg, out_topic, 1)

        self.pub_status = node.create_publisher(MsgString, "status", 3)

        self.stat_pub = node.create_publisher(Stat, "/stat", 3)
        self.pstat_pub = node.create_publisher(
            Stat, self.node_name + "/stat", 3
        )
        self.compress_imagery = compress_imagery
        self.send_image_data = send_image_data

    @property
    def msg_dict(self):
        """Get the most recent message dict"""
        return self.epoch_dict.get(self._current_epoch, {})

    def now_msg(self):
        return self.node.get_clock().now().to_msg()

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
            self.pub_timer.cancel()
            self.node.destroy_timer(self.pub_timer)
            self.pub_timer = None

    def end_of_turn(self, stale_time=1.5):
        """
        Finalize and publish completed packets
        :param stale_time:
        :return:
        """
        now = time.time()
        completed = []
        stale = []
        with self.image_lock:
            for ep in self.epoch_dict:
                msg_dict = self.epoch_dict.get(ep)
                age = now - (ep[0] + ep[1] * 1e-9)
                if all(key in msg_dict for key in self.full_packet_list):
                    self.log.info(
                        "[_] Comp {: >4}: {} {}".format(
                            msg_dict["evt"].event_num, ep, msg_dict.keys()
                        )
                    )
                    completed.append(ep)

                elif age > stale_time:
                    self.log.error(
                        "[_] Messages timed out, epoch {}: {}".format(
                            ep, msg_dict.keys()
                        )
                    )
                    stale.append(ep)
                else:
                    pass

            for candidate in completed + stale:
                msg_dict = self.epoch_dict.pop(candidate)
                self._publish(msg_dict=msg_dict)

        self._recent_epochs = self._recent_epochs[-20:]

    def any_queue_callback(self, msg, modality="evt"):
        modality = modality.lower()
        urlp = urllib_parse.urlparse(msg.header.frame_id)
        qs = urllib_parse.parse_qs(urlp.query)
        self.log.info(
            "<^>{:>3} {:>6}: {:.6f} {}".format(
                modality,
                qs.get("eventNum", ["?"])[0],
                stamp_to_sec(msg.header.stamp),
                qs,
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
        stat.header.stamp = self.now_msg()
        stat.trace_topic = self.node_name + "/queue/" + modality

        self.archiver.disk_check(self.archiver._base, every_nth=4)
        with self.image_lock:
            current_epoch = stamp_key(event_msg.header.stamp)
            msg_dict = self.epoch_dict.get(current_epoch, {})
            if len(msg_dict):
                # If there are already entries in the dict, that means they arrived
                # before this event callback, which is concerning
                self.log.warning("Messages beat event: {}".format(msg_dict.keys()))
            msg_dict.update({"evt": event_msg})
            self.epoch_dict[current_epoch] = msg_dict
            self.log.info(
                "Starting {: >4}: epoch {}, epochs: {}".format(
                    event_msg.event_num, current_epoch, self.epoch_dict.keys()
                )
            )
            if current_epoch in self._recent_epochs:
                self.log.error("Duplicate event! {}".format(event_msg.header))
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
        stat.header.stamp = self.now_msg()
        stat.trace_topic = self.node_name + "/queue/" + modality

        with self.image_lock:
            epoch = stamp_key(image_msg.header.stamp)
            msg_dict = self.epoch_dict.get(epoch, {})
            if "evt" not in msg_dict:
                self.log.warning(
                    "{} Message beat event: {}, epochs: {}".format(
                        modality, epoch, self.epoch_dict.keys()
                    )
                )
            if modality in msg_dict:
                self.log.error(
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

    def check_success(self, msg_dict):
        # type: (dict) -> Tuple[list, list]
        """Returns list of names of all messages present and non-zero in message buffer
        dict, along with list of those that failed"""
        success_list = []
        fail_list = []
        for chan in self.enabled_list + ["evt", "ins"]:
            if chan not in msg_dict:
                self.log.error("Expecting {} Message, not in msg_dict ".format(chan))
                fail_list.append(chan)
                continue

            if chan in ["evt", "ins"]:
                result = True
            else:
                result = check_image_msg(msg_dict[chan], chan, self.log)

            if result is not None:
                success_list.append(chan)
            else:
                self.log.error(f"Registered {chan} as a miss.")

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
            stat.header.stamp = self.now_msg()
            if timer_event is not None:
                self.log.error("Publishing due to timer callback")
            msg_dict["ins"] = self.archiver.latch_ins
            if not any(msg_dict):
                # why does this happen?
                self.log.error("Tried to publish, but no data in buffer")
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

            # Deal with missing event, we still need a header
            if "evt" not in msg_dict:
                self.log.info("Exiting because dummy message.")
                return
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
                    self.log.info("pathdict: {}".format(pathdict))
                else:
                    self.log.info("archived")

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
            self.log.info("All img conversions took %0.3fs." % (time.time() - tic))

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
        self.log.info(infostr)
