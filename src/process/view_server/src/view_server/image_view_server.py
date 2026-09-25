#! /usr/bin/python
from __future__ import division, print_function
import os
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager

import numpy as np
import cv2

# ROS imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from roskv.util import hash_ros_msg
from roskv.rendezvous import ConditionalRendezvous

# Kamera Imports
from custom_msgs.msg import SynchronizedImages, SyncedPathImages
from custom_msgs.srv import (
    RequestImageMetadata,
    RequestCompressedImageView,
    RequestImageView,
)
from nexus.pathimg_bridge import (
    PathImgBridge,
    ExtendedBridge,
    coerce_message,
    InMemBridge,
)

from view_server.img_nexus import Nexus, stamp_to_sec


MEM_TRANSPORT_DIR = os.environ.get("MEM_TRANSPORT_DIR", False)
MEM_TRANSPORT_NS = os.environ.get("MEM_TRANSPORT_NS", False)
if MEM_TRANSPORT_DIR:
    print("MEM_TRANSPORT_DIR", MEM_TRANSPORT_DIR)
    membridge = PathImgBridge(name="VS")
    membridge.dirname = MEM_TRANSPORT_DIR
    SyncedImageMsg = SyncedPathImages
elif MEM_TRANSPORT_NS:
    print("MEM_TRANSPORT_NS", MEM_TRANSPORT_NS)
    membridge = InMemBridge(name="VS")
    membridge.dirname = MEM_TRANSPORT_NS
    SyncedImageMsg = SyncedPathImages
else:
    membridge = ExtendedBridge(name="VS")
    SyncedImageMsg = SynchronizedImages

bridge = CvBridge()


class TimeoutLock(object):
    def __init__(self, default_timeout=None):
        self._lock = threading.RLock()
        self._default_timeout = default_timeout

    def acquire(self, blocking=True, timeout=-1):
        return self._lock.acquire(blocking, timeout)

    @contextmanager
    def acquire_timeout(self, timeout=None):
        timeout = self._default_timeout if timeout is None else timeout
        result = self._lock.acquire(timeout=timeout)
        yield result
        if result:
            self._lock.release()

    def release(self):
        self._lock.release()


def get_interpolation(interpolation):
    if interpolation == 4:
        flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP
    elif interpolation == 3:
        flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    elif interpolation == 2:
        flags = cv2.INTER_AREA | cv2.WARP_INVERSE_MAP
    elif interpolation == 1:
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    else:
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    return flags


class ImageViewServer(object):
    """Provides windowed or reduced-resolution image access over network.

    When a request for imagery is made, it is not returned until a new image is
    received by this node.

    """

    def __init__(
        self,
        node,
        sync_image_topic,
        rgb_service_topic=None,
        rgb_metadata_service_topic=None,
        ir_service_topic=None,
        ir_metadata_service_topic=None,
        uv_service_topic=None,
        uv_metadata_service_topic=None,
        rgb_queue=None,
    ):
        """
        :param node: rclpy node owning the ROS interfaces

        :param sync_image_topic: Topic providing SynchronizedImages messages.
        :type sync_image_topic: str

        :param _service_topic: The service topic providing access to windowed or
            reduced-resolution imagery.
        :type service_topic: str

        :param _metadata_service_topic: The service topic providing metadata for
            the raw imagery stored on this server.
        :type service_topic: str

        """
        self.node = node
        self.log = node.get_logger()
        self.image_lock = threading.RLock()
        self.rgb_msg = None
        self.ir_msg = None
        self.uv_msg = None
        # services must be reentrant so image requests can block while
        # subscription callbacks keep flowing on the executor
        self.cb_group = ReentrantCallbackGroup()

        self.frame2newimg = {"rgb_msg": dict(), "ir_msg": dict(), "uv_msg": dict()}
        self.frame2hash = {"rgb_msg": dict(), "ir_msg": dict(), "uv_msg": dict()}

        if rgb_queue is None:
            raise ValueError("You must provide a rgb_queue parameter")
        self.rgb_queue = rgb_queue

        self.img_stamp_blocks = {
            "rgb_msg": ConditionalRendezvous(1),
            "ir_msg": ConditionalRendezvous(1),
            "uv_msg": ConditionalRendezvous(1),
        }
        self.req_hash_blocks = {
            "rgb_msg": ConditionalRendezvous(1),
            "ir_msg": ConditionalRendezvous(1),
            "uv_msg": ConditionalRendezvous(1),
        }

        if isinstance(SyncedImageMsg(), SyncedPathImages):
            sync_image_topic += "_shm"

        hostname_ns = "/" + socket.gethostname()

        self.enabled = {"rgb": True, "uv": True, "ir": True}
        # todo: deal with channel enable config

        def subscribe_to_single_image(modality="rgb"):
            if self.enabled[modality]:
                topic = hostname_ns + "/{}/image_raw".format(modality)
                self.log.info("Subscribing to Images topic '{}'".format(topic))
                node.create_subscription(
                    Image,
                    topic,
                    lambda msg, m=modality: self.any_queue_callback(msg, m),
                    1,
                    callback_group=self.cb_group,
                )

        def subscribe_to_image_service(service_topic, metadata_service_topic, key):
            if service_topic is not None:
                self.log.info(
                    "Creating RequestImageView service to provide "
                    "'%s' image views on topic '%s'" % (key, service_topic)
                )
                node.create_service(
                    RequestImageView,
                    service_topic,
                    lambda req, resp: self.image_patch_service_request(
                        req, resp, key, False
                    ),
                    callback_group=self.cb_group,
                )

                compressed_service_topic = "%s/compressed" % service_topic
                self.log.info(
                    "Creating RequestCompressedImageView service to "
                    "provide '%s' image views on topic '%s'"
                    % (key, compressed_service_topic)
                )
                node.create_service(
                    RequestCompressedImageView,
                    compressed_service_topic,
                    lambda req, resp: self.image_patch_service_request(
                        req, resp, key, True
                    ),
                    callback_group=self.cb_group,
                )

            if metadata_service_topic is not None:
                self.log.info(
                    "Creating RequestImageMetadata service to "
                    "provide '%s' image metadata via "
                    "RequestImageMetadata on topic '%s'" % (key, metadata_service_topic)
                )
            node.create_service(
                RequestImageMetadata,
                metadata_service_topic,
                lambda req, resp: self.metadata_service_topic_request(req, resp, key),
                callback_group=self.cb_group,
            )

        for modality in self.enabled:
            subscribe_to_single_image(modality)
        subscribe_to_image_service(
            rgb_service_topic, rgb_metadata_service_topic, "rgb_msg"
        )
        subscribe_to_image_service(
            ir_service_topic, ir_metadata_service_topic, "ir_msg"
        )
        subscribe_to_image_service(
            uv_service_topic, uv_metadata_service_topic, "uv_msg"
        )

    @property
    def nop_lock(self):
        return self.image_lock

    def any_queue_callback(self, msg, modality="rgb"):
        modality = modality.lower() + "_msg"
        self.log.info("image callback {}".format(modality))
        for frame in self.frame2newimg[modality]:
            try:
                self.frame2newimg[modality][frame][0] = True
            except Exception:
                pass
        with self.nop_lock:
            setattr(self, modality, msg)

    def image_patch_service_request(self, req, resp, modality, compress):
        """
        see custom_msgs/srv/RequestImageView.srv for more details.

        :param modality: Which image stream from which to return an image view.
        :type modality: str {'RGB','IR','UV'}

        """
        tic = time.time()
        req_hash = hash_ros_msg(req)

        with self.nop_lock:
            if modality == "rgb_msg":
                try:
                    img_msg = self.rgb_queue[0]
                except IndexError as exc:
                    self.log.warning("{}: {}".format(exc.__class__.__name__, exc))
                    img_msg = None
            else:
                img_msg = getattr(self, modality)

        if img_msg is None:
            resp.success = False
            return resp

        try:
            stale_hash = req_hash == self.frame2hash[modality][req.frame][0]
        except Exception:
            stale_hash = False
        try:
            newimg = self.frame2newimg[modality][req.frame][0]
        except Exception:
            newimg = True

        if newimg or not stale_hash:
            try:
                image = membridge.imgmsg_to_cv2(img_msg, "passthrough")
            except Exception as exc:
                self.log.error("{}: {}".format(exc.__class__.__name__, exc))
                resp.success = False
                return resp
        else:
            resp.success = True
            return resp
        try:
            self.frame2newimg[modality][req.frame][0] = False
        except Exception:
            self.frame2newimg[modality][req.frame] = deque([False], maxlen=1)
        try:
            self.frame2hash[modality][req.frame][0] = req_hash
        except Exception:
            self.frame2hash[modality][req.frame] = deque([req_hash], maxlen=1)

        flags = get_interpolation(req.interpolation)
        dsize = (req.output_width, req.output_height)
        if modality == "ir_msg":
            if req.apply_clahe:
                stretch_percentiles = [1, 100]
                img = image.astype("uint16")
                mi = np.percentile(img, stretch_percentiles[0])
                ma = np.percentile(img, stretch_percentiles[1])
                normalized = (img - mi) / (ma - mi)
                normalized = normalized * 255
                normalized[normalized < 0] = 0
                image = np.round(normalized).astype("uint8")

        homography = np.reshape(req.homography, (3, 3)).astype(np.float32)
        raw_image = cv2.warpPerspective(image, homography, dsize=dsize, flags=flags)

        if modality in ("ir_msg", "uv_msg"):
            image2 = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        else:
            image2 = raw_image
        if req.show_saturated_pixels and image2.ndim == 3:
            maxval = 255
            saturation_mask = np.all(image2 == maxval, -1)
            image2[:, :, 1][saturation_mask] = 0
            image2[:, :, 2][saturation_mask] = 0

        if compress:
            out_msg = CompressedImage()
            out_msg.format = "jpeg"
            out_msg.data = np.array(cv2.imencode(".jpg", image2)[1]).tobytes()
            out_msg.header = img_msg.header
        else:
            out_msg = bridge.cv2_to_imgmsg(image2, encoding="rgb8")
            out_msg.header = img_msg.header

        toc = time.time()
        self.log.info(
            "{:.2f} Releasing {: >3}".format(
                stamp_to_sec(img_msg.header.stamp), modality[:3]
            )
        )
        print("Time to process request was %0.3fs" % (toc - tic))
        resp.success = True
        resp.image = out_msg
        return resp

    def metadata_service_topic_request(self, req, resp, modality):
        """
        see custom_msgs/srv/RequestImageMetadata.srv for more details.

        :param modality: Which image stream from which to return an image view.
        :type modality: str {'RGB','IR','UV'}

        """
        self.log.error(
            "request: {} modality: {}".format(req, modality),
            throttle_duration_sec=1.0,
        )
        with self.nop_lock:
            img_msg0 = getattr(self, modality)

        if img_msg0 is None:
            resp.success = False
            resp.height = 0
            resp.width = 0
            resp.encoding = ""
        else:
            resp.success = True
            resp.height = img_msg0.height
            resp.width = img_msg0.width
            resp.encoding = img_msg0.encoding

        # invalidate cache lanes
        img_rendezvous = self.img_stamp_blocks[modality]
        if req.release:
            img_rendezvous.release()

        return resp


def set_up_nexus(node, rgb_queue):
    hostname_ns = "/" + socket.gethostname()
    verbosity = node.declare_parameter("verbosity", 9).value
    rgb_topic = node.declare_parameter(
        "rgb_topic", hostname_ns + "/rgb/image_raw"
    ).value
    ir_topic = node.declare_parameter("ir_topic", hostname_ns + "/ir/image_raw").value
    uv_topic = node.declare_parameter("uv_topic", hostname_ns + "/uv/image_raw").value
    out_topic = hostname_ns + "/synched"
    max_wait = node.declare_parameter("max_frame_period", 444.0).value / 1000.0
    send_image_data = node.declare_parameter("send_image_data", True).value
    compress_imagery = node.declare_parameter("compress_imagery", True).value

    nexus = Nexus(
        node,
        rgb_topic,
        ir_topic,
        uv_topic,
        out_topic,
        compress_imagery,
        send_image_data,
        max_wait,
        rgb_queue=rgb_queue,
        verbosity=verbosity,
    )
    return nexus


def main(args=None):
    rclpy.init(args=args)
    node = Node("image_view_server")

    # -------------------------- Read Parameters -----------------------------
    hostname_ns = "/" + socket.gethostname()
    sync_topic_default = hostname_ns + "/synched"
    sync_image_topic = node.declare_parameter(
        "sync_image_topic", sync_topic_default
    ).value
    rgb_service_topic = node.declare_parameter(
        "rgb_service_topic", sync_topic_default + "/rgb_view_service"
    ).value
    ir_service_topic = node.declare_parameter(
        "ir_service_topic", sync_topic_default + "/ir_view_service"
    ).value
    uv_service_topic = node.declare_parameter(
        "uv_service_topic", sync_topic_default + "/uv_view_service"
    ).value

    rgb_metadata_service_topic = node.declare_parameter(
        "rgb_metadata_service_topic", sync_topic_default + "/rgb_metadata_service"
    ).value
    ir_metadata_service_topic = node.declare_parameter(
        "ir_metadata_service_topic", sync_topic_default + "/ir_metadata_service"
    ).value
    uv_metadata_service_topic = node.declare_parameter(
        "uv_metadata_service_topic", sync_topic_default + "/uv_metadata_service"
    ).value
    # ------------------------------------------------------------------------

    # Share debayered RGB images between nexus and image view
    rgb_queue = deque(maxlen=1)
    set_up_nexus(node, rgb_queue)

    ImageViewServer(
        node,
        sync_image_topic,
        rgb_service_topic,
        rgb_metadata_service_topic,
        ir_service_topic,
        ir_metadata_service_topic,
        uv_service_topic,
        uv_metadata_service_topic,
        rgb_queue=rgb_queue,
    )

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
