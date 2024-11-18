# -*- coding: utf-8 -*-
from __future__ import division, print_function
import datetime
import threading
import rospy
import StringIO
import cv2
import sys
import time
import os
import wx
import numpy as np
from PIL import Image as PILImage

import std_msgs.msg
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridgeError

from custom_msgs.srv import (
    RequestCompressedImageView,
    RequestImageMetadata,
    RequestImageView,
)
from wxpython_gui.cfg import (
    SYS_CFG,
    bridge,
    missed_frame_store,
    channel_format_status,
    format_status,
)


class UpdateImageThread(threading.Thread):
    """This defines a thread to continually check to see if new imagery is
    available on the remote image server.

    This thread manages requests to a remote image server using
    ROS RequestImageView service calls. When first launched, a ROS
    RequestImageMetadata service is called to extract the resolution and
    encoding of the imagery available on the remote server.

    Then, a loop runs calling get_new_raw_image, which attempts to make a
    RequestImageView service call. The remote image server blocks until a new
    image on that channel is available.

    """

    def __init__(self, parent, srv_topic, compressed=False):
        """
        :param srv_topic: ROS topic for RequestImageView service to provide
            the imagery needed.
        :type srv_topic: str

        """
        threading.Thread.__init__(self)
        self.daemon = True
        self._parent = parent
        self._raw_image_height = None
        self._raw_image_width = None
        self._ros_srv_topic = srv_topic
        self._compressed = compressed
        self._stop = False

        # Derive host/channel names
        segments = srv_topic.strip("/").split("/")
        host = segments[0]
        fov = SYS_CFG["arch"]["hosts"][host]["fov"]
        chan = segments[2].split("_")[0]
        self._img_topic = "/".join(["", host, chan, "image_raw"])
        self._node_host = host
        self._chan = chan
        self._fov = fov
        # Get wx ID of panel this thread is serving to
        self._id = str(parent.wx_panel.GetId())
        self.sub = False
        if chan == "ir":
            self.apply_clahe = True
            model = "flir_a6750"
        elif chan == "uv":
            self.apply_clahe = True
            model = "allied_gt4907_uv"
        elif chan == "rgb":
            self.apply_clahe = False
            model = "gsm_ix120"
            # model = "allied_gt6600_rgb"
            self.sub = True
        self._raw_image_height = SYS_CFG["models"][model]["specs"]["height"]
        self._raw_image_width = SYS_CFG["models"][model]["specs"]["width"]
        self._missed_topic = os.path.join("/", self._node_host, self._chan, "missed")
        self._last_header = std_msgs.msg.Header()  # header of last received image

    @property
    def last_header(self):
        return self._last_header

    def run(self):
        """Overrides Thread.run. Don't call this directly its called internally
        when you call Thread.start().

        """
        if not self.sub:
            self._image_service = rospy.ServiceProxy(
                self._ros_srv_topic, RequestImageView, persistent=True
            )
            self._compress_image_service = rospy.ServiceProxy(
                "%s/compressed" % self._ros_srv_topic,
                RequestCompressedImageView,
                persistent=False,
            )
        else:
            sub_topic = os.path.join("/", self._node_host, self._chan, "image_raw")
            self._sub_to_images = rospy.Subscriber(
                sub_topic, Image, self.process_pub_image
            )
        self._sub_missed = rospy.Subscriber(
            self._missed_topic,
            std_msgs.msg.Header,
            missed_frame_store.cb_missed_fov_chan,
            callback_args=(self._fov, self._chan),
        )
        im_rate = 5
        rate = rospy.Rate(im_rate)
        while True:
            if self._stop:
                return None  # Check for a request to stop.
            if self.sub:
                rate.sleep()
                continue
            try:
                ret = self.get_new_raw_image()
                rate.sleep()
                # if ret is None or ret is False:
                #    h = std_msgs.msg.Header()
                #    h.stamp = rospy.Time.now()
                #    self.update_status_msg(h)
                #    format_status()
                #    print("Stopping requests.")
                #    self.stop()
            except AssertionError as e:
                print(e)
                rate.sleep()
                pass  # wx noise
            except Exception as e:
                rate.sleep()
                self._image_service = rospy.ServiceProxy(
                    self._ros_srv_topic, RequestImageView, persistent=True
                )
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                msg = "{}:{}\n{}: {}".format(
                    fname, exc_tb.tb_lineno, exc_type.__name__, e
                )
                print(msg)

    def invalidate_cache(self):
        """Calling this function causes the ImageViewServer to release any blocking requests at the
        barrier. It seems to work best when called twice in a user interaction callback, once at the start to flush
        the current (stale) image request, and then once after the homography has been updated, to expedite the most
        recent request"""
        try:
            pass

        except rospy.service.ServiceException:
            pass

    def get_homography(self, preview=False):
        """Return homography to warp from panel to raw-image coordinates."""
        raise NotImplementedError("This is an abstract base class")

    def process_pub_image(self, image_msg):
        if len(image_msg.data) == 0:
            return
        if self._compressed:
            sio = StringIO.StringIO(image_msg.data)
            im = PILImage.open(sio)
            preview = np.array(im)
            raw_preview = preview.copy()
        else:
            try:
                # Convert your ROS Image message to OpenCV2
                raw_preview = bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
            except CvBridgeError as e:
                return False
        if SYS_CFG["show_saturated_pixels"]:
            # draw white pixels as red
            maxval = 255
            saturation_mask = np.all(raw_preview == maxval, -1)
            raw_preview[:, :, 1][saturation_mask] = 0
            raw_preview[:, :, 2][saturation_mask] = 0
        # Manually warp Preview image to panel width/height
        homography, output_height, output_width = self.get_homography(preview=True)
        inverse_homography = np.linalg.inv(homography)
        # Set linear interpolation.
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        image = cv2.warpPerspective(
            raw_preview,
            inverse_homography,
            dsize=(int(output_width), int(output_height)),
            flags=flags,
        )
        # Run it without preview=true to get the full resolution warp
        homography, output_height, output_width = self.get_homography(preview=False)
        if self._stop:
            return None  # Check for a request to stop.
        self.update_status_msg(image_msg.header)
        wx.CallAfter(self._parent.update_raw_image, image)
        wx.CallAfter(self._parent.update_remote_homography, homography)
        return True

    def get_new_raw_image(self, release=0):
        """Attempts RequestImageView service call to update raw imagery.

        This method considers the panel, where the imagery will be place,
        height and width and requests a downsampled version of the raw imagery
        that best matches the panel. This allows only the pixels needed to be
        transported from the image server.

        :param: release - enum which tells the service how to release any image threads, or return immediately
        without blocking
        see RequestImageView.srv service
        """
        try:
            homography, output_height, output_width = self.get_homography()

            # Invert and flatten
            homography_ = tuple(np.linalg.inv(homography).ravel())
            frame = "%s_%s_%s" % (self._fov, self._chan, self._id)

            # tic = time.time()
            if self._compressed:
                resp = self._compress_image_service(
                    homography=homography_,
                    output_height=output_height,
                    output_width=output_width,
                    interpolation=0,
                    antialias=False,
                    last_header=self._last_header,
                )

            else:
                resp = self._image_service(
                    homography=homography_,
                    output_height=output_height,
                    output_width=output_width,
                    interpolation=0,
                    antialias=False,
                    release=release,
                    frame=frame,
                    apply_clahe=self.apply_clahe,
                    contrast_strength=SYS_CFG["ir_contrast_strength"],
                    show_saturated_pixels=SYS_CFG["show_saturated_pixels"],
                )
            if not resp.success:
                if self._stop:
                    return None  # Check for a request to stop.
                return

            if self._stop:
                return None  # Check for a request to stop.
            image_msg = resp.image
            if len(image_msg.data) == 0:
                return
            if self._compressed:
                sio = StringIO.StringIO(image_msg.data)
                im = PILImage.open(sio)
                image = np.array(im)
                image = image.copy()
            else:
                try:
                    # Convert your ROS Image message to OpenCV2
                    image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
                except CvBridgeError as e:
                    return False

            if self._stop:
                return None  # Check for a request to stop.
            self.update_status_msg(image_msg.header)
            wx.CallAfter(self._parent.update_raw_image, image)
            wx.CallAfter(self._parent.update_remote_homography, homography)
            # toc = time.time()
            # print("Time to update frame was %0.3fs" % (toc - tic))
            return True

        except rospy.service.ServiceException as e:
            # Too noisy
            # rospy.logwarn('Service failed: {}'.format(e))
            self._image_service = rospy.ServiceProxy(
                self._ros_srv_topic, RequestImageView, persistent=True
            )
            return

        except wx._core.PyDeadObjectError as e:
            rospy.logwarn(e)
            return

    def update_status_msg(self, img_header):
        # type: (std_msgs.msg.Header) -> unicode
        """Update the status bar for an image view"""
        t = img_header.stamp.to_sec()
        t = datetime.datetime.utcfromtimestamp(t)
        # string = format_status(timeval=t, num_dropped=0)
        string = channel_format_status(
            self._fov,
            self._chan,
            timeval=t,
        )
        wx.CallAfter(self._parent.update_status_msg, string)
        return string

    def stop(self):
        self._stop = True
