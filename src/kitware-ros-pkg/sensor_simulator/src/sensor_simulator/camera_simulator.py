#!/usr/bin/env python
"""
Library handling imagery simulation.
"""
from __future__ import division, print_function
import numpy as np
from numpy import pi
import cv2
import time

# ROS imports
import rospy
import rospkg
from sensor_msgs.msg import Image
import std_msgs.msg
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError
import genpy


# Instantiate CvBridge
bridge = CvBridge()


class CameraSimulator():
    """Camera that outputs test imagery.

    """
    def __init__(self, res_x, res_y, encoding, image_topic):
        """Initialization.

        :param res_x: Horizontal resolution (i.e., number of columns).
        :type res_x: int

        :param res_y: Vertical resolution (i.e., number of rows).
        :type res_y: int

        :param encoding: ROS image encoding str for the output image.
        :type encoding: str

        :param image_topic: Image topic that the camera publishes to.
        :type image_topic: str

        """
        self._res_x = res_x
        self._res_y = res_y

        if encoding in ['mono8','bayer_grbg8']:
            rand_img = np.random.rand(self.res_y, self.res_x)
            rand_img = np.round(rand_img*255).astype(np.uint8)
        elif encoding in ['rgb8','bgr8']:
            rand_img = np.random.rand(self.res_y, self.res_x, 3)
            rand_img = np.round(rand_img*255).astype(np.uint8)
        if encoding in ['mono16','bayer_grbg16']:
            rand_img = np.random.rand(self.res_y, self.res_x)
            rand_img = np.round(rand_img*65535).astype(np.uint16)
        elif encoding in ['rgb16','bgr16']:
            rand_img = np.random.rand(self.res_y, self.res_x, 3)
            rand_img = np.round(rand_img*65535).astype(np.uint16)

        self._rand_img = rand_img
        self._encoding = encoding
        self._image_topic = rospy.resolve_name(image_topic)
        self._image_publisher = rospy.Publisher(image_topic, Image,
                                                queue_size=100)
        self._seq_ind = 0

    @property
    def res_x(self):
        return self._res_x

    @property
    def res_y(self):
        return self._res_y

    @property
    def encoding(self):
        return self._encoding

    @property
    def rand_img(self):
        return self._rand_img

    @property
    def image_topic(self):
        return self._image_topic

    def publish_image(self, image):
        """Publish image from self.image_list with time closest to t.

        The image is published on topic self._image_topic.

        :param image: Image to publish.
        :type image: array-like

        """
        t = time.time()
        image_message = bridge.cv2_to_imgmsg(image, encoding=self.encoding)

        image_message.header.frame_id = self.image_topic
        image_message.header.stamp = genpy.Time.from_sec(t)
        image_message.header.seq = self._seq_ind
        self._image_publisher.publish(image_message)
        self._seq_ind += 1

    def publish_test_image(self):
        """Publish a generic image to be used as a test image.

        """
        image = self.rand_img
        L = np.prod(image.shape)
        shift = int(np.random.randint(0, L, 1))
        image = np.roll(image, shift)
        self.publish_image(image)
        rospy.loginfo('Published %i x %i image with encoding \'%s\' on image '
                      'topic: %s' %  (image.shape[1],image.shape[0],
                                      self.encoding,self._image_topic))
