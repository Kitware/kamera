#! /usr/bin/python
from __future__ import division, print_function
import numpy as np
import os
import cv2
import time

# ROS imports
import rclpy
import rclpy.logging
from rclpy.node import Node
from custom_msgs.msg import ImageSpaceDetectionList
from cv_bridge import CvBridge, CvBridgeError


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


# Instantiate CvBridge
bridge = CvBridge()


class ChipSaver(object):
    def __init__(self, node, det_topic, image_directory, ext='jpg'):
        rclpy.logging.get_logger('save_chips').info(
            'Saving chips for detection topic det_topics: %s' % det_topic)
        self.image_directory = image_directory
        self.image_subscriber = node.create_subscription(
            ImageSpaceDetectionList, det_topic, self.callback_ros, 10)
        self.ext = ext

    def callback_ros(self, msg):
        """Method that receives messages published on self.image_topic

        :param image_msg: ROS detection message.
        :type image_msg: ImageSpaceDetectionList

        """
        frame_id = msg.header.frame_id
        frame_time = int(np.round(_stamp_to_sec(msg.header.stamp)*100))

        frame_id = frame_id.replace('/','_')

        for det in msg.detections:
            image_msg = det.image_chip
            try:
                # Convert your ROS Image message to OpenCV2
                raw_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            except CvBridgeError as e:
                print(e)
                return None

            fname = ''.join([self.image_directory,'/',frame_id,'_',str(frame_time),
                             '_',str(det.confidence),'.',self.ext])
            print('saving:', fname)
            cv2.imwrite(fname, raw_image)


def main(args=None):
    rclpy.init(args=args)
    node = Node('save_detection_chips_to_disk')

    # -------------------------- Read Parameters -----------------------------
    det_topics = []
    i = 1
    while True:
        topic = node.declare_parameter('detection_topic%i' % i, '').value
        if not topic:
            break
        det_topics.append(topic)
        i += 1

    image_directory = node.declare_parameter('image_directory', '.').value
    image_directory = '%s/%i' % (image_directory, int(time.time()))

    ext = node.declare_parameter('image_extension', 'jpg').value

    try:
        os.makedirs(image_directory)
    except OSError:
        pass
    # ------------------------------------------------------------------------

    savers = [ChipSaver(node, t, image_directory, ext) for t in det_topics]
    (void_ref,) = (savers,)

    rclpy.spin(node)


if __name__ == '__main__':
    main()
