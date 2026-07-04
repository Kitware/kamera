#! /usr/bin/python
from __future__ import division, print_function
import numpy as np
import os
import cv2
import threading
import time
from collections import deque
import random
import string

# ROS imports
import rclpy
import rclpy.logging
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError

log = rclpy.logging.get_logger("rebroadcast_infrequent_detections")

# Custom Imports
from sensor_msgs.msg import Image
from custom_msgs.msg import ImageSpaceDetectionList
from custom_msgs.srv import TransformDetectionList


# Instantiate CvBridge
bridge = CvBridge()


def generate_uid(n=20):
    """Return unique identifier string.

    """
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


class DetectionRebroadcast(object):
    def __init__(self, node, det_in_topic, det_out_topic, image_topics,
                 det_transform_service):
        self.node = node

        # Set up locks.
        self.latest_dets_lock = threading.RLock()
        self.latest_dets = None
        self.tformed_versions_latest_dets = None
        self.image_lock = threading.RLock()
        self._det_tform_serv_lock = threading.RLock()

        # Initialize variables.
        self._initialized = False
        self._det_tform_serv = None
        self._image_msg = None
        self._image = None
        self._detection_list = None
        self.image_deque = deque()

        if det_transform_service is not None:
            self.set_det_tform_service(det_transform_service)

        self.det_pub = node.create_publisher(ImageSpaceDetectionList,
                                             det_out_topic, 10)

        node.create_subscription(ImageSpaceDetectionList, det_in_topic,
                                 self.detection_list_callback, 10)

        log.info('Rebroadcasting detections from topic: \'%s\' on topic: '
                 '\'%s\'' % (det_in_topic,det_out_topic))

        log.info("Starting image processing thread")
        self.thread = threading.Thread(target=self.process_images)
        # Entire Python program exits when only daemon threads are left and we
        # want this thread to shutdown as cleanly as possible.
        #self.thread.daemon = True
        self.thread.start()

        for image_topic in image_topics:
            log.info('Receiving images on topic: %s' % image_topic)
            node.create_subscription(
                Image, image_topic,
                lambda msg, t=image_topic: self.ros_image_callback(msg, t), 1)

    @property
    def lock(self):
        """Return the lock on the latest detection list.

        """
        return self._lock

    @property
    def initialized(self):
        """Return whether the heat map image has been initialized.

        """
        return self._initialized

    def set_det_tform_service(self, topic):
        """Set the detection transform service to use.

        This is the service used to transform a ImageSpaceDetectionList message
        from one frame_id to another.

        :param topic: Topic of the detection transform service.
        :type topic: str

        """
        log.info('Waiting for detection list transformation service '
                 '\'%s\' to come alive' % topic)
        client = self.node.create_client(TransformDetectionList, topic)
        while not client.wait_for_service(timeout_sec=1.0) and rclpy.ok():
            pass
        with self._det_tform_serv_lock:
            self._det_tform_serv = client

    def detection_list_callback(self, msg):
        """Receive a detection list.

        :param msg: Detection list message.
        :type msg: ImageSpaceDetectionList

        """
        log.info('Received detection')
        with self.latest_dets_lock:
            self.latest_dets = msg
            self.tformed_versions_latest_dets = {msg.header.frame_id:msg}

    def ros_image_callback(self, msg, topic):
        """Receive an image.

        :param msg: Image.
        :type msg: Image

        """
        # Lock so that only one message can initialize.
        if self.latest_dets is None:
            log.info('Received image from message '
                     'topic \'%s\', but have not received detections, '
                     'so skipping.' % topic)
            return
        else:
            log.info('Received image from message topic \'%s\'' % topic)

        with self.image_lock:
            self.image_deque.appendleft(msg)
            if len(self.image_deque) > 3:
                self.image_deque.pop()

    def process_images(self):
        while rclpy.ok():
            with self.image_lock:
                if len(self.image_deque) == 0:
                    continue

                image_msg = self.image_deque.pop()

            if image_msg.encoding == 'mono8':
                img = bridge.imgmsg_to_cv2(image_msg, 'mono8')
            elif image_msg.encoding in ['rgb8','bgr8']:
                img = bridge.imgmsg_to_cv2(image_msg, 'rgb8')
            else:
                raise Exception('Unhandled image encoding: %s' %
                                image_msg.encoding)

            with self.latest_dets_lock:
                fid1 = image_msg.header.frame_id
                if fid1 not in self.tformed_versions_latest_dets:
                    try:
                        with self._det_tform_serv_lock:
                            req = TransformDetectionList.Request()
                            req.src_detections = self.latest_dets
                            req.dst_frame_id = fid1
                            future = self._det_tform_serv.call_async(req)
                            while not future.done() and rclpy.ok():
                                time.sleep(0.01)
                            resp = future.result()
                            msg1 = resp.dst_detections
                            self.tformed_versions_latest_dets[fid1] = msg1
                    except Exception as e:
                        log.error('Could not transform detections from '
                                  'source frame_id \'%s\' to destination '
                                  '\'%s\' because %s' %
                                  (self.latest_dets.header.frame_id, fid1, e))
                        raise e

                msg_tformed = self.tformed_versions_latest_dets[fid1]
                msg0 = self.latest_dets
                msg = ImageSpaceDetectionList()
                msg.header.stamp = image_msg.header.stamp
                msg.header.frame_id = msg0.header.frame_id
                msg.image_width = msg0.image_width
                msg.image_height = msg0.image_height
                msg.detections = []

                for i in range(len(msg_tformed.detections)):
                    l = msg_tformed.detections[i].left
                    r = msg_tformed.detections[i].right
                    t = msg_tformed.detections[i].top
                    b = msg_tformed.detections[i].bottom

                    l = np.maximum(l, 0)
                    t = np.maximum(t, 0)
                    r = np.minimum(r, img.shape[1])
                    b = np.minimum(b, img.shape[0])

                    if l >= r or t >= b:
                        # Detection is not contained within the
                        # 'image_msg.header.frame_id' coordinate system.
                        continue

                    det = msg0.detections[i]

                    if img.ndim == 3:
                        det.image_chip = bridge.cv2_to_imgmsg(img[t:b,l:r,:],
                                                                  "rgb8")
                    else:
                        det.image_chip = bridge.cv2_to_imgmsg(img[t:b,l:r])

                    det.uid = generate_uid(20)
                    det.header.stamp = image_msg.header.stamp
                    det.camera_of_origin = image_msg.header.frame_id
                    msg.detections.append(det)

            log.info('Rebroadcasting detection list with %i detections' %
                     len(msg.detections))
            self.det_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Node('rebroadcast_infrequent_detections')

    # -------------------------- Read Parameters -----------------------------
    det_in_topic = node.declare_parameter('det_in_topic', '').value
    det_out_topic = node.declare_parameter('det_out_topic', '').value

    image_topics = []
    i = 1
    while True:
        param = node.declare_parameter('image_in%i_topic' % i, 'unused').value
        if param != 'unused':
            image_topics.append(param)
            i += 1
        else:
            break

    det_transform_service = node.declare_parameter(
        'detection_transform_service', 'none').value

    if det_transform_service == 'none':
        det_transform_service = None
    # ------------------------------------------------------------------------

    det_rebroadcast = DetectionRebroadcast(node, det_in_topic, det_out_topic,
                                           image_topics, det_transform_service)
    (void_ref,) = (det_rebroadcast,)

    rclpy.spin(node)


if __name__ == '__main__':
    main()
