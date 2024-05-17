#! /usr/bin/python
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
import rospy
import rospy
from cv_bridge import CvBridge, CvBridgeError

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
    def __init__(self, det_in_topic, det_out_topic, image_topics,
                 det_transform_service):

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

        self.det_pub = rospy.Publisher(det_out_topic,
                                       ImageSpaceDetectionList,
                                       queue_size=10)

        rospy.Subscriber(det_in_topic, ImageSpaceDetectionList,
                         callback=self.detection_list_callback,
                         queue_size=10)

        rospy.loginfo('Rebroadcasting detections from topic: \'%s\' on topic: '
                      '\'%s\'' % (det_in_topic,det_out_topic))

        rospy.loginfo("Starting image processing thread")
        self.thread = threading.Thread(target=self.process_images)
        # Entire Python program exits when only daemon threads are left and we
        # want this thread to shutdown as cleanly as possible.
        #self.thread.daemon = True
        self.thread.start()

        for image_topic in image_topics:
            rospy.loginfo('Receiving images on topic: %s' % image_topic)
            rospy.Subscriber(image_topic, Image,
                             callback=self.ros_image_callback,
                             callback_args=image_topic, queue_size=1)

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
        rospy.loginfo('Waiting for detection list transformation service '
                      '\'%s\' to come alive' % topic)
        rospy.wait_for_service(topic)
        with self._det_tform_serv_lock:
            self._det_tform_serv = rospy.ServiceProxy(topic,
                                                      TransformDetectionList)

    def detection_list_callback(self, msg):
        """Receive a detection list.

        :param msg: Detection list message.
        :type msg: ImageSpaceDetectionList

        """
        rospy.loginfo('Received detection (seq: %i)' % msg.header.seq)
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
            rospy.loginfo('Received with image (seq: %i) from message '
                          'topic \'%s\', but have not received detections, '
                          'so skipping.' %
                          (msg.header.seq,topic))
            return
        else:
            rospy.loginfo('Received with image (seq: %i) from message '
                          'topic \'%s\'' % (msg.header.seq,topic))

        with self.image_lock:
            self.image_deque.appendleft(msg)
            if len(self.image_deque) > 3:
                self.image_deque.pop()

    def process_images(self):
        while True and not rospy.is_shutdown():
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
                            resp = self._det_tform_serv(self.latest_dets, fid1)
                            msg1 = resp.dst_detections
                            self.tformed_versions_latest_dets[fid1] = msg1
                    except rospy.ServiceException as e:
                        rospy.logerr('Could not transform detections from '
                                     'source frame_id \'%s\' to destination '
                                     '\'%s\' because %s' %
                                     (self.latest_dets.frame_id,fid1, e))
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

            rospy.loginfo('Rebroadcasting detection list with %i detections' %
                          len(msg.detections))
            self.det_pub.publish(msg)


def main():
    # Launch the node.
    node = 'rebroadcast_infrequent_detections'
    rospy.init_node(node, anonymous=False)

    node_name = rospy.get_name()

    # -------------------------- Read Parameters -----------------------------
    det_in_topic = rospy.get_param('%s/det_in_topic' % node_name)
    det_out_topic = rospy.get_param('%s/det_out_topic' % node_name)

    image_topics = []
    i = 1
    while True:
        try:
            param_name = '%s/image_in%i_topic' % (node_name, i)
            param = rospy.get_param(param_name)
            if param != 'unused':
                image_topics.append(param)
                i += 1
            else:
                break
        except KeyError:
            break

    param_name = '%s/detection_transform_service' % node_name
    det_transform_service = rospy.get_param(param_name, None)

    if det_transform_service == 'none':
        det_transform_service = None
    # ------------------------------------------------------------------------

    det_rebroadcast = DetectionRebroadcast(det_in_topic, det_out_topic,
                                           image_topics, det_transform_service)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
