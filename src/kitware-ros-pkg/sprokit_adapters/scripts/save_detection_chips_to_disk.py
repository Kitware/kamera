#! /usr/bin/python
from __future__ import division, print_function
import numpy as np
import os
import cv2
import time

# ROS imports
import rospy
from custom_msgs.msg import ImageSpaceDetectionList
from cv_bridge import CvBridge, CvBridgeError


# Instantiate CvBridge
bridge = CvBridge()


class ChipSaver(object):
    def __init__(self, det_topic, image_directory, ext='jpg'):
        rospy.loginfo('Saving chips for detection topic det_topics: %s' %
                      det_topic)
        self.image_directory = image_directory
        self.image_subscriber = rospy.Subscriber(det_topic,
                                                 ImageSpaceDetectionList,
                                                 self.callback_ros)
        self.ext = ext

    def callback_ros(self, msg):
        """Method that receives messages published on self.image_topic

        :param image_msg: ROS detection message.
        :type image_msg: ImageSpaceDetectionList

        """
        frame_id = msg.header.frame_id
        frame_time = int(np.round(msg.header.stamp.to_sec()*100))

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


def main():
    # Launch the node.
    node = 'save_images_to_disk'
    rospy.init_node(node, anonymous=False)

    node_name = rospy.get_name()

    # -------------------------- Read Parameters -----------------------------
    #print('rospy.get_param_names()', rospy.get_param_names())

    # Load in cueing camera details.
    det_topics = []
    i = 1
    while True:
        try:
            param_name = '%s/detection_topic%i' % (node_name, i)
            det_topics.append(rospy.get_param(param_name))
            i += 1
        except KeyError:
            break

    image_directory = rospy.get_param('%s/image_directory' % node_name)
    image_directory = '%s/%i' % (image_directory,int(time.time()))

    ext = rospy.get_param('%s/image_extension' % node_name)

    try:
        os.makedirs(image_directory)
    except OSError:
        pass
    # ------------------------------------------------------------------------

    for det_topic in det_topics:
        ChipSaver(det_topic, image_directory, ext)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
