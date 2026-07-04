#! /usr/bin/python
from __future__ import division, print_function
import numpy as np
import os
import cv2
import time

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


# Instantiate CvBridge
bridge = CvBridge()


class ImageSaver(object):
    def __init__(self, node, topic_name, image_directory, ext='jpg'):
        print('Saving images for topic:', topic_name)
        self.image_directory = image_directory
        self.image_subscriber = node.create_subscription(
            Image, topic_name, self.image_callback_ros, 10)
        self.ext = ext
    
    def image_callback_ros(self, image_msg):
        """Method that receives messages published on self.image_topic
        
        :param image_msg: ROS image message.
        :type image_msg: Image
        """
        try:
            # Convert your ROS Image message to OpenCV2
            raw_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return None
        
        if raw_image.ndim == 3:
            # BGR to RGB
            raw_image = raw_image[...,::-1]
        
        frame_id = image_msg.header.frame_id
        frame_time = int(np.round(_stamp_to_sec(image_msg.header.stamp)*100))
        
        if raw_image.ndim == 3:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        
        frame_id = frame_id.replace('/','_')
        fname = ''.join([self.image_directory,'/',frame_id,'_',str(frame_time),
                         '.',self.ext])
        print('saving:', fname)
        cv2.imwrite(fname, raw_image)
        


def main(args=None):
    rclpy.init(args=args)
    node = Node('save_images_to_disk')

    # -------------------------- Read Parameters -----------------------------
    # Load in cueing camera details.
    image_topics = []
    i = 1
    while True:
        topic = node.declare_parameter('image_topic%i' % i, '').value
        if not topic:
            break
        image_topics.append(topic)
        i += 1

    image_directory = node.declare_parameter('image_directory', '.').value
    image_directory = ''.join([image_directory, '/', str(int(time.time()))])

    ext = node.declare_parameter('image_extension', 'jpg').value

    try:
        os.makedirs(image_directory)
    except OSError:
        pass
    # ------------------------------------------------------------------------

    savers = [ImageSaver(node, t, image_directory, ext) for t in image_topics]
    (void_ref,) = (savers,)

    rclpy.spin(node)


if __name__ == '__main__':
    main()
