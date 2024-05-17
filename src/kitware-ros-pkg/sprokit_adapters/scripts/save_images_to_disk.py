#! /usr/bin/python
"""
ckwg +31
Copyright 2017 by Kitware, Inc.
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
import time

# ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


# Instantiate CvBridge
bridge = CvBridge()


class ImageSaver(object):
    def __init__(self, topic_name, image_directory, ext='jpg'):
        print('Saving images for topic:', topic_name)
        self.image_directory = image_directory
        self.image_subscriber = rospy.Subscriber(topic_name, Image, 
                                                 self.image_callback_ros)
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
        frame_time = int(np.round(image_msg.header.stamp.to_sec()*100))
        
        if raw_image.ndim == 3:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        
        frame_id = frame_id.replace('/','_')
        fname = ''.join([self.image_directory,'/',frame_id,'_',str(frame_time),
                         '.',self.ext])
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
    image_topics = []
    i = 1
    while True:
        try:
            param_name = ''.join([node_name,'/image_topic',str(i)])
            image_topics.append(rospy.get_param(param_name))
            i += 1
        except:
            break
        
    param_name = ''.join([node_name,'/image_directory'])
    image_directory = rospy.get_param(param_name)
    image_directory = ''.join([image_directory,'/',str(int(time.time()))])
    
    param_name = ''.join([node_name,'/image_extension'])
    ext = rospy.get_param(param_name)
    
    try:
        os.makedirs(image_directory)
    except OSError:
        pass
    # ------------------------------------------------------------------------
    
    for image_topic in image_topics:
        ImageSaver(image_topic, image_directory, ext)
    
    rospy.spin()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
