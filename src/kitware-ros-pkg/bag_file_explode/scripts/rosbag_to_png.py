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
from __future__ import print_function

import argparse
import logging
import multiprocessing as mp
import os
from kamera.sensor_models import euler_from_quaternion
from kamera.sensor_models.nav_conversions import enu_quat_to_ned_quat

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rosbag
import rospy
import yaml
import time


logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# First hard-coded mapping of ROS topic to output filename suffix part.
to_save = {
    '/ros/topic': 'fname1',
}

save_dir = 'output'
filename = 'bagfile'
# Multiple shared memory objects to prevent mp errors
count = mp.Manager().list()
ids = mp.Manager().list()
check_dir = mp.Manager().list()

def save_times(args):
    start = rospy.Time.from_sec(args[0])
    end = rospy.Time.from_sec(args[1])

    with rosbag.Bag(filename, 'r') as bag:
        map(topic_wrapper, bag.read_messages(topics=to_save.keys(),
                                                  start_time=start,
                                                  end_time=end))

def odom_to_yaml(msg, directory):
    """Process INS Odometry message.

    :param msg: Odometry message.
    :type msg: Odometry

    """

    pose = msg.pose.pose
    lat = pose.position.y
    lon = pose.position.x
    alt  = pose.position.z


    # ENU quaternion
    quat = np.array([pose.orientation.x, pose.orientation.y,
		pose.orientation.z, pose.orientation.w])
    yaw = euler_from_quaternion(enu_quat_to_ned_quat(quat),
			axes='rzyx')[0]*180/np.pi

    # Saves navigation info into a yaml file to be dynamically loaded later
    yaml_lat = ('lat: ' + str(lat) + '\n')
    yaml_lon = ('lon: ' + str(lon) + '\n')
    yaml_alt = ('alt: ' + str(alt) + '\n')
    yaml_yaw = ('yaw: ' + str(yaw) + '\n')
    LOG.info("Logging Nav info into %s/nav_odom.yaml"%directory)
    odom_yaml = open(os.path.join(directory, "nav_odom.yaml"), "w+")
    odom_yaml.write(yaml_lat + yaml_lon + yaml_alt + yaml_yaw)
    odom_yaml.close()


def topic_wrapper(args):
    topic, msg, t = args
    # Ensure nav yaml is only written once
    if to_save[topic] == 'nav' and len(count) == 0:
        odom_to_yaml(msg, str(save_dir))
	count.append(0)
    elif to_save[topic] == 'nav' and len(count) != 0:
        pass
    else:
        save_image(msg, to_save[topic], str(save_dir))


def save_image(msg, name, directory):
    image_dir = os.path.join(directory, name)

    # Ensure makedirs is only called once to prevent error
    if not os.path.isdir(image_dir):
        if image_dir not in check_dir:
            try:
                os.makedirs(image_dir)
            except OSError as e:
               if e.errno!=os.errno.EEXIST:
                   raise
               pass
            check_dir.append(image_dir)

    frame_id = name + ": \"" + msg.header.frame_id + '\"\n'
    if frame_id not in ids:
        ids.append(frame_id)

    # Use a CvBridge to convert ROS images to OpenCV images so they can be
    # saved.
    bridge = CvBridge()

    try:
        if msg.encoding == "bgr8":
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        elif msg.encoding == "rgb8":
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        elif msg.encoding == "32FC1":
            # Depth image.
            # NOTE: Assuming Zed camera properties.
            raw_image = bridge.imgmsg_to_cv2(msg, "32FC1")

            # Make sense of nan/inf values
            raw_image[np.isnan(raw_image)] = 0
            raw_image[np.isinf(raw_image)] = 0  # maybe 20?
            # Zed max range should only be 20 (meters)
            assert not (raw_image > 20).any(), \
                "Zed sensor is no supposed to report values over 20! " \
                "(found some...)"
            # Scale remaining non-zero values to 8-bit range,
            # cast to 8-bit image.
            cv_image = (raw_image * (255 / 20.0)).astype(np.uint8)
        else:
            raise RuntimeError("Unexpected image format/encoding: '%s'"
                               % msg.encoding)

        timestr = "%.6f" % msg.header.stamp.to_sec()
        image_name = str(image_dir)+"/"+timestr+"_"+name+".png"
        LOG.info("Saving image: %s" % image_name)
        cv2.imwrite(image_name, cv_image)
    except CvBridgeError as e:
        LOG.error(str(e))


class ImageCreator (object):

    def __init__(self):
        global save_dir
        global filename
        global to_save

        # Get parameters as arguments to 'rosrun my_package bag_to_images.py
        # <save_dir> <filename>', where save_dir and filename exist relative to
        # this executable file.
        parser = argparse.ArgumentParser()
        parser.add_argument('yaml_config',
                            help="YAML config file mapping topics to extract "
                                 "with the output subdirectories to extract "
                                 "to.")
        parser.add_argument('output_dir',
                            help="Directory to output image sub-directories "
                                 "to.")
        parser.add_argument('bag_filepath',
                            help="Filesystem path to the bag file to explode.")
        args, unknown = parser.parse_known_args()

        to_save = yaml.load(open(args.yaml_config))
        save_dir = args.output_dir
        filename = args.bag_filepath


        LOG.info("to-save map: %s" % to_save)
        LOG.info("Output directory = %s" % save_dir)
        LOG.info("Bag filename = %s" % filename)

        self.run_multiprocess()

    def run_multiprocess(self):
        num_processors = mp.cpu_count()
        pool = mp.Pool(num_processors)

        with rosbag.Bag(filename, 'r') as bag:
            starttime = bag.get_start_time()
            endtime = bag.get_end_time()
        LOG.info("Bag start time and end time: %f -> %f" % (starttime, endtime))

        step = (endtime - starttime) / float(num_processors)
        times = [(starttime + (step*i), starttime + (step * (i + 1)))
                 for i in range(num_processors)]
        LOG.info("Time slices:")
        for T in times:
            LOG.info("  %s" % (T,))
        assert times[-1][1] == endtime, \
            "Expected end time to be %f, got %f" % (endtime, times[-1][1])
        pool.map(save_times, times)
        pool.close()
        pool.join()

        ids_text = open(os.path.join(save_dir, "frame_ids.yaml"), "w+")
 	for _id in ids:
            ids_text.write(_id)
        ids_text.close()



if __name__ == '__main__':
    # Go to class functions that do all the heavy lifting. Do error checking.
    image_creator = ImageCreator()
