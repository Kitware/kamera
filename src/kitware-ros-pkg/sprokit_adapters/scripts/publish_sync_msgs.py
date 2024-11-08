#! /usr/bin/python

from __future__ import print_function
import os
import glob
import shutil
import re
import time
import collections
import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge, CvBridgeError

# KAMERA imports.
from custom_msgs.msg import SynchronizedImages

bridge = CvBridge()


def get_fov_dirs(flight_dir):
    """Return the actual FOV direcotry """
    #subdirs = get_subdirs(flight_dir)
    #actual_dirs = [name for name in subdirs if reduce_fov(name, noisy=False)]

    # These are the acceptable options for camera directories that we want to
    # consider.
    acceptable_names = ['cent','center', 'left', 'right', 'center_view',
                        'left_view', 'right_view']

    actual_dirs = []
    for subdir in os.listdir(flight_dir):
        full_path = '%s/%s' % (flight_dir, subdir)
        if os.path.isdir(full_path) and subdir.lower() in acceptable_names:
            actual_dirs.append(subdir)

    return actual_dirs

class ROSPublishSyncMsgs(object):
    def __init__(self, flight_dir, rate, out_sync_image_topic):
        self.flight_dir = flight_dir
        self.rate = rate
        self.pub = rospy.Publisher(out_sync_image_topic,
                                   SynchronizedImages,
                                   queue_size=1)

    def start_publishing(self):
        ros_rate = rospy.Rate(self.rate)

        fov_dirs = get_fov_dirs(self.flight_dir)
        if len(fov_dirs) == 0:
            raise RuntimeError("No valid directories found under given dir %s,"
                               " Exiting." % self.flight_dir)
        fnames = {}

        tic = time.time()
        # organize sync messages
        for d in fov_dirs:
            files = glob.glob("%s/%s/*" % (self.flight_dir, d))
            for f in files:
                dirname = os.path.dirname(f)
                try:
                    s = fnames[dirname]
                except KeyError:
                    fnames[dirname] = {}
                try:
                    t = float(f.split('_')[-2])
                except ValueError:
                    continue
                try:
                    s = fnames[dirname][t]
                except KeyError:
                    fnames[dirname][t] = {}
                cam = f.split('_')[-1].split('.')[0]
                fnames[dirname][t][cam] = f
        rospy.loginfo("Total time to organize: %s" % (time.time() - tic))

        # Sort by t

        for d in fnames:
            print(d)
            fnames[d] = collections.OrderedDict(sorted(fnames[d].items()))
            seq = 0
            for t in fnames[d]:
                sync_msg = SynchronizedImages()
                sync_msg.header.stamp = rospy.Time(t)
                sync_msg.header.seq = seq
                seq += 1
                for cam in fnames[d][t]:
                    fname = fnames[d][t][cam]
                    if cam == 'rgb':
                        encoding = 'rgb8'
                    elif cam == 'uv':
                        encoding = 'mono8'
                    elif cam == 'ir':
                        encoding = 'mono16'
                    im = cv2.imread(fname, -1)
                    if im is None:
                        continue
                    try:
                        msg = bridge.cv2_to_imgmsg(im, encoding=encoding)
                    except CvBridgeError as e:
                        rospy.logerr(e)
                        break
                    if cam == 'rgb':
                        sync_msg.image_rgb = msg
                        sync_msg.file_path_rgb = fname
                    elif cam == 'uv':
                        sync_msg.image_uv = msg
                        sync_msg.file_path_uv = fname
                    elif cam == 'ir':
                        sync_msg.image_ir = msg
                        sync_msg.file_path_ir = fname
                rospy.loginfo("Publishing sync image for time %s" % t)
                self.pub.publish(sync_msg)
                if rospy.is_shutdown():
                    raise SystemExit
                ros_rate.sleep()


def main():
    rospy.init_node("publish_sync_msgs", anonymous=True)

    flight_dir = rospy.get_param("~flight_dir")
    rate = rospy.get_param("~publish_rate")
    out_sync_image_topic = rospy.get_param("~out_topic")

    PSM = ROSPublishSyncMsgs(flight_dir, rate, out_sync_image_topic)

    PSM.start_publishing()

    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        raise SystemExit
