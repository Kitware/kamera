#! /usr/bin/python
"""Explode image/odometry topics from a bag file into PNGs + nav yaml.

ROS2 port: reads bags via the `rosbags` library (pip install rosbags), which
understands both ROS1 .bag files (the legacy data this tool exists for) and
ROS2 bag directories, without needing a ROS environment at all.
"""
from __future__ import print_function

import argparse
import logging
import os

import cv2
import numpy as np
import yaml

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

from kamera.sensor_models import euler_from_quaternion
from kamera.sensor_models.nav_conversions import enu_quat_to_ned_quat

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def stamp_to_sec(stamp):
    # rosbags exposes ROS1 stamps as .sec/.nanosec too
    return stamp.sec + stamp.nanosec * 1e-9


def odom_to_yaml(msg, directory):
    """Process INS Odometry message."""
    pose = msg.pose.pose
    lat = pose.position.y
    lon = pose.position.x
    alt = pose.position.z

    # ENU quaternion
    quat = np.array([pose.orientation.x, pose.orientation.y,
                     pose.orientation.z, pose.orientation.w])
    yaw = euler_from_quaternion(enu_quat_to_ned_quat(quat),
                                axes='rzyx')[0] * 180 / np.pi

    # Saves navigation info into a yaml file to be dynamically loaded later
    LOG.info("Logging Nav info into %s/nav_odom.yaml" % directory)
    with open(os.path.join(directory, "nav_odom.yaml"), "w+") as odom_yaml:
        odom_yaml.write('lat: %s\n' % lat)
        odom_yaml.write('lon: %s\n' % lon)
        odom_yaml.write('alt: %s\n' % alt)
        odom_yaml.write('yaw: %s\n' % yaw)


def save_image(msg, name, directory, ids):
    image_dir = os.path.join(directory, name)
    os.makedirs(image_dir, exist_ok=True)

    frame_id = name + ': "' + msg.header.frame_id + '"\n'
    if frame_id not in ids:
        ids.append(frame_id)

    try:
        if msg.encoding in ("bgr8", "rgb8"):
            cv_image = message_to_cvimage(msg, "bgr8")
        elif msg.encoding == "32FC1":
            # Depth image.
            # NOTE: Assuming Zed camera properties.
            raw_image = message_to_cvimage(msg, "32FC1")

            # Make sense of nan/inf values
            raw_image[np.isnan(raw_image)] = 0
            raw_image[np.isinf(raw_image)] = 0  # maybe 20?
            # Zed max range should only be 20 (meters)
            assert not (raw_image > 20).any(), \
                "Zed sensor is not supposed to report values over 20! " \
                "(found some...)"
            # Scale remaining non-zero values to 8-bit range,
            # cast to 8-bit image.
            cv_image = (raw_image * (255 / 20.0)).astype(np.uint8)
        else:
            raise RuntimeError("Unexpected image format/encoding: '%s'"
                               % msg.encoding)

        timestr = "%.6f" % stamp_to_sec(msg.header.stamp)
        image_name = os.path.join(image_dir, "%s_%s.png" % (timestr, name))
        LOG.info("Saving image: %s" % image_name)
        cv2.imwrite(image_name, cv_image)
    except Exception as e:
        LOG.error(str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_config',
                        help="YAML config file mapping topics to extract "
                             "with the output subdirectories to extract to.")
    parser.add_argument('output_dir',
                        help="Directory to output image sub-directories to.")
    parser.add_argument('bag_filepath',
                        help="Filesystem path to the bag file to explode.")
    args, unknown = parser.parse_known_args()

    with open(args.yaml_config) as fp:
        to_save = yaml.safe_load(fp)
    save_dir = args.output_dir
    filename = args.bag_filepath

    LOG.info("to-save map: %s" % to_save)
    LOG.info("Output directory = %s" % save_dir)
    LOG.info("Bag filename = %s" % filename)

    os.makedirs(save_dir, exist_ok=True)
    ids = []
    wrote_nav = False

    from pathlib import Path
    with AnyReader([Path(filename)]) as reader:
        connections = [c for c in reader.connections if c.topic in to_save]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            name = to_save[connection.topic]
            if name == 'nav':
                if not wrote_nav:
                    odom_to_yaml(msg, str(save_dir))
                    wrote_nav = True
            else:
                save_image(msg, name, str(save_dir), ids)

    with open(os.path.join(save_dir, "frame_ids.yaml"), "w+") as ids_text:
        for _id in ids:
            ids_text.write(_id)


if __name__ == '__main__':
    main()
