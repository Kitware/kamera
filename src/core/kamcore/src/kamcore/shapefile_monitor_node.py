#! /usr/bin/python

import io
import os
import time

import numpy as np
import pygeodesy
import redis
import shapefile
import shapely
import shapely.geometry

import rclpy
from rclpy.node import Node

from custom_msgs.msg import GsofIns
from roskv.impl.redis_envoy import RedisEnvoy
from roskv.util import filter_hosts_by_system

# Location of the geod file.
KAM_DIR = os.environ["DOCKER_KAMERA_DIR"]
geod_filename = os.path.join(KAM_DIR, 'assets/geods/egm84-15.pgm')
geod = pygeodesy.geoids.GeoidPGM(geod_filename)


class ShapefileMonitor(Node):
    """ A class to monitor the shapefile provided by a redis db,
        monitoring if any changes occur and setting archiving if
        the INS reports within the shapefile, turning off archiving
        if it reports outside the shapefile.
    """
    def __init__(self):
        super().__init__("shapefile_monitor")
        self.redis = redis.Redis(os.environ["REDIS_HOST"],
                                 client_name="shapefile_monitor")
        envoy = RedisEnvoy(os.environ["REDIS_HOST"],
                           client_name="shapefile_monitor")
        self.hosts = filter_hosts_by_system(
            envoy.get("/sys/arch/hosts").keys()
        )
        print("hosts: ")
        print(self.hosts)
        self.cnt = 0
        self.use_archive_region = False
        self.archive_region = None

    def load_shapefile(self):
        tic = time.time()
        self.archive_region = None
        b = self.redis.get("/data/shapefile")
        fn = self.redis.get("/sys/shapefile_name")
        sf = None
        if b is not None:
            f = io.BytesIO(b)
            sf = shapefile.Reader(shp=f)
        if sf:
            shapes = sf.shapes()
            if len(shapes) == 1:
                archive_region = shapely.geometry.shape(shapes[0])
            else:
                polygons = []
                for i in range(len(shapes)):
                    polygons.append(shapely.geometry.shape(shapes[i]))
                archive_region = shapely.geometry.MultiPolygon(polygons)
            self.archive_region = archive_region
            if fn is not None:
                self.redis.set("/stat/shapefile_name", fn)
            self.get_logger().info("Successfully loaded shapefile from Redis db!")
        else:
            self.get_logger().warning(
                "Failed to load shapefile from Redis db, waiting for entry.")
        self.get_logger().info("Time for shapefile check was %0.4fs." %
                               (time.time() - tic))

    def listener(self):
        self.sub = self.create_subscription(GsofIns, "/ins",
                                            self.ins_callback, 100)

    def ins_callback(self, msg):
        """ Takes an input GsofIns msg and turns archiving on/off
        depending on if it's in the shapefile or not.

        :param msg: The input ros msg.
        :type msg: custom_msgs.msg/GsofIns.
        """
        self.get_logger().info("Processing incoming INS msgs.",
                               throttle_duration_sec=10)

        # Only check points every 1 s
        if self.cnt < 100:
            self.cnt += 1
            return
        self.cnt = 0

        collection_mode = self.redis.get("/sys/collection_mode").decode('ascii')
        print(collection_mode)
        if collection_mode == "fixed overlap":
            # Get 'True' alt
            h = msg.altitude
            lat = msg.latitude
            lon = msg.longitude
            # Location of geod (i.e., mean sea level, which is
            # generally the ground for us) relative to the
            # ellipsoid. Positive value means that mean sea level is
            # above the WGS84 ellipsoid.
            offset = geod.height(lat, lon)
            alt = h - offset

            speed = msg.total_speed
            fov = float(self.redis.get("/sys/rgb_vfov"))
            overlap = float(self.redis.get("/sys/arch/overlap_percent")) / 100.

            # Width of the field of view in the flight direction.
            w = np.tan(fov/180*np.pi/2.)*alt*2
            print(w, speed, overlap, fov, alt)
            # Distance on ground covered before next acquisition.
            w = float(w*(1-overlap))
            rate = speed/w
            min_fps = float(self.redis.get("/sys/arch/min_frame_rate"))
            max_fps = float(self.redis.get("/sys/arch/max_frame_rate"))
            if rate < min_fps or rate > max_fps:
                self.get_logger().warning(
                    "Overlap percent of %s at alt %s wants "
                    "to set framerate to %s, but min and max "
                    "are %s and %s." % (overlap, alt, rate, min_fps, max_fps))
            fps = np.clip(rate, min_fps, max_fps)
            self.redis.set("/sys/arch/trigger_freq", fps)
            self.get_logger().info("Set triggering rate to map overlap %s to %s." %
                                   (overlap, fps))

        use_archive_region = int(self.redis.get("/sys/arch/use_archive_region"))
        load_sf = int(self.redis.get("/sys/arch/load_shapefile"))
        # Every time /sys/arch/shapefile is updated, load_shapefile must be set
        # to prompt an update
        if load_sf == 1:
            self.load_shapefile()
            self.redis.set("/sys/arch/load_shapefile", 0)

        print(self.archive_region)
        if use_archive_region and self.archive_region:
            point = shapely.geometry.Point(msg.longitude, msg.latitude)
            print(self.archive_region.contains(point))
            if self.archive_region.contains(point):
                print("is_archiving = 1")
                self.redis.set("/sys/arch/is_archiving", 1)
            else:
                print("is_archiving = 0")
                self.redis.set("/sys/arch/is_archiving", 0)
                # make sure nucmode is set to automatic by default
                self.set_nuc_mode("1")
        else:
            # make sure nucmode is set to automatic by default
            self.set_nuc_mode("1")
        is_archiving = int(self.redis.get("/sys/arch/is_archiving")) == 1
        allow_ir_nuc = int(self.redis.get("/sys/arch/allow_ir_nuc")) == 1
        if not allow_ir_nuc and is_archiving:
            print("Turning off NUCing when archiving.")
            # Make sure NUCing is turned off
            self.set_nuc_mode("0")

    def set_nuc_mode(self, val):
        for host in self.hosts:
            topic = "/".join(["", 'sys', 'requested_geni_params',
                              host, "ir", "CorrectionAutoEnabled"])
            self.redis.set(topic, val)


def main(args=None):
    rclpy.init(args=args)
    sm = ShapefileMonitor()
    sm.load_shapefile()
    sm.listener()
    sm.get_logger().info("Waiting for incoming messages on /ins ...")
    try:
        rclpy.spin(sm)
    except KeyboardInterrupt:
        pass
    finally:
        sm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
