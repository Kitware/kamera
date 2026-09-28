#! /usr/bin/python

import os
import socket
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from roskv.impl.redis_envoy import RedisEnvoy
from sensor_msgs.msg import Image
from custom_msgs.msg import GsofEvt

hostname = socket.gethostname()


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class FPSMonitor(Node):
    def __init__(self) -> None:
        super().__init__(f"{hostname}_fps_monitor")
        self.envoy = RedisEnvoy(os.environ["REDIS_HOST"], client_name="fps_monitor")
        self.hostname = hostname
        self.ir_drops = 0
        self.rgb_drops = 0
        self.uv_drops = 0
        self.rgb_queue = deque(maxlen=5)
        self.uv_queue = deque(maxlen=5)
        self.ir_queue = deque(maxlen=5)
        self.evt_queue = deque(maxlen=5)
        self.processed_times = deque(maxlen=50)
        self.previously_archiving = False
        self.init_ros()
        self.update_timer = self.create_timer(1.0, self.update)

    def init_ros(self):
        self.event_sub = self.create_subscription(
            GsofEvt, "/event", self.ingest_event, 2
        )
        self.image_subs = []
        channels = self.envoy.get("/sys/channels").keys()
        for channel in channels:
            self.image_subs.append(
                self.create_subscription(
                    Image,
                    f"/{hostname}/{channel}/image_raw",
                    self.ingest_image,
                    2,
                )
            )

    def ingest_event(self, msg):
        time = stamp_to_sec(msg.gps_time)
        self.get_logger().info("Received event message, time %0.5f." % time)
        is_archiving = self.envoy.get("/sys/arch/is_archiving") == "1"
        # Skip the first event, so we don't accidentally report drops
        if is_archiving and self.previously_archiving:
            self.evt_queue.append(time)

        # on falling edge, clear events
        if not is_archiving and self.previously_archiving:
            self.evt_queue.clear()

        self.previously_archiving = is_archiving

    def ingest_image(self, msg):
        frame_id = msg.header.frame_id
        time = stamp_to_sec(msg.header.stamp)

        modality = ""
        if "uv" in frame_id:
            modality = "uv"
            self.uv_queue.append(time)
        elif "Phase One" in frame_id or "rgb" in frame_id:
            modality = "rgb"
            self.rgb_queue.append(time)
        elif "ir" in frame_id:
            modality = "ir"
            self.ir_queue.append(time)
        else:
            self.get_logger().warning("No valid modality found in image message!")

        self.get_logger().info("Received %s message, time: %0.5f." % (modality, time))

    @staticmethod
    def _fps(times):
        if len(times) < 2:
            return 0
        den = np.mean([times[i] - times[i - 1] for i in range(1, len(times))])
        return round(1 / den, 3) if den != 0 else 0

    def update(self):
        # copy over data structures and sort
        rgb_list = list(self.rgb_queue)
        ir_list = list(self.ir_queue)
        uv_list = list(self.uv_queue)
        times = list(self.evt_queue)

        rgb_fps = self._fps(rgb_list)
        ir_fps = self._fps(ir_list)
        uv_fps = self._fps(uv_list)

        # register missed frames
        # Assume that if we haven't seen this time in the last 5 frames,
        # we missed it. Process everything except the most recent event,
        # since that's assumed to be received before the images
        # Only count frame drops when archiving
        for time in times[:-1]:
            if time in self.processed_times:
                continue
            print("checking event time %0.5f" % time)
            if len(rgb_list) == 0 or time not in rgb_list:
                self.rgb_drops += 1
            if len(ir_list) == 0 or time not in ir_list:
                self.ir_drops += 1
            if len(uv_list) == 0 or time not in uv_list:
                self.uv_drops += 1

            self.processed_times.append(time)

        self.envoy.set(f"/sys/arch/{hostname}/rgb/fps", rgb_fps)
        self.envoy.set(f"/sys/arch/{hostname}/ir/fps", ir_fps)
        self.envoy.set(f"/sys/arch/{hostname}/uv/fps", uv_fps)

        self.envoy.set(f"/sys/arch/{hostname}/rgb/dropped", self.rgb_drops)
        self.envoy.set(f"/sys/arch/{hostname}/ir/dropped", self.ir_drops)
        self.envoy.set(f"/sys/arch/{hostname}/uv/dropped", self.uv_drops)


def main(args=None):
    rclpy.init(args=args)
    mon = FPSMonitor()
    mon.get_logger().info("Waiting for incoming image and event messages ...")
    try:
        rclpy.spin(mon)
    except KeyboardInterrupt:
        pass
    finally:
        mon.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
