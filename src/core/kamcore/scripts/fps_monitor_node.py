#! /usr/bin/python

import os
import socket
import numpy as np
from six.moves.queue import deque

import rospy

from roskv.impl.redis_envoy import RedisEnvoy
from sensor_msgs.msg import Image
from custom_msgs.msg import GSOF_EVT

hostname = socket.gethostname()


class FPSMonitor:
    def __init__(self) -> None:
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

    def init_ros(self):
        self.event_sub = rospy.Subscriber(
            "/event", GSOF_EVT, callback=self.ingest_event, queue_size=2
        )
        channels = self.envoy.get("/sys/channels").keys()
        for channel in channels:
            _ = rospy.Subscriber(
                f"/{hostname}/{channel}/image_raw",
                Image,
                callback=self.ingest_image,
                queue_size=2,
            )

    def ingest_event(self, msg):
        time = msg.gps_time.to_sec()
        rospy.loginfo("Received event message, time %0.5f." % time)
        is_archiving = self.envoy.kv.get("/sys/arch/is_archiving") == "1"
        # Skip the first event, so we don't accidentally report drops
        if is_archiving and self.previously_archiving:
            self.evt_queue.append(time)

        # on falling edge, clear events
        if not is_archiving and self.previously_archiving:
            self.evt_queue.clear()

        self.previously_archiving = is_archiving

    def ingest_image(self, msg):
        frame_id = msg.header.frame_id
        time = msg.header.stamp.to_sec()

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
            rospy.logwarn("No valid modality found in image message!")

        rospy.loginfo("Received %s message, time: %0.5f." % (modality, time))

    def update(self):
        # copy over data structures and sort
        rgb_list = list(self.rgb_queue)
        ir_list = list(self.ir_queue)
        uv_list = list(self.uv_queue)
        times = list(self.evt_queue)

        # calculate FPS
        if len(rgb_list) > 1:
            den = np.mean(
                [rgb_list[i] - rgb_list[i - 1] for i in range(1, len(rgb_list))]
            )
            if den != 0:
                rgb_fps = 1 / den
            else:
                rgb_fps = 0
        else:
            rgb_fps = 0
        if len(ir_list) > 1:
            den = np.mean([ir_list[i] - ir_list[i - 1] for i in range(1, len(ir_list))])
            if den != 0:
                ir_fps = 1 / den
            else:
                ir_fps = 0
        else:
            ir_fps = 0
        if len(uv_list) > 1:
            den = np.mean([uv_list[i] - uv_list[i - 1] for i in range(1, len(uv_list))])
            if den != 0:
                uv_fps = 1 / den
            else:
                uv_fps = 0
        else:
            uv_fps = 0

        # don't need so many sigfigs
        rgb_fps = round(rgb_fps, 3)
        ir_fps = round(ir_fps, 3)
        uv_fps = round(uv_fps, 3)

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

        self.envoy.kv.set(f"/sys/arch/{hostname}/rgb/fps", rgb_fps)
        self.envoy.kv.set(f"/sys/arch/{hostname}/ir/fps", ir_fps)
        self.envoy.kv.set(f"/sys/arch/{hostname}/uv/fps", uv_fps)

        self.envoy.kv.set(f"/sys/arch/{hostname}/rgb/dropped", self.rgb_drops)
        self.envoy.kv.set(f"/sys/arch/{hostname}/ir/dropped", self.ir_drops)
        self.envoy.kv.set(f"/sys/arch/{hostname}/uv/dropped", self.uv_drops)


def main():
    rospy.init_node(f"{hostname}_fps_monitor")
    mon = FPSMonitor()
    rospy.loginfo("Waiting for incoming image and event messages ...")

    while not rospy.is_shutdown():
        mon.update()
        rospy.sleep(1)


if __name__ == "__main__":
    main()
