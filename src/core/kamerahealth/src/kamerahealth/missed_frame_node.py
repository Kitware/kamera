#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import json
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from custom_msgs.msg import Stat


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9



class LowpassIIR(object):
    """
    Digital Infinite impulse response lowpass filter AKA exponential moving
    average. Smooths values.
    """

    def __init__(self, gamma=0.1, init_state=1.0):
        """
        :param gamma: Coefficient for lowpass, (0,1]
        gam=1 -> 100% pass
        """
        self.gamma = gamma
        self._state = init_state

    def update(self, x):
        """
        Push a value into the filter
        :param x: Value of input signal
        :return: Lowpassed signal output
        """
        self._state = (x * self.gamma) + (1.0 - self.gamma) * self._state
        return self._state

    @property
    def state(self):
        return self._state


class MissedFrameAgg(Node):
    def __init__(self):
        super().__init__("missed_listener")
        self.missed_frames = []
        self.intervals = []
        self._last_time = time.time()
        self.period_iir = None
        self.init_latch = True
        self.sub_missed_frames = self.create_subscription(
            Header, "/missed_frames", self.missed_msg_cb, 10)
        self.sub_errstat = self.create_subscription(
            Stat, "/errstat", self.errstat_cb, 10)
        now = int(time.time())
        self.out_file = '/mnt/flight_data/miketest/missed_agg/{}.jsonl'.format(now)

    def lap(self):
        now = time.time()
        last = self._last_time
        self._last_time = now
        elapsed = now - last
        if self.period_iir is None:
            self.period_iir = LowpassIIR(init_state=elapsed)
        else:
            self.period_iir.update(elapsed)
        return elapsed

    def trig_freq_cb(self, msg):
        pass

    def errstat_cb(self, msg):
        header = msg.trace_header
        data = {'type': 'errstat', 'time': str(stamp_to_sec(header.stamp)), 'frame_id': header.frame_id, 'note': msg.note,
                'link': msg.link}
        with open(self.out_file, 'a') as fp:
            json.dump(data, fp)
            fp.write('\n')

        self.get_logger().info('{}: {}'.format(msg.link, msg.note))

    def missed_msg_cb(self, msg):
        self.missed_frames.append(msg)
        elapsed = self.lap()
        self.intervals.append(elapsed)
        iir_s = self.period_iir.state
        hz = 1.0 / iir_s
        med = np.median(self.intervals)
        now = time.time()
        self.get_logger().info(
            "{: <14}:  last interval: {: >2.3f} iir: {: >2.3f}s iir {: >2.3f}Hz Median: {: 2.3f}s {: 2.3f}Hz".format(
                msg.frame_id, elapsed, iir_s, hz, med, 1.0/med,
            )
        )
        data = { 'type': 'missed_frames', 'time': str(stamp_to_sec(msg.stamp)), 'frame_id': msg.frame_id, 'elapsed': elapsed, 'iir_s': iir_s, 'med_hz': 1.0/med}
        with open(self.out_file, 'a') as fp:
            json.dump(data, fp)
            fp.write('\n')


def main(args=None):
    rclpy.init(args=args)
    app = MissedFrameAgg()
    try:
        rclpy.spin(app)
    except KeyboardInterrupt:
        pass
    finally:
        app.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
