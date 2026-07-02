#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading

import serial

import rclpy
from rclpy.node import Node
import std_msgs.msg


class Rate(object):
    def __init__(self, rate=5.0):
        self.rate = rate

    def set_rate(self, msg):
        self.rate = msg.data

    @property
    def period(self):
        return 1.0 / self.rate


def send_pulse(ser, pin='dtr', duration=0.05):
    # type: (serial.Serial, str, float) -> None
    """
    Send a pulse to the pin
    :param ser: Serial interface object
    :param pin: pin to use, must be RTS or DTR
    :param duration: Length of pulse in seconds
    """
    setattr(ser, pin, True)
    timer = threading.Timer(duration, setattr, args=(ser, pin, False))
    timer.daemon = True
    timer.start()


def trigger_serial(node, ser, pulse_frequency=2.0, pin='dtr'):
    """Use this to send a pulse via RTS/DTR pin"""
    pulse_rate = Rate(pulse_frequency)
    pulse_duration = 0.05
    node.create_subscription(std_msgs.msg.Float64, '/daq/trigger_freq',
                             pulse_rate.set_rate, 10)
    while rclpy.ok():
        send_pulse(ser, pin=pin, duration=pulse_duration)
        time.sleep(pulse_rate.period)


def main(args=None):
    rclpy.init(args=args)
    node = Node('daq')

    # subscriptions are serviced by a background executor while the main
    # thread runs the pulse loop
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    pulse_tty = os.environ.get('DAQ_TTY', None)
    try:
        ser = serial.Serial(pulse_tty)
    except Exception:
        ser = None
        print('Unable to find tty: {}'.format(pulse_tty))

    if ser:
        print('Serial connected: {}'.format(ser.name))
        trigger_serial(node, ser)
        sys.exit(0)


if __name__ == '__main__':
    main()
