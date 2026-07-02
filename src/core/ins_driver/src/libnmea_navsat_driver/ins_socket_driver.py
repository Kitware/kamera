#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Software License Agreement (BSD License)
#
# Copyright (c) 2016, Rein Appeldoorn
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the names of the authors nor the names of their
#    affiliated organizations may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import socket
from socket import error as socket_error
import sys
import time
import struct
import threading

import redis
import serial

import rclpy
from rclpy.node import Node
import std_msgs.msg

from libnmea_navsat_driver.gsof import (
    parse_gsof_stream, maybe_gsof, separate_nmea, GsofInsDispatch,
    GsofEventDispatch, GsofSpoofEventDispatch, GsofSpoofInsDispatch,
    GsofEvtSpoofer, GsofHeader, parse_gsof, time_msg_from_sec, time_msg_to_sec)

from custom_msgs.msg import Stat, GsofIns
from nexus.archiver_core import ArchiverBase
from libnmea_navsat_driver.stream_archive import dumpbuf, enumerate_packets


def netcat(hostname, port, content):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))
    s.shutdown(socket.SHUT_WR)
    while 1:
        data = s.recv(1024)
        if data == "":
            break
        print("Received: {}".format(repr(data)))
    print("Connection closed.")
    s.close()


def loginfo(msg, *args, **kwargs):
    print('info: {}'.format(msg))


def logwarn(msg, *args, **kwargs):
    print('warn: {}'.format(msg))


def logerr(msg, *args, **kwargs):
    print('err : {}'.format(msg))


class Rate(object):
    def __init__(self, rate=5):
        self.rate = rate

    def set_rate(self, msg):
        self.rate = msg.data

    @property
    def period(self):
        return 1.0 / self.rate


class FailedToInitInsDriver(Exception):
    def __init__(self, msg=None, host='', port=0, exc=None):
        # type: (str, str, int, Exception) -> None
        """
        Error for failing to initially connect to NMEA server. This is extra
        bad, so we want to handle this outside regular socket errors
        :param msg: custom error message
        """
        if msg is None:
            msg = ('Failed to initialize INS socket client on host {}:{}'
                   '\n Is the INS connected?'.format(host, port))
        if exc is not None:
            msg += '\nOriginal exception: {}'.format(exc)
        super(FailedToInitInsDriver, self).__init__(msg)


def gen_packets(node, host, port, buffer_size=4096, timeout=2.0):
    """
        Packet generator for streaming data from NMEA service
    Args:
        host: hostname of NMEA device
        port: port of NMEA device
        buffer_size: recv() buffer size
        timeout: socket timeout

    Yields:
        NMEA data strings
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        node.get_logger().info('Accessing {}:{}'.format(host, port))
        sock.connect((host, port))
        sock.settimeout(timeout)
        node.get_logger().info('Connected to {}:{}'.format(host, port))

        while True:
            try:
                yield sock.recv(buffer_size).strip()

            except socket.error as exc:
                logerr(
                    "Caught exception socket.error during recv: %s" % exc)

    except socket.error as exc:
        logerr('Critical failure in initualization of host {}:{}'.format(host, port))
        raise FailedToInitInsDriver(host=host, port=port, exc=exc)

    finally:
        sock.close()


class AvxClient(object):

    def __init__(self, node, rc):
        self.node = node
        self.log = node.get_logger()
        self.rc = rc
        self._host = socket.gethostname()
        self.archiver = ArchiverBase(node)
        namespace = self._host
        self.archiver.advertise_services(namespace=namespace)
        self.log.info('Namespace: {}'.format(namespace))

    def now_msg(self):
        return self.node.get_clock().now().to_msg()

    def run(self, host, port, buffer_size=4096, timeout=2.0):
        if '/ins' not in GsofInsDispatch.pubs:
            GsofInsDispatch.add_publisher(self.node, '/ins')
            GsofEventDispatch.add_publisher(self.node, '/event')
        # recv-loop: When we're connected, keep receiving stuff until that fails
        counter = 0
        rc = self.rc
        spoof_events = rc.get("/debug/spoof_events")
        if spoof_events is not None:
            spoof_events = int(spoof_events)
        else:
            spoof_events = 0
        for rawdata in gen_packets(self.node, host, port, buffer_size, timeout):
            event_arrived = self.now_msg()
            if not rclpy.ok():
                break

            raw_ins_path = self.archiver.get_raw_ins_path()
            if not (counter % 100):
                self.log.info('Ins path: {}'.format(raw_ins_path))
                spoof_events = rc.get("/debug/spoof_events")
                if spoof_events is not None:
                    spoof_events = int(spoof_events)
                else:
                    spoof_events = 0
            counter += 1

            dumpbuf(raw_ins_path, rawdata)

            nmea_list, gsof_data = separate_nmea(rawdata)

            dispatches = []
            if maybe_gsof(gsof_data):
                try:
                    dispatches = parse_gsof_stream(gsof_data)
                except struct.error as err:
                    self.log.error("Gsof parse error: {}".format(err))
                except Exception as err:
                    self.log.error("Some other exception in parsing: {}".format(err))
                for d in dispatches:
                    d.msg.sys_time = event_arrived
                    if isinstance(d, GsofEventDispatch) and spoof_events:
                        self.log.warning(
                            "WARNING: Not publishing events because /debug/spoof_events is true.")
                        # let the spoofer handle it
                        continue
                    elif isinstance(d, GsofInsDispatch):
                        if d.msg.gnss_status == 0 and not spoof_events:
                            # we don't have a fix and we're not spoofing, fall back to spoofing events
                            print("We don't have a fix, turning on spoofing.")
                            rc.set("/debug/spoof_events", 1)
                            spoof_events = 1
                        elif d.msg.gnss_status != 0 and spoof_events:
                            # we do have a fix and are spoofing, stop spoofing
                            print("We have a fix, stopping spoofing.")
                            rc.set("/debug/spoof_events", 0)
                            spoof_events = 0

                    d.publish()

            # ignoring NMEA for now

    def replay(self, path_to_data):
        print('REPLAY MODE')
        with open(path_to_data, 'rb') as fp:
            raw_stream = fp.read()

        if '/ins' not in GsofInsDispatch.pubs:
            GsofEventDispatch.add_publisher(self.node, '/event')
            GsofInsDispatch.add_publisher(self.node, '/ins')
        # recv-loop: When we're connected, keep receiving stuff until that fails
        for i, rawdata in enumerate_packets(raw_stream):
            if not rclpy.ok():
                break
            print(i, len(rawdata))

            nmea_list, gsof_data = separate_nmea(rawdata)

            if maybe_gsof(gsof_data):
                dispatches = parse_gsof_stream(gsof_data)
                for d in dispatches:
                    d.publish()
                continue

    def spoof(self, frequency=5):
        rate = Rate(frequency)
        self.log.warning('Going into event spoof mode!')
        GsofSpoofEventDispatch.add_publisher(self.node, '/event')
        self.node.create_subscription(
            std_msgs.msg.Float64, '/daq/trigger_freq', rate.set_rate, 10)
        while rclpy.ok():
            dispatch = GsofSpoofEventDispatch()
            dispatch.publish()
            time.sleep(rate.period)
            self.log.info(str(dispatch))

    def spoof_serial(self, ser, frequency=1000):
        """Use this if you have a pulse plugged into your serial port via DSR"""
        rate = Rate(frequency)
        self.log.warning('Going into event PULSE mode!')
        GsofEventDispatch.add_publisher(self.node, '/event')
        GsofSpoofEventDispatch.add_publisher(self.node, '/event')
        GsofSpoofInsDispatch.add_publisher(self.node, '/ins')
        pulse = False
        stat_pub = self.node.create_publisher(Stat, '/stat', 10)
        self.log.warning('Stat pub engaged')
        evt_spoofer = GsofEvtSpoofer()
        clock_skew = float(os.environ.get('CLOCK_SKEW', 0.0))

        while rclpy.ok():
            # detect edge
            if not pulse:
                if ser.dsr:
                    event_arrived = self.now_msg()
                    fake_packet = evt_spoofer.next_packet()
                    header = GsofHeader(fake_packet)
                    dispatch = parse_gsof(header, fake_packet)
                    dispatch.msg.sys_time = event_arrived
                    stat = Stat()
                    pulse = True

                    now_skewed = time_msg_to_sec(self.now_msg()) + clock_skew
                    dispatch.msg.sys_time = time_msg_from_sec(now_skewed)
                    dispatch.publish()
                    seq = dispatch.msg.event_num
                    stat.trace_header = dispatch.msg.header
                    stat.node = self.node.get_name()
                    stat.link = '/event/{}'.format(seq)
                    stat.trace_topic = '/event'
                    stat_pub.publish(stat)
                    ins_dispatch = GsofSpoofInsDispatch()
                    ins_dispatch.publish()
                    self.log.info('dsr pulse {:>6} {:.3f}'.format(
                        dispatch.msg.event_num,
                        time_msg_to_sec(dispatch.msg.header.stamp)))
            else:
                if not ser.dsr:
                    pulse = False
            time.sleep(rate.period)


def main(args=None):
    redis_host = os.environ.get('REDIS_HOST', 'nuvo0')
    rc = redis.Redis(host=redis_host, client_name='ins')
    print('redis established, term: {}'.format(rc.get('term')))

    rclpy.init(args=args)
    node = Node('ins_socket_driver')
    log = node.get_logger()

    allow_serial_ins_spoof = int(os.environ.get('ALLOW_SERIAL_INS_SPOOF', 0) or 0)
    host = node.declare_parameter('ip', '0.0.0.0').value
    port = node.declare_parameter('port', 10110).value
    buffer_size = node.declare_parameter('buffer_size', 4096).value
    timeout = node.declare_parameter('timeout_sec', 2.0).value
    spoof_rate = max(int(os.environ.get('SPOOF_RATE', 0) or 0), 0)
    replay_path = node.declare_parameter('replay', '').value
    retry = node.declare_parameter('retry', True).value

    # Services/subscriptions (archiver) are handled by a background executor
    # while the main thread runs the blocking socket recv loop.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    client = AvxClient(node, rc)

    pulse_tty = os.environ.get('PULSE_TTY', None)
    if allow_serial_ins_spoof:
        log.warning("ALLOW_SERIAL_INS_SPOOF ON. Serial-based spoof active")
        try:
            ser = serial.Serial(pulse_tty)
        except Exception:
            log.error('Unable to find tty: {}'.format(pulse_tty))
            sys.exit(1)

        print('Serial connected: {}'.format(ser.name))
        client.spoof_serial(ser)
        sys.exit(0)

    if spoof_rate > 0:
        log.warning("\nGlobal spoof enabled. \nSPOOF_RATE={:.3f}".format(spoof_rate))
        client.spoof(spoof_rate)
        sys.exit(0)
    elif replay_path:
        log.warning("\nReplay INS \nreplay_path={}".format(replay_path))
        client.replay(replay_path)
        sys.exit(0)

    while rclpy.ok():
        try:
            client.run(host, port, buffer_size, timeout)

        except FailedToInitInsDriver as err:
            log.error('Failed to connect to INS: {}'.format(err))
            if retry:
                log.warning('Gracefully attempting to reconnect to INS...')
                time.sleep(1)
            else:
                log.error('Gave up trying to connect to INS, terminating')
                raise err
        except socket_error as err:
            log.error('Other socket error trying to connect to INS: {}'.format(err))
            raise err
        except (KeyboardInterrupt, SystemExit):
            log.info('User quitting')
            sys.exit(130)
        except Exception as err:
            print(type(err))
            log.error('Encountered exception, continuing: {}'.format(err))


if __name__ == '__main__':
    main()
