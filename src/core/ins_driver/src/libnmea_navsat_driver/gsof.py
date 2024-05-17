#!/usr/bin/env python
# -*- coding: utf-8 -*-
import struct
import datetime
from typing import Tuple
import rospy
from . import gps_leap_seconds

from kamcore.structures import BasicStamp, BasicHeader, BasicEvent
from std_msgs.msg import Empty
from custom_msgs.msg import GSOF_INS, GSOF_EVT
from msgdispatch.base import DispatchBase


# Constants
gps_epoch = datetime.datetime(1980,1,6)
unix_epoch = datetime.datetime(1970, 1, 1)
gps_leap_td = datetime.timedelta(seconds=gps_leap_seconds.GPS_LEAP_SECONDS)

START_TX        = 0x02
END_TX          = 0x03
GSOF_TYPE_MSG   = 0x40
GSOF_TYPE_INS   = 0x31
GSOF_TYPE_RMS   = 0x32
GSOF_TYPE_EVENT = 0x33


TEST_GSOF_INTS = bytes(bytearray(
    [2, 40, 64, 109, 229, 0, 0, 49, 104, 7, 253, 14, 78, 242, 216, 2, 1, 64, 69,
     110, 172, 58, 222, 106, 2, 192, 82, 113, 95, 3, 136, 113, 59, 64, 84, 68,
     60, 150, 102, 25, 245, 188, 20, 9, 73, 61, 121, 188, 109, 61, 58, 145, 51,
     61, 124, 118, 160, 63, 246, 158, 84, 17, 111, 200, 62, 192, 2, 45, 171, 66,
     52, 8, 28, 64, 93, 182, 36, 117, 138, 130, 100, 64, 88, 155, 124, 109, 98,
     92, 40, 189, 90, 233, 64, 61, 108, 17, 141, 62, 144, 209, 101, 60, 134, 61,
     16, 187, 228, 180, 183, 187, 170, 252, 107, 148, 3]))

TEST_NMEA = bytes(bytearray("""$GNGGA,154056.00,4251.87736134,N,07346.28348206,W,1,12,1.6,118.450,M,-31.849,M,,*4A
$PASHR,154056.000,354.688,T,1.115,-2.610,,0.248,0.248,71.520,1,2*27
""".encode()))

TEST_GSOF_PACKET = TEST_NMEA + TEST_GSOF_INTS


def datetime_to_float(d):
    # type: (datetime.datetime) -> float
    """
    Converts a time to a Unix style float float
    total_seconds will be in decimals (millisecond precision)
    Args:
        d: Datetime object

    Returns:
        unix seconds since epoch
    """
    total_seconds =  (d - unix_epoch).total_seconds()
    #
    return total_seconds


def gps_to_utc(gps_week, gps_time):
    # type: (int, float) -> float
    """
    Convert GPS time
    Useful link: http://leapsecond.com/java/gpsclock.htm
    Args:
        gps_week: weeks since GPS epoch
        gps_time: Seconds since start of gps week

    Returns:
        unixtime (seconds)

    Examples:
        >>> gps_to_utc(2045, 240055.123)
        1553020837.123
    """
    gps_td = datetime.timedelta(days=gps_week * 7, seconds=gps_time)
    time_dt = gps_epoch + gps_td - gps_leap_td
    return (time_dt - unix_epoch).total_seconds()


def utc_to_gps(utc_time):
    # type: (float) -> Tuple[int, float]
    """
    Convert float timestamp to GPS time
    :param utc_time:
    :return: gps_week, gps_time

    >>> now = 1553020837.123
    >>> utc_to_gps(now)
    (2045, 240055.123)
    >>> now == gps_to_utc(*utc_to_gps(now))
    True
    """
    time_dt = unix_epoch + datetime.timedelta(seconds=utc_time)
    gps_td = time_dt + gps_leap_td - gps_epoch
    gps_week = (gps_td // 7).days
    partial_days_td = datetime.timedelta(days=gps_week * 7)
    residual_td = gps_td - partial_days_td
    gps_time = residual_td.total_seconds()
    return gps_week, gps_time


class ClsNullDispatch(DispatchBase):
    """
    Empty message with the same interface so we can map  without
    dealing with Nones
    """
    counter = 0
    message_class = Empty
    msg = Empty()
    pubs = {}

    def __new__(cls, buf=None):
        self = object.__new__(cls)
        return self

    def publish(self):
        pass

    def __nonzero__(self):
        return False

    def __bool__(self):
        return False

    def __len__(self):
        return 0


class GsofHeader(object):
    def __new__(cls, buf):
        if len(buf) < 9:
            return None
        header = struct.unpack('>9B', buf[:9])
        if header[0] != START_TX:
            rospy.logwarn('Start byte does not match STX')
            return None

        self = object.__new__(cls)

        self.message_type = header[2]
        ln = header[3]

        # try:
        checksum, end = struct.unpack('>BB', buf[4+ln:6+ln])
        computed_checksum = sum(bytearray(buf[1:-2])) & 0xff
        # except struct.error as err:
        #     print('failed to unpack {}'.format(err  ))
        if end != END_TX:
            rospy.logwarn('Final byte does not match ETX')
            return None

        self.len = ln
        self.transmission_num = header[4]
        self.page_index = header[5]
        self.max_page_index = header[6]
        self.record_type = header[7]
        self.record_len = header[8]
        self.ok = computed_checksum == checksum

        self.checksum = checksum
        return self

    def __repr__(self):
        return str(self.__dict__)


def unwrap_gsof(buf):
    # type: (bytes) -> Tuple[GsofHeader, bytes]
    header = GsofHeader(buf)
    start = 9
    payload = buf[start:start + header.record_len]
    return header, payload


def wrap_gsof(payload, message_type=GSOF_TYPE_MSG, record_type=GSOF_TYPE_EVENT, transmission_num=0):
    status = 40
    page_index = 0
    max_page_index = 0
    record_len = len(payload)
    mlen = record_len + 5
    header = struct.pack('>8B', status, message_type, mlen, transmission_num, page_index,
                         max_page_index, record_type, record_len)

    data = header + payload
    checksum = sum(bytearray(data)) & 0xff
    trailer = struct.pack('>BB', checksum, END_TX)

    return struct.pack('>B', 2) + header + payload + trailer


def parse_gsof_evt(buf, cls=BasicEvent):
    # type: (bytes, type) -> BasicEvent
    """ The return type is spoofed in order to allow static type checking, this will actually return a
    type `cls` e.g. GSOF_EVT message"""
    msg = cls()  # type: BasicEvent
    data = struct.unpack('>BHdL', buf)

    gps_week = data[1]
    gps_time = data[2]  # is actually seconds, unlike INS packet
    utc_time = gps_to_utc(gps_week, gps_time)  # as unix time

    msg.header.stamp = BasicStamp.now()
    msg.header.frame_id = 'ins_evt'
    msg.gps_time = BasicStamp.from_sec(utc_time)
    msg.sys_time = msg.header.stamp
    msg.time = utc_time
    msg.event_port = data[0]
    msg.event_num = data[3]
    return msg


class GsofEvtSpoofer(object):
    def __init__(self, event_port=23):
        self.event_num = 0
        self.transmission_num = 0
        self.event_port = event_port

    def inc(self):
        self.event_num += 1
        self.event_num &= 0xffff
        self.transmission_num += 1
        self.transmission_num &= 0xff

    def next_msg(self, now=None):
        if now is None:
            import time
            now = time.time()
        self.inc()
        gps_week, gps_time = utc_to_gps(now)
        data = struct.pack('>BHdL', self.event_port, gps_week, gps_time, self.event_num)
        return data

    def next_packet(self, now=None):
        data = self.next_msg(now=now)
        return wrap_gsof(data)


class GsofInsSpoofer(object):
    def __init__(self, envoy, event_port=23):
        self.envoy = envoy
        self.event_num = 0
        self.transmission_num = 0
        self.event_port = event_port

    def inc(self):
        self.event_num += 1
        self.event_num &= 0xffff
        self.transmission_num += 1
        self.transmission_num &= 0xff

    def ins_from_envoy(self):
        dd = self.envoy.get_dict('/debug/spoof/ins')
        msg = GSOF_INS()
        for k,v in dd.items():
            try:
                setattr(msg, k, v)
            except AttributeError as exc:
                print(exc.__class__.__name__, exc)

        return msg

    def next_struct(self, msg=None, now=None):
        if now is None:
            import time
            now = time.time()
        self.inc()

        if msg is None:
            msg = self.ins_from_envoy()
        buf = [0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        gps_week, gps_time = utc_to_gps(now)

        buf[0] = gps_week
        buf[1] = int(gps_time)
        buf[2] = msg.align_status
        buf[3] = msg.gnss_status
        buf[4] = msg.latitude
        buf[5] = msg.longitude
        buf[6] = msg.altitude
        buf[7] = msg.north_velocity
        buf[8] = msg.east_velocity
        buf[9] = msg.down_velocity
        buf[10] = msg.total_speed
        buf[11] = msg.roll
        buf[12] = msg.pitch
        buf[13] = msg.heading
        buf[14] = msg.track_angle
        buf[15] = msg.angular_rate_x
        buf[16] = msg.angular_rate_y
        buf[17] = msg.angular_rate_z
        buf[18] = msg.acceleration_x
        buf[19] = msg.acceleration_y
        buf[20] = msg.acceleration_z

        data    = struct.pack('>HLbbdddffffddddffffff', *buf)

        return data

    def next_packet(self, msg=None, now=None):
        data = self.next_struct(msg=msg, now=now)
        return wrap_gsof(data, record_type=GSOF_TYPE_INS)


class GsofInsDispatch(DispatchBase):
    counter = 0
    message_class = GSOF_INS
    pubs = {}
    label = 'ins'
    __slots__ = (
        'header', 'time', 'align_status', 'gnss_status', 'latitude',
        'longitude', 'altitude', 'north_velocity', 'east_velocity',
        'down_velocity', 'total_speed', 'roll', 'pitch', 'heading',
        'track_angle', 'angular_rate_x', 'angular_rate_y', 'angular_rate_z',
        'acceleration_x', 'acceleration_y', 'acceleration_z')

    def __new__(cls, buf):
        self     = object.__new__(cls)
        self.msg = self.new_message()
        self.buf = bytes()
        if buf is None:
            return self

        self.buf = buf
        data     = struct.unpack('>HLbbdddffffddddffffff', buf)

        gps_week = data[0]
        gps_time = data[1] * 1e-3 # convert ms to s
        utc_time = gps_to_utc(gps_week, gps_time)  # as unix time

        self.msg.header.stamp   = rospy.Time.from_sec(utc_time)
        self.msg.header.seq     = self.next_id()
        self.msg.header.frame_id = 'ins'


        self.msg.time           = utc_time
        self.msg.gps_time       = rospy.Time.from_sec(utc_time)
        self.msg.align_status   = data[2]
        self.msg.gnss_status    = data[3]
        self.msg.latitude       = data[4]
        self.msg.longitude      = data[5]
        self.msg.altitude       = data[6]
        self.msg.north_velocity = data[7]
        self.msg.east_velocity  = data[8]
        self.msg.down_velocity  = data[9]
        self.msg.total_speed    = data[10]
        self.msg.roll           = data[11]
        self.msg.pitch          = data[12]
        self.msg.heading        = data[13]
        self.msg.track_angle    = data[14]
        self.msg.angular_rate_x = data[15]
        self.msg.angular_rate_y = data[16]
        self.msg.angular_rate_z = data[17]
        self.msg.acceleration_x = data[18]
        self.msg.acceleration_y = data[19]
        self.msg.acceleration_z = data[20]
        return self


class GsofEventDispatch(DispatchBase):
    counter = 0
    message_class = GSOF_EVT
    pubs = {}
    label = 'evt'
    __slots__ = ['header', 'time', 'event_port', 'event_num']

    def __new__(cls, buf):
        self     = object.__new__(cls)
        self.msg = self.new_message()
        self.msg.header.stamp = rospy.Time.now()
        self.buf = bytes()

        if buf is None:
            return self

        self.buf = buf

        data     = struct.unpack('>BHdL', buf)

        gps_week = data[1]
        gps_time = data[2]  # is actually seconds, unlike INS packet
        utc_time = gps_to_utc(gps_week, gps_time)  # as unix time

        self.msg.gps_time = rospy.Time.from_sec(utc_time)

        self.msg.time   = utc_time
        self.msg.event_port = data[0]
        self.msg.event_num  = data[3] # todo: should seq id match this?
        # self.msg.header.seq   = self.next_id()

        self.msg.header.stamp = self.msg.gps_time
        self.msg.header.frame_id = '/ins_evt?eventNum={}'.format(self.event_num)
        self.msg.header.seq   = self.event_num # todo: probably, things get really weird if these don't match
        rospy.loginfo('{} {}: {}'.format(self.msg.header.seq, self.msg.event_num, self.msg.time))
        return self


class GsofSpoofEventDispatch(DispatchBase):
    counter = 0
    message_class = GSOF_EVT
    pubs = {}
    label = 'evt_spoof'

    def __new__(cls, stamp=None):

        self     = object.__new__(cls)
        self.msg = self.new_message()
        self.buf = bytes()

        if stamp is None:
            stamp = rospy.Time.now()
        self.msg.header.stamp = stamp
        self.msg.header.seq   = self.next_id()
        self.msg.header.frame_id = 'systime'

        self.msg.sys_time = stamp
        self.msg.gps_time = stamp
        self.msg.time = stamp.to_sec()
        self.msg.event_port = 23  # sentinel value
        self.msg.event_num = self.msg.header.seq & 0xffff
        return self


class GsofSpoofInsDispatch(DispatchBase):
    counter = 0
    message_class = GSOF_INS
    pubs = {}
    label = 'ins_spoof'

    def __new__(cls):

        self     = object.__new__(cls)
        self.msg = self.new_message()
        self.buf = bytes()


        utc_time = datetime_to_float(datetime.datetime.now())  # as unix time

        self.msg.header.stamp = rospy.Time.from_sec(utc_time)
        self.msg.header.seq   = self.next_id()
        self.msg.header.frame_id = 'systime'

        self.msg.time   = utc_time
        self.msg.altitude       = 333.0
        self.msg.total_speed    = 75.0
        self.msg.latitude       = 42.864407
        self.msg.longitude      = -73.7717448

        return self


def stream_gsof_chunker(buf):
    # type: (bytes) -> List(Tuple)
    """
    Break a binary stream into header/buffer pairs chunked into message size
    Args:
        buf:

    Returns:
        List of (header, message_buffer)
    """
    outs = []
    header = None
    while True:
        try:
            header = GsofHeader(buf)
        except struct.error as err:
            print('Failed to decode header: {} {}'.format(len(buf), err))
        if header is None:
            break
        outs.append((header, buf[:6 + header.len]))
        buf = buf[6 + header.len:]

    return outs


NullDispatch = ClsNullDispatch()

def parse_gsof(header, buf):
    # type: (GsofHeader, bytes) -> DispatchBase
    """
    Takes a header and a message, parses it to appropriate GSOF structure

    Returns a NullDispatch if it fails to parse
    Args:
        header: parsed header object
        buf: message bytes

    Returns:
        parsed Dispatch object
    """

    if header.message_type != GSOF_TYPE_MSG:
        rospy.logwarn('invalid message')
        return NullDispatch

    start = 9
    payload = buf[start:start + header.record_len]
    if header.record_type == GSOF_TYPE_INS:
        return GsofInsDispatch(payload)
    elif header.record_type == GSOF_TYPE_EVENT:
        return GsofEventDispatch(payload)
    elif header.record_type == GSOF_TYPE_RMS:
        raise NotImplementedError('RMS parser not available')
    else:
        rospy.logwarn('message type not understood')
        return NullDispatch


def parse_gsof_stream(buf):
    # type: (bytes) -> List[DispatchBase]
    """
    Takes a stream of
    Args:
        buf: stream of 1 or more binary messages

    Returns:
        List of parsed Dispatch objects
    """
    chunks = []
    try:
        chunks = stream_gsof_chunker(buf)
    except struct.error as err:
        print('Failed in parse_gsof_stream: {}'.format(err))
    return [parse_gsof(*chunk) for chunk in chunks]


def maybe_gsof(buf):
    # type: (bytes) -> bool
    if struct.unpack('B', buf[0:1])[0] == START_TX:
        return True
    return False


def separate_nmea(buf):
    # type: (bytes) -> Tuple[List, bytes]
    """
    Split off NMEA packets from binary packets
    It seems NMEA always comes before binary
    Args:
        buf: raw bytestream from AVX socket

    Returns:
        (list_of_nmea, binary)

    Examples:
        >>> stuff = separate_nmea(TEST_GSOF_PACKET)
        >>> stuff[0]
        '$GNGGA,154056.00,4251.87736134,N,07346.28348206,W,1,12,1.6,118.450,M,-31.849,M,,*4A'
        >>> stuff[1]
        '$PASHR,154056.000,354.688,T,1.115,-2.610,,0.248,0.248,71.520,1,2*27'
    """
    nmea_list = []
    tail = bytes(buf)
    while tail:
        if tail[0] == '$':
            temp = tail.split('\n', 1)
            if len(temp) == 2:
                head, tail = temp
                nmea_list.append(head.decode())
            else:
                nmea_list.append(temp[0].decode())
                break
        else:
            break

    return nmea_list, tail


def run_tests():
    from pprint import pprint
    from vprint.base256 import b256encode
    nmea_list, data = separate_nmea(TEST_GSOF_PACKET)

    dispatch = parse_gsof(data)
    pprint(nmea_list)
    print(len(data))
    print(b256encode(data))
    pprint(dispatch.msg)


if __name__ == '__main__':
    run_tests()


