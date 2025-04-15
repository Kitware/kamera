# -*- coding: utf-8 -*-
import struct
import datetime
import sys
import argparse

GPS_LEAP_SECONDS = 18

# Constants
gps_epoch = datetime.datetime(1980, 1, 6)
unix_epoch = datetime.datetime(1970, 1, 1)
gps_leap_td = datetime.timedelta(seconds=GPS_LEAP_SECONDS)

START_TX = 0x02
END_TX = 0x03
GSOF_TYPE_MSG = 0x40
GSOF_TYPE_INS = 0x31
GSOF_TYPE_RMS = 0x32
GSOF_TYPE_EVENT = 0x33


TEST_GSOF_INTS = bytes(
    bytearray(
        [
            2,
            40,
            64,
            109,
            229,
            0,
            0,
            49,
            104,
            7,
            253,
            14,
            78,
            242,
            216,
            2,
            1,
            64,
            69,
            110,
            172,
            58,
            222,
            106,
            2,
            192,
            82,
            113,
            95,
            3,
            136,
            113,
            59,
            64,
            84,
            68,
            60,
            150,
            102,
            25,
            245,
            188,
            20,
            9,
            73,
            61,
            121,
            188,
            109,
            61,
            58,
            145,
            51,
            61,
            124,
            118,
            160,
            63,
            246,
            158,
            84,
            17,
            111,
            200,
            62,
            192,
            2,
            45,
            171,
            66,
            52,
            8,
            28,
            64,
            93,
            182,
            36,
            117,
            138,
            130,
            100,
            64,
            88,
            155,
            124,
            109,
            98,
            92,
            40,
            189,
            90,
            233,
            64,
            61,
            108,
            17,
            141,
            62,
            144,
            209,
            101,
            60,
            134,
            61,
            16,
            187,
            228,
            180,
            183,
            187,
            170,
            252,
            107,
            148,
            3,
        ]
    )
)

TEST_NMEA = bytes(
    bytearray(
        """$GNGGA,154056.00,4251.87736134,N,07346.28348206,W,1,12,1.6,118.450,M,-31.849,M,,*4A
$PASHR,154056.000,354.688,T,1.115,-2.610,,0.248,0.248,71.520,1,2*27
""".encode()
    )
)

TEST_GSOF_PACKET = TEST_NMEA + TEST_GSOF_INTS


class GsofInsDispatch(object):
    counter = 0
    pubs = {}
    label = "ins"
    __slots__ = (
        "buf",
        "header",
        "time",
        "align_status",
        "gnss_status",
        "latitude",
        "longitude",
        "altitude",
        "north_velocity",
        "east_velocity",
        "down_velocity",
        "total_speed",
        "roll",
        "pitch",
        "heading",
        "track_angle",
        "angular_rate_x",
        "angular_rate_y",
        "angular_rate_z",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
    )

    def __new__(cls, buf):
        self = object.__new__(cls)
        self.buf = bytes()
        if buf is None:
            return self

        self.buf = buf
        data = struct.unpack(">HLbbdddffffddddffffff", buf)

        gps_week = data[0]
        gps_time = data[1] * 1e-3  # convert ms to s
        utc_time = gps_to_utc(gps_week, gps_time)  # as unix time

        self.time = utc_time
        self.align_status = data[2]
        self.gnss_status = data[3]
        self.latitude = data[4]
        self.longitude = data[5]
        self.altitude = data[6]
        self.north_velocity = data[7]
        self.east_velocity = data[8]
        self.down_velocity = data[9]
        self.total_speed = data[10]
        self.roll = data[11]
        self.pitch = data[12]
        self.heading = data[13]
        self.track_angle = data[14]
        self.angular_rate_x = data[15]
        self.angular_rate_y = data[16]
        self.angular_rate_z = data[17]
        self.acceleration_x = data[18]
        self.acceleration_y = data[19]
        self.acceleration_z = data[20]
        return self


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


class GsofEventDispatch(object):
    counter = 0
    pubs = {}
    label = "evt"
    __slots__ = ["buf", "header", "time", "event_port", "event_num"]

    def __new__(cls, buf):
        self = object.__new__(cls)
        self.buf = bytes()

        if buf is None:
            return self

        self.buf = buf

        data = struct.unpack(">BHdL", buf)

        gps_week = data[1]
        gps_time = data[2]  # is actually seconds, unlike INS packet
        utc_time = gps_to_utc(gps_week, gps_time)  # as unix time

        self.time = utc_time
        self.event_port = data[0]
        self.event_num = data[3]  # todo: should seq id match this?
        # self.msg.header.seq   = self.next_id()
        return self


class GsofHeader(object):
    __slots__ = [
        "len",
        "transmission_num",
        "page_index",
        "max_page_index",
        "record_type",
        "record_len",
        "ok",
        "len",
        "checksum",
    ]

    def __new__(cls, buf):
        if len(buf) < 9:
            return None
        header = struct.unpack(">9B", buf[:9])
        if header[0] != START_TX:
            print("Start byte does not match STX")
            return None

        self = object.__new__(cls)

        ln = header[3]

        # try:
        checksum, end = struct.unpack(">BB", buf[4 + ln : 6 + ln])
        computed_checksum = sum(bytearray(buf[1:-2])) & 0xFF
        # except struct.error as err:
        #     print('failed to unpack {}'.format(err  ))
        if end != END_TX:
            print("Final byte does not match ETX")
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


def separate_nmea(buf):
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
        if tail[0] == "$":
            temp = tail.split("\n", 1)
            if len(temp) == 2:
                head, tail = temp
                nmea_list.append(head.decode())
            else:
                nmea_list.append(temp[0].decode())
                break
        else:
            break

    return nmea_list, tail


def parse_gsof(header, buf):
    """
    Takes a header and a message, parses it to appropriate GSOF structure

    Returns a NullDispatch if it fails to parse
    Args:
        header: parsed header object
        buf: message bytes

    Returns:
        parsed Dispatch object
    """
    start = 9
    payload = buf[start : start + header.record_len]
    if header.record_type == GSOF_TYPE_INS:
        return GsofInsDispatch(payload)
    elif header.record_type == GSOF_TYPE_EVENT:
        return GsofEventDispatch(payload)
    elif header.record_type == GSOF_TYPE_RMS:
        raise NotImplementedError("RMS parser not available")
    else:
        print("message type not understood")
        return None


def stream_gsof_chunker(buf):
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
            print("Failed to decode header: {} {}".format(len(buf), err))
        if header is None:
            break
        outs.append((header, buf[: 6 + header.len]))
        buf = buf[6 + header.len :]

    return outs


def parse_gsof_stream(buf):
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
        print("Failed in parse_gsof_stream: {}".format(err))
    return [parse_gsof(*chunk) for chunk in chunks]


def maybe_gsof(buf):
    # type: (bytes) -> bool
    if struct.unpack("B", buf[0:1])[0] == START_TX:
        return True
    return False


def enumerate_packets(buf):
    """
    Unpack a binary stream packaged using dumpbuf

    Usage:
        for i, packet in enumerate_packets(buf):
            do_stuff(packet)


    Args:
        buf: Raw binary

    Returns:
        Generator which yields (i, packet)
    """
    count = 0
    p = 0  # pointer
    end = len(buf)
    while p < end:
        sth, length = struct.unpack(">BH", buf[p : p + 3])
        if end - p + 1 < length:
            raise StopIteration("Incomplete packet")
        payload, tail = (
            buf[p + 3 : p + length + 3],
            buf[p + length + 3 : p + length + 8],
        )
        try:
            etb, checksum, cr, nl = struct.unpack(">BHBB", tail)
            check = sum(bytearray(payload)) & 0xFFFF
            if check != checksum:
                print("Warning: invalid checksum")
        except:
            p = p + length + 8
            continue
        p = p + length + 8
        yield count, payload
        count += 1


def convert_dat_to_csv(path_to_data):
    print("Converting .dat file.")
    with open(path_to_data, "rb") as fp:
        raw_stream = fp.read()

        # recv-loop: When we're connected, keep receiving stuff until that fails
        out = []
        header = []
        for i, rawdata in enumerate_packets(raw_stream):
            if not (i % 100 == 0):
                # only save at 1hz
                continue
            # todo: optionally archive stream
            nmea_list, gsof_data = separate_nmea(rawdata)
            # aprint(nmea_list)
            # aprint(str(len(gsof_data)) + '[' + b256encode(gsof_data) + ']')

            if maybe_gsof(gsof_data):
                dispatches = parse_gsof_stream(gsof_data)
                # aprint(dispatch.msg)
                for d in dispatches:
                    try:
                        l = []
                        header = []
                        for slot in GsofInsDispatch.__slots__:
                            if slot == "buf" or slot == "header":
                                continue
                            l.append(str(getattr(d, slot)))
                            header.append(slot)
                        out.append(l)
                    except AttributeError as e:
                        pass
                    except Exception as e:
                        print(e)
                        pass
                continue

        lines = []
        head = ",".join(header)
        lines.append(head)
        for entry in out:
            lines.append("\n")
            line = ",".join(entry)
            lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "dat_fname", type=str, help="The path to an existing .dat file."
    )
    args = parser.parse_args()
    lines = convert_dat_to_csv(args.dat_fname)
    cname = args.dat_fname.replace(".dat", ".csv")
    with open(cname, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
