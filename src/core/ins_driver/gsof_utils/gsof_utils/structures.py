import struct
import warnings
from six import string_types

from gsof_utils.constants import START_TX, END_TX


def render_inner_str(s_or_obj, quote='"'):
    if isinstance(s_or_obj, string_types):
        return quote + s_or_obj + quote
    return repr(s_or_obj)


class ManditoryInitializer(object):
    def __init__(self, **kwargs):
        sslots = set(self.__slots__)
        diffs = sslots.symmetric_difference(kwargs)
        if diffs:
            raise ValueError("missing fields: {}".format([x for x in diffs]))

        for k, v in kwargs.items():
            setattr(self, k, v)


class DefaultInitializer(object):
    def __init__(self, **kwargs):
        for dk, dv in self.__defaults__.items():
            if dk not in kwargs:
                kwargs[dk] = dv

        for k, v in kwargs.items():
            setattr(self, k, v)


class LitePDict(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __eq__(self, other):
        return self.items() == other.items()

    def __str__(self):
        kves = ["{}={}".format(k, render_inner_str(v)) for k, v in self.items()]
        kwargstr = ", ".join(kves)
        return "{}({})".format(self.__class__.__name__, kwargstr)

    def __repr__(self):
        return str(self)

    def keys(self):
        return self.__slots__

    def items(self):
        return [(k, getattr(self, k)) for k in self.__slots__]

    def values(self):
        # convenience method, not optimal
        return [el[1] for el in self.items()]

    @classmethod
    def from_mapping(cls, d):
        tmp = cls.__new__(cls)  # we need a totally blank object
        for key, value in d.items():
            tmp[key] = value
        return tmp

    def copy(self):
        return self.__class__.from_mapping(self)


class GsofRecord(LitePDict):
    __slots__ = ["record_type", "record_len", "data"]

    def __init__(self, record_type=0, record_len=0, data=b""):
        self.record_type = record_type  # type: int
        self.record_len = record_len  # type: int
        self.data = data  # type: bytes


class GsofPacket(LitePDict, DefaultInitializer):
    __slots__ = ["message_type", "len", "transmission_num", "page_index", "max_page_index", "recs"]
    __defaults__ = {
        "message_type": 0,
        "len": 0,
        "transmission_num": 0,
        "page_index": 0,
        "max_page_index": 0,
        "recs": [],
    }

    @classmethod
    def from_buf(cls, buf):
        if len(buf) < 9:
            warnings.warn("GsofChunkerError: Packet too short with length: {}".format(len(buf)))
            return None
        first_byte = buf[0:1]
        if first_byte != START_TX:
            # rospy.logwarn('Start byte does not match STX')
            warnings.warn("first byte did not match: got: {} want: {}".format(repr(first_byte), repr(START_TX)))
            return None
        start = 0
        header = struct.unpack(">9B", buf[:9])

        self = cls.__new__(cls)

        self.message_type = header[2]
        gsof_len = header[3]

        # try:
        checksum, last_byte = struct.unpack(">BB", buf[4 + gsof_len : 6 + gsof_len])
        end = start + gsof_len + 6
        last_byte = buf[end - 1 : end]
        if last_byte != END_TX:
            warnings.warn("last byte did not match: got: {} want: {}".format(repr(last_byte), repr(END_TX)))
            return None

        checksum = sum(bytearray(buf[start + gsof_len + 4 : start + gsof_len + 5]))
        computed_checksum = sum(bytearray(buf[start + 1 : start + gsof_len + 4])) & 0xFF
        if checksum != computed_checksum:
            warnings.warn("checksum did not match: got: {} want: {}".format(repr(computed_checksum), repr(checksum)))

        self.len = gsof_len
        self.transmission_num = header[4]
        self.page_index = header[5]
        self.max_page_index = header[6]
        records = []
        idx = 7
        while idx < gsof_len:

            record_type, record_len = struct.unpack(">BB", buf[idx : idx + 2])
            records.append(GsofRecord(record_type, record_len, buf[idx : idx + record_len + 2]))
            idx += record_len + 2

        self.recs = records
        self.ok = computed_checksum == checksum

        self.checksum = checksum
        return self


class UtcMessage(LitePDict, DefaultInitializer):
    __slots__ = ["gps_week", "gps_time", "offset", "flags", "time", "type"]
    __defaults__ = {"gps_week": 0, "gps_time": 0.0, "offset": 0, "flags": 0, "time": None, "type": "utc_gsof"}


class TimeSyncInfo(LitePDict, DefaultInitializer):
    __slots__ = ["gps_offset", "gps_offset_kind", "ntp_offset", "ntp_stratum", "ntp_valid", "ntp_addr"]
    __defaults__ = {
        "gps_offset": 0,
        "gps_offset_kind": "null",
        "ntp_offset": 0,
        "ntp_stratum": 16,
        "ntp_valid": False,
        "ntp_addr": None,
    }
