from __future__ import division

import threading
import time
from datetime import datetime, timedelta
import numbers
from kamcore.datatypes import ToDictMxn, TryIntoAttrMxn, DefaultInitializer

MILLION = 1000000
BILLION = 1000000000
BILLIONF = 1000000000.0
SECONDS_PER_DAY = 86400
UNIX_EPOCH = datetime(1970, 1, 1)


class _Default(object):
    def __str__(self):
        return "<default>"


_default = _Default()


def _canon(secs, nsecs):
    # canonical form: nsecs is always positive, nsecs < 1 second
    secs_over = nsecs // 1000000000
    secs += secs_over
    nsecs -= secs_over * 1000000000
    return secs, nsecs


class BasicStamp(ToDictMxn, TryIntoAttrMxn):
    """ Static-typed data class for use with interoperating with rospy.Time objects.
    This is to facilitate development and testing without ROS dependencies
    Can be assigned to header.stamp and successfully serialized.

    If you need to do arithmetic, use into(rospy.Time()) or into(rospy.Duration()). 
    I am not bothering to implement the whole interface. 
    """

    __defaults__ = {"secs": 0, "nsecs": 0}
    __slots__ = ["secs", "nsecs"]

    # mimic same API as messages when being introspected
    _slot_types = ["int32", "int32"]

    def __init__(self, secs=0, nsecs=0):  # noqa: D205, D400
        """
        :param secs: seconds. If secs is a float, then nsecs must not be set or 0,
          larger seconds will be of type long on 32-bit systems, ``int/long/float``
        :param nsecs: nanoseconds, ``int``
        """
        if not isinstance(secs, numbers.Integral):
            # float secs constructor
            if nsecs != 0:
                raise ValueError("if secs is a float, nsecs cannot be set")
            float_secs = secs
            secs = int(float_secs)
            nsecs = int((float_secs - secs) * 1000000000)
        else:
            secs = int(secs)
            nsecs = int(nsecs)

        self.secs, self.nsecs = _canon(secs, nsecs)

    @classmethod
    def from_any(cls, other):
        """Try to coerce from some other time-like object. If 'other' is a singular number (int or float),
        then treat it as seconds"""
        if isinstance(other, cls):
            return other.copy()
        if isinstance(other, numbers.Real):
            return cls.from_sec(other)
        other_to_nsec = getattr(other, "to_nsec", None)
        if other_to_nsec is not None:
            return cls(secs=0, nsecs=other_to_nsec())

        if isinstance(other, (datetime, timedelta)):
            if isinstance(other, datetime):
                td = other - UNIX_EPOCH
            else:
                td = other

            return cls.from_sec(td.total_seconds())

    def to_nsec(self):
        # type: () -> int
        """Total nanoseconds since epoch"""
        return int(self.secs * BILLION + self.nsecs)

    def to_sec(self):
        """Total seconds (with decimal) since epoch"""
        return self.secs * BILLIONF + float(self.nsecs)

    @classmethod
    def from_stamp(cls, stamp):
        # type: (BasicStamp) -> BasicStamp
        """From any structure with secs, nsecs fields. """
        return cls(stamp.secs, stamp.nsecs)

    @classmethod
    def from_sec(cls, float_secs):
        """
        Create new TVal instance using time.time() value (float seconds).

        :param float_secs: time value in time.time() format, ``float``
        :returns: :class:`BasicStamp` instance for specified time
        """
        secs = int(float_secs)
        nsecs = int((float_secs - secs) * BILLION)
        return cls(secs, nsecs)

    @classmethod
    def from_timedelta(cls, td):
        # type: (timedelta) -> BasicStamp
        """Coerce from datetime.timedelta"""
        nsecs = td.microseconds * 1000
        secs = td.days * SECONDS_PER_DAY + td.seconds
        return cls(secs, nsecs)

    @classmethod
    def now(cls):
        return cls.from_sec(time.time())

    def is_zero(self):
        return self.secs == 0 and self.nsecs == 0

    def canon(self):
        """
        Canonicalize the field representation in this instance.

        Should only be used when manually setting secs/nsecs slot values, as
        in deserialization.
        """
        self.secs, self.nsecs = _canon(self.secs, self.nsecs)

    def set(self, secs, nsecs):
        """
        Set time using separate secs and nsecs values.

        :param secs: seconds since epoch, ``int``
        :param nsecs: nanoseconds since seconds, ``int``
        """
        self.secs = secs
        self.nsecs = nsecs

    def __lt__(self, other):
        """< test for time values."""

        return self.__cmp__(other) < 0

    def __le__(self, other):
        """<= test for time values."""

        return self.__cmp__(other) <= 0

    def __gt__(self, other):
        """> test for time values."""

        return self.__cmp__(other) > 0

    def __ge__(self, other):
        """>= test for time values."""

        return self.__cmp__(other) >= 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def __cmp__(self, other):
        other_to_nsec = getattr(other, "to_nsec", None)
        if other_to_nsec is None:
            raise TypeError("Cannot compare with type {}: lacks to_nsec() method".format(type(other)))
        a = self.to_nsec()
        b = other_to_nsec()
        return (a > b) - (a < b)

    def __eq__(self, other):
        other_to_nsec = getattr(other, "to_nsec", None)
        if other_to_nsec is None:
            return False
        return self.to_nsec() == other_to_nsec()


class BasicHeader(ToDictMxn, TryIntoAttrMxn, DefaultInitializer):
    __defaults__ = {"seq": 0, "frame_id": "", "stamp": BasicStamp()}
    __slots__ = ["seq", "frame_id", "stamp"]
    __seq__ = -1

    @classmethod
    def new(cls, frame_id=""):
        cls.__seq__ += 1
        return cls(seq=cls.__seq__, frame_id=frame_id, stamp=BasicStamp.now())


class BasicEvent(ToDictMxn, TryIntoAttrMxn):
    __slots__ = ["header", "time", "sys_time", "gps_time", "event_port", "event_port", "event_num"]

    def __init__(self, header=None, sys_time=None, gps_time=None, event_port=None, event_num=None, time=None):
        self.header = header or BasicHeader()
        self.sys_time = sys_time or BasicStamp()
        self.gps_time = gps_time or BasicStamp()
        self.event_port = (event_port or 0) & 0xFF
        self.event_num = (event_num or 0) & 0xFFFF
        self.time = time or 0.0


class LockyDict(dict):
    def __init__(self, *args, **kwargs):
        super(LockyDict, self).__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def __setitem__(self, name, value):
        with self._lock:
            # print(f'<>{type(self).__name__}["{name}"]"')
            super(LockyDict, self).__setitem__(name, value)

    def __getitem__(self, name):
        with self._lock:
            return super(LockyDict, self).__getitem__(name)

    def get(self, key, default=_default):
        with self._lock:
            if default is _default:
                v = super(LockyDict, self).get(key)
                return v

            v = super(LockyDict, self).get(key, default)
            return v
