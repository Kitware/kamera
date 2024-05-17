import time
import datetime
from kamcore.structures import BasicStamp, BILLION


def float_to_stamp(float_secs):
    secs = int(float_secs)
    nsecs = int((float_secs - secs) * BILLION)
    return BasicStamp(secs, nsecs)
