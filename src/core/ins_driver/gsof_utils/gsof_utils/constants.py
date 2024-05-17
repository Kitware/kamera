from enum import Enum

START_TX = b"\x02"
END_TX = b"\x03"

GSOF_START_PATTERN = b"(?P<startseq>\x02[\s\S]@)(?P<len>[\s\S])"
RMC_START_PATTERN = b"\$G.RMC"
TIME_PPS = "UTC %y.%m.%d %H:%M:%S"  # %z is not py2 compat


class GSOF_TYPE(Enum):
    UTC = 16
    INS = 49


class UTC_BIT_MASK(Enum):
    TimeValid = 1
    UtcOffsetValid = 2
