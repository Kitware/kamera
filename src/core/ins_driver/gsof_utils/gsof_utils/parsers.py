#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import struct
import warnings

import datetime
from typing import Tuple, List, Optional
import pytz
from dateutil.tz import gettz

from gsof_utils.constants import (
    START_TX,
    END_TX,
    GSOF_START_PATTERN,
    RMC_START_PATTERN,
    TIME_PPS,
    GSOF_TYPE,
    UTC_BIT_MASK,
)
from gsof_utils.structures import GsofRecord, GsofPacket, UtcMessage, LitePDict

PAT_GSOF_START = re.compile(GSOF_START_PATTERN)
PAT_RMC_START = re.compile(RMC_START_PATTERN)

# a GPS TZ. separate from UTC, but we don't know the leap seconds just yet
# GPS_NAIVE_TZ = datetime.timezone(datetime.timedelta(seconds=0), 'gps_naive')


class CustomTzInfo(pytz.tzinfo.StaticTzInfo):
    def __init__(self, name="", offset=None, dst=None):
        self.zone = name
        self._utcoffset = offset or datetime.timedelta(0)
        self._dst = dst or datetime.timedelta(0)
        self._tzname = self.zone


def make_gps_tz(leap_seconds=0):
    name = "GPS{:d}".format(leap_seconds)
    return CustomTzInfo(name, datetime.timedelta(seconds=leap_seconds))


GPS_NAIVE_TZ = CustomTzInfo("GPS_NAIVE")

gps_epoch_naive = datetime.datetime(1980, 1, 6, tzinfo=GPS_NAIVE_TZ)
unix_epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)


def maybe_parse_pps(bb):
    # type: (bytes) -> Optional[datetime.datetime]
    if b"UTC" != bb[:3]:
        # warnings.warn('Does not look like PPS packet'.format(bb))
        return None
    try:
        s = bb[:21].decode()
    except UnicodeDecodeError as exc:
        warnings.warn("{}: {}".format(exc.__class__.__name__, exc))
        return None
    try:
        return datetime.datetime.strptime(s, TIME_PPS).replace(tzinfo=pytz.UTC)
    except ValueError as exc:
        warnings.warn("{}: {}".format(exc.__class__.__name__, exc))
        return None


def gsof_chunker(buf):
    # type: (bytes) -> Tuple[List[bytes], bytes]
    if len(buf) < 9:
        warnings.warn("GsofChunkerError: Packet too short with length: {}".format(len(buf)))
        return [], buf

    gsofs = []
    possible_starts = re.finditer(PAT_GSOF_START, buf)
    for hit in possible_starts:
        if not hit:
            continue
        maybe_len = hit.groupdict().get("len", None)
        if maybe_len is None:
            continue
        gsof_len = struct.unpack(">B", maybe_len)[0]
        if gsof_len < 3:
            # can't be a real packet
            continue
        start = hit.start()
        end = start + gsof_len + 6
        last_byte = buf[end - 1 : end]
        if last_byte != END_TX:
            # expected end byte
            #             print(last_byte)
            warnings.warn("last byte did not match: got: {} want: {}".format(repr(last_byte), repr(END_TX)))
            continue
        checksum = sum(bytearray(buf[start + gsof_len + 4 : start + gsof_len + 5]))
        computed_checksum = sum(bytearray(buf[start + 1 : start + gsof_len + 4])) & 0xFF
        if checksum != computed_checksum:
            warnings.warn("checksum did not match: got: {} want: {}".format(repr(computed_checksum), repr(checksum)))

        gsof = buf[start:end]
        gsofs.append(gsof)

    resid = bytearray(buf)
    for gsof in gsofs:
        resid = resid.replace(gsof, b"")
    return gsofs, bytes(resid)


def maybe_parse_rmc(buf, utc_date=None):
    if not utc_date:
        utc_now = datetime.datetime.now(gettz()).astimezone(pytz.UTC)
        utc_date = utc_now.date()
    #     warnings.warn("UTC day not set")
    #     return None
    starts = list(re.finditer(PAT_RMC_START, buf))
    if not starts:
        return None
    try:
        start = starts[0].start()
        tn = buf[start + 7 : start + 17].decode()
        hh, mm, ss, ms = map(int, [tn[:2], tn[2:4], tn[4:6], tn[7:9]])
        now = datetime.datetime(
            year=utc_date.year, month=utc_date.month, day=utc_date.day, hour=hh, minute=mm, second=ss, tzinfo=pytz.UTC
        )
        return now
    except Exception as exc:
        warnings.warn("{}: {}".format(exc.__class__.__name__, exc))


def avx_chunker(buf):
    gsofs, resid = gsof_chunker(buf)
    maybe_pps = maybe_parse_pps(resid)
    if maybe_pps is None:
        rmc = maybe_parse_rmc(resid)
    else:
        rmc = None

    return {"gsofs": gsofs, "rmc": rmc, "pps": maybe_pps}


def gps_to_dt(gps_week, gps_time, offset=0, flags=0):
    # type: (int, float, int, int) -> float
    gps_td = datetime.timedelta(days=gps_week * 7, seconds=gps_time)
    if flags & UTC_BIT_MASK.UtcOffsetValid.value:
        tz = make_gps_tz(offset)
        origin = gps_epoch_naive.replace(tzinfo=tz)
    else:
        origin = gps_epoch_naive
    time_dt = origin + gps_td
    return time_dt


def parse_rec_utc(utc_record):
    # type: (GsofRecord) -> UtcMessage
    assert utc_record.record_type == GSOF_TYPE.UTC.value
    sup = struct.unpack(">BBlhhB", utc_record.data)
    gps_time = sup[2] * 1e-3
    gps_week = sup[3]
    offset = sup[4]
    flags = sup[5]
    #     stamp = gps_to_utc(gps_week, gps_time)
    time_dt = gps_to_dt(gps_week, gps_time, offset, flags)
    return UtcMessage(gps_week=gps_week, gps_time=gps_time, offset=offset, flags=flags, time=time_dt)


def parse_rec_ins(ptut):
    sup = struct.unpack(">BBhlBB", ptut[:10])
    gps_time = sup[3] * 1e-3
    gps_week = sup[2]
    #     stamp = gps_to_utc(gps_week, gps_time)
    return {"type": "ins", "gps_week": gps_week, "gps_time": gps_time, "other_fields": "..."}


def parse_raw_record(record):
    # type (GsofRecord) -> dict
    # import ipdb
    # ipdb.set_trace()
    # print("Record: {}".format(record))
    if record.record_type == int(GSOF_TYPE.UTC.value):
        return parse_rec_utc(record)
    elif record.record_type == int(GSOF_TYPE.INS.value):
        return parse_rec_ins(record)
    else:
        raise ValueError("Unable to parse message type: {}".format(record.record_type))


def parse_raw_packet(gsof_packet):
    # type (GsofPacket) -> List
    return [parse_raw_record(rec) for rec in gsof_packet.recs]


def process_buffer(buf):
    chunks = avx_chunker(buf)
    parsed_gsofs = []
    for gsof_raw in chunks['gsofs']:
        gsof_packet = GsofPacket.from_buf(gsof_raw)
        parsed_gsofs.extend(parse_raw_packet(gsof_packet))
    chunks['parsed_gsofs'] = parsed_gsofs
    if chunks['rmc']:
        for msg in parsed_gsofs:
            if isinstance(msg, UtcMessage):
                if msg.offset:
                    chunks['offset'] = msg.offset
                    chunks['offset_kind'] = 'gps'
                else:
                    offset = msg.time - chunks['rmc']
                    offset = datetime.timedelta(seconds=offset.seconds)
                    chunks['offset'] = offset
                    chunks['offset_kind'] = 'computed'
    return chunks


def self_test():
    """maintaining python2/3 compatible tests is painful"""
    tmp = b"UTC 21.01.13 21:05:09"
    basic_gsof = b"\x02(@\x0e,\x00\x00\x10\t\x13\xf9\x8eS\x08\\\x00\x00\x00\x0c\x03"
    latent_rmc = (
        b"\x02(@\x0e,\x00\x00\x10\t\x13\xf9\x8eS\x08\\\x00\x00\x00\x0c\x03$GPRMC,210504.00,V,,,,,,,130121,,,N*7F\r\n"
    )
    out_pps = maybe_parse_pps(tmp)
    chunks = avx_chunker(basic_gsof)

    gsof_packet = GsofPacket.from_buf(basic_gsof)

    assert out_pps == datetime.datetime(2021, 1, 13, 21, 5, 9, tzinfo=pytz.UTC)
    assert maybe_parse_rmc(latent_rmc) == datetime.datetime(2021, 1, 13, 21, 5, 4, tzinfo=pytz.UTC)
    assert chunks["gsofs"] == [basic_gsof]

    expected = GsofPacket(
        message_type=64,
        len=14,
        transmission_num=44,
        page_index=0,
        max_page_index=0,
        recs=[GsofRecord(record_type=16, record_len=9, data=b"\x10\t\x13\xf9\x8eS\x08\\\x00\x00\x00")],
    )
    assert gsof_packet == expected


if __name__ == "__main__":
    self_test()
    print("ok")
