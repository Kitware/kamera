#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import codecs

import pytz

from gsof_utils.parsers import avx_chunker
from gsof_utils.structures import GsofPacket, GsofRecord
from gsof_utils.parsers import parse_raw_record, parse_raw_packet

sample_gsof_rmc_5hz = """02 28 40 0e 29 00 00 10 09 13 eb a3 d3 08 5c 00 00 00 90 03 24 47 50 52 4d 43 2c 32 30 34 39 35 32 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 35 0d 0a
02 28 40 0e 4d 00 00 10 09 13 eb b7 5b 08 5c 00 00 00 50 03 24 47 50 52 4d 43 2c 32 30 34 39 35 37 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 30 0d 0a
02 28 40 0e 71 00 00 10 09 13 eb ca e3 08 5c 00 00 00 0f 03 24 47 50 52 4d 43 2c 32 30 35 30 30 32 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 38 0d 0a
02 28 40 0e 95 00 00 10 09 13 eb de 6c 08 5c 00 00 00 d0 03 24 47 50 52 4d 43 2c 32 30 35 30 30 37 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 44 0d 0a
02 28 40 0e b9 00 00 10 09 13 eb f1 f3 08 5c 00 00 00 8e 03 24 47 50 52 4d 43 2c 32 30 35 30 31 32 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 39 0d 0a
02 28 40 0e dd 00 00 10 09 13 ec 05 7b 08 5c 00 00 00 4f 03 24 47 50 52 4d 43 2c 32 30 35 30 31 37 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 43 0d 0a
02 28 40 0e 01 00 00 10 09 13 ec 19 03 08 5c 00 00 00 0f 03 24 47 50 52 4d 43 2c 32 30 35 30 32 32 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 41 0d 0a"""

sample_pps_gsof_rmc_1hz = """02 28 40 0e 2c 00 00 10 09 13 f9 8e 53 08 5c 00 00 00 0c 03 24 47 50 52 4d 43 2c 32 31 30 35 30 34 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 46 0d 0a
55 54 43 20 32 31 2e 30 31 2e 31 33 20 32 31 3a 30 35 3a 30 35 20 3f 3f 0d 0a
02 28 40 0e 34 00 00 10 09 13 f9 92 3b 08 5c 00 00 00 00 03 24 47 50 52 4d 43 2c 32 31 30 35 30 35 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 45 0d 0a
55 54 43 20 32 31 2e 30 31 2e 31 33 20 32 31 3a 30 35 3a 30 36 20 3f 3f 0d 0a
02 28 40 0e 3c 00 00 10 09 13 f9 96 23 08 5c 00 00 00 f4 03 24 47 50 52 4d 43 2c 32 31 30 35 30 36 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 44 0d 0a
55 54 43 20 32 31 2e 30 31 2e 31 33 20 32 31 3a 30 35 3a 30 37 20 3f 3f 0d 0a
02 28 40 0e 45 00 00 10 09 13 f9 9a 0b 08 5c 00 00 00 e9 03 24 47 50 52 4d 43 2c 32 31 30 35 30 37 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 43 0d 0a
55 54 43 20 32 31 2e 30 31 2e 31 33 20 32 31 3a 30 35 3a 30 38 20 3f 3f 0d 0a
02 28 40 0e 4d 00 00 10 09 13 f9 9d f3 08 5c 00 00 00 dc 03 24 47 50 52 4d 43 2c 32 31 30 35 30 38 2e 30 30 2c 56 2c 2c 2c 2c 2c 2c 2c 31 33 30 31 32 31 2c 2c 2c 4e 2a 37 33 0d 0a
55 54 43 20 32 31 2e 30 31 2e 31 33 20 32 31 3a 30 35 3a 30 39 20 3f 3f 0d 0a"""

utc_valid_packet = b'\x02(@\x0e\xa9\x00\x00\x10\t\x0f\x1d\xa1\t\x08\\\x00\x12\x03\x87\x03'

b'$GNRMC,222622.00,A,4251.87878501,N,07346.28551606,W,0.016,127.620,120121,13.4465,W,A*2B'
b'\x02(@\x0e\xa9\x00\x00\x10\t\x0f\x1d\xa1\t\x08\\\x00\x12\x03\x87\x03'
utc_actual_time = datetime.datetime(2021, 1, 12, 22, 26, 22, 9000, tzinfo=pytz.utc)
nmea_actual_time = datetime.datetime(2021, 1, 12, 22, 26, 22, 0, tzinfo=pytz.utc)
# {'gps_week': 2140, 'gps_time': 253600.009}
# {'time': datetime.datetime(2021, 1, 12, 22, 26, 22, 9000, tzinfo=datetime.timezone.utc), 'utc': datetime.datetime(2021, 1, 12, 22, 26, 22, 9000, tzinfo=datetime.timezone.utc), 'gps': datetime.datetime(2021, 1, 12, 22, 26, 40, 9000, tzinfo=datetime.timezone.utc), 'offset': 18, 'flags': b'\x03'}

def lmap(f, i):
    return list(map(f, i))


def decode_hexline(line):
    return codecs.decode(line.replace(" ", ""), "hex_codec")


packets_gsof_rmc_5hz = lmap(decode_hexline, sample_gsof_rmc_5hz.split("\n"))
packets_pps_gsof = lmap(decode_hexline, sample_pps_gsof_rmc_1hz.split("\n"))


def self_test(N=-1):
    for packets in [packets_gsof_rmc_5hz[:N], packets_pps_gsof[:N]]:
        for packet in packets:
            chunks = avx_chunker(packet)
            gsof_parsed = [GsofPacket.from_buf(gs) for gs in chunks['gsofs']]
            if chunks['pps']:
                print(chunks['pps'])
            else:
                print(chunks['rmc'])
                print(gsof_parsed[0])
                print(parse_raw_packet(gsof_parsed[0]))
            print('---')

    parsed_utc_packet = GsofPacket.from_buf(utc_valid_packet)
    utc_message = parse_raw_packet(parsed_utc_packet)[0]
    utc_message.time = utc_message.time.replace(microsecond=0)
    print(utc_message.time, nmea_actual_time, utc_message.time.astimezone(pytz.UTC))
    assert utc_message.time == nmea_actual_time


if __name__ == "__main__":
    self_test()
    print("ok")

