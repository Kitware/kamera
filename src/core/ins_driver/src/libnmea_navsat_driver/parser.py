# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Eric Perko
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

from typing import List
import re
import time
import calendar
import math
import logging
# for doctests and debugging
from pprint import pprint
from vprint import aprint
from vprint.loggers import get_verbose_logger
# logger = logging.getLogger('rosout')
logger = get_verbose_logger('rosout2', verbose=True)


def safe_float(field):
    try:
        return float(field)
    except ValueError:
        return float('NaN')


def safe_int(field):
    try:
        return int(field)
    except ValueError:
        return 0


def convert_latitude(field):
    return safe_float(field[0:2]) + safe_float(field[2:]) / 60.0


def convert_longitude(field):
    return safe_float(field[0:3]) + safe_float(field[3:]) / 60.0


def convert_time(nmea_utc):
    # Get current time in UTC for date information
    utc_struct = time.gmtime()  # immutable, so cannot modify this one
    utc_list = list(utc_struct)
    # If one of the time fields is empty, return NaN seconds
    if not nmea_utc[0:2] or not nmea_utc[2:4] or not nmea_utc[4:6]:
        return float('NaN')
    else:
        hours = int(nmea_utc[0:2])
        minutes = int(nmea_utc[2:4])
        seconds = int(nmea_utc[4:6])
        utc_list[3] = hours
        utc_list[4] = minutes
        utc_list[5] = seconds
        unix_time = calendar.timegm(tuple(utc_list))
        return unix_time


def convert_time_float(nmea_utc):
    t = convert_time(nmea_utc)
    if math.isnan(t):
        return t
    else:
        return t + float(nmea_utc[6:])


def convert_status_flag(status_flag):
    if status_flag == "A":
        return True
    elif status_flag == "V":
        return False
    else:
        return False


def convert_knots_to_mps(knots):
    return safe_float(knots) * 0.514444444444


# Need this wrapper because math.radians doesn't auto convert inputs
def convert_deg_to_rads(degs):
    return math.radians(safe_float(degs))

"""Format for this dictionary is a sentence identifier (e.g. "GGA") as the key, with a
list of tuples where each tuple is a field name, conversion function and index
into the split sentence"""
parse_maps = {
    "GGA": [
        ("fix_type", int, 6),
        ("ulatitude", convert_latitude, 2),
        ("latitude_direction", str, 3),
        ("ulongitude", convert_longitude, 4),
        ("longitude_direction", str, 5),
        ("orthometric_height", safe_float, 9),
        ("mean_sea_level", safe_float, 11),
        ("hdop", safe_float, 8),
        ("num_satellites", safe_int, 7),
        ("utc_time", convert_time_float, 1),
        ],
    "RMC": [
        ("utc_time", convert_time_float, 1),
        ("fix_valid", convert_status_flag, 2),
        ("ulatitude", convert_latitude, 3),
        ("latitude_direction", str, 4),
        ("ulongitude", convert_longitude, 5),
        ("longitude_direction", str, 6),
        ("speed", convert_knots_to_mps, 7),
        ("true_course", convert_deg_to_rads, 8),
        ],
    "GST": [
        ("utc_time", convert_time_float, 1),
        ("ranges_std_dev", safe_float, 2),
        ("semi_major_ellipse_std_dev", safe_float, 3),
        ("semi_minor_ellipse_std_dev", safe_float, 4),
        ("semi_major_orientation", safe_float, 5),
        ("lat_std_dev", safe_float, 6),
        ("lon_std_dev", safe_float, 7),
        ("alt_std_dev", safe_float, 8),
        ],
    "HDT": [
        ("heading", safe_float, 1),
        ],
    "PASHR": [
        ("utc_time", convert_time_float, 1),
        ("heading", safe_float, 2),
        ("roll", safe_float, 4),
        ("pitch", safe_float, 5),
        ("gnss_status", int, 10),
        ("imu_alignment_status", int, 11),
        ],
    "EVT": [
        ("utc_time", convert_time_float, 2),
        ("event", int, 3),
        ("event_counter", int, 4),
        ],
    }


def rectify_latlonalt(geo_data):
    # type: (dict) -> dict
    if 'latitude_direction' not in geo_data:
        # no need to rectify
        return geo_data
    latitude = geo_data['ulatitude']
    if geo_data['latitude_direction'] == 'S':
        latitude = -latitude

    longitude = geo_data['ulongitude']
    if geo_data['longitude_direction'] == 'W':
        longitude = -longitude
    geo_data.update({'latitude': latitude, 'longitude': longitude})

    # Altitude is above ellipsoid, so adjust for mean-sea-level
    ortho = geo_data.get('orthometric_height', None)
    if ortho is not None:
        altitude = ortho + geo_data['mean_sea_level']
        geo_data.update({'altitude': altitude})

    return geo_data


def parse_sentence_type(fields):
    # type: (List[str]) -> str
    """
    Parse sentence type from list of fields
    Args:
        fields: Lists of NMEA string fields

    Returns:
        proper field name

    Examples:
        >>> parse_sentence_type(['$PASHR','191019.500','57.100','T','1.161'])
        'PASHR'
        >>> parse_sentence_type(['$PTNL','EVT','19','1','4','2045','1','18*72'])
        'EVT'
        >>> parse_sentence_type(['$GNGGA','191020.00','4251','N','07346','W'])
        'GGA'
    """
    if fields[0] == "$PASHR":
        sentence_type = "PASHR"
    elif fields[0] == "$PTNL":
        sentence_type = fields[1]
    else:
        # Ignore the $ and talker ID portions (e.g. GP)
        sentence_type = fields[0][3:]
    return sentence_type


def is_valid_nmea(nmea_sentence):
    # type: (str) -> bool
    """
    Determines if string is a valid NMEA sentence
    Args:
        nmea_sentence:

    Returns:
        true if valid

    Examples:
        >>> assert is_valid_nmea('$GNGGA,195639.00,4....2*12')
        >>> assert is_valid_nmea('$GNGGA,195637.00,07.*42')
        >>> assert is_valid_nmea('$PASHR,195636.000,1,2*29')
        >>> assert is_valid_nmea('$PTNL,EVT,,156.490,T,1.125,-2.6,2*29')
        >>> assert is_valid_nmea('$PTNL,AVR,,+157.,Yaw,-2.6292,Tilt,2.2,16*36')

    """
    match = re.match('^\$(GP|GN|GL|P).*\*[0-9A-Fa-f]{2}$', nmea_sentence)
    return not not match


def parse_nmea_sentence(nmea_sentence):
    # Check for a valid nmea sentence
    nmea_sentence = nmea_sentence.strip()
    if not is_valid_nmea(nmea_sentence):
        logger.debug("Regex didn't match, sentence not valid NMEA? Sentence was: %s"
                     % repr(nmea_sentence))
        return None
    stripped_sentence = nmea_sentence[:-3]  # Strip checksum
    fields = [field.strip(',') for field in stripped_sentence.split(',')]
    sentence_type = parse_sentence_type(fields)

    if sentence_type not in parse_maps:
        logger.debug("Sentence type %s not in parse map, ignoring."
                     % repr(sentence_type))
        return None

    parse_map = parse_maps[sentence_type]

    parsed_sentence = {}
    for entry in parse_map:
        parsed_sentence[entry[0]] = entry[1](fields[entry[2]])

    if sentence_type in ['GGA', 'RMC']:
        parsed_sentence = rectify_latlonalt(parsed_sentence)

    return {sentence_type: parsed_sentence}


def some_doctests():
    """
    Examples:
        >>> evt = '$PTNL,EVT,191023.083423,1,40104,2045,1,18*7E'
        >>> pprint(parse_nmea_sentence(evt)['EVT'])
        {'event': 1, 'event_counter': 40104, 'utc_time': 1552936223.083423}
        >>> evt = '''$PTNL,EVT,191023.083423,1,40104,2045,1,18*7E '''
        >>> pprint(parse_nmea_sentence(evt)['EVT'])
        {'event': 1, 'event_counter': 40104, 'utc_time': 1552936223.083423}
    Examples:
        >>> pashr = '$PASHR,191022.500,58.072,T,1.165,-2.568,,0.247,0.247,77.169,1,2*10'
        >>> pprint(parse_nmea_sentence(pashr)['PASHR'])
         {'gnss_status': 1,
          'heading': 58.072,
          'imu_alignment_status': 2,
          'pitch': -2.568,
          'roll': 1.165,
          'utc_time': 1552936222.5}
    Examples:
        >>> gga = '$GNGGA,193012.00,4251.87728463,N,07346.28597204,W,1,16,0.9,109.055,M,-31.849,M,,*44'
        >>> pprint(parse_nmea_sentence(gga)['GGA'])
        {'altitude': 109.055,
         'fix_type': 1,
         'hdop': 0.9,
         'latitude': 42.8646214105,
         'latitude_direction': 'N',
         'longitude': 73.77143286733333,
         'longitude_direction': 'W',
         'mean_sea_level': -31.849,
         'num_satellites': 16,
         'utc_time': 1552937412}
         >>> avr = '$PTNL,AVR,201140.00,+53.6362,Yaw,-2.5690,Tilt,+1.0876,Roll,0.000,1,2.2,16*0E'


    """
    pass
