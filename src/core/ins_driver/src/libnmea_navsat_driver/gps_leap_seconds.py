#!/usr/bin/env python
# This file is automatically edited at line 5
#
# VVVVVV this line is modified automatically
GPS_LEAP_SECONDS = 18
# /\/\/\ this line is modified automatically
#
TAI_MINUS_GPS = 19

def get_gps_offset_astropy():
    """
    Procure GPS-UTC leap seconds from astropy
    Returns:

    """
    import time
    import astropy.time
    from datetime import datetime, timedelta
    import dateutil.parser

    seconds = int(time.time())
    t = astropy.time.Time(seconds, format='unix')
    gps_epoch = datetime(1980,1,6)
    gps_time = timedelta(seconds=t.gps) + gps_epoch
    gps_time = dateutil.parser.parse(gps_time.isoformat())
    iso_time = dateutil.parser.parse(t.iso)
    leaps = gps_time - iso_time
    return leaps.seconds


def get_gps_offset_iers():
    """
    Procure GPS-UTC leap seconds from IERS directly.
    Pull down the bulletinc.dat, parse it and try to extract the offset.

    There should be a line like:

         from 2017 January 1, 0h UTC, until further notice : UTC-TAI = -37 s


    Returns:

    """
    import requests
    import re
    rq = requests.get('https://hpiers.obspm.fr/iers/bul/bulc/bulletinc.dat')
    text = rq.text or ''
    print(text)
    res = re.findall(r'(?<=UTC-TAI = -)(\d+)(?= s)', text)
    if res:
        return int(res[0]) - TAI_MINUS_GPS
    return None


def get_gps_offset():
    leap_seconds = get_gps_offset_iers()
    if leap_seconds is None:
        leap_seconds = get_gps_offset_astropy()

    print('GPS-UTC leap seconds: {}'.format(leap_seconds))
    return leap_seconds


if __name__ == '__main__':
    """
    Determine GPS offset, then write it back to this file.
    This needs to only need to be run once in a while. 
    
    GPS is ahead of UTC by 18 seconds (as of Mar 2019)
    TAI is always ahead of GPS by 19 seconds
    UTC < GPS < TAI
    """
    leap_seconds = get_gps_offset()

    with open(__file__, 'r') as fp:
        raw_lines = fp.readlines()

    # magic line number
    raw_lines[4] = 'GPS_LEAP_SECONDS = {}\n'.format(leap_seconds)
    with open(__file__, 'w') as fp:
        fp.writelines(raw_lines)



