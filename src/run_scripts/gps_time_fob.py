import serial
from datetime import datetime
import os

def set_date(now):
    # type: (datetime) -> int
    cm = 'sudo date -s "{}"'.format(now.isoformat())
    return os.system(cm)

ser = serial.Serial('/dev/ttyUSB0', baudrate=4800)

while True:
    line = ser.readline()
    if line[:6] == b'$GPRMC':
        tn = line[7:17].decode()
        # 005611.154
        nowish = datetime.now()
        date = nowish.date()
        hh, mm, ss, ms = map(int, [tn[:2], tn[2:4], tn[4:6], tn[7:]])
        now = datetime(year=date.year, month=date.month, day=date.day, hour=hh, minute=mm, second=ss, microsecond=1000 * tn)
        set_date(now)
        print(tn, datetime.now().isoformat())