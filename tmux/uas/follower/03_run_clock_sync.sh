# Start kamerad service here too
service kamerad start
# Syncs the Master clock (CLOCK_REALTIME) to the RGB camera (enp1s0)
/usr/sbin/phc2sys -q -s CLOCK_REALTIME -c enp1s0 -O 0
