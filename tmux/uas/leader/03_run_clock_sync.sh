# Start kamerad service here too
service kamerad start
# Syncs the master clock (CLOCK_REALTIME) to the slave eth port (enp3s0)
/usr/sbin/phc2sys -q -s CLOCK_REALTIME -c enp3s0 -O 0
