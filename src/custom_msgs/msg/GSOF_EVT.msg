# Event output from POS AVX Intertial navigation system. Message components
# taken from page 44 of the POS AVX 210 User Guide.

# Standard header
std_msgs/Header header

# UTC Time - Time since 00:00:00 Thursday, 1 January 1970, in seconds - DEPRECATED
float64 time

# system time of the system receiving the event
time sys_time

# time associated with the gps event message
time gps_time

# Event port associated with captured pulse
uint8 event_port

# Event counter
uint64 event_num
