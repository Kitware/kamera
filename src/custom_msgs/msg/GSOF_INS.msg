# Event output from POS AVX Intertial navigation system. Message components
# taken from page 42 of the POS AVX 210 User Guide.

# Standard header
std_msgs/Header header

# UTC Time - Time since 00:00:00 Thursday, 1 January 1970, in seconds - DEPRECATED
float64 time

# system time of the system receiving the event
time sys_time

# time associated with the gps event message
time gps_time

# INS quality indicator.
# {0: 'gps only', 1: 'coarse leveling', 2: 'degraded', 3: 'aligned', 4: 'full nav'}
uint8 align_status

# Satellite fix status and quality indicator
# {0: 'Fix not available', 1: 'GNSS SPS Mode', 2: 'Differential GPS SPS',
#  3: 'GNSS PPS Mode', 4: 'Fixed RTK Mode', 5: 'Float RTK', 6: 'DR Mode' }
uint8 gnss_status

# Latitude [degrees]. Positive is north of equator; negative is south.
float64 latitude

# Longitude [degrees]. Positive is east of prime meridian; negative is west.
float64 longitude

# Altitude [m]. Positive is above the WGS 84 ellipsoid
# (quiet NaN if no altitude is available).
float64 altitude

# Velocity components are in meters/second
float64 north_velocity
float64 east_velocity
float64 down_velocity

# Meters/second
float64 total_speed

# Degrees
float64 roll
float64 pitch
float64 heading
float64 track_angle

# X, Y, Z are longitudinal (roll), transverse (pitch), and vertical (yaw)

# degrees/second
float64 angular_rate_x
float64 angular_rate_y
float64 angular_rate_z

# meters/second/second
float64 acceleration_x
float64 acceleration_y
float64 acceleration_z
