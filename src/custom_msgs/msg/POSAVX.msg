# State output from POS AVX Intertial navigation system. Message components
# taken from page 42 of the POS AVX 210 User Guide.

# Header for this tuple of images
std_msgs/Header header

# Time since 00:00:00 Thursday, 1 January 1970
float64 time

# INS quality indicator
# 0 - GPS Only
# 1 - Coarse leveling
# 2 - Degraded
int8 imu_alignment_status

# GNSS Quality Indicator
# 0 - Fix not available
# 1 - GNSS SPS Mode
# 2 - Differential GPS,SPS
# 3 - GNSS PPS Mode
# 4 - Fixed RTK Mode
# 5 - Float RTK.
# 6 - DR Mode
int8 gnss_status

# Latitude (-90,90] (degrees)
float64 latitude

# Longitude (-180,180] (degrees)
float64 longitude

# Altitude (meters)
float64 altitude

# North velocity (meters/sec)
float32 north_velocity

# East velocity (meters/sec)
float32 east_velocity

# Down velocity (meters/sec)
float32 down_velocity

# Total speed (meters/sec)
float32 total_speed

# Roll (-180,180] (degrees)
float64 roll

# Pitch (-180,180] (degrees)
float64 pitch

# Heading [0,360) (degrees)
float64 heading

# Track Angle [0,360) (degrees/sec)
float32 track_angle

# Angular Rate X (degrees/sec)
float32 angular_rate_x

# Angular Rate Y (degrees/sec)
float32 angular_rate_y

# Angular Rate Z (degrees/sec)
float32 angular_rate_z

# Acceleration X (meters/sec^2)
float32 acceleration_x

# Acceleration Y (meters/sec^2)
float32 acceleration_y

# Acceleration Z (meters/sec^2)
float32 acceleration_z