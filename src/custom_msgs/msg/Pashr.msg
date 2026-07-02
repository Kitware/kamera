# State output from POS AVX Intertial navigation system. Message components
# taken from page 37 of the POS AVX 210 User Guide.

# Header for this tuple of images
std_msgs/Header header

# Time since 00:00:00 Thursday, 1 January 1970
float64 time

# True heading in degrees
float32 heading

# Roll in degrees
float32 roll

# Pitch in degrees
float32 pitch

# GNSS quality indicator
# 0 - Fix not available
# 1 - GNSS SPS Mode
# 2 - Differential GPS,SPS
# 3 - GNSS PPS Mode
# 4 - Fixed RTK Mode
# 5 - Float RTK.
# 6 - DR Mode
int8 gnss_status

# INS quality indicator
# 0 - GPS Only
# 1 - Coarse leveling
# 2 - Degraded
# 3 - Aligned
# 4 - Full Navigation Mode
int8 imu_alignment_status
