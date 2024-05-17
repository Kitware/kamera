# Synchronized Images from multi-camera array
# Header for this tuple of images
std_msgs/Header header

# File path to each image's on-disk location.
string file_path_rgb
string file_path_ir
string file_path_uv

# Actual data
sensor_msgs/Image image_rgb
sensor_msgs/Image image_ir
sensor_msgs/Image image_uv
