# Image transported by a saved path on disk
# Header for this image
std_msgs/Header header

# Header associated with the most recent event
std_msgs/Header evt_header


# File path to image's on-disk location.
string file_path_rgb
string file_path_ir
string file_path_uv

# Actual data
PathImage image_rgb
PathImage image_ir
PathImage image_uv

# metadata expansion field
string meta_json
