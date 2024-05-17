# Image transported by a saved path on disk
# Header for this image
std_msgs/Header header

# Header associated with the most recent event
std_msgs/Header evt_header

## from sensor_msgs/Image
uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns
string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes

# File path to image's on-disk location.
string filename

# metadata expansion field
string meta_json
