Header header

# Identifies the camera image stream that the detector operated on.
# Typically, this is the ROS topic of the source imagery.
string camera_of_origin

# Category of the detected object or occurrence.
string category

# Specification of the detection bounding box.
int32 left
int32 right
int32 top
int32 bottom

# Confidence in the detected object or occurrence of type category.
float64 confidence

# Indicates the detector that was used.
string detector

# Unique identifier string.
string uid

# Image chip drawn from the source imagery passed to the detector covering the
# bounding box or an enlarged region centered on it.
sensor_msgs/Image image_chip

# Defines the source-image coordinates for the upper-left pixel of the image
# chip. This is useful if the image chip represents a larger area than the
# bounding box or if the bounding box has been transformed to another image
# coordinate system.
int32 image_chip_row_offset
int32 image_chip_col_offset

# Descriptor vector for this detection (vector of floats)
float64[] descriptor
