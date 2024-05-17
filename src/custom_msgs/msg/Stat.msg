# General purpose stat/health message

# Header for this message
std_msgs/Header header

# This is the header of the event being traced, which may occur at different times
std_msgs/Header trace_header

# This header corresponds to the end of the span
std_msgs/Header end_header

# Unique trace ID
string id

# Name of node generating this message
string node

# Link to another trace
string link

# Name of the event being traced
string trace_topic

# General purpose note field
string note

# metadata expansion field
string meta_json
