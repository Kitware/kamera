#!/bin/bash

# ROS2: uses DDS discovery; ensure ROS_DOMAIN_ID matches the system

ros2 service call /nuvo2/uv/uv_view_service/get_image_view custom_msgs/srv/RequestImageView "homography: [1,0,0,0,1,0,0,0,1]
output_height: 2
output_width: 2
interpolation: 0
antialias: false
last_header:
  seq: 0
  stamp: 0
  frame_id: ''
release: 0"


