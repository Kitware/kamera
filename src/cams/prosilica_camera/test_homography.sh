#!/bin/bash

export ROS_MASTER_URI=http://nuvo0:11311/

rosservice call /nuvo2/uv/uv_view_service/get_image_view "homography: [1,0,0,0,1,0,0,0,1]
output_height: 2
output_width: 2
interpolation: 0
antialias: false
last_header:
  seq: 0
  stamp: 0
  frame_id: ''
release: 0"


