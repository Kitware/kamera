#!/bin/bash
set -e

CATKIN_DEVEL="$(catkin locate -d)"

# Make sure driver executable has setcap run on it
NODE_EXE="${CATKIN_DEVEL}/.private/kw_genicam_driver/lib/kw_genicam_driver/simple_driver_node"
if [ ! -f "${NODE_EXE}" ]
then
  echo "ERROR: Expected location of driver node was not found: ${NODE_EXE}"
  exit 1
fi
echo "INFO: Running setcap on driver node exe"
sudo setcap cap_net_raw+ep "${NODE_EXE}"

# Make sure network tweaked to support packets sufficiently.
NET_IFACE=enp4s0
sudo /usr/dalsa/GigeV/bin/gev_nettweak ${NET_IFACE}

roslaunch kw_genicam_driver genicam_simple.launch \
  debug:=true \
  namespace:=/test \
  firmware_mode:=bayer \
  camera_serial:=S1125704 \
  xmlFeatures_autoBrightness:=true \
  imageTransfer_numImageBuffers:=8 \
  frame_id:=/test/camera/cueing/0 \
  output_topic_raw:=camera/cueing/0/bayer_image_raw \
  output_topic_debayer:=camera/cueing/0/image_raw \
  output_frame_rate:=3
