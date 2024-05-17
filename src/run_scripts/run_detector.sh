#!/bin/bash

# Not sure why this is necassary. Docker should already handle this.
source /entrypoint.sh

#PIPEFILE='/root/kamera_ws/src/kitware-ros-pkg/sprokit_adapters/pipelines/embedded_dual_stream/arctic_seal_yolo_eo_only.pipe'
#PIPEFILE='/root/kamera_ws/src/kitware-ros-pkg/sprokit_adapters/pipelines/embedded_dual_stream/arctic_seal_yolo_ir_only.pipe'
#PIPEFILE='/root/kamera_ws/src/kitware-ros-pkg/sprokit_adapters/pipelines/embedded_dual_stream/arctic_seal_yolo_ir_to_eo_frame_trigger.pipe'
PIPEFILE='/root/kamera_ws/src/kitware-ros-pkg/sprokit_adapters/pipelines/embedded_dual_stream/arctic_seal_yolo_ir_to_tiny_eo_region_tigger.pipe'

# Directory to store detection csv files.
DETECTION_CSV_DIR="/root/$(date +%Y%m%d%H%M%S)"

# Directory to store detection image list files.
IMAGE_LIST_DIR=$DETECTION_CSV_DIR

# Create directories.
mkdir -p ${DETECTION_CSV_DIR}
mkdir -p ${IMAGE_LIST_DIR}

roslaunch sprokit_adapters sprokit_detector_fusion_adapter.launch \
                    kwiver:=${WS_DEVEL} \
                    detector_node:=detector \
                    detection_pipefile:="${PIPEFILE}" \
                    embed_det_chips:=true \
                    pad_det_chip_percent:=50 \
                    det_topic:=/subsys0/detections \
                    detector_id_string:=seal \
                    synchronized_images_in1:=/subsys0/synched \
                    rgb_port_ind:=1 \
                    ir_port_ind:=2 \
                    uv_port_ind:=0 \
                    detection_csv_dir:=${DETECTION_CSV_DIR} \
                    image_list_dir:=${IMAGE_LIST_DIR}
