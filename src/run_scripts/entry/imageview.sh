#!/bin/bash

# Nexus node startup script

echo "/\ /\ /\   IMAGE VIEW    /\ /\ /\ "
source /entry/project.sh
source /aliases.sh
NODE_HOSTNAME=${NODE_HOSTNAME:-undefined}

if [[ $(redis-cli --raw -h $REDIS_HOST get /detector/read_from_nas ) -eq 1 ]]; then
  echo "Reading from NAS enabled, not sending any image bytes in sync msg."
  SEND_IMAGE_DATA="false"
else
  echo "Reading from NAS disabled, sending image bytes in sync msg."
  SEND_IMAGE_DATA="true"
fi

if [[ $(redis-cli --raw -h $REDIS_HOST get /imageview/compress_imagery ) -eq 1 ]]; then
  echo "Compression is enabled, will compress/decompress imagery."
  COMPRESS_IMAGERY="true"
else
  echo "Compression is disabled, will not compress/decompress imagery."
  COMPRESS_IMAGERY="false"
fi

roslaunch --wait view_server image_view_server.launch \
    norespawn:="${NORESPAWN}" \
	system_name:=${NODE_HOSTNAME} \
	send_image_data:=${SEND_IMAGE_DATA} \
	compress_imagery:=${COMPRESS_IMAGERY}
