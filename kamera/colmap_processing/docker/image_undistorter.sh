# Pass the path to the Colmap data folder.

docker pull colmap/colmap:latest

docker run -it \
    --gpus all \
    --network="host" \
    -e "DISPLAY" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -w /working \
    -v $1:/working \
    colmap/colmap:latest \
    colmap image_undistorter \
        --image_path images0 \
        --input_path sparse/$2 \
        --output_path . \
        --output_type COLMAP \
        --max_image_size 4000
