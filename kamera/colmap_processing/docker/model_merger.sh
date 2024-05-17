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
    colmap model_merger \
        --input_path1 sparse/$2 \
        --input_path2 sparse/$3 \
        --output_path sparse/merged
