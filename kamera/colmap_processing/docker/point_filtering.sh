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
    colmap point_filtering \
	--input_path $2 \
        --output_path sparse \
        --min_track_len=5 \
        --max_reproj_error=4 \
        --min_tri_angle=3 \
