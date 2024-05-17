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
    colmap model_aligner \
	--input_path $2 \
        --ref_images_path $3 \
        --output_path $4 \
        --min_common_images=3 \
        --robust_alignment=1 \
        --robust_alignment_max_error=5
