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
    colmap stereo_fusion \
        --workspace_path . \
        --workspace_format COLMAP \
        --output_path fused.ply \
        --input_type photometric \
        --StereoFusion.max_image_size -1 \
        --StereoFusion.min_num_pixels 10 \
        --StereoFusion.max_num_pixels 10000 \
        --StereoFusion.max_traversal_depth 100 \
        --StereoFusion.max_reproj_error 2 \
        --StereoFusion.max_depth_error 0.0099999997764825821 \
        --StereoFusion.max_normal_error 10 \
        --StereoFusion.check_num_images 3 \
        --StereoFusion.cache_size 100 \

# photometric (default=geometric) {photometric, geometric}
# max_image_size -1 \ # (default=-1)
# min_num_pixels 10 \ # (default=5)
# max_num_pixels 10000 \ # (default=10000)
# max_traversal_depth 100 \ # (default=100)
# max_reproj_error 2 \ # (default=2)
# max_depth_error 0.0099999997764825821 \ # (default=0.0099999997764825821)
# max_normal_error 10 \ # (default=10)
# check_num_images 50 \ # (default=50)
# cache_size 100 \ # (default=32)
