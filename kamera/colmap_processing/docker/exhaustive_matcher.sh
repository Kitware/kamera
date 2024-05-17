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
    colmap exhaustive_matcher \
        --database_path database.db \
        --SiftMatching.num_threads=-1 \
        --SiftMatching.use_gpu=1 \
        --SiftMatching.gpu_index=-1 \
        --SiftMatching.max_ratio=0.80000000000000004 \
        --SiftMatching.max_distance=0.69999999999999996 \
        --SiftMatching.cross_check=1 \
        --SiftMatching.max_error=4 \
        --SiftMatching.max_num_matches=32768 \
        --SiftMatching.confidence=0.999 \
        --SiftMatching.max_num_trials=10000 \
        --SiftMatching.min_inlier_ratio=0.25 \
        --SiftMatching.min_num_inliers=15 \
        --SiftMatching.multiple_models=0 \
        --SiftMatching.guided_matching=0 \
        --ExhaustiveMatching.block_size=50
