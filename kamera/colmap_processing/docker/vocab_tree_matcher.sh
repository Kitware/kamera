# Pass the path to the Colmap data folder.

docker pull colmap/colmap:latest

# Location of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker run -it \
    --gpus all \
    --network="host" \
    -e "DISPLAY" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -w /working \
    -v $1:/working \
    -v $DIR:/opt/colmap \
    colmap/colmap:latest \
    colmap vocab_tree_matcher \
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
        --VocabTreeMatching.num_images=100 \
        --VocabTreeMatching.num_nearest_neighbors=5 \
        --VocabTreeMatching.num_checks=256 \
        --VocabTreeMatching.num_images_after_verification=0 \
        --VocabTreeMatching.max_num_features=-1 \
        --VocabTreeMatching.vocab_tree_path /opt/colmap/vocab_tree_flickr100K_words1M.bin
