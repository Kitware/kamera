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
    colmap bundle_adjuster \
        --input_path $2 \
        --output_path $3 \
        --BundleAdjustment.max_num_iterations=100 \
        --BundleAdjustment.max_linear_solver_iterations=200 \
        --BundleAdjustment.function_tolerance=0 \
        --BundleAdjustment.gradient_tolerance=0 \
        --BundleAdjustment.parameter_tolerance=0 \
        --BundleAdjustment.refine_focal_length=1 \
        --BundleAdjustment.refine_principal_point=0 \
        --BundleAdjustment.refine_extra_params=1 \
        --BundleAdjustment.refine_extrinsics=1

