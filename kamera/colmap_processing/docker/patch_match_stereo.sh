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
    colmap patch_match_stereo \
        --workspace_path . \
        --workspace_format COLMAP \
        --pmvs_option_name option-all \
        --PatchMatchStereo.max_image_size -1 \
        --PatchMatchStereo.gpu_index -1 \
        --PatchMatchStereo.depth_min -1 \
        --PatchMatchStereo.depth_max -1 \
        --PatchMatchStereo.window_radius 5 \
        --PatchMatchStereo.window_step 1 \
        --PatchMatchStereo.sigma_spatial -1 \
        --PatchMatchStereo.sigma_color 0.20000000298023224 \
        --PatchMatchStereo.num_samples 15 \
        --PatchMatchStereo.ncc_sigma 0.60000002384185791 \
        --PatchMatchStereo.min_triangulation_angle 1 \
        --PatchMatchStereo.incident_angle_sigma 0.89999997615814209 \
        --PatchMatchStereo.num_iterations 5 \
        --PatchMatchStereo.geom_consistency 1 \
        --PatchMatchStereo.geom_consistency_regularizer 0.30000001192092896 \
        --PatchMatchStereo.geom_consistency_max_cost 3 \
        --PatchMatchStereo.filter 1 \
        --PatchMatchStereo.filter_min_ncc 0.10000000149011612 \
        --PatchMatchStereo.filter_min_triangulation_angle 3 \
        --PatchMatchStereo.filter_min_num_consistent 2 \
        --PatchMatchStereo.filter_geom_consistency_max_cost 1 \
        --PatchMatchStereo.cache_size 128 \
        --PatchMatchStereo.write_consistency_graph 0

# Options
#  --random_seed arg (=0)
#  --project_path arg
#  --workspace_path arg                  Path to the folder containing the 
#                                        undistorted images
# --workspace_format arg (=COLMAP)      {COLMAP, PMVS}
#  --pmvs_option_name arg (=option-all)
#  --PatchMatchStereo.max_image_size arg (=-1)
#  --PatchMatchStereo.gpu_index arg (=-1)
#  --PatchMatchStereo.depth_min arg (=-1)
#  --PatchMatchStereo.depth_max arg (=-1)
#  --PatchMatchStereo.window_radius arg (=5)
#  --PatchMatchStereo.window_step arg (=1)
#  --PatchMatchStereo.sigma_spatial arg (=-1)
#  --PatchMatchStereo.sigma_color arg (=0.20000000298023224)
#  --PatchMatchStereo.num_samples arg (=15)
#  --PatchMatchStereo.ncc_sigma arg (=0.60000002384185791)
#  --PatchMatchStereo.min_triangulation_angle arg (=1)
#  --PatchMatchStereo.incident_angle_sigma arg (=0.89999997615814209)
#  --PatchMatchStereo.num_iterations arg (=5)
#  --PatchMatchStereo.geom_consistency arg (=1)
#  --PatchMatchStereo.geom_consistency_regularizer arg (=0.30000001192092896)
#  --PatchMatchStereo.geom_consistency_max_cost arg (=3)
#  --PatchMatchStereo.filter arg (=1)
#  --PatchMatchStereo.filter_min_ncc arg (=0.10000000149011612)
#  --PatchMatchStereo.filter_min_triangulation_angle arg (=3)
#  --PatchMatchStereo.filter_min_num_consistent arg (=2)
#  --PatchMatchStereo.filter_geom_consistency_max_cost arg (=1)
#  --PatchMatchStereo.cache_size arg (=32)
#  --PatchMatchStereo.write_consistency_graph arg (=0)

