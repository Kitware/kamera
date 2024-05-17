## Mapper Parameters

**min_num_matches** (default=15) : Only load image pairs with a minimum number of matches.

**ignore_watermarks** (default=0) : Whether to ignore watermark image pairs.

**multiple_models** (default=1) : Whether to allow multiple models to be generated.

**max_num_models** (default=50) : Maximum numbers of models to be generate.

**max_model_overlap** (default=20) : The maximum number of overlapping images between sub-models. If the current sub-models shares more than this number of images with another model, then the reconstruction is stopped.

**min_model_size** (default=10) : The minimum number of registered images of a sub-model, otherwise the sub-model is discarded.

**init_image_id1** (default=-1) : The image identifiers used to initialize the reconstruction. Note that only one or both image identifiers can be specified. In the former case, the second image is automatically determined.

**init_image_id2** (default=-1) :  See **init_image_id2**.

**init_num_trials** (default=200) : The number of trials to initialize the reconstruction.

**extract_colors** (default=1) : Whether to extract colors for reconstructed points.

**num_threads** (default=-1) : The number of threads to use during reconstruction.

**min_focal_length_ratio** (default=0.10000000000000001) : Thresholds for filtering images with degenerate intrinsics.

**max_focal_length_ratio** (default=10) : Thresholds for filtering images with degenerate intrinsics.

**max_extra_param** (default=1)

**ba_refine_focal_length** (default=1) : Refine focal lengths during bundle adjustment.

**ba_refine_principal_point** (default=0) : Refine principal point during bundle adjustment.

**ba_refine_extra_params** (default=1) : Refine extra parameters during bundle adjustment.

**ba_min_num_residuals_for_multi_threading** (default=50000) : The minimum number of residuals per bundle adjustment problem to enable multi-threading solving of the problems.

**ba_local_num_images** (default=6) : The number of images to optimize in local bundle adjustment.

**ba_local_max_num_iterations** (default=25) : The maximum number of local bundle adjustment iterations.

**ba_global_use_pba** (default=0) : Whether to use PBA in global bundle adjustment.

**ba_global_pba_gpu_index** (default=-1) : The GPU index for PBA bundle adjustment.

**ba_global_images_ratio** (default=1.1000000000000001) : The growth rates after which to perform global bundle adjustment.

**ba_global_points_ratio** (default=1.1000000000000001) : The growth rates after which to perform global bundle adjustment.

**ba_global_images_freq** (default=500) : The growth rates after which to perform global bundle adjustment.

**ba_global_points_freq** (default=250000) : The growth rates after which to perform global bundle adjustment.

**ba_global_max_num_iterations** (default=50) : The maximum number of global bundle adjustment iterations.

**ba_global_max_refinements** (default=5) : The thresholds for iterative bundle adjustment refinements.

**ba_global_max_refinement_change** (default=0.00050000000000000001) : The thresholds for iterative bundle adjustment refinements.

**ba_local_max_refinements** (default=2) : The thresholds for iterative bundle adjustment refinements.

**ba_local_max_refinement_change** (default=0.001) : The thresholds for iterative bundle adjustment refinements.

**snapshot_path** : Path to a folder with reconstruction snapshots during incremental reconstruction. Snapshots will be saved according to the specified frequency of registered images.

**snapshot_images_freq** (default=0) : See **snapshot_path**.

**fix_existing_images** (default=0) : If reconstruction is provided as input, fix the existing image poses.

**init_min_num_inliers** (default=100) : Minimum number of inliers for initial image pair.

**init_max_error** (default=4) : Maximum error in pixels for two-view geometry estimation for initial image pair.

**init_max_forward_motion** (default=0.94999999999999996) : Maximum forward motion for initial image pair.

**init_min_tri_angle** (default=16) : Minimum triangulation angle (degrees) for initial image pair.

**init_max_reg_trials** (default=2) : Maximum number of trials to use an image for initialization.

**abs_pose_max_error** (default=12) : Maximum reprojection error in absolute pose estimation.

**abs_pose_min_num_inliers** (default=30) : Minimum number of inliers in absolute pose estimation.

**abs_pose_min_inlier_ratio** (default=0.25) : Minimum inlier ratio in absolute pose estimation.

**filter_max_reproj_error** (default=4) : Maximum reprojection error in pixels for observations.

**filter_min_tri_angle** (default=1.5) : Minimum triangulation angle in degrees for stable 3D points.

**max_reg_trials** (default=3) : Maximum number of trials to register an image.

Triangulation includes creation of new points, continuation of existing points, and merging of separate points if given image bridges tracks. Note that the given image must be registered and its pose must be set in the associated reconstruction.

**tri_max_transitivity** (default=1) : Maximum transitivity to search for correspondences.

**tri_create_max_angle_error** (default=2) : Maximum angular error (degrees) to create new triangulations.

**tri_continue_max_angle_error** (default=2) : Maximum angular error (degrees) to continue existing triangulations.

**tri_merge_max_reproj_error** (default=4) : Maximum reprojection error in pixels to merge triangulations.

**tri_complete_max_reproj_error** (default=4) : Maximum reprojection error to complete an existing triangulation.

**tri_complete_max_transitivity** (default=5) : Maximum transitivity for track completion.

**tri_re_max_angle_error** (default=5) : Maximum angular error (degrees) to re-triangulate under-reconstructed image pairs.

**tri_re_min_ratio** (default=0.20000000000000001) : Minimum ratio of common triangulations between an image pair over the number of correspondences between that image pair to be considered as under-reconstructed.

**tri_re_max_trials** (default=1) : Maximum number of trials to re-triangulate an image pair.

**tri_min_angle** (default=1.5) : Minimum pairwise triangulation angle for a stable triangulation.

**tri_ignore_two_view_tracks** (default=1) : Whether to ignore two-view tracks.