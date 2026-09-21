# How the rig calibration works, step by step

This follows one run of `kamera-calibrate <flight_dir>` from the raw KAMERA flight
folder to the camera models, using plain language. File names in `kamera/calibration/`
are given so you can read along in the code.

## What goes in

A KAMERA flight folder, for example `052025_Calibration/`, with one folder per view
(`center_view`, `left_view`, `right_view`) and, for every trigger, four files with a
common stem:

```
taiga_calibration_2025_fl118_C_20250503_203245.017993_meta.json
taiga_calibration_2025_fl118_C_20250503_203245.017993_rgb.jpg
taiga_calibration_2025_fl118_C_20250503_203245.017993_uv.jpg
taiga_calibration_2025_fl118_C_20250503_203245.017993_ir.tif
```

The meta json holds the trigger time (`evt.time`), the INS reading nearest to it
(`ins`: latitude, longitude, altitude, heading, pitch, roll), and the camera metadata.

## What comes out

Everything lands in `<flight_dir>/calibration/models/`:

- one yaml per camera (`<rig>_<camera>.yaml`), readable by the existing
  `camera_models.load_from_file`
- `<rig>_rig.yaml` with the rig geometry and the INS boresight
- `dive_registration/*.json`, one homography file per camera pair per channel
- `gifs/`, flip animations of one camera warped onto another
- `<rig>_calibration_report.pdf`

Everything else under `<flight_dir>/calibration/` is intermediate and can be deleted;
the tool rebuilds whatever is missing and skips whatever exists.

## The idea in one paragraph

Structure from motion (COLMAP) can work out where every picture was taken from and
what it was looking at, just from the pictures overlapping each other. It does this
in its own arbitrary coordinate system, so we hand it the INS positions to pin the
model to the real world. Because all nine cameras fire on the same trigger, the nine
pictures of one trigger share one rig position and orientation; COLMAP 4 can enforce
that ("rigs" and "frames"). Once the model is solved with that constraint, the fixed
rotation and offset of each camera relative to the reference camera drops out, and
comparing the rig orientation with the INS orientation over hundreds of frames gives
the boresight. The only thing COLMAP cannot do for us is match thermal pictures to
visible ones, and with the rig constraint it does not need to.

## Step 1: find the frames (`flight.py`, `discover_flight`)

Read every `*_meta.json`. Group them by trigger time, rounded to a millisecond, so the
L, C and R files of one trigger become one **frame**. Name each image
`<channel>_<modality>`, for example `C_rgb` or `L_ir`. Collect the INS readings from
every json into one time-ordered trajectory. Frames missing any of the nine images are
dropped, so every frame used has all nine.

The INS trajectory (`InsTrajectory`) converts latitude/longitude/altitude to metres in a
local east-north-up frame centred on the flight, and heading/pitch/roll into a
rotation, using the same convention as the rest of KAMERA (`sensor_models.nav_state`).
Asked for the pose at any time, it interpolates between the two nearest samples. With
the meta json this is coarse, one sample per second; a denser INS log would drop in
here unchanged.

## Step 2: lay the images out for COLMAP (`flight.py`, `build_image_tree`)

COLMAP wants one folder per camera and, for rigs, the *same file name* across folders
for pictures of the same frame. So the tree is
`calibration/images/<camera>/<trigger time>.jpg`. RGB files are symlinks. UV and IR are
rewritten: the UV frames are very dark (median value 9 of 255) and the IR frames are
16-bit, so both are stretched between their 0.1 and 99.9 percentiles and given a mild
local contrast boost (CLAHE). Without this, SIFT found about 30 features per UV frame
instead of thousands. This runs in parallel because there are thousands of files.

## Step 3: features and INS priors (`sfm.py`, `extract_features`, `write_pose_priors`)

SIFT features are extracted per camera folder on the GPU, with the image downsampled
to 3200 px on the long side (a 12768 px RGB frame gives about 12,000 features). Each
camera folder gets one COLMAP camera with the OPENCV model (focal length, principal
point, k1, k2, p1, p2), seeded with a rough focal length per modality from the config
so the first frames register cleanly.

Then every image gets a **position prior**: the INS position at its trigger time, with a
2 m standard deviation. COLMAP uses these priors in two ways later: to decide which
images to try to match, and to keep the model in real-world metres and orientation.

## Step 4: matching (`sfm.py`, `match_features`, `prune_cross_spectral`)

Rather than matching every image against every other (8,800 images would be 39
million pairs), each image is matched against its 90 nearest neighbours by INS
position within 250 m. That covers the frames just before and after, and also the
crossovers of the figure eights, which are what make the geometry strong.

The neighbours include every camera, so the same-frame RGB and UV pictures get matched
too, which is useful: they share features and tie the UV into the RGB model directly.
Thermal-to-visible pairs also get "matched", but those matches are garbage (about 25
random inliers), and left in they pull IR images to wrong places. They are deleted from
the database right after matching.

## Step 5: pass 1, mapping with independent cameras (`sfm.py`, `run_mapping`)

COLMAP's incremental mapper builds the 3D model: it picks a good starting pair,
triangulates points, adds the next image by matching its features to points already in
3D, and periodically re-optimises everything (bundle adjustment). The position priors
are switched on, so the model comes out in INS coordinates rather than an arbitrary
frame, with the right scale.

At this stage every camera is still independent. The result is normally two models:
one with all the EO cameras (RGB and UV of L, C and R, linked by same-frame matches
and by the overlap strips between channels) and one with the IR cameras, which only
match each other. Both are in INS coordinates thanks to the priors, so they can be
compared.

Two practical notes. The mapper needs a point to be seen from three pictures to add
a third picture; at 300 to 400 m above ground with one frame per second the along-track
overlap is under 50%, so those legs only register through crossovers with higher
passes. And the global bundle adjustment is set to run every 30% of growth instead of
10%, which halved the run time on the full flight (about 2.5 hours for 8,800 images).

## Step 6: work out the rig from pass 1 (`sfm.py`, `derive_rig`, `robust_mean`)

For every camera and every frame where both that camera and the reference camera
(`C_rgb`) were placed, compute the camera's pose relative to the reference. On a rigid
rig that relative pose is the same every frame, so the hundreds of estimates should
agree. Take the densest cluster of them (the estimate with the most neighbours within
one degree, then the mean of that cluster) rather than a plain median, because a badly
registered part of a model can put half the estimates 20 degrees off, and the cluster
ignores those. For the IR cameras this comparison goes across the two models; it works
because both models are in INS coordinates, and it is accurate to about 0.3 degrees,
which is plenty for a starting point.

The tool prints, per camera, how many of the shared frames fell in the cluster and how
tightly they agree, and warns when the scatter is over half a degree or under half the
frames made the cluster. Pass 2 only refines a seed it can triangulate from: the
triangulator drops tracks over 4 px of reprojection error, about 0.13 degrees for IR,
and a seed a degree off loses the crossover tracks that pin the offset, so the offset
stalls near the seed rather than blowing up. A warning here means the IR result needs
checking, not that the run failed.

## Step 7: pass 2, the rig bundle adjustment (`sfm.py`, `rigged_model`, `refine_rig`)

Now tell COLMAP about the rig:

1. Write the rig definition into the database: `C_rgb` is the reference sensor and
   every other camera has the starting `cam_from_rig` from step 6. COLMAP groups the
   images into frames by their shared file name.
2. Put the rig onto the largest pass 1 model. Its frames now hold one pose each, taken
   from the `C_rgb` image. Add the images pass 1 never placed, mostly IR: they inherit
   their pose from the frame pose and the rig offsets.
3. Triangulate every image afresh from those poses. IR features now become 3D points
   too, because the IR images have poses even though nothing matched them to EO.
4. Bundle adjust with the INS position priors, refining the frame poses and the
   `cam_from_rig` of every camera, with intrinsics held fixed.
5. Triangulate again from the refined poses, and bundle adjust once more with the
   intrinsics free (focal length and distortion).

The result is one model with all nine cameras. On the full May 2025 flight: 740
frames, 6,660 images, 0.71 px mean reprojection error, matching a 250-frame subset
to about 0.05 degrees on the rig angles.

Two things that did not work, so nobody repeats them: continuing COLMAP's incremental
mapper from the rigged model (it throws the model away as "insufficient size"), and
COLMAP's plain bundle adjuster on the rigged model (without a fixed gauge it diverges).
The triangulate-then-adjust route above is stable.

## Step 8: read off the calibration (`rig.py`, `calibrate_rig`)

From the final model:

- **Intrinsics** per camera: focal lengths, principal point, distortion, straight from
  COLMAP's OPENCV camera. The per-camera reprojection error is computed over every
  observation of that camera, and the observation count is reported next to it. That
  count is the check on a stalled IR seed: the reprojection error only covers tracks
  that survived triangulation, so it stays small even when most IR tracks were dropped,
  while the observation count collapses.
- **Rig geometry**: `cam_from_rig` for each camera, which maps rig coordinates (the
  `C_rgb` camera frame) into that camera. From it, the rotation relative to the
  reference (the L and R channels come out at about 30 degrees, UV within half a degree
  of its RGB, IR within about a degree) and the camera centre in the rig.
- **INS boresight**: for every frame, take the rig's orientation in the world from the
  model and the INS orientation at the same time, and compute the rotation between
  them. That should be one fixed rotation; the densest-cluster mean of it over all
  frames is `ins_from_rig`, and the spread of the individual frames about it (about
  0.15 degrees median) is the honest per-frame uncertainty. The same comparison of
  positions gives the lever arm from the INS to the rig, which is noise dominated.
- **Camera models in the INS frame**: each camera's rotation into the INS body is
  `ins_from_rig` composed with the camera's rotation into the rig, and its position is
  the lever arm plus its rig centre rotated into the body frame. These two numbers are
  the `camera_quaternion` and `camera_position` in the yaml, exactly as the existing
  KAMERA georegistration code expects.

`write_camera_yaml` and `write_rig_yaml` put all of this on disk, with the original
keys first so old readers still work and the provenance after.

## Step 9: homographies for DIVE (`registration.py`, `cli.py`)

For each channel and each pair `ir->uv`, `ir->rgb`, `uv->rgb`: take a grid of pixels in
the first camera, cast them out to a nominal ground range through the calibrated
model, project them into the second camera, and fit one 3x3 homography to the result.
The fit residual says how much a single matrix loses to lens distortion. The range
matters because the cameras do not expose at exactly the same instant, which shows up
as an along-track offset of about a metre in the rig (see `exposure_timing.md`); it
defaults to the calibration flight's median scene range and should be set to the
survey altitude. The files use DIVE's registration format version 2, one matrix-only
pair each.

The GIFs warp the first camera's picture onto the second with that homography for
five frames spread across the flight and flip between the two, so a misregistration is
visible at a glance.

## Step 10: the report (`report.py`)

A PDF with the camera table, the rig angles and a sketch of the optical axes, the
boresight numbers with per-frame residual plots, one page per homography pair with the
warped overlay, and a page on the error budget: INS sample staleness, model drift, the
weak observability of lever arms, and exposure timing.

## Running it again

Every stage checks for its outputs and skips itself when they exist, so rerunning the
same command after a crash or a code change in a late stage takes minutes, not hours.
Delete `calibration/pass1` to redo the mapping, `calibration/rig` to redo the rig
adjustment, or pass `--force` to redo everything. `--max_frames` and `--frame_start`
select a subset for quick experiments; pick frames from the high-altitude part of the
flight for those.
