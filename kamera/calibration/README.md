# Rig calibration

Calibrates every camera on a KAMERA rig from one calibration flight (figure eights at
several altitudes) and expresses them in the INS body frame. One COLMAP model holds all
modalities: the trigger-synchronized images of each event form a *frame* with a single rig
pose, so IR ties into the EO model through the rig without cross-modal matching.

```bash
conda activate kamera
kamera-calibrate /data/052025_Calibration                 # everything
kamera-calibrate /data/052025_Calibration --max_frames 150 --frame_stride 3   # quick look
kamera-calibrate --help
```

For a plain-language walkthrough of every stage, see [how_it_works.md](how_it_works.md).

## Stages (each resumes from `<flight>/calibration/`)

1. **frames** — `*_meta.json` grouped by trigger time into frames; camera names are
   `<channel>_<modality>` (`C_rgb`, `L_ir`, ...). IR is stretched to 8 bit; EO is symlinked.
2. **features** — SIFT per camera with an initial focal length and distortion per modality, then an INS
   position prior per image (`InsTrajectory` interpolates the meta.json samples).
3. **match** — spatial matching from the priors, across all cameras, so figure-eight
   crossovers are matched as well as neighbours in time. Thermal-to-visible pairs are dropped:
   SIFT cannot match them and their few spurious inliers mislead the mapper.
4. **pass1** — incremental mapping with independent cameras and position priors. EO and IR
   come out as separate models, both in INS ENU. Needs three-view overlap along track: at
   64 m/s and 1 frame/s that means flying above roughly 600 m AGL for these lenses; lower legs
   only register through crossovers with higher passes.
5. **pass2** — `cam_from_rig` for every camera is averaged from pass 1 (frames shared with
   the reference camera, both models being in INS ENU), the rig and frames are written to the
   database and onto the largest pass-1 model, and the images pass 1 never posed (IR) are added
   to their frames. Every image is then triangulated from the rig poses and bundle adjusted
   against the INS position priors, twice: rig poses and `sensor_from_rig` first, then with the
   intrinsics free as well.
6. **calibrate** — INS boresight (`ins_from_rig`) and lever arm as a robust average over
   frames, per-camera models, and `rig.yaml`.
7. **registration** — per channel `ir->rgb` and `uv->rgb` homographies as DIVE
   camera-registration JSON (format v2), plus flip GIFs.
8. **report** — PDF with intrinsics, rig angles, boresight residuals, overlays and the error budget.

## Outputs (`<flight>/calibration/camera_models/`)

- `<rig>_<camera>.yaml` — `standard` camera model readable by
  `kamera.colmap_processing.camera_models.load_from_file`, with the rig and calibration
  provenance in extra keys.
- `<rig>_rig.yaml` — `cam_from_rig` per camera, `ins_from_rig`, lever arm, quality statistics.
- `dive_registration/<left>_to_<right>_registration.json`, `gifs/`, `<rig>_calibration_report.pdf`.

## Exposure timing

The cameras do not expose at the same instant after the shared trigger; see
[exposure_timing.md](exposure_timing.md) for the measurements, the manuals, and what the
pipeline does about it.

## Conventions

- `camera_quaternion` (x, y, z, w) rotates camera vectors into the INS body frame
  (forward, right, down); `camera_position` is in that frame, metres.
- COLMAP's `cam_from_rig` maps rig (= reference camera) coordinates into the camera.
- The INS body-to-ENU rotation is `NED_TO_ENU * R_z(heading) R_y(pitch) R_x(roll)`, identical
  to `kamera.sensor_models.nav_state`.
