# Rig Calibration

Calibrates every camera on a KAMERA rig from one calibration flight (figure eights at several altitudes) and expresses them in the INS body frame. One COLMAP model holds all modalities: the trigger-synchronized images of each event form a *frame* with a single rig pose, so IR ties into the EO model through the rig without any cross-modal matching.

```bash
conda activate kamera
kamera-calibrate /data/052025_Calibration                 # everything
kamera-calibrate /data/052025_Calibration --max_frames 150 --frame_stride 3   # quick look
kamera-calibrate --help
```

For a walkthrough of every stage, see [how_it_works.md](how_it_works.md).

## Stages

Each stage resumes from `<flight>/calibration/`, skipping whatever already exists.

1. **frames** — `*_meta.json` grouped by trigger time into frames; camera names are `<channel>_<modality>` (`C_rgb`, `L_ir`, ...). IR and UV are contrast stretched to 8 bit; RGB is symlinked.
2. **features** — SIFT per camera with an initial focal length and distortion per modality, then an INS position prior per image (`InsTrajectory` interpolates the meta.json samples).
3. **match** — spatial matching from the priors, across all cameras, so figure-eight crossovers are matched as well as neighbours in time. Thermal-to-visible pairs are dropped: SIFT cannot match them and their few spurious inliers mislead the mapper.
4. **pass1** — incremental mapping with independent cameras and position priors. EO and IR come out as separate models, both in INS ENU. Needs three-view overlap along track: at 64 m/s and 1 frame/s that means flying above roughly 600 m AGL for these lenses; lower legs only register through crossovers with higher passes.
5. **pass2** — `cam_from_rig` for every camera is averaged from pass 1 (frames shared with the reference camera, both models being in INS ENU), the rig and frames are written to the database and onto the largest pass-1 model, and the images pass 1 never posed (IR) are added to their frames. Every image is then triangulated from the rig poses and bundle adjusted against the INS position priors, twice: rig poses and `sensor_from_rig` first, then with the intrinsics free as well.
6. **calibrate** — INS boresight (`ins_from_rig`) and lever arm as a robust average over frames, per-camera models, and `rig.yaml`.
7. **registration** — per channel `ir->rgb` and `uv->rgb` homographies as DIVE camera-registration JSON (format v2), plus flip GIFs.
8. **report** — PDF with the flight summary, intrinsics, rig geometry and registration overlays.

## Outputs

All output is directed to `<flight>/calibration/camera_models/`:

- `<rig>_<camera>.yaml` — `standard` camera model readable by `kamera.colmap_processing.camera_models.load_from_file`, with the rig and calibration provenance in extra keys.
- `<rig>_rig.yaml` — `cam_from_rig` per camera, `ins_from_rig`, lever arm, quality statistics.
- `dive_registration/<left>_to_<right>_registration.json`, `gifs/`, `<rig>_calibration_report.pdf`.

## Exposure Timing

The cameras do not expose at the same instant after the shared trigger; see [exposure_timing.md](exposure_timing.md) for the measurements, the manuals, and what the pipeline does about it.

## Error Sources

What limits the accuracy of the result, and how each source shows up in the outputs.

**INS attitude at the trigger.** Each meta.json carries one INS sample taken shortly before the trigger, so the attitude used for a frame can be up to 10 ms old. In a figure-eight turn at about 5 degrees per second that is up to 0.05 degrees, roughly 25 RGB pixels on the ground, and it enters every frame's boresight estimate as noise. An event-stamped INS sample or a full-rate INS log would remove it; `InsTrajectory` accepts either without code changes.

**Model drift.** The INS position priors pin the model's scale, heading and position, but its orientation still drifts slowly along the flight, and that drift is what dominates the per-frame boresight scatter reported in the rig yaml. The rig constraint removes any freedom between cameras within a frame, so the relative camera geometry, and therefore the homographies, is far better determined than the absolute boresight.

**Exposure timing.** A camera that exposes later than the reference camera sees the ground further along track, by ground speed times the delay. A bundle adjustment on a moving rig cannot tell that from a camera mounted that far forward, so the delay shows up as an along-track lever arm. The rig table in the report reads that lever arm back into a time difference at the flight's ground speed, positive when the camera exposes after the reference. Only the relative timing between cameras is observable, since a delay shared by the whole rig is absorbed by the position priors. The camera yaml positions carry these offsets, which is correct at similar ground speeds.

**Lever arms.** Beyond that timing signal the lever arms are weakly determined: at 400 to 900 m range a 30 cm baseline subtends less than one IR pixel. Read the reported translations with that in mind. The INS lever arm is the median offset of the rig origin from the INS position over all frames.

**Homographies.** A homography maps one camera onto another exactly only for flat ground at one range, and the timing baseline above makes the range matter. Each pair is fit for the range in its page title (the survey altitude if given, otherwise the calibration flight's median scene range). The fit residual, rms and 95th percentile in RGB pixels, measures the lens distortion a single matrix cannot carry, and the overlay shows it visually as coloured fringes.

## Conventions

- `camera_quaternion` (x, y, z, w) rotates camera vectors into the INS body frame (forward, right, down); `camera_position` is in that frame, in metres.
- COLMAP's `cam_from_rig` maps rig (= reference camera) coordinates into the camera.
- The INS body-to-ENU rotation is `NED_TO_ENU * R_z(heading) R_y(pitch) R_x(roll)`, identical to `kamera.sensor_models.nav_state`.
