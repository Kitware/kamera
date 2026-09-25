# Camera exposure timing on the KAMERA rig

Written 2026-09-16 from the May 2025 calibration flight (`052025_Calibration`).

## The short version

All nine cameras get the same trigger pulse, but they do not take their pictures at
the same moment. Each camera type has its own delay between the trigger and the middle
of its exposure:

| camera | when the middle of the exposure happens | how we know |
|---|---|---|
| RGB (Phase One iXM-GS120, electronic shutter) | about 20 ms after the trigger | Phase One guide |
| UV (Prosilica GT4907) | about 10 ms after the trigger (half of a 20 ms exposure) | Prosilica manual + measurement |
| IR (FLIR A6750) | within a few ms of the trigger | FLIR manual + measurement |

On top of that, the INS reading saved with each image is the last 100 Hz sample
*before* the trigger, so it is on average about 8 to 10 ms old.

At 65 m/s, 20 ms is 1.3 m on the ground. So the RGB, UV and IR pictures of one
"frame" were taken from three slightly different places along the flight line, and
the INS reading belongs to a fourth.

For now the calibration handles this in software (see "What we do about it now").
The right long-term fix is in hardware (see "What we should do later").

## How we found it

1. **The GIFs looked wrong.** The IR-to-RGB registration GIFs showed the IR image
   sitting a metre or two off from the RGB image, even though the calibration itself
   reported sub-pixel fits.

2. **We measured the offset instead of eyeballing it.** For each GIF we lined up the
   edges of the warped IR frame with the RGB frame (phase correlation on gradient
   images) and converted the shift to metres using the INS altitude. Result: the shift
   was about 1.5 m, almost entirely along the direction of flight, and it was the
   *same in metres* at 400 m and at 900 m altitude. A camera pointing error would grow
   with altitude. A constant distance along the flight line is what a time delay looks
   like.

3. **The 3D model said the same thing.** Bundle adjustment had placed all three IR
   camera centers about 1.0 m behind the RGB camera along the flight line, and the UV
   centers about 0.6 m behind, on a rig whose real spacing is a few tens of
   centimetres. A rig flying in a straight line cannot tell "this camera exposed 15 ms
   earlier" from "this camera is mounted 1 m further back", so the adjustment turned
   the timing into a fake lever arm. We confirmed the direction with nothing but the
   model's camera positions and the INS velocity: IR behind RGB by 0.93 to 1.01 m, UV
   behind RGB by 0.60 to 0.67 m, consistent over all 740 frames.

4. **We checked whether the RGB is "on time".** It cannot be told from the model:
   the INS position priors define where the model sits, so a delay shared by every
   camera is invisible. Only the differences between cameras are measurable. Trying to
   read the shared delay off the turns (a delay shows up as a pointing error that
   scales with turn rate) gave a weak +16 ms with most of the residual unexplained,
   so the manuals had to settle the absolute numbers.

5. **We checked how long the IR actually exposes.** A 30 ms integration at 65 m/s
   would smear the IR picture by 2 m, which is 8 pixels at 400 m altitude, and would
   visibly blur edges along the flight line. The raw IR frames have the same sharpness
   along and across track, so the integration in use is a few milliseconds at most.

6. **We read the manuals** (next section) and the numbers lined up.

## Supporting documentation

- **Phase One, iXM-GS120 Operation Guide, Rev 1.0.0, section 3.1 "Exposure Sequence".**
  The table "Hardware Pulses and Delay Parameter Signals" gives Trigger IN to Mid
  Exposure as "~20 msec + 0.5 x Exposure Time" for the electronic shutter and
  "~25 msec + 0.5 x Exposure Time" for the leaf shutter. The camera was run with the
  electronic shutter and a 0.3 ms exposure (from the image metadata), so its
  mid-exposure is about 20 ms after the trigger. The same table shows the camera
  outputs a **Mid-Exposure Pulse** on its own signal line.

- **Teledyne FLIR, A6000 and A8500 Series User's Manual, section 5.4.2 "Frame Sync
  Starts".** There is no single "latency" number. The camera has two sync modes: Frame
  Sync Starts Integration ("take a picture now") and Frame Sync Starts Readout (the
  sync reads out the previous frame and the exposure is placed automatically). Either
  way the exposure sits within a few ms of the sync edge, plus or minus half the
  integration time. Section 6.5.5 describes the Sync In as a rising-edge TTL signal
  with no stated delay. The KAMERA driver (`genicam_a6750.launch`) sets the frame sync
  source to External and does not set the sync mode or the integration time, so both
  come from the preset stored in the camera. Note: the older "A6xx series" manual
  covers the uncooled A615/A655 (640 x 480) and does not apply to the A6750
  (640 x 512).

- **Allied Vision, Prosilica GT Technical Manual V3.3.3, "Trigger timing concept"
  (camera interfaces chapter).** Trigger latency is defined as the delay from the user
  trigger to the start of exposure; the sibling models list 0.7 to 25.8 microseconds.
  So the UV exposure starts at the trigger for all practical purposes and its middle
  is half the exposure later. The GT4907 itself was removed from this manual in
  V3.2.1 as a discontinued model, so it has no spec table there. KAMERA runs the UV
  with auto-exposure (driver default, capped at 100 ms); the measured 10 ms offset
  matches a 20 ms exposure, which is also the value written in the comments of
  `prosilica.launch`.

Putting the three together: RGB middle at +20 ms, UV middle at +10 ms, IR middle
near 0 ms. The model measured IR 15 ms before RGB and UV 10 ms before RGB. That agrees
to within the "~" in the Phase One table.

## What we do about it now

- The rig calibration keeps the timing as along-track offsets in the camera
  positions. `rig.yaml` reports each camera's forward offset and the exposure time
  difference it implies (`exposure_offset_from_reference_ms`, negative = earlier than
  the RGB). The camera yaml positions carry the same offsets, which is correct as long
  as the survey flies at a similar ground speed.
- The DIVE homographies are fit for a nominal ground range
  (`--registration_range_m`, default: the calibration flight's median scene range)
  instead of at infinity, because a homography at infinity throws the offset away.
  Set it to the survey altitude above ground for the best registration there.
- `InsTrajectory` interpolates between INS samples, so a denser INS log removes the
  8 to 10 ms staleness without code changes.

## What we should do later

1. Log the full-rate INS stream (or post-process with POSPac) so the navigation
   solution can be interpolated to any timestamp.
2. Feed the Phase One mid-exposure pulse into an INS event input. That timestamps
   the RGB exposure directly and removes the "~20 ms".
3. Trigger the IR from that same pulse; its exposure then starts at the RGB
   mid-exposure and ends a few ms later.
4. Fix the UV exposure (no auto) and delay its trigger by 20 ms minus half the
   exposure, using the GT trigger-delay feature, so its middle lands on the pulse too.
   At minimum, record the UV exposure in the meta json.

With all three mid-exposures on one timestamped pulse, the "one pose per frame"
assumption in the rig model becomes exactly true, the rig offsets become physical,
and the homographies stop depending on speed and altitude.
