# Camera Rig Calibration

This document walks through calibrating a KAMERA camera rig from a raw
calibration flight to per-camera models, using
`TO26Su1_RicesWhale_calibration/fl004` as the running example.

The calibration recovers, for every camera on the rig:

- **intrinsics** — focal length, principal point, and OpenCV distortion,
  solved by COLMAP bundle adjustment; and
- **the mount (extrinsics)** — the camera's orientation (and lever arm)
  relative to the aircraft INS, solved as a single *boresight* per
  modality group.

It produces per-camera `*_v3.yaml` models, one self-contained
`<flight>_<date>_<config>_rig.json` describing the whole mount, and
registration QC gifs.

---

## Installation

Same native post-processing install as the main
[README](README.md#installation) — GDAL and pycolmap come from
conda-forge, [uv](https://docs.astral.sh/uv/) installs the rest into
`.venv`. Requires [conda](https://conda-forge.org/download/). Works on
Windows and Linux (and macOS).

**Linux / macOS:**

```bash
git clone https://github.com/Kitware/kamera.git
cd kamera
conda env create -f environment.yml
conda activate kamera
make install
source .venv/bin/activate
```

**Windows (PowerShell or Anaconda Prompt):**

```powershell
git clone https://github.com/Kitware/kamera.git
cd kamera
conda env create -f environment.yml
conda activate kamera
pip install -e .
```

Afterwards, `conda activate kamera` is all you need. Conda installs the
CUDA build of pycolmap automatically with NVIDIA driver 575+ (CUDA
12.9), otherwise the CPU build. **The GPU only matters for this
calibration** (feature extraction / matching over a full flight) — the
rest of post-processing runs fine on CPU, and so will a small flight,
just slowly.

## Prerequisites

- The `kamera` env, installed and activated as above.
- The raw flight must contain the INS `*_meta.json` files (one per
  image); everything reads the INS from those, not the `.dat`.

---

## TL;DR

```bash
conda activate kamera

# 1. Stage raw imagery into the images0 layout (symlinks, keeps all frames)
python kamera/postflight/scripts/prepare_flight.py \
    /Volumes/extreme2tb/TO26Su1_RicesWhale_calibration/fl004/images_21deg_N56RF \
    /Volumes/extreme2tb/TO26Su1_RicesWhale_calibration/fl004

# 2. Calibrate: build database, map with INS priors, boresight, export
python kamera/postflight/scripts/calibrate_rig.py \
    /Volumes/extreme2tb/TO26Su1_RicesWhale_calibration/fl004
```

Outputs land in `<flight_dir>/kamera_models/`. Step 2 is GPU-heavy for a
from-scratch flight — run it on a machine with a CUDA GPU.

---

## Input: the raw flight folder

A raw flight looks like this (fl004):

```
fl004/
├── images_21deg_N56RF/          # raw imagery, one folder per camera view
│   ├── center_view/             #   *_C_*_<mod>.jpg  +  *_C_*_meta.json
│   ├── left_view/               #   *_L_*_<mod>.jpg  +  *_L_*_meta.json
│   ├── right_view/              #   *_R_*_<mod>.jpg  +  *_R_*_meta.json
│   └── sys_config.json
├── ins_raw/
│   └── ins_raw_*.dat
└── TO26Su1_..._fl004_log.txt
```

Each `center/left/right_view` folder holds every image that camera took,
across whatever modalities the flight carried — `rgb`, `uv`, `ir`, or a
mix — alongside a `_meta.json` per image carrying the INS state at
exposure time. The example commands below are shown for the RGB imagery
in fl004, but UV and IR are picked up automatically and handled the same
way (see modality groups, below).

Image names encode their provenance:

```
TO26Su1_RicesWhale_calibration_fl004_C_20260704_193924.627932_rgb.jpg
└──────────── effort ────────────┘ └flt┘└ch┘└─ date ─┘└── time ──┘└mod┘
```

All cameras on the rig — every station and modality — are
hardware-synchronized: at each trigger they share an identical
timestamp, which is how images are grouped into frames.

---

## Step 1 — `prepare_flight.py`: stage into `images0`

The calibration pipeline reads a per-*camera* folder layout
(`images0/<prefix>_<channel>_<modality>/`), not the raw per-*view*
layout. `prepare_flight.py` reorganizes the raw imagery into it:

```bash
python kamera/postflight/scripts/prepare_flight.py <raw_imagery_dir> <flight_dir>
```

For fl004:

```bash
python kamera/postflight/scripts/prepare_flight.py \
    /Volumes/extreme2tb/TO26Su1_RicesWhale_calibration/fl004/images_21deg_N56RF \
    /Volumes/extreme2tb/TO26Su1_RicesWhale_calibration/fl004 \
    --focal-px 15360        # optional: enables the overlap report
```

This creates, by **symlink** (nothing is copied):

```
fl004/
├── colmap_rgb/images0/                 # EO group (rgb + uv)
│   ├── 21deg_N56RF_center_rgb/
│   ├── 21deg_N56RF_left_rgb/
│   ├── 21deg_N56RF_right_rgb/
│   └── ...                              #   + *_uv folders if the flight has UV
└── colmap_ir/images0/                   # IR group, only if the flight has IR
    └── ...                              #   21deg_N56RF_center_ir/ ...
```

Modalities are split into groups so SIFT never tries to match across the
EO/IR gap: `rgb` + `uv` → `colmap_rgb`, `ir` → `colmap_ir`. Whichever
modalities the flight actually contains are staged; a group with no
imagery simply isn't created. (fl004 as shown here is RGB, so only
`colmap_rgb` appears.)

### Paring down the frame set

Calibration flights can carry far more frames than SfM needs, so
`prepare_flight.py` can subsample. Selection keeps synchronized triggers
together (all cameras at a kept trigger), which the rig logic requires.

| flag | effect |
|------|--------|
| `--spacing METERS` | keep a trigger only after the aircraft has moved this far (3-D). Evens coverage regardless of speed. |
| `--every N` | keep every Nth survivor |
| `--max-frames N` | cap the count by uniform decimation |
| `--focal-px F` | report estimated forward overlap and warn if a selection is too sparse for SfM |
| `--copy` | copy instead of symlink |

**Do not over-prune.** SfM needs sufficient image-to-image overlap to
match and triangulate; the binding constraint is the *lowest-altitude*
pass, where the ground footprint is smallest. The overlap report exists
to catch this — if lowest-altitude overlap drops below ~40%, the model
will fragment.

> **fl004 specifically:** the flight is three ~10-minute figure-8 blocks
> at ~565 m, ~435 m, and ~275 m — all of it is calibration, no transit
> to drop. At its native ~44 m trigger spacing, forward overlap is
> already only ~55%, so **keep every frame** (the default). Here the
> "glut" is not spatial oversampling; the compute saving comes from the
> spatial matcher (Step 2), not from dropping frames.

---

## Step 2 — `calibrate_rig.py`: calibrate

```bash
python kamera/postflight/scripts/calibrate_rig.py <flight_dir>
```

For each modality group it runs, end to end:

1. **Build the COLMAP database** (if none exists) — SIFT feature
   extraction (one OPENCV camera per folder) and **spatial matching**
   guided by INS priors, so matching is limited to spatial neighbors
   instead of every pair.
2. **Write INS pose priors** — each image's ENU position at exposure
   time, from the INS.
3. **Map** — prior-position incremental reconstruction. The priors fix
   the model directly in the INS ENU frame (no separate alignment step)
   and keep it from fragmenting.
4. **Solve the boresight** — one robust rotation solve of the rig
   relative to the INS, over every synchronized frame.
5. **Export** — a `StandardCamera` yaml per camera, mount =
   `ins_from_rig ∘ rig_from_sensor`.

After all groups it writes the combined rig JSON and the QC gifs.

Useful flags:

| flag | effect |
|------|--------|
| `--reuse-aligned` | reuse an existing `aligned/` model instead of re-mapping (fast; the boresight is gauge-independent) |
| `--save-dir DIR` | where models are written (default `<flight_dir>/kamera_models`) |
| `--prior-std M` | INS position prior std, meters (default 2.0) |
| `--groups rgb:colmap_rgb ir:colmap_ir` | override which workspaces to calibrate |
| `--no-gifs` / `--num-gifs N` | skip or set the number of QC gifs per camera |

Each group runs only if its workspace exists — the EO group needs
`colmap_rgb`, the IR group needs `colmap_ir` — so a flight calibrates
whatever modalities it carries with no extra flags. (For fl004 as shown
here, only the EO group runs.)

---

## Outputs

All under `<flight_dir>/kamera_models/`:

```
kamera_models/
├── 21deg_N56RF_center_rgb_v3.yaml      # one per camera, all modalities
├── 21deg_N56RF_left_rgb_v3.yaml        #   (*_uv, *_ir yamls too when present)
├── 21deg_N56RF_right_rgb_v3.yaml
├── fl004_20260704_21deg_N56RF_rig.json # complete rig model (see below)
└── registration_gifs_v3/               # QC: each camera flipped against
    └── *_vs_*_0.gif                     #     its colocated reference
```

There is one yaml per physical camera, so a 3-station EO+UV+IR flight
produces nine; the single rig JSON always covers every calibrated
camera across all groups.

### Per-camera yaml (`*_v3.yaml`)

The runtime camera model: image size, `fx/fy/cx/cy`, OpenCV distortion,
`camera_quaternion` (camera → INS mount), and `camera_position`.

### Rig model (`<flight>_<date>_<config>_rig.json`)

The authoritative, self-contained description of the whole mount:

- **provenance** — flight, effort, config, date, UTC calibration time,
  method, pycolmap version, git commit;
- **reference frame** — INS-body platform, ENU world with its
  `lat0/lon0/h0` origin, and the quaternion convention spelled out;
- **per group** (rgb, ir) — reference camera, model source, boresight
  (`ins_from_rig`), lever arm, and boresight residual statistics;
- **per camera** — intrinsics, the INS mount, the rig extrinsics
  (`sensor_from_rig`, rotation + translation), median reprojection
  error, and image count.

### Registration gifs

Each non-reference camera is flipped against its colocated
reference-modality image warped into its view. Well-calibrated cameras
hold ground features still across the flip; jitter means
misregistration. This is the visual acceptance check.

---

## How it works (in one paragraph)

The cameras are rigidly mounted and synchronized, so the flight maps
onto COLMAP's rig model. Rather than force the rig onto the mapper
(rig-constrained mapping from scratch fragments badly), we map once with
INS position priors to get a single ENU model, then recover the rig in
closed form: each camera's pose relative to the rig
(`sensor_from_rig`) by robust averaging over synchronized frames, and a
single rig-to-INS **boresight** per modality group. Because every
group's boresight is solved against the *same physical INS frame*, EO
and IR cameras come out mutually consistent with no cross-modal
matching — which is why this replaces the older per-camera search and
IR transfer/manual-keypoint steps.

---

## Verifying and troubleshooting

- **Check the gifs first.** They are the fastest read on whether a
  calibration is good.
- **Reprojection error** in the rig JSON is INS-attitude-noise-limited
  (tens of px at long EO focal lengths); it is a health signal, not the
  primary accuracy metric. The boresight residual (fractions of a
  degree) and the gifs are.
- **Fragmented model / few images registered** → overlap too low. Re-run
  Step 1 keeping more frames, and check the overlap report.
- **Iterating on the boresight/export** without re-mapping: pass
  `--reuse-aligned`, or point the driver at an existing model.
- **Focal length for the overlap report** (`--focal-px`) only affects
  the printed estimate, not the staging or the calibration. Use the
  camera's real focal in pixels when known (~15360 for the EO cameras
  here); it is otherwise cosmetic.
