# Camera Rig Calibration

Calibrate a KAMERA camera rig from a raw calibration flight to
per-camera models. For every camera it recovers:

- **intrinsics** — focal length, principal point, OpenCV distortion
  (from COLMAP bundle adjustment); and
- **the mount** — orientation and lever arm relative to the aircraft
  INS, from a single *boresight* solve per modality group.

Outputs: a per-camera `*_v3.yaml`, one self-contained
`<flight>_<date>_<config>_rig.json`, and registration QC gifs.
`TO26Su1_RicesWhale_calibration/fl004` is the running example.

<!-- IMAGE: sparse reconstruction / camera frustums over terrain -->
![Reconstruction overview](assets/calibration/reconstruction_overview.jpg)
*The prior-mapped SfM model — camera poses and 3-D points in the INS ENU frame.*

---

## Installation

Same native post-processing setup as the main
[README](README.md#installation): GDAL and pycolmap from conda-forge,
[uv](https://docs.astral.sh/uv/) for the rest. Requires
[conda](https://conda-forge.org/download/); works on Windows and Linux.

```bash
git clone https://github.com/Kitware/kamera.git && cd kamera
conda env create -f environment.yml
conda activate kamera
make install && source .venv/bin/activate   # Linux/macOS
# Windows:  pip install -e .
```

Afterwards `conda activate kamera` is all you need. Conda picks the CUDA
pycolmap build with NVIDIA driver 575+, else CPU. **A GPU only matters
here** — feature extraction/matching over a full flight is slow on CPU.

> The raw flight must contain the INS `*_meta.json` files (one per
> image); everything reads the INS from those, not the `.dat`.

---

## TL;DR

```bash
conda activate kamera
FLIGHT=/Volumes/extreme2tb/TO26Su1_RicesWhale_calibration/fl004

# 1. Stage raw imagery into the images0 layout (symlinks)
python kamera/postflight/scripts/prepare_flight.py $FLIGHT/images_21deg_N56RF $FLIGHT

# 2. Calibrate: build database, map with INS priors, boresight, export
python kamera/postflight/scripts/calibrate_rig.py $FLIGHT
```

Outputs land in `$FLIGHT/kamera_models/`.

---

## Input: the raw flight

```
fl004/
├── images_21deg_N56RF/     # one folder per camera view
│   ├── center_view/        #   *_C_*_<mod>.jpg + *_C_*_meta.json
│   ├── left_view/          #   *_L_*_<mod>.jpg + ...
│   └── right_view/         #   *_R_*_<mod>.jpg + ...
├── ins_raw/ins_raw_*.dat
└── ..._fl004_log.txt
```

Each view folder holds every image that camera took, in whatever
modalities were flown (`rgb`, `uv`, `ir`, or a mix), each with a
`_meta.json` carrying the INS state at exposure. All cameras are
hardware-synchronized — at each trigger they share one timestamp, which
is how images group into frames. Names encode provenance:

```
TO26Su1_RicesWhale_calibration_fl004_C_20260704_193924.627932_rgb.jpg
└──────────── effort ────────────┘ └flt┘└ch┘└─ date ─┘└── time ──┘└mod┘
```

---

## Step 1 — `prepare_flight.py`: stage into `images0`

The pipeline reads a per-*camera* layout, not the raw per-*view* one.
This reorganizes it (by symlink; nothing is copied) into per-modality
groups so SIFT never matches across the EO/IR gap:

```bash
python kamera/postflight/scripts/prepare_flight.py <raw_imagery_dir> <flight_dir>
```

```
fl004/
├── colmap_rgb/images0/          # EO group: rgb + uv
│   ├── 21deg_N56RF_center_rgb/  #   + *_uv folders if flown
│   ├── 21deg_N56RF_left_rgb/
│   └── 21deg_N56RF_right_rgb/
└── colmap_ir/images0/           # IR group, only if ir was flown
```

A group with no imagery simply isn't created (fl004 is RGB, so only
`colmap_rgb`).

### Paring the frame set

Selection keeps synchronized triggers together (all cameras at a kept
trigger). Flags:

| flag | effect |
|------|--------|
| `--spacing METERS` | keep a trigger only after this much 3-D travel |
| `--every N` | keep every Nth survivor |
| `--max-frames N` | cap by uniform decimation |
| `--focal-px F` | report forward overlap; warn if too sparse for SfM |
| `--copy` | copy instead of symlink |

**Don't over-prune.** SfM needs image overlap to triangulate; the
binding constraint is the *lowest-altitude* pass (smallest footprint).
If lowest-altitude overlap drops below ~40%, the model fragments — pass
`--focal-px` to watch it.

<!-- IMAGE: fl004 trajectory colored by altitude, three figure-8 bands -->
![fl004 trajectory](assets/calibration/fl004_trajectory.png)
*fl004 — three figure-8 blocks at ~565/435/275 m. All calibration, no transit.*

> **fl004:** at its native ~44 m spacing, overlap is already ~55%, so
> **keep every frame** (the default). The compute saving here is the
> spatial matcher in Step 2, not dropping frames.

---

## Step 2 — `calibrate_rig.py`: calibrate

```bash
python kamera/postflight/scripts/calibrate_rig.py <flight_dir>
```

Per modality group, end to end:

1. **Build the COLMAP database** (if none) — SIFT extraction (one OPENCV
   camera per folder) + **spatial matching** on INS priors, so matching
   is limited to spatial neighbors instead of every pair.
2. **Write INS pose priors** — each image's ENU position at exposure.
3. **Map** — prior-position reconstruction, directly in the INS ENU
   frame (no separate alignment) and resistant to fragmenting.
4. **Boresight** — one robust rig-to-INS rotation over all frames.
5. **Export** — a yaml per camera, mount = `ins_from_rig ∘ rig_from_sensor`.

Then it writes the rig JSON and QC gifs. Flags:

| flag | effect |
|------|--------|
| `--reuse-aligned` | reuse an existing `aligned/` model instead of re-mapping (fast) |
| `--save-dir DIR` | output dir (default `<flight_dir>/kamera_models`) |
| `--prior-std M` | INS position prior std, meters (default 2.0) |
| `--groups rgb:colmap_rgb ir:colmap_ir` | override which workspaces run |
| `--no-gifs` / `--num-gifs N` | skip / set QC gifs per camera |

A group runs only if its workspace exists, so a flight calibrates
whatever modalities it carries with no extra flags.

---

## Outputs

Under `<flight_dir>/kamera_models/`:

```
kamera_models/
├── 21deg_N56RF_center_rgb_v3.yaml       # one per physical camera
├── ...                                  #   (nine for a 3-station EO+UV+IR rig)
├── fl004_20260704_21deg_N56RF_rig.json  # complete rig model
└── registration_gifs_v3/*.gif           # QC
```

**Per-camera yaml** — the runtime model: image size, `fx/fy/cx/cy`,
distortion, `camera_quaternion` (camera→INS mount), `camera_position`.

**Rig JSON** — the authoritative, self-contained mount description:
provenance (flight, date, pycolmap version, git commit); reference frame
(INS body, ENU origin, quaternion convention); per group the boresight,
lever arm, and residual stats; per camera the intrinsics, INS mount, rig
extrinsics (`sensor_from_rig`), reprojection error, and image count.

**Registration gifs** — each non-reference camera flipped against its
colocated reference image warped into its view. Features hold still when
calibrated; jitter means misregistration. The visual acceptance check.

<!-- IMAGE: registration flip gif, e.g. uv vs rgb -->
![Registration gif](assets/calibration/registration_example.gif)
*UV flipped against RGB warped into its view — ground features stay locked.*

---

## How it works

The cameras are rigidly mounted and synchronized, so the flight maps
onto COLMAP's rig model. Rather than force the rig onto the mapper
(rig-constrained mapping from scratch fragments badly), we map once with
INS priors into one ENU model, then recover the rig in closed form: each
camera's `sensor_from_rig` by robust averaging over synchronized frames,
and one rig-to-INS **boresight** per group. Because every group's
boresight is solved against the *same physical INS frame*, EO and IR
come out mutually consistent with no cross-modal matching — which is why
this replaces the old per-camera search and IR transfer/keypoint steps.

<!-- IMAGE: frame diagram — camera / rig / INS / ENU and the transforms -->
![Mount frames](assets/calibration/mount_frames.png)
*camera → rig (`sensor_from_rig`) → INS (`ins_from_rig` boresight) → ENU world.*

---

## Troubleshooting

- **Check the gifs first** — the fastest read on calibration quality.
- **Reprojection error** (rig JSON) is INS-noise-limited (tens of px at
  long EO focals); a health signal, not the accuracy metric. The
  boresight residual (fractions of a degree) and the gifs are.
- **Fragmented model / few images** → overlap too low; re-run Step 1
  keeping more frames.
- **Iterating on boresight/export** without re-mapping → `--reuse-aligned`.
- **`--focal-px`** only affects the printed overlap estimate, nothing
  else (~15360 for these EO cameras).
