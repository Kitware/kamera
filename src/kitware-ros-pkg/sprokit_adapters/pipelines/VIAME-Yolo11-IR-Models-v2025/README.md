# VIAME-Yolo11-IR-Models-v2025

2025-generation IR hotspot detector (ultralytics YOLO11s), successor to the
darknet `arctic_seal_ir` model in VIAME-JoBBS-Models-v2021.01.20.

Received 2026-09-24 alongside the 2025 validation set
(`/data2/datasets/2025_validation_data`: CHESS2016 + polar_bear_2019 +
test_* + ice_seals_2025 fl102/fl207 frames, single class `hotspot`,
YOLO-format labels).

- `models/yolo11s_IR_2025_best.pt` — training checkpoint (19 MB; keep out
  of git like the JoBBS `.weights` files)
- `models/yolo11s_IR_2025_best.torchscript` — the same model exported to
  TorchScript, which is what inference runs (also kept out of git)
- `scripts/2025_IR_batch_norm_yolo11s.py` — batch inference with plain
  PyTorch (16-bit tif -> despeckle -> percentile norm -> YOLO -> VIAME CSV)
- `scripts/despeckle_and_norm_001.py` — standalone preprocessor; the
  inference script imports its functions, so both stay in sync

The checkpoint is a pickled ultralytics object, so it only loads with
ultralytics installed. Export it to TorchScript once, on any machine that
has ultralytics; the image size and class names travel with the export:

```
python -c "from ultralytics import YOLO; YOLO('models/yolo11s_IR_2025_best.pt').export(format='torchscript')"
```

Inference then needs only torch, torchvision and opencv (all in the VIAME
image):

```
python scripts/2025_IR_batch_norm_yolo11s.py models/yolo11s_IR_2025_best.torchscript \
    <tif_folder> [<tif_folder> ...] -o detections.csv --conf 0.01
```

Folders can also come from `--manifest <file>` (one path per line).

Preprocessing contract (must match training):
despeckle_16bit(median-3 diff > 500) then percentile normalize
[0.001, 99.999] -> uint8 -> replicate to 3 channels. This REPLACES the
2021 pipeline's `npy_percentile_norm` (1st percentile -> max).
Inference: conf threshold 0.01 at scoring time; pick operating point from
the validation PR curve.

## Evaluation vs the 2021 darknet model (2026-09-24)

Scored on the 2025 validation set (2,592 images, 2,900 GT hotspots, 882
negatives); eval code + plots in /data2/code/kamera_work/ir_eval/.
GT boxes are tight ~8x8 px; the 2021 model emits ~18x18 px boxes, so
IoU-only scoring misreads it — center-hit (det center inside GT dilated
to >=20x20 px) measures detection, IoU measures localization.

| metric                    | 2021 darknet | 2025 YOLO11s |
|---------------------------|--------------|--------------|
| AP center-hit (detection) | 0.799        | **0.866**    |
| max-F1 center-hit         | 0.862        | **0.891** (conf 0.166, P 0.862 / R 0.921) |
| AP @ IoU 0.25             | 0.183        | **0.848**    |
| AP @ IoU 0.5              | 0.000        | **0.622**    |
| ultralytics val mAP50     | n/a          | 0.735        |

Per-subset center-hit AP gains: polar_bear 0.695->0.814, test
0.749->0.861, ice_seals_2025 0.897->0.977, CHESS2016 0.888->0.898.
Recommended operating point: **conf 0.166** (max-F1).

Caveats: validation PNGs came pre-normalized with THIS model's
preprocessing, so the 2021 model ran slightly out-of-distribution
(it still hits 86% detection recall); training/val split provenance
of the 2025 set is unverified upstream.

Fusion regression (Image_sequence_ice_seals, 14 frames): finds the same
3 hotspots as the 2021 model — no new false cues at conf 0.166 — with
tighter boxes; all 3 EO-corroborated within 0.3 m of the registered cue
(run in /data2/code/kamera_work/seal_detect/fusion_v2025/).
