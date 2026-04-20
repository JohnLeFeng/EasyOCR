# EasyOCR — OpenVINO Inference

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE)

**PyTorch-free** OpenVINO inference pipeline for [EasyOCR](https://github.com/JaidedAI/EasyOCR) models.
Runs the full CRAFT text detection + english_g2 recognition flow entirely via
OpenVINO on Intel CPU, GPU, or NPU.

## Overview

| Step | Description |
|---|---|
| Detection | CRAFT detector finds text bounding boxes |
| Cropping | Extracts grayscale text regions |
| Recognition | CTC-based recognizer reads each crop |
| Contrast retry | Low-confidence regions get a second pass with contrast enhancement |
| Filtering | Results below the confidence threshold are discarded |

## Installation

```bash
pip install openvino opencv-python numpy
```

## Model Conversion

Convert EasyOCR PyTorch models to OpenVINO IR format:

```bash
python ov_convert.py
```

This produces `ov_model/detector.xml` and `ov_model/recognizer_en.xml`.

## Usage

### Detection Only

```bash
python ov_infer_detect.py -i photo.png -d CPU
```

Outputs an annotated image with detected text bounding boxes.
Run `python ov_infer_detect.py --help` for all options.

### Detection + Recognition

```bash
python ov_infer_recog.py -i photo.png --det_device CPU --rec_device CPU
```

### Device Options

Detection and recognition run on **separate devices**, allowing mixed deployment
(e.g. detection on NPU, recognition on CPU with dynamic shapes):

| Flag | Default | Description |
|---|---|---|
| `--det_device` | `NPU` | Device for the CRAFT text detector |
| `--rec_device` | `NPU` | Device for the english_g2 recognizer |

Supported devices: `CPU`, `GPU`, `NPU`.

### Detection Parameters

| Flag | Default | Description |
|---|---|---|
| `--canvas_size` | `2560` | Max image size for detection |
| `--mag_ratio` | `1.0` | Image magnification ratio |
| `--text_threshold` | `0.7` | Text confidence threshold |
| `--low_text` | `0.4` | Text low-bound score |
| `--link_threshold` | `0.4` | Link confidence threshold |
| `--slope_ths` | `0.1` | Max slope for box merging |
| `--ycenter_ths` | `0.5` | Max y-center shift for merging |
| `--height_ths` | `0.5` | Max height difference for merging |
| `--width_ths` | `0.5` | Max horizontal distance for merging |
| `--add_margin` | `0.1` | Bounding box margin extension |
| `--min_size` | `20` | Min box size in pixels |

### Recognition Parameters

| Flag | Default | Description |
|---|---|---|
| `--imgH` | `64` | Recognition input height |
| `--imgW` | `320` | Recognition input max width |
| `--char_width` | `10` | Pixel width per character (`batch_max_length = imgW / char_width`) |
| `--dynamic_width` | off | Per-crop dynamic input width (CPU/GPU only; `--imgW` becomes cap) |
| `--decoder` | `greedy` | Decode strategy: `greedy`, `beamsearch`, `wordbeamsearch` |
| `--beamWidth` | `5` | Beam width for search decoders |
| `--contrast_ths` | `0.1` | Confidence threshold for contrast retry |
| `--adjust_contrast` | `0.5` | Target contrast for retry pass |
| `--filter_ths` | `0.03` | Discard results below this confidence |
| `--slice_trigger_ratio` | `1.25` | Split long crops into slices when width exceeds `imgW * ratio` |
| `--slice_overlap_ratio` | `0.2` | Overlap ratio between adjacent slices |
| `--merge_before_decode` | off | Stitch probability matrices from slices before CTC decoding |
| `--merge_method` | `blank` | Stitch method: `blank` (splice at CTC blank boundary) or `avg` (average overlap). Only with `--merge_before_decode` |

### Long-Crop Slicing

When a detected text crop is much wider than `--imgW`, the script splits it into
overlapping horizontal slices for recognition. Two merge strategies are available:

**Text-merge (default):** Each slice is decoded independently; overlapping text
is resolved by comparing per-character confidence at the boundary.

**Merge-before-decode (`--merge_before_decode`):** Raw probability matrices from
each slice are stitched together *before* CTC decoding:

- `--merge_method blank` — Finds the timestep in the overlap zone where both
  slices have the highest blank-token probability (a natural CTC boundary) and
  splices cleanly there. Preserves CTC alignment on both sides.
- `--merge_method avg` — Averages the overlapping timesteps. Simpler but can
  blur CTC transitions and drop characters.

The `blank` method produces the most accurate text on sliced regions, especially
for long lines at small `--imgW` values.

### Output

| Flag | Default | Description |
|---|---|---|
| `-o` | `output` | Output directory |

The script saves an annotated image with bounding boxes and a text file with
recognized text and confidence scores.

### Examples

```bash
# NPU for both (default)
python ov_infer_recog.py -i photo.png

# CPU detection + GPU recognition with dynamic width
python ov_infer_recog.py -i photo.png --det_device CPU --rec_device GPU --dynamic_width

# Beam search decoder with stricter filtering
python ov_infer_recog.py -i photo.png --det_device CPU --rec_device CPU \
    --decoder beamsearch --beamWidth 10 --filter_ths 0.1

# Merge probability matrices before decoding (best for small imgW)
python ov_infer_recog.py -i photo.png --det_device CPU --rec_device CPU \
    --imgW 200 --merge_before_decode --merge_method blank
```

## File Structure

| File | Description |
|---|---|
| `ov_convert.py` | Convert PyTorch models to OpenVINO IR |
| `ov_infer_detect.py` | Detection-only inference script |
| `ov_infer_recog.py` | Full detection + recognition pipeline |
| `ov_inference_recognition.py` | Experimental recognition script (slicing/overlap research) |
| `ov_model/` | OpenVINO IR model files |
| `easyocr/` | Utility functions for pre/postprocessing |
| `scripts/compare_merge_methods.py` | Benchmark script comparing merge strategies |

## Acknowledgement and References

This project is based on [EasyOCR](https://github.com/JaidedAI/EasyOCR) and research from several papers and open-source repositories.

Detection uses the CRAFT algorithm from this [official repository](https://github.com/clovaai/CRAFT-pytorch) and their [paper](https://arxiv.org/abs/1904.01941).

The recognition model is a CRNN ([paper](https://arxiv.org/abs/1507.05717)) composed of feature extraction ([ResNet](https://arxiv.org/abs/1512.03385)/VGG), sequence labeling ([LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)), and CTC decoding ([paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)). Training pipeline based on [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

Beam search code is based on this [repository](https://github.com/githubharald/CTCDecoder).
