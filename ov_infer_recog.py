import cv2
import sys
import time
import math
import logging
import argparse
import itertools

import numpy as np
import openvino as ov

from pathlib import Path

from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import resize_aspect_ratio
from easyocr.utils import group_text_box, get_image_list, diff, ctcBeamSearch, word_segmentation

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

# Character set for english_g2 recognizer model (must match the model training config)
ENGLISH_G2_CHARACTERS = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \u20acABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help',
                      help='Show this help message and exit.')
    args.add_argument('--det_device', type=str, default="NPU",
                      help='Optional. Device for detection model inference. Default is "NPU".')
    args.add_argument('--rec_device', type=str, default="NPU",
                      help='Optional. Device for recognition model inference. Default is "NPU".')
    args.add_argument('-i', '--input_image', type=str, required=True,
                      help='Path to the input image for OCR.')
    args.add_argument('-o', '--output_dir', type=str, default="output",
                      help='Path to the output directory for saving results.')
    args.add_argument('--detector_model', type=str, default="ov_model/detector.xml",
                      help='Optional. Path to the OpenVINO detector model. '
                           'Default is "ov_model/detector.xml".')
    args.add_argument('--recognizer_model', type=str, default="ov_model/recognizer_en.xml",
                      help='Optional. Path to the OpenVINO recognizer model. '
                           'Default is "ov_model/recognizer_en.xml".')
    # Detection parameters
    args.add_argument("--mag_ratio", type=float, default=1.,
                      help="Optional. Image magnification ratio. Default is 1.")
    args.add_argument("--canvas_size", type=int, default=2560,
                      help="Optional. Maximum image size. Default is 2560.")
    args.add_argument("--text_threshold", type=float, default=0.7,
                      help="Optional. Text confidence threshold. Default is 0.7.")
    args.add_argument("--low_text", type=float, default=0.4,
                      help="Optional. Text low-bound score. Default is 0.4.")
    args.add_argument("--link_threshold", type=float, default=0.4,
                      help="Optional. Link confidence threshold. Default is 0.4.")
    args.add_argument("--slope_ths", type=float, default=0.1,
                      help="Optional. Maximum slope to considered merging. Default is 0.1.")
    args.add_argument("--ycenter_ths", type=float, default=0.5,
                      help="Optional. Maximum shift in y direction. Default is 0.5.")
    args.add_argument("--height_ths", type=float, default=0.5,
                      help="Optional. Maximum different in box height. Default is 0.5.")
    args.add_argument("--width_ths", type=float, default=0.5,
                      help="Optional. Maximum horizontal distance to merge boxes. Default is 0.5.")
    args.add_argument("--add_margin", type=float, default=0.1,
                      help="Optional. Extend bounding boxes in all direction. Default is 0.1.")
    args.add_argument("--min_size", type=int, default=20,
                      help="Optional. Filter text box smaller than minimum value in pixel. Default is 20.")
    # Recognition parameters
    args.add_argument("--imgH", type=int, default=64,
                      help="Optional. Recognition input image height. Default is 64.")
    args.add_argument("--imgW", type=int, default=320,
                      help="Optional. Recognition input image max width. Default is 320.")
    args.add_argument("--char_width", type=int, default=10,
                      help="Optional. Estimated pixel width per character, used to compute "
                           "max output length (batch_max_length = imgW / char_width). "
                           "Lower value = more characters allowed. Default is 10.")
    args.add_argument("--decoder", type=str, default="greedy",
                      choices=["greedy", "beamsearch", "wordbeamsearch"],
                      help='Optional. Decoder strategy: greedy, beamsearch, or wordbeamsearch. '
                           'Default is "greedy".')
    args.add_argument("--beamWidth", type=int, default=5,
                      help="Optional. Beam width for beam search decoders. Default is 5.")
    args.add_argument("--contrast_ths", type=float, default=0.1,
                      help="Optional. Confidence threshold below which contrast retry is triggered. "
                           "Default is 0.1.")
    args.add_argument("--adjust_contrast", type=float, default=0.5,
                      help="Optional. Target contrast for retry pass. Default is 0.5.")
    args.add_argument("--filter_ths", type=float, default=0.03,
                      help="Optional. Filter out results with confidence below this threshold. "
                           "Default is 0.03.")
    args.add_argument("--dynamic_width", action="store_true",
                      help="Optional. Use per-crop dynamic input width for recognition "
                           "instead of the fixed --imgW for all crops. "
                           "Only works on CPU/GPU (NPU always uses fixed width). "
                           "--imgW becomes the upper-bound cap.")
    args.add_argument("--slice_trigger_ratio", type=float, default=1.25,
                      help="Optional. Split a detected text crop into overlapping slices when "
                           "its width exceeds imgW * this ratio. Default is 1.25.")
    args.add_argument("--slice_overlap_ratio", type=float, default=0.2,
                      help="Optional. Horizontal overlap ratio between adjacent recognition "
                           "slices for long crops. Default is 0.2.")
    args.add_argument("--merge_before_decode", action="store_true",
                      help="Optional. Merge raw model probability outputs from slices "
                           "before CTC decoding, instead of decoding each slice "
                           "independently and merging text. May improve accuracy "
                           "for sliced long crops.")
    args.add_argument("--merge_method", type=str, default="blank",
                      choices=["blank", "avg"],
                      help='Optional. Method for stitching probability matrices: '
                           '"blank" = splice at highest blank-token boundary (preserves CTC alignment), '
                           '"avg" = average overlapping timesteps. Default is "blank". '
                           'Only used when --merge_before_decode is enabled.')

    return parser.parse_args()


def preprocess_detection(image, canvas_size, mag_ratio):
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    input_blob = np.expand_dims(np.transpose(img_resized, (2, 0, 1)), 0)
    ratio_h = ratio_w = 1 / target_ratio
    return input_blob, ratio_w, ratio_h


def postprocess_detections(detect_result, ratio_w, ratio_h, text_threshold, link_threshold, low_text,
                           slope_ths, ycenter_ths, height_ths, width_ths, add_margin, min_size):
    poly = False
    estimate_num_chars = None
    optimal_num_chars = None

    y, feature = detect_result[0], detect_result[1]

    boxes_list, polys_list = [], []
    for out in y:
        score_text = out[:, :, 0]
        score_link = out[:, :, 1]
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    text_box_list = []
    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            p = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(p)
        text_box_list.append(single_img_result)

    horizontal_list_agg, free_list_agg = [], []
    for text_box in text_box_list:
        horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                    ycenter_ths, height_ths,
                                                    width_ths, add_margin,
                                                    (optimal_num_chars is None))
        if min_size:
            horizontal_list = [i for i in horizontal_list if max(
                i[1] - i[0], i[3] - i[2]) > min_size]
            free_list = [i for i in free_list if max(
                diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
        horizontal_list_agg.append(horizontal_list)
        free_list_agg.append(free_list)

    return horizontal_list_agg, free_list_agg


def contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high - low) / np.maximum(10, high + low), high, low


def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200. / np.maximum(10, high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(np.full(img.shape, 0),
                         np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


def preprocess_recognition(crop_img, imgH=64, imgW=320, do_adjust_contrast=False, adjust_contrast_target=0.5):
    """Preprocess a cropped grayscale text image for the recognizer.

    Replicates EasyOCR's AlignCollate + NormalizePAD in pure numpy.
    Returns an array of shape [1, 1, imgH, imgW].
    """
    if do_adjust_contrast:
        crop_img = adjust_contrast_grey(crop_img, target=adjust_contrast_target)

    h, w = crop_img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)

    resized_img = cv2.resize(crop_img, (resized_w, imgH), interpolation=cv2.INTER_CUBIC)

    # Normalize: transforms.ToTensor() maps [0,255]->[0,1], then sub(0.5).div(0.5) maps to [-1,1]
    normalized = resized_img.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5

    # Right-pad: replicate last column (matches NormalizePAD behaviour)
    pad_img = np.zeros((imgH, imgW), dtype=np.float32)
    pad_img[:, :resized_w] = normalized
    if resized_w < imgW:
        pad_img[:, resized_w:] = normalized[:, resized_w - 1:resized_w]

    return pad_img[np.newaxis, np.newaxis, :, :]


def split_long_crop(crop_img, max_width, trigger_ratio=1.25, overlap_ratio=0.2):
    """Split very wide recognition crops into overlapping horizontal slices."""
    crop_width = crop_img.shape[1]
    if crop_width <= max_width * trigger_ratio:
        return [{"start_x": 0, "end_x": crop_width, "image": crop_img}]

    overlap_pixels = int(round(max_width * overlap_ratio))
    overlap_pixels = max(0, min(overlap_pixels, max_width - 1))
    step = max(1, max_width - overlap_pixels)

    slices = []
    start_x = 0
    while start_x < crop_width:
        end_x = min(start_x + max_width, crop_width)
        start_x = max(0, end_x - max_width)
        if slices and start_x <= slices[-1][0]:
            break
        slices.append((start_x, crop_img[:, start_x:end_x]))
        if end_x >= crop_width:
            break
        start_x += step

    return [{"start_x": start_x, "end_x": start_x + item.shape[1], "image": item}
            for start_x, item in slices]


def estimate_overlap_char_count(text_length, overlap_pixels, slice_width):
    if text_length <= 0 or overlap_pixels <= 0 or slice_width <= 0:
        return 0

    estimated = int(round(text_length * overlap_pixels / float(slice_width)))
    return max(1, min(text_length, estimated))


def normalize_char_confidences(text, char_confidences, fallback_confidence):
    if not text:
        return []

    if len(char_confidences) == len(text):
        return [float(value) for value in char_confidences]

    if not char_confidences:
        return [float(fallback_confidence)] * len(text)

    return [float(value) for value in itertools.islice(itertools.cycle(char_confidences), len(text))]


def merge_slice_texts(slice_predictions):
    """Merge left-to-right slice predictions using overlap-region confidence."""
    merged_text = ""
    merged_char_confidences = []
    previous_prediction = None

    for prediction in slice_predictions:
        text = prediction["text"]
        if not text:
            continue

        char_confidences = normalize_char_confidences(
            text, prediction.get("char_confidences", []), prediction.get("confidence", 0.0))
        if not merged_text:
            merged_text = text
            merged_char_confidences = char_confidences
            previous_prediction = prediction
            continue

        overlap_pixels = max(0, previous_prediction["end_x"] - prediction["start_x"])
        previous_overlap_chars = estimate_overlap_char_count(
            len(previous_prediction["text"]), overlap_pixels, previous_prediction["end_x"] - previous_prediction["start_x"])
        current_overlap_chars = estimate_overlap_char_count(
            len(text), overlap_pixels, prediction["end_x"] - prediction["start_x"])

        if overlap_pixels <= 0 or previous_overlap_chars == 0 or current_overlap_chars == 0:
            merged_text += text
            merged_char_confidences.extend(char_confidences)
            previous_prediction = prediction
            continue

        previous_suffix_text = merged_text[-previous_overlap_chars:]
        previous_suffix_confidences = merged_char_confidences[-previous_overlap_chars:]
        current_prefix_text = text[:current_overlap_chars]
        current_prefix_confidences = char_confidences[:current_overlap_chars]

        previous_overlap_confidence = sum(previous_suffix_confidences) / len(previous_suffix_confidences)
        current_overlap_confidence = sum(current_prefix_confidences) / len(current_prefix_confidences)

        if current_overlap_confidence > previous_overlap_confidence:
            merged_text = (merged_text[:-previous_overlap_chars] + current_prefix_text +
                           text[current_overlap_chars:])
            merged_char_confidences = (merged_char_confidences[:-previous_overlap_chars] +
                                       current_prefix_confidences +
                                       char_confidences[current_overlap_chars:])
        else:
            merged_text += text[current_overlap_chars:]
            merged_char_confidences.extend(char_confidences[current_overlap_chars:])

        previous_prediction = prediction

    return merged_text


def stitch_slice_probs(slice_probs_list, slice_infos, imgW, method='blank'):
    """Stitch softmax probability matrices from overlapping slices before decoding.

    Methods:
        'avg'   — Average the overlapping timesteps (original).
        'blank' — Find the best blank-token cut point in each overlap zone and
                   splice cleanly there, preserving CTC alignment on both sides.

    Blank-based splice logic:
        For each adjacent slice pair (left, right) with overlap:
        1. Compute overlap_T = number of overlapping timesteps (proportional to
           pixel overlap / slice pixel width).
        2. In left slice's last overlap_T timesteps, get blank prob at each t.
        3. In right slice's first overlap_T timesteps, get blank prob at each t.
        4. For each possible cut position k in [0, overlap_T]:
           score(k) = left_blank[k] + right_blank[k]
           This finds where BOTH slices agree there's a CTC boundary.
        5. Pick k with highest score. Keep left[:T-overlap_T+k] + right[k:].
    """
    if len(slice_probs_list) == 1:
        return slice_probs_list[0]

    T = slice_probs_list[0].shape[1]  # timesteps per slice
    combined = slice_probs_list[0][0].copy()  # [T, C]

    for i in range(1, len(slice_probs_list)):
        curr = slice_probs_list[i][0]  # [T, C]

        overlap_px = max(0, slice_infos[i - 1]["end_x"] - slice_infos[i]["start_x"])
        slice_w = slice_infos[i]["end_x"] - slice_infos[i]["start_x"]

        overlap_T = int(round(T * overlap_px / slice_w)) if slice_w > 0 and overlap_px > 0 else 0
        overlap_T = min(overlap_T, T - 1, len(combined) - 1)

        if overlap_T > 0 and method == 'blank':
            # blank is index 0 in character list
            left_overlap = combined[-overlap_T:]      # [overlap_T, C]
            right_overlap = curr[:overlap_T]           # [overlap_T, C]

            left_blank = left_overlap[:, 0]   # blank prob per timestep
            right_blank = right_overlap[:, 0]

            # Score each cut position: want both sides to be at a blank boundary
            scores = left_blank + right_blank
            best_k = int(np.argmax(scores))

            # Splice: keep left up to cut, keep right from cut
            combined = np.concatenate([
                combined[:len(combined) - overlap_T + best_k],
                curr[best_k:]
            ], axis=0)
        elif overlap_T > 0 and method == 'avg':
            avg = (combined[-overlap_T:] + curr[:overlap_T]) / 2.0
            combined = np.concatenate([combined[:-overlap_T], avg, curr[overlap_T:]], axis=0)
        else:
            combined = np.concatenate([combined, curr], axis=0)

    return combined[np.newaxis, :, :]  # [1, total_T, C]


def get_recognition_width(crop_img, imgH, imgW, char_width, use_dynamic_width):
    if use_dynamic_width:
        h, w = crop_img.shape[:2]
        crop_imgW = min(math.ceil(imgH * w / float(h)), imgW)
        crop_imgW = max(crop_imgW, imgH)
    else:
        crop_imgW = imgW

    crop_bml = max(1, int(crop_imgW / char_width))
    return crop_imgW, crop_bml


def recognize_crop(ov_recognizer, crop_img, args, character_list, imgW, batch_max_length,
                   use_dynamic_width=False, do_adjust_contrast=False):
    crop_slices = split_long_crop(crop_img, imgW,
                                  trigger_ratio=args.slice_trigger_ratio,
                                  overlap_ratio=args.slice_overlap_ratio)

    merge_before = getattr(args, 'merge_before_decode', False) and len(crop_slices) > 1

    if merge_before:
        # Merge-before-decode: stitch probability matrices, then decode once
        slice_probs = []
        slice_infos = []
        total_time = 0.0

        for slice_info in crop_slices:
            slice_img = slice_info["image"]
            slice_imgW, slice_bml = get_recognition_width(slice_img, args.imgH, imgW,
                                                          args.char_width, use_dynamic_width)
            img_blob = preprocess_recognition(slice_img, imgH=args.imgH, imgW=slice_imgW,
                                              do_adjust_contrast=do_adjust_contrast,
                                              adjust_contrast_target=args.adjust_contrast)
            text_for_pred = np.zeros((1, slice_bml + 1), dtype=np.int64)

            start_t = time.time()
            recog_output = ov_recognizer({0: img_blob, 1: text_for_pred})
            total_time += time.time() - start_t

            preds = recog_output[0]
            probs = softmax(preds, axis=2)
            slice_probs.append(probs)
            slice_infos.append(slice_info)

        merge_method = getattr(args, 'merge_method', 'blank')
        combined_probs = stitch_slice_probs(slice_probs, slice_infos, imgW, method=merge_method)
        decoded = decode_predictions(combined_probs, character_list,
                                     decoder=args.decoder, beamWidth=args.beamWidth,
                                     apply_softmax=False)
        text, confidence, char_confidences = decoded[0]
        return text, confidence, total_time, len(crop_slices)

    # Original path: decode each slice independently, merge text
    slice_results = []
    total_time = 0.0

    for slice_info in crop_slices:
        slice_img = slice_info["image"]
        slice_imgW, slice_bml = get_recognition_width(slice_img, args.imgH, imgW,
                                                      args.char_width, use_dynamic_width)
        img_blob = preprocess_recognition(slice_img, imgH=args.imgH, imgW=slice_imgW,
                                          do_adjust_contrast=do_adjust_contrast,
                                          adjust_contrast_target=args.adjust_contrast)
        text_for_pred = np.zeros((1, slice_bml + 1), dtype=np.int64)

        start_t = time.time()
        recog_output = ov_recognizer({0: img_blob, 1: text_for_pred})
        total_time += time.time() - start_t

        preds = recog_output[0]
        decoded = decode_predictions(preds, character_list,
                                     decoder=args.decoder, beamWidth=args.beamWidth)
        text, confidence, char_confidences = decoded[0]
        slice_results.append({
            "start_x": slice_info["start_x"],
            "end_x": slice_info["end_x"],
            "text": text,
            "confidence": confidence,
            "char_confidences": char_confidences,
        })

    merged_text = merge_slice_texts(slice_results)
    slice_confidences = [item["confidence"] for item in slice_results if item["text"]]
    if slice_confidences:
        merged_confidence = float(sum(slice_confidences) / len(slice_confidences))
    elif slice_results:
        merged_confidence = float(sum(item["confidence"] for item in slice_results) / len(slice_results))
    else:
        merged_confidence = 0.0

    return merged_text, merged_confidence, total_time, len(crop_slices)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def custom_mean(x):
    return x.prod() ** (2.0 / np.sqrt(len(x)))


def decode_predictions(preds, characters, ignore_idx=None, decoder='greedy', beamWidth=5,
                       separator_list=None, dict_list=None, apply_softmax=True):
    """Decode model output using greedy, beam search, or word beam search.

    Args:
        preds: numpy array of shape [batch, T, num_classes] (raw logits or probabilities).
        characters: list where characters[0] = '[blank]'.
        ignore_idx: list of character indices to ignore (e.g. blank + separators).
        decoder: 'greedy', 'beamsearch', or 'wordbeamsearch'.
        beamWidth: beam width for beam search decoders.
        separator_list: dict mapping lang -> [start_sep, end_sep] for word beam search.
        dict_list: dict or list of words for word beam search.
        apply_softmax: if False, preds are already softmax probabilities.

    Returns:
        list of (text, confidence, char_confidences) tuples.
    """
    if ignore_idx is None:
        ignore_idx = [0]
    if separator_list is None:
        separator_list = {}
    if dict_list is None:
        dict_list = []

    results = []
    preds_prob = softmax(preds, axis=2) if apply_softmax else preds

    for i in range(preds_prob.shape[0]):
        prob = preds_prob[i]  # [T, num_classes]

        if decoder == 'greedy':
            # Greedy CTC decode on raw softmax (blank token must remain for proper CTC collapse)
            indices = np.argmax(prob, axis=1)
            max_probs = np.max(prob, axis=1)

            chars = []
            char_probs = []
            prev_idx = -1
            for j, idx in enumerate(indices):
                if idx != prev_idx:
                    if idx != 0:
                        chars.append(characters[idx])
                        char_probs.append(max_probs[j])
                prev_idx = idx

            text = ''.join(chars)
            confidence = float(custom_mean(np.array(char_probs))) if char_probs else 0.0
            char_confidences = [float(value) for value in char_probs]

        elif decoder == 'beamsearch':
            # Beam search handles ignore_idx internally via ctcBeamSearch
            text = ctcBeamSearch(prob, characters, ignore_idx, None, beamWidth=beamWidth)
            # Confidence from non-blank argmax positions
            indices = np.argmax(prob, axis=1)
            max_probs = np.max(prob, axis=1)
            char_probs = max_probs[indices != 0]
            confidence = float(custom_mean(char_probs)) if len(char_probs) > 0 else 0.0
            char_confidences = [float(confidence)] * len(text)

        elif decoder == 'wordbeamsearch':
            argmax = np.argmax(prob, axis=1)
            text = ''
            if len(separator_list) == 0:
                space_char = ' '
                space_idx = characters.index(space_char) if space_char in characters else -1
                if space_idx >= 0:
                    data = np.argwhere(argmax != space_idx).flatten()
                    group = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
                    group = [list(item) for item in group if len(item) > 0]
                    for j, list_idx in enumerate(group):
                        matrix = prob[list_idx, :]
                        t = ctcBeamSearch(matrix, characters, ignore_idx, None,
                                          beamWidth=beamWidth, dict_list=dict_list)
                        if j == 0:
                            text += t
                        else:
                            text += ' ' + t
                else:
                    text = ctcBeamSearch(prob, characters, ignore_idx, None,
                                          beamWidth=beamWidth, dict_list=dict_list)
            else:
                words = word_segmentation(argmax, separator_list)
                for word in words:
                    matrix = prob[word[1][0]:word[1][1] + 1, :]
                    if word[0] == '':
                        wdict = []
                    else:
                        wdict = dict_list.get(word[0], []) if isinstance(dict_list, dict) else dict_list
                    t = ctcBeamSearch(matrix, characters, ignore_idx, None,
                                      beamWidth=beamWidth, dict_list=wdict)
                    text += t

            # Confidence for wordbeamsearch
            indices = np.argmax(prob, axis=1)
            max_probs = np.max(prob, axis=1)
            char_probs = max_probs[indices != 0]
            confidence = float(custom_mean(char_probs)) if len(char_probs) > 0 else 0.0
            char_confidences = [float(confidence)] * len(text)

        results.append((text, confidence, char_confidences))

    return results


def draw_box(img, boxes):
    maximum_y, maximum_x, _ = img.shape

    for _obj in boxes:
        x_min = max(0,_obj[0])
        x_max = min(_obj[1],maximum_x)
        y_min = max(0,_obj[2])
        y_max = min(_obj[3],maximum_y)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    return img


def main():
    args = parse_args()

    core = ov.Core()

    detector_path = Path(args.detector_model)
    recognizer_path = Path(args.recognizer_model)
    det_device = args.det_device
    rec_device = args.rec_device
    input_image = Path(args.input_image)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Output directory created at %s", output_dir)

    # Character list: '[blank]' at index 0, then the actual characters
    character_list = ['[blank]'] + list(ENGLISH_G2_CHARACTERS)

    # ── Load image ──────────────────────────────────────────────────────
    img = cv2.imread(str(input_image))
    if img is None:
        log.error("Failed to load image from %s", input_image)
        return 1
    log.info("Loaded image from %s", input_image)

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Prepare input ───────────────────────────────────────────────────
    input_blob, ratio_w, ratio_h = preprocess_detection(img, args.canvas_size, args.mag_ratio)
    _, _, target_h, target_w = input_blob.shape

    # ── Load & compile detection model ──────────────────────────────────
    ov_detector_model = core.read_model(detector_path)
    log.info("Read detector model from %s", detector_path)

    prep = ov.preprocess.PrePostProcessor(ov_detector_model)
    prep.input(0).tensor().set_layout(ov.Layout("NCHW"))
    prep.input(0).preprocess().scale([255, 255, 255])
    prep.input(0).preprocess().mean([0.485, 0.456, 0.406]).scale([0.229, 0.224, 0.225])
    ov_detector_model = prep.build()

    if det_device == "NPU":
        ov_detector_model.reshape([1, 3, target_h, target_w])
        log.info("Reshaped detector input to [1, 3, %d, %d] for NPU.", target_h, target_w)
        ov_detector = core.compile_model(ov_detector_model, det_device, {"NPU_TILES": 1})
    else:
        ov_detector = core.compile_model(ov_detector_model, det_device)
    log.info("Compiled detector model on %s", det_device)

    imgW = args.imgW
    batch_max_length = max(1, int(imgW / args.char_width))

    use_dynamic_width = args.dynamic_width and rec_device != "NPU"
    if args.dynamic_width and rec_device == "NPU":
        log.warning("--dynamic_width is not supported on NPU; using fixed width %d.", imgW)
    if use_dynamic_width:
        log.info("Dynamic width enabled (cap=%d, char_width=%d)", imgW, args.char_width)
    else:
        log.info("Recognition input width: %d, batch_max_length: %d (char_width=%d)",
                 imgW, batch_max_length, args.char_width)
    log.info("Long-crop slicing enabled when width > %.2fx imgW (overlap=%.2f)",
             args.slice_trigger_ratio, args.slice_overlap_ratio)
    if args.merge_before_decode:
        log.info("Merge-before-decode: stitching probability matrices before CTC decoding (method=%s)",
                 args.merge_method)

    # ── Load & compile recognizer model ─────────────────────────────────
    ov_recog_model = core.read_model(recognizer_path)
    log.info("Read recognizer model from %s", recognizer_path)

    if rec_device == "NPU":
        ov_recog_model.reshape({0: [1, 1, args.imgH, imgW],
                                1: [1, batch_max_length + 1]})
        log.info("Reshaped recognizer input for NPU: image [1, 1, %d, %d], text [1, %d]",
                 args.imgH, imgW, batch_max_length + 1)
        ov_recognizer = core.compile_model(ov_recog_model, rec_device, {"NPU_TILES": 1})
    else:
        ov_recognizer = core.compile_model(ov_recog_model, rec_device)
    log.info("Compiled recognizer model on %s", rec_device)

    # ── Run detection ─────────────────────────────────────────────────────
    start_t = time.time()
    detect_result = ov_detector(input_blob)
    detect_t = time.time() - start_t
    log.info("Detection inference time: %.3f s", detect_t)

    horizontal_list_agg, free_list_agg = postprocess_detections(
        detect_result, ratio_w, ratio_h,
        args.text_threshold, args.link_threshold, args.low_text,
        args.slope_ths, args.ycenter_ths, args.height_ths,
        args.width_ths, args.add_margin, args.min_size)

    cv2.imwrite(output_dir / "detection_result.jpg", draw_box(img.copy(), horizontal_list_agg[0]))

    horizontal_list = horizontal_list_agg[0]
    free_list = free_list_agg[0]
    log.info("Detected %d horizontal + %d free text regions",
             len(horizontal_list), len(free_list))

    # ── Crop text regions ───────────────────────────────────────────────
    image_list, max_width = get_image_list(horizontal_list, free_list, img_grey,
                                           model_height=args.imgH)
    if not image_list:
        log.info("No text regions to recognise.")
        return 0
    else:
        log.info("Cropped %d text regions for recognition.", len(image_list))

    # ── Run recognition (first pass) ─────────────────────────────────────
    results = []
    total_recog_time = 0.0
    sliced_region_count = 0

    for box, crop_img in image_list:
        text, confidence, recog_time, slice_count = recognize_crop(
            ov_recognizer, crop_img, args, character_list, imgW, batch_max_length,
            use_dynamic_width=use_dynamic_width)
        total_recog_time += recog_time
        if slice_count > 1:
            sliced_region_count += 1
        results.append((box, text, confidence))

    log.info("Recognition inference time: %.3f s (%d regions)", total_recog_time, len(image_list))
    if sliced_region_count > 0:
        log.info("Applied long-crop slicing to %d regions in the first pass", sliced_region_count)

    # ── Contrast re-try for low-confidence regions ──────────────────────
    # """
    low_confident_idx = [i for i, r in enumerate(results) if r[2] < args.contrast_ths]
    if low_confident_idx:
        log.info("Contrast retry: %d regions below confidence threshold %.2f",
                 len(low_confident_idx), args.contrast_ths)
        retry_sliced_region_count = 0
        for idx in low_confident_idx:
            box, crop_img = image_list[idx]
            text2, confidence2, recog_time, slice_count = recognize_crop(
                ov_recognizer, crop_img, args, character_list, imgW, batch_max_length,
                use_dynamic_width=use_dynamic_width,
                do_adjust_contrast=True)
            total_recog_time += recog_time
            if slice_count > 1:
                retry_sliced_region_count += 1

            # Keep the result with higher confidence
            _, text1, confidence1 = results[idx]
            if confidence2 > confidence1:
                results[idx] = (box, text2, confidence2)
                log.info("  Region %d improved: %.4f -> %.4f", idx, confidence1, confidence2)

        log.info("Total recognition time (incl. retry): %.3f s", total_recog_time)
        if retry_sliced_region_count > 0:
            log.info("Applied long-crop slicing to %d retry regions", retry_sliced_region_count)
    # """
 
    # ── Filter low-confidence results ────────────────────────────────────
    if args.filter_ths > 0:
        before_count = len(results)
        results = [r for r in results if r[2] >= args.filter_ths]
        filtered_count = before_count - len(results)
        if filtered_count > 0:
            log.info("Filtered out %d regions below confidence %.4f", filtered_count, args.filter_ths)

    # ── Print results ───────────────────────────────────────────────────
    for _, text, confidence in results:
        log.info("  [%.4f] %s", confidence, text)

    # ── Save annotated image ────────────────────────────────────────────
    img_out = img.copy()
    for box, text, confidence in results:
        xs = [int(pt[0]) for pt in box]
        ys = [int(pt[1]) for pt in box]
        x_min, x_max = max(min(xs), 0), min(max(xs), img.shape[1])
        y_min, y_max = max(min(ys), 0), min(max(ys), img.shape[0])
        cv2.rectangle(img_out, (x_min, y_min), (x_max, y_max), (0, 128, 0), 2)
        cv2.putText(img_out, f"{text} ({confidence:.2f})",
                    (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1, cv2.LINE_AA)

    img_path = str(output_dir / ("recognition_result_" + str(imgW) + ".jpg"))
    cv2.imwrite(img_path, img_out)
    log.info("Annotated image saved to %s", img_path)

    # ── Save text results ───────────────────────────────────────────────
    txt_path = str(output_dir / ("recognition_result_" + str(imgW) + ".txt"))
    with open(txt_path, 'w', encoding='utf-8') as f:
        for box, text, confidence in results:
            f.write(f"[{confidence:.4f}] {text}\n")
    log.info("Text results saved to %s", txt_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
