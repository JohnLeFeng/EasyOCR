import cv2
import sys
import logging
import argparse

import numpy as np
import openvino as ov

from pathlib import Path

from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import resize_aspect_ratio
from easyocr.utils import group_text_box, diff

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', 
                      help='Show this help message and exit.')
    args.add_argument('-d', '--device', type=str, default="GPU", 
                      help='Optional. Device for OpenVINO inference. Default is "GPU".')
    args.add_argument('-i', '--input_image', type=str, required=True, 
                      help='Path to the input image for text detection.')
    args.add_argument('-o', '--output_dir', type=str, default="output", 
                      help='Path to the output directory for saving results.')
    args.add_argument('-m', '--models_path', type=str, default="ov_model/detector.xml",
                      help='Optional. Path to the OpenVINO model file to use for text detection. '
                           'Default is "ov_model/detector.xml".')
    args.add_argument("--mag_ratio", type=float, default=1.,
                      help="Optional. Image magnification ratio. Default is 1.",)
    args.add_argument("--canvas_size", type=int, default=2560,
                      help="Optional. Maximum image size. Image bigger than this value will be "
                           "resized down. Default is 2560.")
    args.add_argument("--text_threshold", type=float, default=0.7,
                      help="Optional. Text confidence threshold. Default is 0.7.")
    args.add_argument("--low_text", type=float, default=0.4,
                      help="Optional. Text low-bound score. Default is 0.4.")
    args.add_argument("--link_threshold", type=float, default=0.4,
                      help="Optional. Link confidence threshold. Default is 0.4.")
    args.add_argument("--slope_ths", type=float, default=0.1,
                      help="Optional. Maximum slope (delta y/delta x) to considered merging. "
                           "Low value means tiled boxes will not be merged. Default is 0.1.")
    args.add_argument("--ycenter_ths", type=float, default=0.5,
                      help="Optional. Maximum shift in y direction. Boxes with different level "
                           "should not be merged. Default is 0.5.")
    args.add_argument("--height_ths", type=float, default=0.5,
                      help="Optional. Maximum different in box height. Boxes with very different "
                           "text size should not be merged. Default is 0.5.")
    args.add_argument("--width_ths", type=float, default=0.5,
                      help="Optional. Maximum horizontal distance to merge boxes. Default is 0.5.")
    args.add_argument("--add_margin", type=float, default=0.1,
                      help="Optional. Extend bounding boxes in all direction by certain value. "
                           "This is important for language with complex script (E.g. Thai). Default is 0.1.")
    args.add_argument("--min_size", type=int, default=20,
                      help="Optional. Filter text box smaller than minimum value in pixel. Default is 20.")

    return parser.parse_args()

def preprocess_input(image, canvas_size, mag_ratio):
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
        # make score and link map
        score_text = out[:, :, 0]
        score_link = out[:, :, 1]

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
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
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
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

    model_path = Path(args.models_path)
    device = args.device
    input_image = Path(args.input_image)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Output directory isn't existed and created at {}".format(output_dir))

    img = cv2.imread(input_image)
    log.info("Loaded image from {}".format(input_image))

    input_blob, ratio_w, ratio_h = preprocess_input(img, args.canvas_size, args.mag_ratio)

    _, _, target_h, target_w = input_blob.shape 

    ov_detector_model = core.read_model(model_path)
    log.info("Read EasyOCR detection model from {}".format(model_path))

    prep = ov.preprocess.PrePostProcessor(ov_detector_model)
    prep.input(0).tensor().set_layout(ov.Layout("NCHW"))
    prep.input(0).preprocess().scale([255, 255, 255])
    prep.input(0).preprocess().mean([0.485, 0.456, 0.406]).scale([0.229, 0.224, 0.225])

    ov_detector_model = prep.build()

    if device == "NPU":
        ov_detector_model.reshape([1, 3, target_h, target_w])
        log.info ("Reshape model input to static input shape({}, {}, {}, {}) for NPU inference.".format(1, 3, target_h, target_w))

    ov_detector = core.compile_model(ov_detector_model, device)
    log.info("Compiled EasyOCR detection model on {}".format(device))

    detect_result = ov_detector(input_blob)

    horizontal_list_agg, _ = postprocess_detections(detect_result, ratio_w, ratio_h, 
                                                    args.text_threshold, args.link_threshold, args.low_text, 
                                                    args.slope_ths, args.ycenter_ths, args.height_ths, 
                                                    args.width_ths, args.add_margin, args.min_size)

    cv2.imwrite(output_dir / "detection_result.jpg", draw_box(img.copy(), horizontal_list_agg[0]))
    log.info("Detection results saved to {}".format(output_dir / "detection_result.jpg"))

    return 0

if __name__ == "__main__":
    sys.exit(main())