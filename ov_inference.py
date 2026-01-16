
import cv2

import numpy as np
import openvino as ov

from pathlib import Path
from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import resize_aspect_ratio
from easyocr.utils import group_text_box, diff

core = ov.Core()


models_dir = "ov_model"
detector_device = "GPU"
input_image = "a.png"

img = cv2.imread(input_image)

canvas_size = 2560
mag_ratio = 1.
# img = cv2.resize(img, (2560, 1728)) # 1, 3, 1728, 2560
# print (img.shape)
# img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                                              interpolation=cv2.INTER_LINEAR,
                                                              mag_ratio=mag_ratio)
print ("img_resized: ", img_resized.shape)
input_blob = np.expand_dims(np.transpose(img_resized, (2, 0, 1)), 0)
ratio_h = ratio_w = 1 / target_ratio

detector_path = Path(models_dir) / "detector.xml"

ov_detector_model = core.read_model(detector_path)

prep = ov.preprocess.PrePostProcessor(ov_detector_model)
prep.input(0).tensor().set_layout(ov.Layout("NCHW"))
prep.input(0).preprocess().scale([255, 255, 255])
prep.input(0).preprocess().mean([0.485, 0.456, 0.406]).scale([0.229, 0.224, 0.225])

ov_detector_model = prep.build()

ov_detector = core.compile_model(ov_detector_model, detector_device)

detect_result = ov_detector(input_blob)

y, feature = detect_result[0], detect_result[1]
print ("y: ", y.shape)
print ("feature: ", feature.shape)

# min_size = 20, text_threshold = 0.7, low_text = 0.4,\
# link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
# slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
# width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None,
# threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,

text_threshold = 0.7
link_threshold = 0.4
low_text = 0.4
poly = False
estimate_num_chars = None

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

# print (polys_list)
# print (boxes_list)

text_box_list = []

for polys in polys_list:
    single_img_result = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        single_img_result.append(poly)
    text_box_list.append(single_img_result)

# min_size = 20, text_threshold = 0.7, low_text = 0.4,\
# link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
# slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
# width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None,
# threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,

slope_ths = 0.1
ycenter_ths = 0.5
height_ths = 0.5
width_ths = 0.5
add_margin = 0.1
optimal_num_chars=None
min_size = 20

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

print ("horizontal_list_agg: ", horizontal_list_agg[0])
# print (len(horizontal_list_agg))
# print (len(horizontal_list_agg[0]))
# print (len(horizontal_list_agg[0][0]))

maximum_y,maximum_x, _ = img.shape

for _obj in horizontal_list_agg[0]:
    # cv2.rectangle(img, (_obj[0], _obj[1]), (_obj[0] + _obj[2], _obj[1] + _obj[3]), (255, 0, 0), 2)
    # cv2.rectangle(img, (_obj[1], _obj[0]), (_obj[1] + _obj[3], _obj[0] + _obj[2]), (255, 0, 0), 2)

    x_min = max(0,_obj[0])
    x_max = min(_obj[1],maximum_x)
    y_min = max(0,_obj[2])
    y_max = min(_obj[3],maximum_y)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)


cv2.imwrite("dd.jpg", img)
# cv2.rectangle(image, (5, 5), (220, 220), (255, 0, 0), 2)