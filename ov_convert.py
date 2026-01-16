import torch
import easyocr

import openvino as ov

from pathlib import Path


language = "en"
models_dir = "ov_model"

reader = easyocr.Reader([language], quantize=False, gpu=False)
recognizer_path = Path(models_dir) / "recognizer_{}.xml".format(language)
detector_path = Path(models_dir) / "detector.xml"

if not recognizer_path.exists():
    ov_model = ov.convert_model(reader.recognizer, example_input=(torch.zeros([1, 1, 64, 320]), torch.zeros([1, 33], dtype=torch.long)))
    ov.save_model(ov_model, recognizer_path)

if not detector_path.exists():
    ov_model = ov.convert_model(reader.detector, example_input=torch.zeros([1, 3, 1728, 2560]))
    ov.save_model(ov_model, detector_path)
