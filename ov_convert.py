import sys
import torch
import easyocr
import logging
import argparse

import openvino as ov

from pathlib import Path

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', 
                      help='Show this help message and exit.')
    args.add_argument('-l', '--language', type=str, default="en", 
                      help='Optional. Language for OCR. Default is "en". '
                           'Please see EasyOCR documentation for other supported languages.')
    args.add_argument('-md', '--models_dir', type=str, default="ov_model",
                      help='Optional. Directory for OpenVINO models.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    language = args.language
    models_dir = args.models_dir

    reader = easyocr.Reader([language], quantize=False, gpu=False)
    recognizer_path = Path(models_dir) / "recognizer_{}.xml".format(language)
    detector_path = Path(models_dir) / "detector.xml"

    if not recognizer_path.exists():
        ov_model = ov.convert_model(reader.recognizer, example_input=(torch.zeros([1, 1, 64, 320]), torch.zeros([1, 33], dtype=torch.long)))
        ov.save_model(ov_model, recognizer_path)
        log.info(f"Recognizer model for {language} saved to {recognizer_path}")
    else:
        log.info(f"Recognizer model for {language} already exists at {recognizer_path}")

    if not detector_path.exists():
        ov_model = ov.convert_model(reader.detector, example_input=torch.zeros([1, 3, 1728, 2560]))
        ov.save_model(ov_model, detector_path)
        log.info(f"Detector model saved to {detector_path}")
    else:
        log.info(f"Detector model already exists at {detector_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())