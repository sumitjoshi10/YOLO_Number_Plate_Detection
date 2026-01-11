from yolo_number_plate_detection.constants.constants import *
from yolo_number_plate_detection.utility.utils import read_yaml
from yolo_number_plate_detection.components.model_loader import load_pretrained_model


config = read_yaml(CONFIG_FILE_PATH)

def train():
    pretrained_model = load_pretrained_model(config.YOLO8.model_saved_path)
    print(pretrained_model)