from yolo_number_plate_detection.constants.constants import *
from yolo_number_plate_detection.utility.utils import read_yaml
from yolo_number_plate_detection.components.model_loader import load_pretrained_model
from yolo_number_plate_detection.components.model_trainer import train_model


config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

def train():
    pretrained_model = load_pretrained_model(config.yolo8.model_saved_path)
    trained_model = train_model(
        pretrained_model = pretrained_model, 
        params= params.training_params_yolo8
    )
