import cv2
from yolo_number_plate_detection.constants.constants import *
from yolo_number_plate_detection.utility.utils import read_yaml
from yolo_number_plate_detection.components.model_loader import load_pretrained_model
from yolo_number_plate_detection.components.model_prediction import run_inferencing

config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

def predict(image_path):
    loaded_model = load_pretrained_model(config.inference_yolo8_model.model_path)   
    results = run_inferencing(
        trained_model=loaded_model,
        params=params.inference_params_yolo8,
        image_path=image_path
    )
    
    return results[0].plot()

if __name__ == "__main__":
    predict("data/test/images/Cars199.png")