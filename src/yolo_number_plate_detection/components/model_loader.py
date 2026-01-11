from ultralytics import YOLO


def load_pretrained_model(model_path: str):
    model = YOLO(model_path)
    return model
   
    