from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io



from yolo_number_plate_detection.pipeline.prediction_pipeline import predict

app = FastAPI(title="YOLO Image Prediction API")

# Load model once (IMPORTANT for performance)
model = YOLO("artifacts/trained_yolo8_model/weights/best.pt")  # your trained YOLOv8 model


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()

    # Convert to OpenCV image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Run YOLO inference
    # results = predict(img)
    results = model(img)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

            # Label
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Convert back to image bytes
    _, encoded_img = cv2.imencode(".jpg", img)

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")