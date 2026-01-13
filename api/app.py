from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import io
import numpy as np

from src.yolo_number_plate_detection.pipeline.prediction_pipeline import predict

app = FastAPI()

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()
        
        # Save temp file (keep as-is for now)
        image_path = "temp_file.jpg"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        # Get annotated image (NumPy array)
        annotated_img = predict(image_path)
        
        # Encode image to JPEG
        success, encoded_img = cv2.imencode(".jpg", annotated_img)
        if not success:
            return {"error": "Failed to encode image"}
        
        # Return image as response
        return StreamingResponse(
            io.BytesIO(encoded_img.tobytes()),
            media_type="image/jpeg"
        )


    except Exception as e:
        return {"error": str(e)}