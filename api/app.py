from altair import value
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
import cv2
import io
import numpy as np

from src.yolo_number_plate_detection.pipeline.prediction_pipeline import predict

app = FastAPI()

class ImageFile(BaseModel):
    file: UploadFile

    @field_validator("file")
    @classmethod
    def validate_image(cls, file: UploadFile):
        if not file.content_type.startswith("image/"):
            raise ValueError("File must be an image")
        return file


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        ImageFile(file=file)
        
        # Read uploaded image
        image_bytes = await file.read()
        
        # Save temp file
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