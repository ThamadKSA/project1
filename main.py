from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import io
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model (تأكدي إن ملف model.pt مرفوع مع المشروع)
model = YOLO("model.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)
    predictions = []

    for box in results[0].boxes.data:
        class_id = int(box[5])
        confidence = float(box[4])
        x1, y1, x2, y2 = [int(coord) for coord in box[:4]]

        predictions.append({
            "class_id": class_id,
            "confidence": round(confidence, 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    return JSONResponse(content={"predictions": predictions})
