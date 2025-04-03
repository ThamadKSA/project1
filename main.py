
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import requests
import io
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Download model if not found ------------------
model_path = "model.pt"
drive_url = "https://drive.google.com/file/d/13o7_pMIAVKgQ91ZoTptMNCbglvHqIQNy/view?usp=sharing"  

if not os.path.exists(model_path):
    print("🔽 Downloading model from Google Drive...")
    with open(model_path, "wb") as f:
        f.write(requests.get(drive_url).content)
    print("✅ Model downloaded.")

# ------------------ Load YOLOv8 model ------------------
model = YOLO(model_path)

# ------------------ API route ------------------
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

# ------------------ Run server ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
