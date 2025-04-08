from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import download_model_from_gdrive

app = FastAPI()

# روابط Google Drive
od_id = "13PHjb6k65CgW_xzom3rPUdqO-jP07tF5"

# تحميل المودل
os.makedirs("models", exist_ok=True)
download_model_from_gdrive(od_id, "models/od_model.pt")

# تحميل مودل YOLO الخاص بـ OD
od_model = YOLO("models/od_model.pt")

@app.get("/")
def root():
    return {"message": "YOLO OD Model Loaded ✅"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    results = od_model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    return JSONResponse(content={
        "message": "OD model detected successfully ✅",
        "detected_boxes": len(boxes)
    })
