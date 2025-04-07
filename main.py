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
ocr_id = "1-4m87-gC-ui0ANOYZ03E6B7QvbZXVESP"

# تحميل المودلات
os.makedirs("models", exist_ok=True)
download_model_from_gdrive(od_id, "models/od_model.pt")
download_model_from_gdrive(ocr_id, "models/ocr_model.pt")

# تحميل YOLO models
od_model = YOLO("models/od_model.pt")
ocr_model = YOLO("models/ocr_model.pt")

@app.get("/")
def root():
    return {"message": "YOLO OD & OCR Models Loaded"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # كشف أماكن النقوش
    results = od_model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    predictions = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped_img = img[y1:y2, x1:x2]

        # تشغيل OCR على كل نقش
        ocr_result = ocr_model(cropped_img)[0]
        names = ocr_result.names
        for b in ocr_result.boxes:
            class_id = int(b.cls[0])
            label = names[class_id]
            predictions.append(label)

    return JSONResponse(content={"predicted_labels": predictions})
