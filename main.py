from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import download_model_from_gdrive

app = FastAPI()

# رابط Google Drive لمودل OCR فقط
ocr_id = "1-4m87-gC-ui0ANOYZ03E6B7QvbZXVESP"

# تحميل المودل إذا ما كان موجود
os.makedirs("models", exist_ok=True)
download_model_from_gdrive(ocr_id, "models/ocr_model.pt")

# تحميل مودل OCR فقط
ocr_model = YOLO("models/ocr_model.pt")

@app.get("/")
def root():
    return {"message": "OCR-only model loaded ✅"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # تشغيل OCR على الصورة الكاملة
    ocr_results = ocr_model(img)[0]
    predictions = []

    for ocr_box in ocr_results.boxes:
        class_id = int(ocr_box.cls[0])
        label = ocr_results.names[class_id]

        x1, y1, x2, y2 = map(int, ocr_box.xyxy[0])
        conf = float(ocr_box.conf[0])

        predictions.append({
            "label": label,
            "confidence": round(conf, 3),
            "box": [x1, y1, x2, y2]
        })

    return JSONResponse(content={
        "ocr_predictions": predictions
    })
