from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import download_model_from_gdrive

app = FastAPI()

# روابط Google Drive للمودلين
od_id = "13PHjb6k65CgW_xzom3rPUdqO-jP07tF5"
ocr_id = "1-4m87-gC-ui0ANOYZ03E6B7QvbZXVESP"

# تحميل المودلين إذا ما كانوا موجودين
os.makedirs("models", exist_ok=True)
download_model_from_gdrive(od_id, "models/od_model.pt")
download_model_from_gdrive(ocr_id, "models/ocr_model.pt")

# تحميل مودلي YOLO
od_model = YOLO("models/od_model.pt")
ocr_model = YOLO("models/ocr_model.pt")

@app.get("/")
def root():
    return {"message": "Smart OD → OCR API loaded 🎯"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    height, width, _ = img.shape

    # تشغيل OD
    od_results = od_model(img)[0]
    od_boxes = []
    ocr_predictions = []

    for box in od_results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)

        # margin لإعطاء مساحة إضافية حول كل نقش
        margin = 10
        x1m = max(0, x1 - margin)
        y1m = max(0, y1 - margin)
        x2m = min(width, x2 + margin)
        y2m = min(height, y2 + margin)

        cropped = img[y1m:y2m, x1m:x2m]  # مقصوصة مع هامش
        od_boxes.append([x1, y1, x2, y2])

        # تشغيل OCR على المقصوصة بهامش
        ocr_results = ocr_model(cropped)[0]

        for ocr_box in ocr_results.boxes:
            class_id = int(ocr_box.cls[0])
            label = ocr_results.names[class_id]
            conf = float(ocr_box.conf[0])

            # تحويل إحداثيات OCR إلى الصورة الأصلية
            x1_ocr, y1_ocr, x2_ocr, y2_ocr = map(int, ocr_box.xyxy[0])
            abs_x1 = x1m + x1_ocr
            abs_y1 = y1m + y1_ocr
            abs_x2 = x1m + x2_ocr
            abs_y2 = y1m + y2_ocr

            ocr_predictions.append({
                "label": label,
                "confidence": round(conf, 3),
                "box": [abs_x1, abs_y1, abs_x2, abs_y2]
            })

    return JSONResponse(content={
        "od_boxes": od_boxes,
        "ocr_predictions": ocr_predictions
    })
