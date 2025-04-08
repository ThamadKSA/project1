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
    return {"message": "OD + OCR models loaded 🎉"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # تشغيل مودل OD لاستخراج مناطق النقوش
    od_results = od_model(img)[0]
    boxes = od_results.boxes.xyxy.cpu().numpy()

    predictions = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped = img[y1:y2, x1:x2]

        # تشغيل مودل OCR على الجزء المقصوص
        ocr_results = ocr_model(cropped)[0]
        for ocr_box in ocr_results.boxes:
            class_id = int(ocr_box.cls[0])
            label = ocr_results.names[class_id]

            x1_ocr, y1_ocr, x2_ocr, y2_ocr = map(int, ocr_box.xyxy[0])
            conf = float(ocr_box.conf[0])

            predictions.append({
                "label": label,
                "confidence": round(conf, 3),
                "box": [x1_ocr, y1_ocr, x2_ocr, y2_ocr]
            })

    return JSONResponse(content={"predictions": predictions})
