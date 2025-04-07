from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import download_model

app = FastAPI()

# روابط Hugging Face
od_url = "https://huggingface.co/Randa88/OD/resolve/main/best.pt"
ocr_url = "https://huggingface.co/Randa88/OCR/resolve/main/best-2.pt"

# تحميل المودلات
os.makedirs("models", exist_ok=True)

# أرقام تقريبية للحجم بالبايت (مثلاً 20MB = 20 * 1024 * 1024)
download_model(od_url, "models/od_model.pt", expected_size=20 * 1024 * 1024)
download_model(ocr_url, "models/ocr_model.pt", expected_size=20 * 1024 * 1024)

# تحميل YOLO models
od_model = YOLO("models/od_model.pt")
ocr_model = YOLO("models/ocr_model.pt")

@app.get("/")
def root():
    return {"message": "YOLO OD & OCR Models Loaded"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 1. قراءة الصورة
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # 2. تشغيل OD model على الصورة الأصلية
    results = od_model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

    predictions = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped_img = img[y1:y2, x1:x2]

        # 3. تشغيل OCR على كل نقش مقصوص
        ocr_result = ocr_model(cropped_img)[0]
        names = ocr_result.names
        boxes_ocr = ocr_result.boxes
        for box_ocr in boxes_ocr:
            class_id = int(box_ocr.cls[0])
            label = names[class_id]
            predictions.append(label)

    return JSONResponse(content={"predicted_labels": predictions})
