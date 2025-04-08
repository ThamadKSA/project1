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

    # تشغيل OD
    od_results = od_model(img)[0]
    od_boxes = []
    boxes = od_results.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        od_boxes.append([x1, y1, x2, y2])  # حفظ بوكس النقش

    predictions = []

    # نمر على كل نقش ونسوي OCR داخله فقط
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped = img[y1:y2, x1:x2]

        # تشغيل OCR على الصورة المقصوصة
        ocr_results = ocr_model(cropped)[0]
        for ocr_box in ocr_results.boxes:
            class_id = int(ocr_box.cls[0])
            label = ocr_results.names[class_id]

            # إحداثيات داخل القص
            x1_local, y1_local, x2_local, y2_local = map(int, ocr_box.xyxy[0])
            conf = float(ocr_box.conf[0])

            # تحويل الإحداثيات للصورة الأصلية
            x1_ocr = x1 + x1_local
            y1_ocr = y1 + y1_local
            x2_ocr = x1 + x2_local
            y2_ocr = y1 + y2_local

            predictions.append({
                "label": label,
                "confidence": round(conf, 3),
                "box": [x1_ocr, y1_ocr, x2_ocr, y2_ocr],
                "source_od_box": idx  # مهم للمقارنة لاحقًا
            })

    return JSONResponse(content={
        "od_boxes": od_boxes,
        "ocr_predictions": predictions
    })
