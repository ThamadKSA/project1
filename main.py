from fastapi import FastAPI
import os
from utils import download_model
from ultralytics import YOLO

app = FastAPI()

# روابط Hugging Face
od_url = "https://huggingface.co/Randa88/OD/resolve/main/best.pt"
ocr_url = "https://huggingface.co/Randa88/OCR/resolve/main/best-2.pt"

# تأكد من وجود المجلد
os.makedirs("models", exist_ok=True)

# تحميل المودلات
download_model(od_url, "models/od_model.pt")
download_model(ocr_url, "models/ocr_model.pt")

# تحميل مودلات YOLO
od_model = YOLO("models/od_model.pt")
ocr_model = YOLO("models/ocr_model.pt")

@app.get("/")
def root():
    return {"message": "YOLO OD & OCR Models Loaded"}
