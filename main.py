# ✅ FastAPI API that chains OD then OCR using YOLOv8 models (with single-time model loading)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import gdown
import io
import os

app = FastAPI()

# ------------------ CORS Middleware ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Download Models ------------------
od_model_path = "od_model.pt"
od_drive_id = "13PHjb6k65CgW_xzom3rPUdqO-jP07tF5"

ocr_model_path = "ocr_model.pt"
ocr_drive_id = "1-4m87-gC-ui0ANOYZ03E6B7QvbZXVESP"

def download_model_if_needed(drive_id, output):
    if not os.path.exists(output):
        print(f"🔽 Downloading {output} from Google Drive...")
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, output, quiet=False)
        print(f"✅ {output} downloaded successfully.")

# ------------------ Load Models Once on Startup ------------------
@app.on_event("startup")
async def load_models_once():
    download_model_if_needed(od_drive_id, od_model_path)
    download_model_if_needed(ocr_drive_id, ocr_model_path)

    app.state.od_model = YOLO(od_model_path)
    app.state.ocr_model = YOLO(ocr_model_path)
    print("✅ Models loaded and ready.")

# ------------------ Predict Endpoint ------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # ✅ Run OD model only (no IoU or OCR for now)
    od_model = app.state.od_model
    od_results = od_model(image)

    od_boxes = []
    for box in od_results[0].boxes.data:
        conf = float(box[4])
        cls = int(box[5])
        x1, y1, x2, y2 = map(int, box[:4])
        od_boxes.append({
            "box": [x1, y1, x2, y2],
            "confidence": round(conf, 3),
            "class_id": cls
        })

    return JSONResponse(content={"od_only_results": od_boxes})
