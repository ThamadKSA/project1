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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Download OD and OCR models from Drive ------------------
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

# ------------------ Load models once at startup ------------------
@app.on_event("startup")
async def load_models_once():
    download_model_if_needed(od_drive_id, od_model_path)
    download_model_if_needed(ocr_drive_id, ocr_model_path)

    app.state.od_model = YOLO(od_model_path)
    app.state.ocr_model = YOLO(ocr_model_path)
    print("✅ Models loaded and ready.")

# ------------------ Helper: IoU Filter ------------------
def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = float(box1Area + box2Area - interArea)
    return interArea / unionArea if unionArea != 0 else 0

def filter_overlapping_boxes(boxes, threshold=0.5):
    filtered = []
    boxes = sorted(boxes, key=lambda b: b['confidence'], reverse=True)
    while boxes:
        current = boxes.pop(0)
        filtered.append(current)
        boxes = [b for b in boxes if calculate_iou(current['box'], b['box']) < threshold]
    return filtered

# ------------------ Main Predict API ------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # فقط اختبار: تشغيل نموذج OD فقط بدون IoU ولا OCR
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
