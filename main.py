from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
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

# ------------------ Paths to Models ------------------
od_model_path = "od_model.pt"
ocr_model_path = "ocr_model.pt"

# ------------------ Load models at startup ------------------
od_model = None
ocr_model = None

@app.on_event("startup")
async def load_models_once():
    global od_model, ocr_model
    if os.path.exists(od_model_path) and os.path.exists(ocr_model_path):
        od_model = YOLO(od_model_path)
        ocr_model = YOLO(ocr_model_path)
        print("✅ Models loaded successfully.")
    else:
        raise FileNotFoundError("❌ Model files not found. Please upload 'od_model.pt' and 'ocr_model.pt' manually.")

# ------------------ IoU Filter ------------------
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

# ------------------ Predict API ------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Step 1: Run OD model
    od_results = od_model(image)
    od_boxes = []
    for box in od_results[0].boxes.data:
        conf = float(box[4])
        cls = int(box[5])
        x1, y1, x2, y2 = map(int, box[:4])
        od_boxes.append({"box": [x1, y1, x2, y2], "confidence": conf, "class_id": cls})

    # Step 2: IoU filtering
    filtered_boxes = filter_overlapping_boxes(od_boxes)

    # Step 3: OCR on each cropped region
    predictions = []
    for item in filtered_boxes:
        x1, y1, x2, y2 = item['box']
        crop = image.crop((x1, y1, x2, y2))
        ocr_results = ocr_model(crop)

        chars = []
        for char in ocr_results[0].boxes.data:
            char_conf = float(char[4])
            char_id = int(char[5])
            chars.append({"char_id": char_id, "confidence": round(char_conf, 3)})

        predictions.append({
            "inscription_bbox": item['box'],
            "inscription_confidence": round(item['confidence'], 3),
            "translated_chars": chars
        })

    return JSONResponse(content={"results": predictions})

# ------------------ Run locally ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
