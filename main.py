from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io, base64, uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load your model ──────────────────────────
MODEL_PATH = "v2_best.pt"   # just the filename since it's in the same folder
model = YOLO(MODEL_PATH)

CLASS_NAMES = {
    0: {"code": "D00", "name": "Longitudinal Crack",  "description": "Crack running along the road"},
    1: {"code": "D10", "name": "Transverse Crack",    "description": "Crack running across the road"},
    2: {"code": "D20", "name": "Alligator Crack",     "description": "Interconnected cracking pattern"},
    3: {"code": "D40", "name": "Pothole",              "description": "Depression or hole in surface"},
}

def get_severity(count, confidences):
    if count == 0:
        return "none"
    avg = sum(confidences) / len(confidences)
    if count >= 5 or avg > 0.8:
        return "severe"
    if count >= 3 or avg > 0.6:
        return "moderate"
    return "minor"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    results = model(img)[0]

    detections = []
    confidences = []
    unique_types = set()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        meta   = CLASS_NAMES.get(cls_id, {
            "code": f"C{cls_id}",
            "name": f"Class {cls_id}",
            "description": "Unknown damage type"
        })
        detections.append({**meta, "confidence": round(conf, 3)})
        confidences.append(conf)
        unique_types.add(meta["code"])

    # Render annotated image → base64 JPEG
    annotated_array = results.plot()
    buf = io.BytesIO()
    Image.fromarray(annotated_array).save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({
        "is_damaged":       len(detections) > 0,
        "damage_count":     len(detections),
        "overall_severity": get_severity(len(detections), confidences),
        "unique_types":     list(unique_types),
        "detections":       detections,
        "annotated_image":  img_b64,
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)