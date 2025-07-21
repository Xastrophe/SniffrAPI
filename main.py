from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io, os
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

app = FastAPI()

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Path to found dogs collection
FOUND_DOG_DIR = "./found_dogs"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def is_allowed_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def image_to_tensor(image: Image.Image):
    return preprocess(image).unsqueeze(0).to(device)

def calculate_similarity(image1: torch.Tensor, image2: torch.Tensor) -> float:
    with torch.no_grad():
        emb1 = model.encode_image(image1).float()
        emb2 = model.encode_image(image2).float()
        emb1 /= emb1.norm(dim=-1, keepdim=True)
        emb2 /= emb2.norm(dim=-1, keepdim=True)
        return torch.nn.functional.cosine_similarity(emb1, emb2).item()


# Crop dog face using OpenCV's human face detector first, fallback to YOLO
def detect_and_crop_dog_face(image: Image.Image) -> Image.Image:
    import numpy as np
    import cv2

    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Load OpenCV's default frontal face detector (for humans)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return image.crop((x, y, x + w, y + h))

    # Fallback: use YOLO to detect whole dog if no face found
    results = yolo_model(image_np)
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            label = yolo_model.names[int(cls)]
            if label == "dog":
                x1, y1, x2, y2 = map(int, box)
                return image.crop((x1, y1, x2, y2))

    return image  # fallback to original image

@app.post("/compare-lost-dog")
async def compare_lost_dog(file: UploadFile = File(...), threshold: float = 0.70):
    if not is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    contents = await file.read()
    try:
        lost_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Could not open the uploaded image")

    cropped_lost_image = detect_and_crop_dog_face(lost_image)
    lost_tensor = image_to_tensor(cropped_lost_image)
    best_matches = []
    best_match = None
    best_score = -1

    for found_filename in os.listdir(FOUND_DOG_DIR):
        if not is_allowed_file(found_filename):
            continue
        found_path = os.path.join(FOUND_DOG_DIR, found_filename)
        try:
            found_image = Image.open(found_path).convert("RGB")
            cropped_found_image = detect_and_crop_dog_face(found_image)
            found_tensor = image_to_tensor(cropped_found_image)
            score = calculate_similarity(lost_tensor, found_tensor)
            print(f"Comparing {file.filename} with {found_filename}: score = {score}")
            if score > threshold:
                best_score = score
                best_matches.append(found_filename)
        except Exception as e:
            continue  # Skip unreadable files

    if best_score >= threshold:
        return {
            "match": best_matches,
            "similarity": round(best_score, 4)
        }
    else:
        return JSONResponse(content={"match": None, "similarity": round(best_score, 4)}, status_code=404)