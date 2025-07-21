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
model, preprocess = clip.load("ViT-L/14", device=device, download_root="models") #use 'ViT-L/14@336px' for better accuracy

# Load YOLOv8 model
yolo_model = YOLO("models/yolov8n.pt")

# Path to found dogs collection
FOUND_DOG_DIR = "./found_dogs"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def is_allowed_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def image_to_tensor(image: Image.Image):
    return preprocess(image).unsqueeze(0).to(device)

def calculate_similarity(image1: torch.Tensor, image2: torch.Tensor) -> dict:
    with torch.no_grad():
        # CLIP embedding
        emb1 = model.encode_image(image1).float()
        emb2 = model.encode_image(image2).float()
        emb1 /= emb1.norm(dim=-1, keepdim=True)
        emb2 /= emb2.norm(dim=-1, keepdim=True)
        clip_score = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        # Placeholder for second model score (e.g., DINOv2 or another embedding model)
        alt_score = clip_score  # Replace this with real second model score if available

        # Weighted average for merged score
        merged_score = round((0.6 * clip_score + 0.4 * alt_score), 4)

        return {
            "clip_score": round(clip_score, 4),
            "alt_score": round(alt_score, 4),
            "merged_score": merged_score
        }


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

from typing import List

@app.post("/pets/compare-lost-dog")
async def compare_lost_dog(files: List[UploadFile] = File(...), threshold: float = 0.85):
    results = []
    for file in files:
        print(f"Received file: {file.filename} with threshold: {threshold}")
        if not is_allowed_file(file.filename):
            continue

        contents = await file.read()
        try:
            lost_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except:
            continue  # Skip invalid images

        cropped_lost_image = detect_and_crop_dog_face(lost_image)
        #lost_tensor = image_to_tensor(cropped_lost_image) # Uncomment if you want to use cropped image
        lost_tensor = image_to_tensor(lost_image)
        best_matches = []
        scores = []

        for found_filename in os.listdir(FOUND_DOG_DIR):
            if not is_allowed_file(found_filename):
                continue
            found_path = os.path.join(FOUND_DOG_DIR, found_filename)
            try:
                found_image = Image.open(found_path).convert("RGB")
                found_tensor = image_to_tensor(found_image)
                result = calculate_similarity(lost_tensor, found_tensor)
                merged_score = result["merged_score"]
                print(f"Comparing {file.filename} with {found_filename}: merged = {merged_score}, clip = {result['clip_score']}, alt = {result['alt_score']}")
                
                if merged_score > threshold:
                    scores.append(merged_score)
                    best_matches.append({
                        "filename": found_filename,
                        "merged_score": merged_score,
                        "clip_score": result["clip_score"],
                        "alt_score": result["alt_score"]
                    })
            except Exception as e:
                continue

        best_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        results.append({
            "origin": file.filename,
            "matches": best_matches if best_score >= threshold else None,
            "average_merged_score": best_score
        })

    return results