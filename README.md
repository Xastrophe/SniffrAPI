


# 🐾 Sniffr API

Welcome to **Sniffr** — a smart, image-based lost & found API for dogs. This project uses AI (CLIP + YOLOv8) to match lost dog photos against a collection of found dog images and help bring furry friends back home faster. Built with ❤️ using **FastAPI**, **PyTorch**, and **OpenAI CLIP**.

---

## 🚀 Features

- 🐶 Upload one or more images of a lost dog and compare them against a gallery of found dogs
- 🔍 Uses **CLIP (ViT-L/14)** embeddings for visual similarity
- 📦 Optionally extendable with **DINOv2**, **DogFaceNet**, or breed classifiers
- 🧠 Smart face cropping using OpenCV and fallback detection with YOLOv8
- 📤 Multiple image upload support (via Postman, curl, or UI)

---

## 🛠 Tech Stack

- **FastAPI** – blazing fast web framework
- **PyTorch** – powering deep learning under the hood
- **OpenAI CLIP** – for image embeddings
- **YOLOv8** – for object detection (dog body/face)
- **OpenCV** – fallback face detection
- **PIL / torchvision** – image preprocessing

---

## 📂 Project Structure

Sniffr/
├── main.py               # FastAPI application
├── models/               # Pretrained model files (CLIP + YOLO)
│   ├── ViT-L-14.pt
│   └── yolov8n.pt
├── found_dogs/           # Folder of known found dog images
├── requirements.txt
└── README.md

---

## ▶️ Running the API

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/sniffr-api.git
   cd sniffr-api

	2.	Create a virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows


	3.	Install dependencies

pip install -r requirements.txt


	4.	Run the app with Uvicorn

uvicorn main:app --reload


	5.	Open your browser
Go to http://localhost:8000/docs to test the interactive Swagger UI.

⸻

📸 API Usage

Endpoint

POST /pets/compare-lost-dog

Form-Data Fields

Field	Type	Required	Description
files	File[]	✅	One or more images of the lost dog
threshold	float	❌	(default: 0.85) Match threshold between 0–1

Example in Postman
	•	Set method to POST
	•	URL: http://localhost:8000/pets/compare-lost-dog
	•	Body → form-data:
	•	files → select multiple files (type: File)
	•	threshold → optional (type: Text, e.g., 0.88)

⸻

📦 Requirements
	•	Python 3.8+
	•	Internet (to download CLIP weights on first run)
	•	Some dog photos 🐕

Install dependencies:

pip install -r requirements.txt


⸻

🧠 Future Ideas
	•	Integrate DINOv2 or DogFaceNet for more precise matching
	•	Add breed classification for additional filtering
	•	Store matches in a database (e.g., PostgreSQL)
	•	Notify pet owners via SMS/email on match
	•	Upload endpoint for found dog photos

⸻

❤️ Credits
	•	OpenAI CLIP
	•	Ultralytics YOLOv8
	•	FastAPI
	•	All the good boys and girls 🐕🐾

⸻

📬 License

MIT License – use, modify, and share with attribution!

⸻

“May all lost dogs find their way home.”

---
