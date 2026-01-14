# 🚘 YOLO Number Plate Detection

This project implements a deep learning–based system for detecting vehicle number plates using the **YOLO (You Only Look Once)** object detection architecture.  
It enables fast and accurate detection of license plates from images and videos, making it suitable for real-time intelligent transportation and surveillance systems.

---

## 🧩 Problem Statement

Vehicle number plate detection is a critical component in applications such as:

- 🚓 Traffic surveillance and law enforcement  
- 🅿️ Parking management systems  
- 🚗 Toll collection automation  
- 🌆 Smart city infrastructure  

Traditional computer vision approaches struggle with variations in lighting, camera angles, motion blur, and background noise. A robust, scalable, and real-time solution is required to handle these challenges efficiently.

---

## ✅ Solution Implemented

This project delivers an **AI-powered number plate detection system** that:

✔ Uses the **YOLO object detection model** for fast and accurate detection  
✔ Supports **image** inputs  
✔ Provides **real-time inference**  
✔ Is modular and extensible for OCR or tracking integration  
✔ Can be retrained with custom datasets  

---

## 🚀 Features

🔍 Real-time vehicle number plate detection  
📸 Image support  
⚡ High-speed inference using YOLO  
🧠 Custom training support  
📦 Clean and modular architecture  
🛠️ Easily extendable for OCR integration  

---

## 🧠 Model & Approach

- YOLO (You Only Look Once) deep learning architecture  
- Custom annotated dataset in YOLO format  
- Bounding box prediction for license plates  
- Training and inference pipelines using PyTorch and OpenCV  

The trained model detects number plates by predicting bounding boxes around plate regions in each frame.

---

## 📁 Project Structure
```text
YOLO_Number_Plate_Detection/
│
├── api/ # API or inference scripts
├── artifacts/ # Trained model weights
├── config/ # Configuration files
├── data/ # Dataset and annotations
├── experiment/ # Training experiments and logs
├── src/
│ └── yolo_number_plate_detection/
│     └── Components
|     └── Configuration
|     └── Constants
|     └── Pipeline
|     └── Utils
├── poetry.lock # Python dependencies
├── pyproject.toml
├── README.md # Project documentation
└── LICENSE # Apache License 2.0
```

## ▶️ Entry Point

The main application entry point is:

```python
app/app.py
```

This file initializes the fastapi app, we can use sqaggerUI to upload the image and perform prediction.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sumitjoshi10/YOLO_Number_Plate_Detection.git
cd YOLO_Number_Plate_Detection
```

### 2️⃣ Install Poetry (if not already installed)

```bash
pip install poetry
poetry --version
```

### 3️⃣ Install dependencies

```bash
poetry install
```
### 4️⃣ Activate the virtual environment

```bash
poetry env activate
```

### 5️⃣ Run FastAPI server

```bash
poetry run uvicorn api.app:app --reload
```

---

## ▶️ Usage
🔹 Detect number plate via API

Send a POST request to /predict with an image file:


---

## 🛠️ Tech Stack

- **Python**
- **Poetry**
- **FastAPI**
- **PyTorch**
- **YOLO (Ultralytics)**
- **OpenCV**
- **NumPy**

---

## 🔮 Future Enhancements

- 🔠 OCR integration for reading plate numbers
- 📡 Live camera / RTSP stream support
- 📦 Docker deployment
- 📊 Detection analytics dashboard
- 🌍 Multi-country plate support

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new feature branch (`feature/new-feature`)
3. Commit your changes
4. Push and open a Pull Request

---

📜 **License**: Apache License 2.0

---

👤 **Author** Sumit Joshi
🔗 GitHub: https://github.com/sumitjoshi10