🧠 NeuroScan: Early Brain Tumor Detection

Using YOLOv10 and Deep Learning

🔍 Overview

NeuroScan is an AI-based system designed for early detection and classification of brain tumors from MRI scans.
The project leverages YOLOv10, a state-of-the-art deep learning model, to detect and classify tumors into three types:

Class 0: No Tumor (Healthy Brain Tissue)

Class 1: Benign Tumor (Non-cancerous)

Class 2: Malignant Tumor (Cancerous)

This project aims to assist radiologists and medical professionals in making faster, more accurate diagnoses.

🚀 Features

✅ Tumor detection using YOLOv10
✅ Multiclass classification (No Tumor / Benign / Malignant)
✅ User-friendly GUI built with Tkinter
✅ Model training, testing, and evaluation
✅ Graphs for accuracy, loss, and entropy metrics
✅ Dataset preprocessing and augmentation (for model training)

🧩 System Requirements

Python 3.10+

TensorFlow / PyTorch

OpenCV

NumPy

Matplotlib

Tkinter

YOLOv10 dependencies

⚙️ Installation Guide

Clone the repository

git clone https://github.com/Syed096/NeuroScan-Brain-Tumor-Detection.git
cd NeuroScan-Brain-Tumor-Detection


Create a virtual environment

py -3.10 -m venv venv
venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Run the main application

python main.py

🧠 Dataset

MRI Brain Tumor Dataset (3 classes: No Tumor, Benign, Malignant)

Dataset preprocessed and augmented (rotation, flipping, scaling, shearing, etc.)

Used for YOLOv10 training and evaluation

📊 Results
Metric	Value
Model Accuracy	~96%
Detection Speed	Fast (Real-time capable)
Framework	YOLOv10 + Deep Learning
🖥️ GUI Preview

(You can add screenshots here later — for example, of your Tkinter interface or detection results.)
Example:

📁 Upload MRI Image → 🧠 Detect Tumor → ✅ Display Result (Type & Confidence)

🧑‍💻 Author

Amina Syed
🎓Software Engineer
📍 Pakistan
💡 Passionate about Artificial Intelligence & Space Technology
📧 aminasyed096@gmail.com

🌟 How to Support

If you found this project helpful, please star this repository ⭐ on GitHub to show your support!
