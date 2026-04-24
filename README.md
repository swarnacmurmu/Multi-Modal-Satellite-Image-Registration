# Multimodal Satellite Image Registration

A deep learning project that aligns **SAR (Sentinel-1)** satellite images with corresponding **Optical (Sentinel-2)** images using an **Affine Registration Network** built with **PyTorch**, integrated with a **FastAPI web interface** for inference.

---

# Overview

Satellite images from different sensors capture the same region in different ways:

- **SAR** works in all weather and day/night conditions.
- **Optical** captures rich visual scene details.

This project performs **multimodal image registration** to spatially align SAR images with Optical reference images for improved analysis.

---

# Dataset

Terrain-wise paired Sentinel dataset:

```text
data/raw/v_2/
├── agri/
│   ├── s1/   (SAR)
│   └── s2/   (Optical)
├── barrenland/
│   ├── s1/
│   └── s2/
├── grassland/
│   ├── s1/
│   └── s2/
└── urban/
    ├── s1/
    └── s2/


```
# Tech Stack

```text
Programming Language : Python 3.11
Deep Learning       : PyTorch
GPU Acceleration    : CUDA
Computer Vision     : OpenCV
Numerical Computing : NumPy
Backend Framework   : FastAPI
ASGI Server         : Uvicorn
Frontend            : HTML / CSS / JavaScript
Image Handling      : Pillow
Visualization       : Matplotlib
Progress Tracking   : tqdm
Configuration       : YAML
```

# Project Architecture

```text
Input Dataset
(SAR + Optical Image Pairs)
        ↓
Data Validation
(Check paired folders / file count)
        ↓
Preprocessing
(Resize / Normalize / Grayscale)
        ↓
Dataset Split
(Train / Validation / Test)
        ↓
Synthetic Misalignment
(Rotation / Translation / Scale)
        ↓
Affine Registration Network
(CNN Feature Extractor + Regressor)
        ↓
Predict 2x3 Affine Matrix
        ↓
Spatial Warping Layer
(Align SAR → Optical)
        ↓
Loss Computation
(Edge Loss + Theta Loss)
        ↓
Best Trained Model Saved
        ↓
Inference Pipeline
(Upload Pair / Predict / Register)
        ↓
FastAPI Web Application
(Display Registered Output)
```


# Folder Structure

```text
Image_Registration/
│
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Preprocessed images
│   └── splits/              # Train / Val / Test JSON files
│
├── checkpoints/            # Saved trained models
│
├── outputs/
│   ├── inference/          # Prediction results
│   └── training_samples/   # Sample training outputs
│
├── training/
│   ├── dataset.py          # Custom PyTorch dataset loader
│   ├── model.py            # Registration model
│   ├── losses.py           # Loss functions
│   ├── preprocess.py       # Image preprocessing
│   ├── split_dataset.py    # Dataset split script
│   ├── train.py            # Training pipeline
│   └── validate.py         # Model evaluation
│
├── inference/
│   └── predict.py          # Single pair prediction script
│
├── backend/
│   ├── app.py              # FastAPI backend
│   ├── services/           # Prediction service
│   ├── templates/          # HTML frontend
│   └── static/             # CSS / JS files
│
├── scripts/
│   ├── check_gpu.py        # GPU verification
│   └── check_dataset.py    # Dataset validation
│
├── config.yaml            # Project configuration
├── requirements.txt       # Dependencies
└── run_app.py             # Launch web application
```


# Model Input / Output

```text
Input
-----
Moving Image : SAR
Fixed Image  : Optical

Output
------
Registered SAR Image
Predicted Affine Matrix
```



Edge Loss   : Aligns structural features
Theta Loss  : Measures affine parameter accuracy

Total Loss  = Edge Loss + λ × Theta Loss



# How To Run
```text
1. Install Dependencies:
   pip install -r requirements.txt

2. Check GPU:
   python scripts/check_gpu.py

3. Check Dataset:
   python scripts/check_dataset.py

4. Preprocess Dataset:
   python -m training.preprocess

5. Split Dataset:
   python -m training.split_dataset

6. Train Model:
   python -m training.train

7. Validate Model:
   python -m training.validate

8. Run Inference:
   python -m inference.predict --moving path_to_sar_image --fixed path_to_optical_image

9. Run Web App:
   python run_app.py

10. Open in Browser:
    http://127.0.0.1:8000

```
