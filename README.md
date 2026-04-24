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


