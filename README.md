# Multi-Modal-Satellite-Image-Registration

A deep learning-based image registration system that aligns SAR (Sentinel-1) satellite imagery with corresponding Optical (Sentinel-2) imagery using an affine transformation network built with PyTorch, with a FastAPI frontend/backend for interactive inference.
This project uses paired SAR-optical patches from the SEN1-2 style terrain-wise dataset and learns to geometrically align cross-modal remote sensing images.

Project Highlights
->Deep learning based cross-modal image registration
->Aligns SAR image to Optical reference image
->Uses Affine Transformation Prediction Network
->GPU accelerated training using CUDA
->FastAPI web interface for image upload and inference
->Works on terrain-wise Sentinel-1 / Sentinel-2 datasets
->Modular training + inference + deployment pipeline
